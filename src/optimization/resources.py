import asyncio
from dataclasses import dataclass
import logging
import psutil
import torch
import gc
import pynvml
from threading import Lock
from typing import Dict, Optional, List, Union
from datetime import datetime, timedelta
import aiofiles
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from numba import jit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResourceThresholds:
    """Configurable resource thresholds."""

    cpu_high: float = 0.85  # Warns at 85% CPU usage
    cpu_critical: float = 0.95  # Critical at 95% CPU usage
    memory_high: float = 0.90  # Warns at 80% memory usage
    memory_critical: float = 0.95  # Critical at 90% memory usage
    gpu_memory_high: float = 0.85  # Warns at 85% GPU memory usage
    gpu_memory_critical: float = 0.95  # Critical at 95% GPU memory usage
    gpu_temp_high: float = 80.0  # Warns at 80°C
    gpu_temp_critical: float = 90.0  # Critical at 90°C


@dataclass
class ResourceMetrics:
    """Enhanced resource metrics with detailed system information."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: float  # in GB
    memory_total: float  # in GB
    cpu_count: int
    cpu_frequency: float  # current CPU frequency
    disk_usage: float  # disk usage percentage
    gpu_metrics: Optional[Dict[int, Dict[str, Union[str, float, int]]]] = None
    io_counters: Optional[Dict[str, int]] = None

    @property
    def is_cpu_critical(self) -> bool:
        """Check if CPU usage is at critical level."""
        return self.cpu_percent > ResourceThresholds.cpu_critical * 100

    @property
    def is_memory_critical(self) -> bool:
        """Check if memory usage is at critical level."""
        return self.memory_percent > ResourceThresholds.memory_critical * 100


class ResourceManager:
    def __init__(self, monitoring_interval: float = 1, metrics_history_hours: int = 1):
        """Initialize the resource manager with monitoring settings."""
        self._lock = Lock()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._metrics_history: List[ResourceMetrics] = []
        self.monitoring_interval = monitoring_interval
        self._metrics_retention = timedelta(hours=metrics_history_hours)
        self.thresholds = ResourceThresholds()

        self.cpu_count = psutil.cpu_count(
            logical=False
        )  # Use physical cores for more accurate CPU metrics
        self.total_memory = psutil.virtual_memory().total

        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0

        self.nvml_initialized = False
        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("NVIDIA Management Library initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA Management Library: {e}")

        logger.info(
            f"Resource Manager initialized with {self.cpu_count} physical CPUs, {self.total_memory / (1024**3):.1f}GB RAM, {self.gpu_count} GPUs"
        )

    async def _collect_gpu_metrics(
        self,
    ) -> Optional[Dict[int, Dict[str, Union[str, float, int]]]]:
        if not self.gpu_available or not self.nvml_initialized:
            return None

        gpu_metrics = {}
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                metrics = self._get_gpu_metrics(handle)
                gpu_metrics[i] = metrics
            except Exception as e:
                logger.error(f"Error collecting GPU metrics for GPU {i}: {e}")
        return gpu_metrics if gpu_metrics else None

    def _get_gpu_metrics(self, handle):
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics = {
            "name": pynvml.nvmlDeviceGetName(
                handle
            ),  # Remove .decode("utf-8") if it's already a string
            "memory_total": memory_info.total / (1024**3),
            "memory_used": memory_info.used / (1024**3),
            "memory_free": memory_info.free / (1024**3),
        }

        for metric_name, func, args in [
            ("gpu_utilization", pynvml.nvmlDeviceGetUtilizationRates, [handle]),
            (
                "temperature",
                pynvml.nvmlDeviceGetTemperature,
                [handle, pynvml.NVML_TEMPERATURE_GPU],
            ),
            ("power_usage", pynvml.nvmlDeviceGetPowerUsage, [handle]),
            ("fan_speed", pynvml.nvmlDeviceGetFanSpeed, [handle]),
            (
                "graphics_clock",
                pynvml.nvmlDeviceGetClockInfo,
                [handle, pynvml.NVML_CLOCK_GRAPHICS],
            ),
            (
                "memory_clock",
                pynvml.nvmlDeviceGetClockInfo,
                [handle, pynvml.NVML_CLOCK_MEM],
            ),
        ]:
            try:
                if metric_name == "power_usage":
                    metrics[metric_name] = func(*args) / 1000.0  # Convert to Watts
                elif metric_name == "gpu_utilization":
                    utilization = func(*args)
                    metrics[metric_name] = utilization.gpu
                    metrics["memory_utilization"] = utilization.memory
                else:
                    metrics[metric_name] = func(*args)
            except Exception:
                logger.debug(f"Could not get {metric_name}")

        return metrics

    async def _collect_metrics(self) -> ResourceMetrics:
        try:
            vm = psutil.virtual_memory()
            cpu_freq = psutil.cpu_freq()
            disk = psutil.disk_usage("/")
            io = psutil.disk_io_counters()

            # Correct usage of cpu_percent
            cpu_percent = psutil.cpu_percent(interval=0.1)

            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=vm.percent,
                memory_available=vm.available / (1024**3),
                memory_total=vm.total / (1024**3),
                cpu_count=self.cpu_count,
                cpu_frequency=cpu_freq.current if cpu_freq else 0,
                disk_usage=disk.percent,
                io_counters=(
                    {"read_bytes": io.read_bytes, "write_bytes": io.write_bytes}
                    if io
                    else None
                ),
                gpu_metrics=await self._collect_gpu_metrics(),
            )
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            raise

    async def start_monitoring(self):
        """Start the resource monitoring process."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_resources())
        logger.info("Resource monitoring started")

    async def _monitor_resources(self):
        while self._monitoring:
            try:
                metrics = await self._collect_metrics()
                self._metrics_history.append(metrics)
                self._metrics_history = [
                    m
                    for m in self._metrics_history
                    if datetime.now() - m.timestamp < self._metrics_retention
                ]

                adjusted_interval = self._adjust_monitoring_interval(metrics)
                # self._check_thresholds(metrics)  # Ensure this method call is here
                await asyncio.sleep(adjusted_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def stop_monitoring(self):
        """Stop the resource monitoring process gracefully."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")

    def _check_thresholds(self, metrics: ResourceMetrics):
        """Check all resource thresholds and log appropriate warnings."""
        # CPU threshold checks
        if metrics.cpu_percent > self.thresholds.cpu_critical * 100:
            logger.warning(f"Critical CPU usage: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent > self.thresholds.cpu_high * 100:
            logger.info(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        # Memory threshold checks
        if metrics.memory_percent > self.thresholds.memory_critical * 100:
            logger.warning(f"Critical memory usage: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent > self.thresholds.memory_high * 100:
            logger.info(f"High memory usage: {metrics.memory_percent:.1f}%")

        # GPU threshold checks
        if metrics.gpu_metrics:
            for gpu_id, gpu_info in metrics.gpu_metrics.items():
                # Check GPU memory usage
                if "memory_used" in gpu_info and "memory_total" in gpu_info:
                    memory_percent = (
                        gpu_info["memory_used"] / gpu_info["memory_total"]
                    ) * 100
                    if memory_percent > self.thresholds.gpu_memory_critical * 100:
                        logger.warning(
                            f"Critical GPU {gpu_id} memory usage: {memory_percent:.1f}%"
                        )
                    elif memory_percent > self.thresholds.gpu_memory_high * 100:
                        logger.info(
                            f"High GPU {gpu_id} memory usage: {memory_percent:.1f}%"
                        )

                # Check GPU temperature
                if "temperature" in gpu_info:
                    temp = gpu_info["temperature"]
                    if temp > self.thresholds.gpu_temp_critical:
                        logger.warning(f"Critical GPU {gpu_id} temperature: {temp}°C")
                    elif temp > self.thresholds.gpu_temp_high:
                        logger.info(f"High GPU {gpu_id} temperature: {temp}°C")

    async def _monitor_resources(self):
        """Main monitoring loop that collects and processes metrics periodically."""
        while self._monitoring:
            try:
                # Collect new metrics
                metrics = await self._collect_metrics()
                self._metrics_history.append(metrics)

                # Clean up old metrics
                current_time = datetime.now()
                self._metrics_history = [
                    m
                    for m in self._metrics_history
                    if current_time - m.timestamp < self._metrics_retention
                ]

                # Adjust monitoring frequency based on system load
                adjusted_interval = self._adjust_monitoring_interval(metrics)

                # Check resource thresholds
                # self._check_thresholds(metrics)

                # Wait for next collection cycle
                await asyncio.sleep(adjusted_interval)

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)

    def _adjust_monitoring_interval(self, metrics: ResourceMetrics) -> float:
        """Dynamically adjust the monitoring interval based on system load."""
        base_interval = self.monitoring_interval

        if metrics.is_cpu_critical or metrics.is_memory_critical:
            return max(0.1, base_interval / 2)  # More frequent checks under high load

        if metrics.gpu_metrics:
            for gpu_info in metrics.gpu_metrics.values():
                if gpu_info.get("gpu_utilization", 0) > 90:
                    return max(0.1, base_interval / 2)

        if (
            metrics.cpu_percent < 50
            and metrics.memory_percent < 50
            and (
                not metrics.gpu_metrics
                or all(
                    info.get("gpu_utilization", 0) < 50
                    for info in metrics.gpu_metrics.values()
                )
            )
        ):
            return min(2, base_interval * 1.5)

        return base_interval

    def get_optimal_batch_size(self, sample_size_bytes: int) -> int:
        """Calculate the optimal batch size based on current resource availability."""
        try:
            if not self._metrics_history:
                return self._calculate_default_batch_size(sample_size_bytes)

            recent_metrics = self._metrics_history[-1]

            memory_factor = 1 - (recent_metrics.memory_percent / 100)
            cpu_factor = 1 - (recent_metrics.cpu_percent / 100)

            if self.gpu_available and self.nvml_initialized:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    available_memory = memory_info.free
                    base_batch_size = int((available_memory * 0.9) / sample_size_bytes)
                except Exception as e:
                    logger.error(f"Error getting GPU memory info: {e}")
                    available_memory = recent_metrics.memory_available * (1024**3)
                    base_batch_size = int((available_memory * 0.7) / sample_size_bytes)
            else:
                available_memory = recent_metrics.memory_available * (1024**3)
                base_batch_size = int((available_memory * 0.7) / sample_size_bytes)

            adjusted_batch_size = int(base_batch_size * min(memory_factor, cpu_factor))

            if recent_metrics.gpu_metrics:
                gpu_utilization_factor = 1.0
                for gpu_info in recent_metrics.gpu_metrics.values():
                    if "gpu_utilization" in gpu_info:
                        gpu_utilization_factor = min(
                            gpu_utilization_factor,
                            1 - (gpu_info["gpu_utilization"] / 100),
                        )
                adjusted_batch_size = int(adjusted_batch_size * gpu_utilization_factor)

            return max(1, min(adjusted_batch_size, 1000))

        except Exception as e:
            logger.error(f"Error calculating batch size: {e}")
            return self._calculate_default_batch_size(sample_size_bytes)

    def _calculate_default_batch_size(self, sample_size_bytes: int) -> int:
        """Calculate a conservative default batch size when optimal calculation fails."""
        total_memory = psutil.virtual_memory().total
        return max(1, min(32, int((total_memory * 0.1) / sample_size_bytes)))

    async def export_metrics(self, filepath: str):
        """Export collected metrics history to a JSON file."""
        try:
            metrics_data = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "memory_available": m.memory_available,
                    "memory_total": m.memory_total,
                    "cpu_frequency": m.cpu_frequency,
                    "disk_usage": m.disk_usage,
                    "gpu_metrics": m.gpu_metrics,
                    "io_counters": m.io_counters,
                }
                for m in self._metrics_history
            ]

            async with aiofiles.open(filepath, "w") as f:
                await f.write(json.dumps(metrics_data, indent=2))

            # logger.info(f"Resource metrics exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    async def cleanup(self):
        """Perform comprehensive cleanup of all resources."""
        try:
            await self.stop_monitoring()

            if self.gpu_available:
                try:
                    torch.cuda.empty_cache()
                    # logger.debug("CUDA memory cache cleared")
                except Exception as e:
                    logger.error(f"Error clearing CUDA cache: {e}")

                if self.nvml_initialized:
                    try:
                        pynvml.nvmlShutdown()
                        # logger.info("NVIDIA Management Library shut down successfully")
                    except Exception as e:
                        logger.error(f"Error shutting down NVML: {e}")

            gc.collect()
            # logger.info("Resource manager cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example usage
if __name__ == "__main__":

    async def main():
        manager = ResourceManager()
        await manager.start_monitoring()
        await asyncio.sleep(10)  # Monitor for 10 seconds
        await manager.stop_monitoring()
        await manager.export_metrics("metrics.json")
        await manager.cleanup()

    asyncio.run(main())
