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

logger = logging.getLogger(__name__)


@dataclass
class ResourceThresholds:
    """Configurable resource thresholds."""

    # CPU thresholds
    cpu_high: float = 0.85  # Warns at 85% CPU usage
    cpu_critical: float = 0.95  # Critical at 95% CPU usage

    # Memory thresholds
    memory_high: float = 0.80  # Warns at 80% memory usage
    memory_critical: float = 0.90  # Critical at 90% memory usage

    # GPU thresholds
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
    """Enhanced resource manager with adaptive monitoring and optimization."""

    def __init__(self, monitoring_interval: int = 5, metrics_history_hours: int = 1):
        """Initialize the resource manager with monitoring settings.

        Args:
            monitoring_interval: How often to collect metrics (in seconds)
            metrics_history_hours: How long to keep metrics history (in hours)
        """
        # Initialize core attributes
        self._lock = Lock()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._metrics_history: List[ResourceMetrics] = []
        self.monitoring_interval = monitoring_interval
        self._metrics_retention = timedelta(hours=metrics_history_hours)
        self.thresholds = ResourceThresholds()

        # Initialize system information
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total

        # Initialize GPU capabilities
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0

        # Initialize NVML for advanced GPU monitoring
        self.nvml_initialized = False
        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("NVIDIA Management Library initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA Management Library: {e}")

        logger.info(
            f"Resource Manager initialized with {self.cpu_count} CPUs, "
            f"{self.total_memory / (1024**3):.1f}GB RAM, "
            f"{self.gpu_count} GPUs"
        )

    async def _collect_gpu_metrics(
        self,
    ) -> Optional[Dict[int, Dict[str, Union[str, float, int]]]]:
        """Collect detailed GPU metrics using NVML.

        Returns:
            Dictionary containing detailed metrics for each GPU, or None if collection fails
        """
        if not self.gpu_available or not self.nvml_initialized:
            return None

        gpu_metrics = {}
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Base metrics that should always be available
                try:
                    metrics = {}
                    name = pynvml.nvmlDeviceGetName(handle)
                    # Handle the case where name is bytes
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")

                    metrics = {
                        "name": name,
                        "memory_total": memory_info.total / (1024**3),
                        "memory_used": memory_info.used / (1024**3),
                        "memory_free": memory_info.free / (1024**3),
                    }

                except Exception as e:
                    logger.debug(f"Could not get GPU name: {e}")
                    metrics["name"] = "Unknown GPU"

                # Extended metrics that might not be available on all GPUs
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics.update(
                        {
                            "gpu_utilization": utilization.gpu,
                            "memory_utilization": utilization.memory,
                        }
                    )
                except pynvml.NVMLError as e:
                    logger.debug(f"Could not get GPU utilization for GPU {i}: {e}")

                try:
                    metrics["temperature"] = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except pynvml.NVMLError as e:
                    logger.debug(f"Could not get temperature for GPU {i}: {e}")

                try:
                    metrics["power_usage"] = (
                        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    )
                except pynvml.NVMLError as e:
                    logger.debug(f"Could not get power usage for GPU {i}: {e}")

                try:
                    metrics["fan_speed"] = pynvml.nvmlDeviceGetFanSpeed(handle)
                except pynvml.NVMLError as e:
                    logger.debug(f"Could not get fan speed for GPU {i}: {e}")

                try:
                    metrics["graphics_clock"] = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_GRAPHICS
                    )
                    metrics["memory_clock"] = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_MEM
                    )
                except pynvml.NVMLError as e:
                    logger.debug(f"Could not get clock speeds for GPU {i}: {e}")

                gpu_metrics[i] = metrics

            return gpu_metrics
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
            return None

    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect comprehensive system metrics including CPU, memory, disk, and GPU."""
        try:
            # Collect basic system metrics
            vm = psutil.virtual_memory()
            cpu_freq = psutil.cpu_freq()
            disk = psutil.disk_usage("/")
            io = psutil.disk_io_counters()

            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=psutil.cpu_percent(),
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

            return metrics

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            raise

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

    async def start_monitoring(self):
        """Start the resource monitoring process."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_resources())
        logger.info("Resource monitoring started")

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
                self._check_thresholds(metrics)

                # Wait for next collection cycle
                await asyncio.sleep(adjusted_interval)

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)

    def _adjust_monitoring_interval(self, metrics: ResourceMetrics) -> float:
        """Dynamically adjust the monitoring interval based on system load."""
        base_interval = self.monitoring_interval

        # Increase monitoring frequency under high load
        if metrics.is_cpu_critical or metrics.is_memory_critical:
            return max(1, base_interval / 2)

        # Check GPU utilization
        if metrics.gpu_metrics:
            for gpu_info in metrics.gpu_metrics.values():
                if "gpu_utilization" in gpu_info and gpu_info["gpu_utilization"] > 90:
                    return max(1, base_interval / 2)

        # Decrease frequency under low load
        if (
            metrics.cpu_percent < 50
            and metrics.memory_percent < 50
            and (
                not metrics.gpu_metrics
                or all(
                    gpu_info.get("gpu_utilization", 0) < 50
                    for gpu_info in metrics.gpu_metrics.values()
                )
            )
        ):
            return min(10, base_interval * 1.5)

        return base_interval

    def get_optimal_batch_size(self, sample_size_bytes: int) -> int:
        """Calculate the optimal batch size based on current resource availability.

        This method takes into account:
        - Available system and GPU memory
        - Current CPU utilization
        - GPU utilization (if available)
        - Historical performance metrics

        Args:
            sample_size_bytes: Size of a single sample in bytes

        Returns:
            int: Optimal batch size for processing
        """
        try:
            # If no metrics history available, use default calculation
            if not self._metrics_history:
                return self._calculate_default_batch_size(sample_size_bytes)

            recent_metrics = self._metrics_history[-1]

            # Calculate resource utilization factors (0.0 to 1.0)
            memory_factor = 1 - (recent_metrics.memory_percent / 100)
            cpu_factor = 1 - (recent_metrics.cpu_percent / 100)

            # Calculate base batch size based on available memory
            if self.gpu_available and self.nvml_initialized:
                try:
                    # Get GPU memory information from NVML
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    available_memory = memory_info.free
                    base_batch_size = int((available_memory * 0.7) / sample_size_bytes)
                except Exception as e:
                    logger.error(f"Error getting GPU memory info: {e}")
                    # Fallback to system memory if GPU memory query fails
                    available_memory = recent_metrics.memory_available * (1024**3)
                    base_batch_size = int((available_memory * 0.7) / sample_size_bytes)
            else:
                # Use system memory if GPU is not available
                available_memory = recent_metrics.memory_available * (1024**3)
                base_batch_size = int((available_memory * 0.7) / sample_size_bytes)

            # Adjust batch size based on CPU and memory utilization
            adjusted_batch_size = int(base_batch_size * min(memory_factor, cpu_factor))

            # Further adjust based on GPU utilization if available
            if recent_metrics.gpu_metrics:
                gpu_utilization_factor = 1.0
                for gpu_info in recent_metrics.gpu_metrics.values():
                    if "gpu_utilization" in gpu_info:
                        # Reduce batch size if GPU is heavily utilized
                        gpu_utilization_factor = min(
                            gpu_utilization_factor,
                            1 - (gpu_info["gpu_utilization"] / 100),
                        )
                adjusted_batch_size = int(adjusted_batch_size * gpu_utilization_factor)

            # Ensure batch size stays within reasonable bounds
            return max(1, min(adjusted_batch_size, 1000))

        except Exception as e:
            logger.error(f"Error calculating batch size: {e}")
            return self._calculate_default_batch_size(sample_size_bytes)

    def _calculate_default_batch_size(self, sample_size_bytes: int) -> int:
        """Calculate a conservative default batch size when optimal calculation fails.

        Args:
            sample_size_bytes: Size of a single sample in bytes

        Returns:
            int: Conservative default batch size
        """
        # Use 10% of total system memory as a safe default
        total_memory = psutil.virtual_memory().total
        return max(1, min(32, int((total_memory * 0.1) / sample_size_bytes)))

    async def export_metrics(self, filepath: str):
        """Export collected metrics history to a JSON file.

        Args:
            filepath: Path where the metrics JSON file will be saved
        """
        try:
            # Convert metrics history to serializable format
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

            # Write metrics to file asynchronously
            async with aiofiles.open(filepath, "w") as f:
                await f.write(json.dumps(metrics_data, indent=2))

            logger.info(f"Resource metrics exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    async def cleanup(self):
        """Perform comprehensive cleanup of all resources.

        This method:
        1. Stops the monitoring process
        2. Clears GPU memory cache if available
        3. Shuts down NVML if it was initialized
        4. Forces garbage collection
        """
        try:
            # Stop monitoring first
            await self.stop_monitoring()

            # Clean up GPU resources
            if self.gpu_available:
                try:
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    logger.debug("CUDA memory cache cleared")
                except Exception as e:
                    logger.error(f"Error clearing CUDA cache: {e}")

                # Shutdown NVML if it was initialized
                if hasattr(self, "nvml_initialized") and self.nvml_initialized:
                    try:
                        pynvml.nvmlShutdown()
                        logger.info("NVIDIA Management Library shut down successfully")
                    except Exception as e:
                        logger.error(f"Error shutting down NVML: {e}")

            # Force garbage collection
            gc.collect()
            logger.info("Resource manager cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
