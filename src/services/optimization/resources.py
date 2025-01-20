"""
Resource management module for system monitoring and optimization.

This module provides classes for monitoring and managing system resources
including CPU, memory, and GPU utilization.
"""

import asyncio
import gc
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, Optional, List, Union
import logging

# Third-party imports
import psutil
import pynvml
import torch


logger = logging.getLogger(__name__)


@dataclass
class ResourceThresholds:
    """Configurable resource thresholds."""

    cpu_high: float = 85.0  # Changed to percentage
    cpu_critical: float = 95.0
    memory_high: float = 90.0
    memory_critical: float = 95.0
    gpu_memory_high: float = 85.0
    gpu_memory_critical: float = 95.0
    gpu_temp_high: float = 80.0
    gpu_temp_critical: float = 90.0


@dataclass
class ResourceMetrics:
    """Resource metrics with detailed system information."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: float
    memory_total: float
    cpu_count: int
    cpu_frequency: float
    disk_usage: float
    gpu_metrics: Optional[Dict[int, Dict[str, Union[str, float, int]]]] = (
        None
    )
    io_counters: Optional[Dict[str, int]] = None

    def is_cpu_critical(self, thresholds: ResourceThresholds) -> bool:
        """Check if CPU usage is at critical level."""
        return self.cpu_percent > thresholds.cpu_critical

    def is_memory_critical(self, thresholds: ResourceThresholds) -> bool:
        """Check if memory usage is at critical level."""
        return self.memory_percent > thresholds.memory_critical


class ResourceManager:
    """Manages system resources including CPU, memory, and GPU utilization."""

    def __init__(
        self,
        monitoring_interval: float = 1.0,
        metrics_history_hours: int = 1,
    ):
        """Initialize the resource manager.

        Args:
            monitoring_interval: Time between metric collections in seconds
            metrics_history_hours: How many hours of metrics to retain

        Raises:
            RuntimeError: If system resource access fails
        """
        self._lock = Lock()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._metrics_history: List[ResourceMetrics] = []
        self.monitoring_interval = monitoring_interval
        self._metrics_retention = timedelta(hours=metrics_history_hours)
        self.thresholds = ResourceThresholds()

        # System info initialization
        self.cpu_count = psutil.cpu_count(logical=False)
        self.total_memory = psutil.virtual_memory().total

        # GPU initialization with proper error handling
        self.gpu_available = False
        self.gpu_count = 0
        self.nvml_initialized = False

        try:
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_count = torch.cuda.device_count()
                # Initialize NVIDIA Management Library
                pynvml.nvmlInit()
                self.nvml_initialized = True
                logger.info(
                    "NVIDIA Management Library initialized successfully with %d GPUs",
                    self.gpu_count,
                )

                # Set optimal GPU settings
                for i in range(self.gpu_count):
                    torch.cuda.set_device(i)
                    # Enable TF32 for better performance on Ampere GPUs
                    if torch.cuda.get_device_capability()[0] >= 8:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    # Enable cuDNN autotuner
                    torch.backends.cudnn.benchmark = True

        except Exception as e:
            logger.warning("GPU initialization failed: %s", str(e))
            self.gpu_available = False
            self.gpu_count = 0

        logger.info(
            "Resource Manager initialized with %d physical CPUs, %.1fGB RAM, %d GPUs",
            self.cpu_count,
            self.total_memory / (1024**3),
            self.gpu_count,
        )

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
            except Exception as e:
                logger.error("Error stopping monitoring: %s", str(e))

        logger.info("Resource monitoring stopped")

    async def cleanup(self):
        """Clean up resources and shut down monitoring."""
        try:
            await self.stop_monitoring()

            if self.gpu_available:
                try:
                    torch.cuda.empty_cache()
                    if self.nvml_initialized:
                        pynvml.nvmlShutdown()
                except Exception as e:
                    logger.error("Error during GPU cleanup: %s", str(e))

            gc.collect()
            logger.info("Resource manager cleanup completed")
            return True

        except Exception as e:
            logger.error("Error during cleanup: %s", str(e))


async def main():
    """Main function to demonstrate ResourceManager usage."""
    try:
        manager = ResourceManager()
        await manager.cleanup()
        logger.info("ResourceManager demo completed successfully")
    except Exception as e:
        logger.error("Error in main: %s", str(e))


if __name__ == "__main__":
    asyncio.run(main())
