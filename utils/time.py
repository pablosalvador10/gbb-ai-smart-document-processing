import functools
import logging
import threading
import time

import numpy as np
from tabulate import tabulate

from utils.ml_logging import get_logger

logger = get_logger()

import time


class RunMeasure:
    def __init__(self):
        self.execution_times = []

    def measure_time(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.execution_times.append(elapsed_time)
            print(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds")
            return result

        return wrapper

    def get_execution_times(self):
        return self.execution_times

    def calculate_statistics(self):
        if not self.execution_times:
            print("No execution times recorded.")
            return

        execution_times = self.execution_times
        avg_time_per_run = np.mean(execution_times)
        median_time_per_run = np.median(execution_times)
        min_time = np.min(execution_times)
        max_time = np.max(execution_times)
        p95_time_per_run = np.percentile(execution_times, 95)
        p99_time_per_run = np.percentile(execution_times, 99)

        table = [
            ["Statistic", "Time (seconds)"],
            ["Average time per run", f"{avg_time_per_run:.4f}"],
            ["Median time per run", f"{median_time_per_run:.4f}"],
            ["Minimum time per run", f"{min_time:.4f}"],
            ["Maximum time per run", f"{max_time:.4f}"],
            ["95th percentile time per run", f"{p95_time_per_run:.4f}"],
            ["99th percentile time per run", f"{p99_time_per_run:.4f}"],
        ]

        print(tabulate(table, headers="firstrow", tablefmt="grid"))


# Function to calculate and print statistics
def calculate_statistics(successful_times, error_count):
    if successful_times:
        avg_time_per_run = np.mean(successful_times)
        median_time_per_run = np.median(successful_times)
        min_time = np.min(successful_times)
        max_time = np.max(successful_times)
        p95_time_per_run = np.percentile(successful_times, 95)
        p99_time_per_run = np.percentile(successful_times, 99)

        table = [
            ["Statistic", "Value"],
            ["Average time per run (seconds)", f"{avg_time_per_run:.4f}"],
            ["Median time per run (seconds)", f"{median_time_per_run:.4f}"],
            ["Minimum time per run (seconds)", f"{min_time:.4f}"],
            ["Maximum time per run (seconds)", f"{max_time:.4f}"],
            ["95th percentile time per run (seconds)", f"{p95_time_per_run:.4f}"],
            ["99th percentile time per run (seconds)", f"{p99_time_per_run:.4f}"],
            ["Number of errors", error_count],
        ]

        print(tabulate(table, headers="firstrow", tablefmt="grid"))
    else:
        logger.error("No successful execution times recorded.")
