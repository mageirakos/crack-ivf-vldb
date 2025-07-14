import math

def calculate_execution_time(operations, memory_bytes, compute_peak_flops, memory_peak_bw, num_cores, max_cores, bw_utilization=1.0):
    """
    Calculate kernel execution time based on operations, memory transfer size, and number of cores used.
    
    Parameters:
    - operations: Total number of operations (e.g., FLOPs).
    - memory_bytes: Total number of bytes to transfer.
    - compute_peak_flops: Peak compute performance in FLOPs/s.
    - memory_peak_bw: Peak memory bandwidth in bytes/s.
    - num_cores: Number of cores utilized for the computation.
    - max_cores: Maximum number of available cores.
    - bw_utilization: Fraction of peak memory bandwidth utilized (default 100%).
    
    Returns:
    - Execution time in seconds, assuming peak utilization of resources.
    """
    # Adjust compute performance based on the number of cores utilized
    compute_time = operations / (compute_peak_flops * (num_cores / max_cores))
    
    # Adjust memory bandwidth utilization
    memory_time = memory_bytes / (memory_peak_bw * bw_utilization)
    
    # Total execution time (assuming no overlap)
    total_time = compute_time + memory_time
    
    return compute_time, memory_time, total_time


def main():
    # Hardware specifications (modify these based on your CPU)
    # TODO: figure out a way to set it
    compute_peak_flops = 94.99e12  # Peak compute performance (FLOPs/s) for AMD EPYC 7V13
    memory_peak_bw = 409.6e9      # Peak memory bandwidth (Bytes/s) for DDR4-3200 (8 channels)
    
    # compute_peak_flops = 474.56e9  # 474.56 GFLOPS, Peak compute (FLOPs/s)
    # memory_peak_bw = 256e9        # 256 GB/s, Peak memory bandwidth (Bytes/s)

    # compute_peak_flops = 2.45e12  # 2.45 TFLOPS, Peak compute (FLOPs/s)
    # memory_peak_bw = 136e9       # 136 GB/s, Peak memory bandwidth (Bytes/s)


    max_cores = 128  # Number of available CPU cores (in this case, for the EPYC 7V13)

    # Examples of workloads (operations, memory_bytes, number of cores used, and bandwidth utilization)
    examples = [
        {"operations": 2e12, "memory_bytes": 4 * 1024**3, "num_cores": 16, "bw_utilization": 0.8},  # Example 1: 2 trillion FLOPs, 4 GB, 64 cores, 80% BW
        {"operations": 1e12, "memory_bytes": 2 * 1024**3, "num_cores": 16, "bw_utilization": 0.7},  # Example 2: 1 trillion FLOPs, 2 GB, 32 cores, 70% BW
        {"operations": 5e12, "memory_bytes": 8 * 1024**3, "num_cores": 16, "bw_utilization": 0.9}, # Example 3: 5 trillion FLOPs, 8 GB, 128 cores, 90% BW
        {"operations": 1e12, "memory_bytes": 0.48 * 1024**3, "num_cores": 128, "bw_utilization": 0.8}, # Example 3: 5 trillion FLOPs, 8 GB, 128 cores, 90% BW
    ]
    
    # Process each example
    for i, example in enumerate(examples, 1):
        operations = example["operations"]
        memory_bytes = example["memory_bytes"]
        num_cores = example["num_cores"]
        bw_utilization = example["bw_utilization"]
        
        # Calculate execution time
        compute_time, memory_time, total_time = calculate_execution_time(
            operations, memory_bytes, compute_peak_flops, memory_peak_bw, num_cores, max_cores, bw_utilization
        )
        
        # Print the results
        print(f"Example {i}:")
        print(f"  Operations: {operations:.2e} FLOPs")
        print(f"  Memory: {memory_bytes / 1024**3:.2f} GB")
        print(f"  Cores Used: {num_cores}/{max_cores} cores")
        print(f"  Bandwidth Utilization: {bw_utilization * 100:.0f}%")
        print(f"  Compute Time: {compute_time:.6f} seconds")
        print(f"  Memory Time: {memory_time:.6f} seconds")
        print(f"  Total Execution Time: {total_time:.6f} seconds")
        print(f"  Total Execution Time(ms): {total_time*1000:.6f} ms")
        print()


if __name__ == "__main__":
    main()
