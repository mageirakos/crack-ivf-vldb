#!/bin/bash
# 
# To add more datasets, copy and paste the following line:
# run_command "python run_baselines.py --store --dbname <your-dbname>"

# Function to run a command and check for errors
run_command() {
    echo "Running: $1"
    eval $1

    if [ $? -ne 0 ]; then
        echo "Error: Command failed. Exiting."
        exit 1
    fi

    echo "Command completed successfully: $1"
}



# 25/02/25 - re-run final for paper
run_command "python run_baselines.py --store --index_name BruteForce IVFFlat --dbname SIFT1M --runid 250227_rerun_baselines --nthreads 16 --niter 10 --nprobe 1 2 4 8 16 20 32 48 64 80 128 256 512 1024 --nlist 100 1000 5000 10000 16000"
run_command "python run_baselines.py --store --index_name BruteForce IVFFlat --dbname SIFT10M --runid 250227_rerun_baselines --nthreads 16 --niter 10 --nprobe 1 2 4 8 16 20 32 48 64 80 128 256 512 1024 --nlist 100 1000 5000 10000 16000"
run_command "python run_baselines.py --store --index_name BruteForce IVFFlat --dbname glove-25-angular --runid 250227_rerun_baselines --nthreads 16 --niter 10 --nprobe 1 2 4 8 16 32 64 128 256 512 1024 --nlist 100 1000 5000 10000 16000"
run_command "python run_baselines.py --store --index_name BruteForce IVFFlat --dbname glove-50-angular --runid 250227_rerun_baselines --nthreads 16 --niter 10 --nprobe 1 2 4 8 16 32 64 128 256 512 1024 --nlist 100 1000 5000 10000 16000"
run_command "python run_baselines.py --store --index_name BruteForce IVFFlat --dbname glove-100-angular --runid 250227_rerun_baselines --nthreads 16 --niter 10 --nprobe 1 2 4 8 16 32 64 128 256 512 1024 --nlist 100 1000 5000 10000 16000"
run_command "python run_baselines.py --store --index_name BruteForce IVFFlat --dbname deep-image-96-angular --runid 250227_rerun_baselines --nthreads 16 --niter 10 --nprobe 1 2 4 8 16 32 64 128 256 512 1024 --nlist 100 1000 5000 10000 16000"
