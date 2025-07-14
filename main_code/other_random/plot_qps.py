# $ python plot_qps.py --skew default --plotid 20250205 --dbname SIFT1M
# plot QPS only ( should not have recall filter )
# 1) qps - recall
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import seaborn as sns
import numpy as np
import warnings
from datetime import datetime

# Optional: Additional customizations
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")  # Add gridlines for readability

def save_plot(plt: plt, output_dir: str, plot_name: str, additional_info: str = ""):
    """
    Save plot with timestamp and optional additional information.
    
    Args:
        plt: Matplotlib pyplot object
        output_dir: Directory to save the plot
        plot_name: Base name for the plot
        additional_info: Optional additional information for filename
    """
    filename = f"{plot_name}_{additional_info}.jpg".replace(" ", "_")
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Saved plot to: {filepath}")


# Step 1: Data Processing - Create the unique index identifier
def create_unique_index_identifier_for_build_time_and_qps_plots(df):
    # Avoid modifying the original DataFrame by creating a copy
    df_for_plot = df.copy()

    # Generate a consistent index identifier based on 'index_name', 'nlist', 'niter', and 'nthreads'
    df_for_plot['index_identifier'] = df_for_plot.apply(
        lambda row: (
            f"{row['index_name']}-nlist_{int(row['nlist']) if pd.notna(row['nlist']) else 'N/A'}"
            f"-niter_{int(row['niter']) if pd.notna(row['niter']) else 'N/A'}"
            f"-nthreads_{int(row['nthreads']) if pd.notna(row['nthreads']) else 'N/A'}"
        ), 
        axis=1
    )
    return df_for_plot


# Step 3: QPS vs Recall Plot - Plotting Function
def plot_qps(df_for_plot):
    # Split the dataset by 'nthreads' to create different plots for each thread count
    unique_nthreads = df_for_plot['nthreads'].unique()

    # Create a plot for each unique value of 'nthreads'
    for nthread in unique_nthreads:
        # Filter data for the current 'nthreads' value
        subset = df_for_plot[df_for_plot['nthreads'] == nthread]

        # Initialize the plot
        plt.figure(figsize=(10, 6))

        # Plot QPS vs Recall_100 for each index
        for index in subset['index_identifier'].unique():
            index_data = subset[subset['index_identifier'] == index]

            # Check if 'CrackIVF' is part of the index identifier
            if 'CrackIVF' in index:
                # Make the line distinct for 'CrackIVF' by changing the marker and line width
                plt.plot(
                    index_data['recall_10'], index_data['QPS'],
                    marker='D',  # Diamond markers for CrackIVF
                    label=str(index), 
                    linestyle='-', 
                    markersize=6,  # Larger markers for emphasis
                    linewidth=3  # Thicker line for distinction
                )
            else:
                # Default plot style for other indexes (maintaining your original style)
                plt.plot(
                    index_data['recall_10'], index_data['QPS'],
                    marker='o',  # Circle markers for other indices
                    label=str(index), 
                    linestyle='-.', 
                    markersize=1  # Regular marker size for others
                )

        # Set the plot title and labels
        plt.title(f"{df_for_plot.dataset_name.iloc[0]} QPS vs. Recall for Different Indexes - batch_size= {subset.batch_size.iloc[0]} - nthreads={nthread}")
        plt.xlabel("Recall@100")
        plt.ylabel("Queries Per Second (QPS)")
        plt.yscale("log")  # Using logarithmic scale for QPS

        # Show grid, legend, and layout
        plt.grid(True)
        plt.legend(title="Index Name", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Show the plot
        # plt.show()
        save_plot(plt, PLOT_OUTPUT_DIR, "qps_recall", f"threads_{nthread}")



################################################
################################################
################################################
################################################

def load_data(baseline_file, our_file_summary, our_file_detailed):
    """
    Load the datasets from provided file paths.
    """
    df_baselines = pd.read_csv(baseline_file)
    df_ours = pd.read_csv(our_file_summary)
    df_ours_detailed = pd.read_csv(our_file_detailed)
    return df_baselines, df_ours, df_ours_detailed


def preprocess_for_build_time_and_qps(df_baselines, df_ours, nthreads_filter=32):
    """
    Preprocess the data for the build time and QPS plots.
    """
    interesting_cols = [
        'dataset_name', 'index_name', 'nlist', 'niter', 'nthreads', 'batch_size',
        'QPS', 'recall_1', 'recall_10', 'recall_100', 'total_build_time', 
        'total_search_time', 'total_time'
    ]
    df_baselines_subset = df_baselines[interesting_cols]
    df_ours_subset = df_ours[interesting_cols]
    
    df_baselines_subset["source"] = "Baseline"
    df_ours_subset["source"] = "Ours"

    combined_df = pd.concat([df_baselines_subset, df_ours_subset], ignore_index=True)
    df_for_plot = create_unique_index_identifier_for_build_time_and_qps_plots(combined_df)
    
    # # Filter data
    # df_for_plot = df_for_plot[
    #     (df_for_plot["nthreads"] == nthreads_filter) &
    #     ((df_for_plot["niter"] == 0) | (df_for_plot["index_name"].str.startswith("BruteForce")))
    # ]
    return df_for_plot


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder: {path}")
    else:
        print(f"Folder already exists: {path}")

# Main Script
def main(baseline_file, our_file_summary, our_file_detailed, FILTER, plot_output_dir, xscale="log", yscale="log"):
    df_baselines, df_ours, df_ours_detailed = load_data(baseline_file, our_file_summary, our_file_detailed)

    df_qps = preprocess_for_build_time_and_qps(df_baselines, df_ours)
    df_qps = df_qps.query(FILTER)
    plot_qps(df_qps)


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Compare baseline results and our results.")
    parser.add_argument(
        "--dbname",
        type=str,
        required=True,
        help="Dataset name (e.g., SIFT1M)"
    )
    parser.add_argument(
        "--skew",
        type=str,
        required=True,
        help="Skew value (e.g., default, high, low)"
    )
    parser.add_argument(
        "--plotid",
        type=int,
        required=False,
        default=None,
        help="unique id if you wish for plot directory to be idfferent (default None)"
    )
    parser.add_argument(
        "--xscale",
        type=str,
        required=False,
        default="log",
        help="for plot, default log])"
    )
    parser.add_argument(
        "--yscale",
        type=str,
        required=False,
        default="log",
        help="for plot, default log])"
    )
    # TODO: add more options, log/linear, targenum queries, whatever

    args = parser.parse_args()

    # Folder and file paths
    baseline_folder = f'./results/{args.dbname}_{args.skew}_BASELINES/'
    our_folder = f'./results/{args.dbname}_{args.skew}_OURS/'
    

    baseline_file = os.path.join(baseline_folder, 'baseline_results.csv')
    our_file_summary = os.path.join(our_folder, 'qps_summary_results.csv')
    our_file_detailed = os.path.join(our_folder, 'super_detailed_results.csv')

    # Recall filter
    # all
    FILTER = "recall_10 < 0.99 and nthreads == 16 and niter == 10 or index_name == 'CrackIVF' or (index_name == 'BruteForce' and nthreads ==16)"
    
    # Output directory for plots
    if args.plotid is None:
        PLOT_OUTPUT_DIR = f'./results/{args.dbname}_{args.skew}_OURS/plots'
    else:
        PLOT_OUTPUT_DIR = f'./results/{args.dbname}_{args.skew}_OURS/plots_{args.plotid}'
    create_folder_if_not_exists(PLOT_OUTPUT_DIR)
    
    # Call the main function
    main(baseline_file, our_file_summary, our_file_detailed, FILTER, PLOT_OUTPUT_DIR, xscale=args.xscale, yscale=args.yscale)

