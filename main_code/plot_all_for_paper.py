# python plot_all.py --skew default --plotid 20250211 --yscale log --dbname SIFT1M
# plot ours vs baselines for a specific dataset: 
# 1) qps - recall
# 2) total build time
# 3) cummulative time - query sequence
# 4) dataset skew << this has to be done during the run where you have access to the index
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import seaborn as sns
import numpy as np
import warnings
from datetime import datetime
import re

# Optional: Additional customizations
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")  # Add gridlines for readability

PLOT_FOR_PAPER = True
dbmame = ""

def save_plot(plt, output_dir: str, plot_name: str, additional_info: str = ""):
    """
    Save plot with timestamp and optional additional information.
    
    Args:
        plt: Matplotlib pyplot object
        output_dir: Directory to save the plot
        plot_name: Base name for the plot
        additional_info: Optional additional information for filename
    """
    filename = f"{plot_name}{'_' if len(additional_info) > 0 else ''}{additional_info}.{'jpg' if PLOT_FOR_PAPER else 'jpg'}".replace(" ", "_")
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved plot to: {filepath}")

def change_font_size(ax, font_size=24):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    # set the legend font size
    return ax


# Step 1: Data Processing - Create the unique index identifier
def create_unique_index_identifier_for_build_time_and_qps_plots(df):
    # Avoid modifying the original DataFrame by creating a copy
    df_for_plot = df.copy()

    if PLOT_FOR_PAPER:
        df_for_plot['index_identifier'] = df_for_plot.apply(
            lambda row: (
                f"{row['index_name']} {'(' + str(int(row['nlist'])) + ')' if pd.notna(row['nlist']) else ''}"
            ), 
            axis=1
        )
        df_for_plot.loc[df_for_plot['index_identifier'] == 'IVFFlat (100)', 'index_identifier'] = 'CrackIVF-START (100)'
    else:
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


# Step 2: Build Time Plot - Plotting Function
def plot_build_time(df_for_plot):
    # Split the dataset by 'nthreads' to create different plots for each thread count
    unique_nthreads = df_for_plot['nthreads'].unique()

    # Create a color palette to differentiate between indexes
    palette = sns.color_palette("Set2")

    # Create a plot for each unique value of 'nthreads'
    for nthread in unique_nthreads:
        # Filter the data for the current 'nthreads' value
        df_nthreads = df_for_plot[df_for_plot['nthreads'] == nthread].copy()

        # Restrict the palette size to match the number of unique indexes
        num_unique_indexes = df_nthreads['index_identifier'].nunique()
        current_palette = sns.color_palette("Set2", num_unique_indexes)

        # Plot the total build time for each unique index
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_nthreads, x='index_identifier', y='total_build_time', hue='index_identifier', palette=current_palette, s=100, edgecolor='black')

        # Set plot labels and title
        plt.xlabel('Index (Unique: index_name, nlist, niter)')
        plt.ylabel('Total Build Time (seconds)')
        plt.title(f'Total Build Time for nthreads={nthread}')

        # Rotate x-axis labels to avoid overlapping
        plt.xticks(rotation=45, ha='right')

        # Add legend
        plt.legend(title="Index Name", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        # plt.show()
        save_plot(plt, PLOT_OUTPUT_DIR, "build_time", f"threads_{nthread}")


def create_unique_index_identifier_for_cummulative_time_plot(df, CHECK=False):
    '''
    for cummulative plot
    unique index identifier: index name, nlist, niter, nthreads
    - if multiple runs keep the latest one
    '''
    # Avoid modifying the original DataFrame by creating a copy
    df_for_plot = df.copy()

    # Generate the index identifier, using only 'index_name' if nlist, niter, nprobe, and nthreads are not provided
    df_for_plot['index_identifier'] = df_for_plot.apply(
        lambda row: (
            f"{row['index_name']}" if all(pd.isna(row[col]) for col in ['nlist', 'niter', 'nprobe', 'nthreads'])
            else f"{row['index_name']}-nlist_{int(row['nlist']) if pd.notna(row['nlist']) else 'N/A'}"
                 f"-niter_{int(row['niter']) if pd.notna(row['niter']) else 'N/A'}"
                 f"-nprobe_{int(row['nprobe']) if pd.notna(row['nprobe']) else 'N/A'}"
                 f"-nthreads_{int(row['nthreads']) if pd.notna(row['nthreads']) else 'N/A'}"
        ), 
        axis=1
    )
    if CHECK:
        # Check for multiple runs for the same unique index identifier
        unique_index_identifiers = df_for_plot['index_identifier'].unique()
        for index_identifier in unique_index_identifiers:
            index_data = df_for_plot[df_for_plot['index_identifier'] == index_identifier]
            # If multiple runs exist for the same index_identifier, log a warning and keep the latest run
            if len(index_data) > 1:
                # Log a warning if needed
                # warnings.warn(f"Multiple runs detected for index '{index_identifier}'. Keeping the latest run.") # optional warning
                # print(f"Details for '{index_identifier}':\n{index_data[['index_identifier', 'run_date']]}")
                
                # Keep only the latest run by 'run_date'
                latest_run = index_data.sort_values(by='run_date', ascending=False).iloc[0]
                df_for_plot = df_for_plot[~((df_for_plot['index_identifier'] == index_identifier) & (df_for_plot.index != latest_run.name))]
            
            # Check that the remaining data satisfies the conditions
            index_data = df_for_plot[df_for_plot['index_identifier'] == index_identifier]
            if index_data['median_search_time_ms'].nunique() > 1:
                warnings.warn(f"'median_search_time_ms' is not unique for index {index_identifier}")
            if index_data['total_build_time'].nunique() > 1:
                warnings.warn(f"'total_build_time' is not unique for index {index_identifier}")
            assert index_data['total_build_time'].iloc[0] > 0, f"Error: 'total_build_time' must be positive for index {index_identifier}"

    # Create a new column to represent the unique combination of 'index_name' and 'nlist', this is for the colouring
    df_for_plot['index_group'] = df_for_plot.apply(
        lambda row: f"{row['index_name']}-nlist_{int(row['nlist']) if pd.notna(row['nlist']) else 'N/A'}", 
        axis=1
    )

    return df_for_plot

def create_cumulative_time_df(df_for_plot, target_num_queries=None, start_from_batch=0, sample_rate=None):
    """
    Creates a new DataFrame where each row is a query_sequence_id with calculated cumulative time.
    Includes batch size, index identifier, and other relevant metadata.
    
    :param df_for_plot: The processed DataFrame containing the necessary columns.
    :param target_num_queries: The target number of queries to use (optional).
    :param start_from_batch: The starting batch sequence number (default: 0).
    :param sample_rate: If provided, only every nth batch will be included in the output.
    :return: A new DataFrame with the cumulative time.
    """
    # Create a list to store all rows for the new DataFrame
    cumulative_time_data = []
    
    unique_nthreads = df_for_plot['nthreads'].unique()
    # print(f"{df_for_plot=}")
    total_num_queries = df_for_plot['total_num_queries'].iloc[0]  # Assuming it's unique
    # Use target_num_queries if provided, otherwise fall back to total_num_queries
    num_queries_to_plot = target_num_queries if target_num_queries else total_num_queries

    # Loop through each nthreads and calculate cumulative times for each query sequence
    for nthread in unique_nthreads:
        df_nthreads = df_for_plot[df_for_plot['nthreads'] == nthread].copy()
        batch_size = df_nthreads['batch_size'].iloc[0]  # Assuming it's unique per nthreads
        
        # Calculate batch sequences up to the total_num_queries (end at the total number of queries)
        batch_sequences = np.arange(0, num_queries_to_plot + batch_size, batch_size)
        batch_sequences = batch_sequences[batch_sequences >= start_from_batch]  # Apply start_from_batch

        # Apply sampling if a sample_rate is provided (select every nth batch)
        if sample_rate is not None:
            sampled_batch_sequences = batch_sequences[::sample_rate]
        else:
            sampled_batch_sequences = batch_sequences  # No sampling, use all batches

        for index_identifier in df_nthreads['index_identifier'].unique():
            index_data = df_nthreads[df_nthreads['index_identifier'] == index_identifier]
            
            # Get the total build time and median search time
            total_build_time_ms = index_data['total_build_time'].iloc[0] * 1000  # Convert to milliseconds
            # median_search_time_ms = index_data['median_search_time_ms'].iloc[0] # dont use median
            mean_search_time_ms = index_data['mean_search_time_ms'].iloc[0] # NOTE: use mean

            # Initialize cumulative time
            cumulative_time_ms = [total_build_time_ms]  # Start with total build time for the first batch
            current_time = total_build_time_ms  # Start from the total build time
            
            # Calculate cumulative time for each batch sequence
            for i, seq_num in enumerate(batch_sequences[1:]):  # Skip the first batch since it's already initialized
                current_time += mean_search_time_ms  # Add the median search time for the batch
                if seq_num in sampled_batch_sequences:
                    cumulative_time_ms.append(current_time)
            
            # Add each batch's cumulative time into the final dataframe
            for i, seq_num in enumerate(sampled_batch_sequences):
                
                
                row = {
                    'batch_id': seq_num,
                    'index_identifier': index_identifier,
                    'index_group': index_data['index_group'].iloc[0],
                    'batch_size': batch_size,
                    'nthreads': nthread,
                    'total_time': index_data['total_time'].iloc[0],  # Should be unique if index identifier is correctly created
                    'est_time_curr_batch_ms': cumulative_time_ms[i],  # Cumulative time for the current batch
                    'total_build_time': total_build_time_ms,
                    # 'mean_search_time_ms': mean_search_time_ms,
                    # 'median_search_time': median_search_time_ms,
                    'nlist': index_data['nlist'].iloc[0],  # Assuming nlist is the same for all rows of this index
                    'niter': index_data['niter'].iloc[0],  # Assuming niter is the same for all rows of this index
                    'nprobe': index_data['nprobe'].iloc[0],  # Assuming nprobe is the same for all rows of this index
                    'total_num_queries': index_data['total_num_queries'].iloc[0],
                    'index_name': index_data['index_name'].iloc[0],
                    'dataset_name': index_data['dataset_name'].iloc[0],
                }
                cumulative_time_data.append(row)

    # Convert the list of rows into a DataFrame
    cumulative_time_df = pd.DataFrame(cumulative_time_data)
    
    return cumulative_time_df


def plot_per_nthread_cumulative_time_from_cumulative_df(cumulative_time_df, color_by_index_group=True, target_num_queries=None, sample_rate=None, time_unit="ms", xscale="log", yscale="log", print_calculated_total_time=False, start_from_batch=0, x_axis_col="batch_id", y_axis_col="est_time_curr_batch_ms"):
    unique_nthreads = cumulative_time_df['nthreads'].unique()
    total_num_queries = cumulative_time_df['total_num_queries'].iloc[0]  # Assuming it's unique
    
    # Use target_num_queries if provided, otherwise fall back to total_num_queries
    num_queries_to_plot = target_num_queries if target_num_queries else total_num_queries

    # Generate a color palette if coloring by index group
    if color_by_index_group:
        unique_index_groups = cumulative_time_df['index_group'].unique()
        color_palette = sns.color_palette("tab10", len(unique_index_groups))  # Use seaborn's color palette

        index_group_colors = {group: color for group, color in zip(unique_index_groups, color_palette)}
    else:
        index_group_colors = None  # No color group logic if not using index group coloring

    # Create a plot for each unique value of 'nthreads'
    for nthread in unique_nthreads:
        df_nthreads = cumulative_time_df[cumulative_time_df['nthreads'] == nthread].copy()
        batch_size = df_nthreads['batch_size'].iloc[0]  # Assuming it's unique per nthreads

        # Initialize the plot
        plt.figure(figsize=(15, 6))  # Make the plot wider

        # # Dictionary to store total_time for each index
        index_total_times = {}

        # Plot cumulative time for each unique index identifier within the current nthreads
        for index_identifier in df_nthreads['index_identifier'].unique():
            # Filter the data for the current index identifier
            index_data = df_nthreads[df_nthreads['index_identifier'] == index_identifier]
            
            if color_by_index_group:
                # Get the group (index_name + nlist) to assign the correct color
                index_group = index_data['index_group'].iloc[0]
                color = index_group_colors[index_group]
            else:
                # Default color is the same as in the original method (sequential color)
                color = plt.cm.get_cmap('tab10')(hash(index_identifier) % 10)
            
            batch_sequences = index_data[x_axis_col]
            cumulative_time_ms = index_data[y_axis_col]

            if index_data['index_identifier'].str.startswith('BruteForce').iloc[0]:
                # make it more clear which is the fully trained one
                alpha_style = 0.7
                markersize =1
                markerstyle = 's'
                linestyle = '-'
                linewidth=1
            elif index_data['index_identifier'].str.startswith('CrackIVF').iloc[0]:
                # make it more clear which is the fully trained one
                alpha_style = 1
                markersize = 2
                markerstyle = 's'
                linestyle = '-'
                linewidth=2
            elif index_data['niter'].iloc[0] == 10:
                # make it more clear which is the fully trained one
                alpha_style = 0.7
                markersize = 1
                markerstyle = 's'
                linestyle = '-'
                linewidth=1
            else:
                alpha_style = 0.5
                markersize = 1
                markerstyle = 'o'
                linestyle = '--'
                linewidth=1
                
            plt.plot(batch_sequences, cumulative_time_ms, 
                     label=f'{index_identifier}', 
                     color=color,  # Use the color for the current index group
                     marker=markerstyle, 
                     linestyle=linestyle, 
                     linewidth=linewidth,  # Reduce line width
                     markersize=markersize,  # Smaller marker size
                     alpha=alpha_style  # Add transparency based on niter to make FULL TRAINED clear
                     )

            # Store the total time for the current index_identifier
            index_total_times[index_identifier] = cumulative_time_ms.iloc[-1]

            # Add the final value on the right of each dot for this index identifier
            x_range = max(batch_sequences) - min(batch_sequences)
            offset = 0.01 * x_range  # Adjust 0.01 as needed for more or less spacing
            plt.text(batch_sequences.iloc[-1]+offset, cumulative_time_ms.iloc[-1], 
                     f'{cumulative_time_ms.iloc[-1]:,.0f}', 
                     color=color, ha='left', va='center', fontsize=9)

        # Set plot labels and title
        plt.xlabel('Query Sequence')
        plt.ylabel(f'{y_axis_col}')
        # plt.title(f'Cumulative Time for Different Indexes on a {num_queries_to_plot} query sequence - {batch_size=} - nthreads={nthread}')
        plt.title(f'{x_axis_col}-{y_axis_col} for Different Indexes on a {num_queries_to_plot} query sequence - {batch_size=} - nthreads={nthread}')

        if print_calculated_total_time:
            # Add total_time box at the top-left of the plot
            total_time_text = "\n".join([f"{index}: {time:,.0f} ms" for index, time in index_total_times.items()])
            plt.gca().text(0.02, 0.96, f"total_time\n{total_time_text}", transform=plt.gca().transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='left', 
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0."))

        # Add legend (now showing index identifiers)
        plt.legend(title="Index Identifier", bbox_to_anchor=(1.01, 1), loc='upper left')

        # Adjust layout
        plt.tight_layout()
        plt.yscale(yscale)
        plt.xscale(xscale)

        # Show the plot
        # plt.show()
        save_plot(plt, PLOT_OUTPUT_DIR, f"{x_axis_col}_{y_axis_col}", f"threads_{nthread}")


################################################
################################################
################################################
################################################

def load_data(baseline_file, our_file_summary, our_file_detailed, our_file_summary_qps):
    """
    Load the datasets from provided file paths.
    """
    df_baselines = pd.read_csv(baseline_file)
    df_ours = pd.read_csv(our_file_summary)
    df_ours_detailed = pd.read_csv(our_file_detailed)
    df_ours_qps = pd.read_csv(our_file_summary_qps)
    return df_baselines, df_ours, df_ours_detailed, df_ours_qps


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


def preprocess_for_cumulative_time(df_ours, df_ours_detailed, df_baselines, FILTER):
    """
    Preprocess the data for cumulative time plots.
    """
    df_ours_summary = create_unique_index_identifier_for_cummulative_time_plot(df_ours)
    df_ours_detailed = create_unique_index_identifier_for_cummulative_time_plot(df_ours_detailed)
    merged_df_ours = pd.merge(
        df_ours_summary, df_ours_detailed, on='index_identifier', how='inner', suffixes=('_summary', '_detailed')
    )

    # Resolve column conflicts
    for column in merged_df_ours.columns:
        if column.endswith('_summary') and column.replace('_summary', '_detailed') in merged_df_ours.columns:
            detailed_column = column.replace('_summary', '_detailed')
            if (merged_df_ours[column] == merged_df_ours[detailed_column]).all():
                merged_df_ours = merged_df_ours.drop(columns=detailed_column).rename(
                    columns={column: column.replace('_summary', '')}
                )
            else:
                raise ValueError(f"Conflict in column '{column.replace('_summary', '')}': Values differ.")

    # Filter and prepare cumulative time data
    filtered_df_ours = merged_df_ours.query(FILTER)
    df_ours_for_plot = create_unique_index_identifier_for_cummulative_time_plot(filtered_df_ours)

    # Baseline cumulative time preparation
    df_baselines_for_plot = create_unique_index_identifier_for_cummulative_time_plot(df_baselines, CHECK=True)
    # NOTE: add exception for bruteforce to make sure we see it in the plots
    
    # df_baselines_filtered = df_baselines_for_plot.query(f"{FILTER} or index_identifier.str.startswith('BruteForce')", engine='python') # include brute force
    df_baselines_filtered = df_baselines_for_plot.query(f"{FILTER}")

    print(df_ours_for_plot['index_identifier'].value_counts())
    print(df_ours_for_plot[['index_identifier','nthreads']].value_counts())
    return df_ours_for_plot, df_baselines_filtered


def plot_cumulative_time(df_ours, df_baselines_filtered, target_num_queries, start_from_batch=0, sample_rate=None,xscale="log", yscale="log"):
    """
    Plot cumulative time from processed data.
    """
    cumulative_time_df = create_cumulative_time_df(
        df_baselines_filtered, target_num_queries=target_num_queries,
        start_from_batch=start_from_batch, sample_rate=sample_rate
    )
    cumulative_time_df['cummulative_time_ms_mean'] = cumulative_time_df['est_time_curr_batch_ms']
    cumulative_time_df['qid'] = cumulative_time_df['batch_id']
    
    combined_cumulative_time_df = pd.concat([cumulative_time_df, df_ours])
    # TODO: add this for overhead etc. like in the other plots it's missing here
    print(combined_cumulative_time_df['index_identifier'].value_counts())
    print(combined_cumulative_time_df[['index_identifier','nthreads']].value_counts())
    print("cumulative df = ", combined_cumulative_time_df)
    # exit()
    if PLOT_FOR_PAPER:
        plot_per_query_time(combined_cumulative_time_df, 'cummulative_time_ms_mean', nthreads=args.nthreads, target_num_queries=target_num_queries, algo = "CrackIVF*|BruteForce|IVFFlat*", FILTER='nlist == 1e3 or nlist == 5e3 or nlist == 1e4 or nlist==1.6e4 or index_name == "BruteForce" or index_name == "CrackIVF"')
    else:
        plot_per_nthread_cumulative_time_from_cumulative_df(
            combined_cumulative_time_df, color_by_index_group=True, target_num_queries=None,
            sample_rate=None, time_unit="ms", xscale=xscale, yscale=yscale, start_from_batch=0,
            x_axis_col="qid", y_axis_col="cummulative_time_ms_mean"
        )


def plot_bar_charts(df, metrics, plot_output_dir):
    """
    Plot bar charts for specific metrics grouped by index_identifier.
    """
    grouped = df.groupby('index_identifier')
    metrics_mapping = {
        'total_ms_mean': 'Total',
        'overhead_ms_mean': 'Build',
        'search_time_ms_mean': 'Search',
    }
    for index_identifier, group in grouped:
        column_sums = [group[metric].sum() / 1000 for metric in metrics]

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.bar([metrics_mapping[m] for m in metrics], column_sums, color='skyblue')
        
        for bar, value in zip(bars, column_sums):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=24)

        # ax.set_xlabel('Metrics')
        ax.set_ylabel('Time (s)')
        # ax.set_title(f'{df["dataset_name"].iloc[0]} Time breakdown')
        # ax.set_xticklabels(labels = ax.get_xticklabels(), rotation=45, ha='right')
        change_font_size(ax, font_size=36)
        plt.tight_layout()
        save_plot(plt, plot_output_dir, "time_breakdown")


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder: {path}")
    else:
        print(f"Folder already exists: {path}")

def add_baseline_dist_comp(ax):
    # pass # pass for ablation because too slow to load dataset
    # CORRECTED: 
    from vasili_helpers import load_dataset
    xb, _, _, _ = load_dataset(args.dbname) # slow on large datasets...
    cmap = {
        1000: 'purple',
        5000: 'orange',
        10000: 'green',
        16000: 'red'
    }
    for nlist in [1000, 5000, 10000, 16000]:
        # TODO: so far, we always train the index for 10 iterations, hence "* 10". 
        n_points = xb.shape[0] 
        niter = 10
        max_points = 256
        P_train = min(n_points , nlist * max_points)
        ax.axhline(y=niter*P_train*nlist, color=cmap[nlist], label=f"IVFFlat ({nlist})", linewidth=4)

def plot_per_query_time(df_in, y_axis_col="total_ms_mean", nthreads=16, target_num_queries=1e5, algo = "CrackIVF*", FILTER=None):
    plot_name_mapping = {
        'total_ms_mean': 'Total Time (ms)',
        'overhead_ms_mean': 'Overhead Time (ms)',
        'search_time_ms_mean': 'Search Time (ms)',
        'cummulative_time_ms_mean': 'Cumulative Time (ms)',
        'cum_dist_comp': 'Cumulative Dist Comp',
    }

    if y_axis_col not in df_in.columns:
        print(f"Warning: '{y_axis_col}' not found in DataFrame columns.")
        return
    
    df = df_in.copy()
    total_num_queries = df['total_num_queries'].iloc[0]
    batch_size = df['batch_size'].iloc[0]
    
    num_queries_to_plot = target_num_queries if target_num_queries else total_num_queries
    df = df[(df['nthreads'] == nthreads) & (df['qid'] < num_queries_to_plot)]
    
    if FILTER:
        df = df.query(FILTER)
    
    df = df[df['index_identifier'].apply(lambda x: True if re.search(algo, x) else False)]
    INCLUDE_NPROBE = False
    if INCLUDE_NPROBE:
        df['index_identifier_for_plot'] = df.apply(
            lambda row: (
                f"{row['index_name']}" if any(pd.isna(row[col]) for col in ['nlist', 'niter', 'nprobe', 'nthreads']) or row['index_name'] == 'CrackIVF'
                else f"{row['index_name']} ({int(row['nlist']) if pd.notna(row['nlist']) else 'N/A'}"
                    f"/{int(row['nprobe']) if pd.notna(row['nprobe']) else 'N/A'})"
            ), 
            axis=1
        )
    else:
        df['index_identifier_for_plot'] = df.apply(
            lambda row: (
                f"{row['index_name']}" if any(pd.isna(row[col]) for col in ['nlist', 'niter', 'nprobe', 'nthreads']) or row['index_name'] == 'CrackIVF'
                else f"{row['index_name']} ({int(row['nlist']) if pd.notna(row['nlist']) else 'N/A'})"
            ), 
            axis=1
        )

    if len(df[df['index_name'] == 'IVFFlat']) > 0:
        print(df['index_name'])
        ivf = df[df['index_name'] == 'IVFFlat'].copy()
        ivf = ivf.groupby(['index_name', 'nlist']).agg({"nprobe": "max"}).reset_index(inplace=False)
        print("IVF = ", ivf)
        df = df.merge(ivf, on=['index_name', 'nlist'], how='left', suffixes=('', '_max'))
        df = df[(df['nprobe_max'].isnull())|(df['nprobe'] == df['nprobe_max'])]
        print(df)
    
    df = df.sort_values(by='index_identifier_for_plot', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 10))

    if y_axis_col.startswith("cum"):
        sns.lineplot(data=df, x='qid', y=y_axis_col, hue='index_identifier_for_plot', ax=ax, linewidth=4)
        if y_axis_col == "cum_dist_comp":
            add_baseline_dist_comp(ax)
    else:
        # for R3D10 - Report the number of cracks and refines for each dataset
        with open(os.path.join(PLOT_OUTPUT_DIR, "crack_refine.txt"), 'w') as f:
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Number of Queries: {num_queries_to_plot}\n")
            f.write(f"Algorithm: {algo}\n")
            f.write(f"Filter: {FILTER}\n")
            crack_time_count = df[df['crack_time_ms_mean'] >= 1].shape[0]
            f.write(f"Number of batches where crack_time_ms_mean >= 1: {crack_time_count}\n")
            f.write("Percentage of batches where crack_time_ms_mean >= 1: {:.5f}%\n".format(crack_time_count / df.shape[0] * 100))
            refine_time_count = df[df['refine_time_ms_mean'] >= 1].shape[0]
            f.write(f"Number of batches where refine_time_ms_mean >= 1: {refine_time_count}\n")
            f.write("Percentage of batches where refine_time_ms_mean >= 1: {:.5f}%\n".format(refine_time_count / df.shape[0] * 100))
        
        print("the dataframe has ", df.shape[0], "rows")
        print("the dataframe has columns ", df.columns)
        assert df['index_identifier_for_plot'].nunique() == 1
        sns.scatterplot(data=df, x='qid', y=y_axis_col, ax=ax, label="search ({:.4f}%)".format((df.shape[0]-crack_time_count-refine_time_count)/df.shape[0]*100))
        sns.scatterplot(data=df[df['crack_time_ms_mean'] >= 1], x='qid', y=y_axis_col, ax=ax, color='red', label='crack ({:.4f}%)'.format(crack_time_count/df.shape[0]*100))
        sns.scatterplot(data=df[df['refine_time_ms_mean'] >= 1], x='qid', y=y_axis_col, ax=ax, color='green', label='refine ({:.4f}%)'.format(refine_time_count/df.shape[0]*100))
        
    # ax.set_title(f'{df.dataset_name.iloc[0]} {plot_name_mapping[y_axis_col]} - batch_size={batch_size} - nthreads={nthreads}')
    ax.set_xlabel('Query Sequence')
    ax.set_ylabel(f'{plot_name_mapping[y_axis_col]}')
    if y_axis_col == 'cummulative_time_ms_mean':
        ax.legend(title="", loc='upper left' if not dbname.startswith('glove') else "lower right" , fontsize=22, ncols=2 if not dbname.startswith('glove') else 1)
    elif y_axis_col == 'cum_dist_comp':
        ax.legend(title="", loc='lower left', bbox_to_anchor=(0, 0.15), fontsize=22, ncols=2)
    else:
        ax.legend(title="", loc='upper left' if not dbname == "glove-100-angular" else "lower left", fontsize=22, ncols=1)
    change_font_size(ax, font_size=36)
    ax.grid(True)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.tight_layout()
    save_plot(plt, PLOT_OUTPUT_DIR, f"{y_axis_col}_per_query")

# Main Script
def main(our_file_summary_qps, baseline_file, our_file_summary, our_file_detailed, FILTER, plot_output_dir, xscale="log", yscale="log"):
    df_baselines, df_ours, df_ours_detailed, df_ours_qps = load_data(baseline_file, our_file_summary, our_file_detailed, our_file_summary_qps)
    df_ours_detailed['dataset_name'] = df_ours['dataset_name'].iloc[0]
    # Build time and QPS plots
    df_qps = preprocess_for_build_time_and_qps(df_baselines, df_ours_qps) # df_our_qps dont filter 
    # NOTE: QPS FILTER SHOULD BE SAME AS FILTER, IT JUST DOES NOT FILTER FOR RECALL ( to be able to see all data for recall < 99)
    QPS_FILTER = "recall_10 < 0.99 and nthreads == 16 and niter == 10 or index_name == 'CrackIVF' or (index_name == 'BruteForce' and nthreads ==16)"
    df_qps = df_qps.query(QPS_FILTER)
    plot_qps(df_qps) # do not apply recall filter on QPS ( to be able to 'scan' )
    
    # FILTER FOR THE REST OF THE PLOTS    
    if not PLOT_FOR_PAPER:
        df_build = preprocess_for_build_time_and_qps(df_baselines, df_ours)
        df_build = df_build.query(FILTER)
        plot_build_time(df_build)

    # Cumulative time plots
    df_ours_cumulative, df_baselines_filtered = preprocess_for_cumulative_time(
        df_ours, df_ours_detailed, df_baselines, FILTER
    )

    # print(df_ours_cumulative.columns) # DEBUG
    # print(df_ours_cumulative['recall_10_perQ']) # DEBUG
    # exit()
    # print(df_ours_cumulative)
    target_num_queries = int(df_ours_cumulative["total_num_queries"].iloc[0]) # NOTE: SAME AS RUN FOR CRACKIVF
    # print(f"{target_num_queries=}")
    # for y_axis_column in ['total_ms_mean', 'overhead_ms_mean', 'search_time_ms_mean']:
    #     plot_per_nthread_cumulative_time_from_cumulative_df(
    #         df_ours_cumulative, color_by_index_group=True, target_num_queries=None,
    #         sample_rate=None, time_unit="ms", xscale=xscale, yscale=yscale, start_from_batch=0,
    #         x_axis_col="qid", y_axis_col=y_axis_column
    #     )
    plot_per_query_time(df_ours_cumulative, y_axis_col="total_ms_mean", nthreads=args.nthreads, target_num_queries=1e6, algo = "CrackIVF*")
    plot_per_query_time(df_ours_cumulative, y_axis_col="cum_dist_comp", nthreads=args.nthreads, target_num_queries=1e6, algo = "CrackIVF*")
    
    # Create a subplot for each y-axis column
    # fig, axes = plt.subplots(len(['recall_10_perQ', 'recall_100_perQ']), figsize=(8, 6))

    # # Loop through y-axis columns and plot them
    # x_axis_column="qid"
    # for i, y_axis_column in enumerate(['recall_10_perQ', 'recall_100_perQ']):
    #     axes[i].scatter(df_ours_cumulative[x_axis_column], df_ours_cumulative[y_axis_column])
    #     axes[i].set_title(f'Scatter plot of {y_axis_column}')
    #     axes[i].set_xlabel(x_axis_column)
    #     axes[i].set_ylabel(y_axis_column)

    # plt.tight_layout()
    # save_plot(plt, plot_output_dir, f"{x_axis_column}_{y_axis_column}")

    if not PLOT_FOR_PAPER:
        # PLOT RECALLS
        fig, axes = plt.subplots(len(['recall_10_perQ', 'recall_100_perQ']), figsize=(12, 10))

        x_axis_column = "qid"
        for i, y_axis_column in enumerate(['recall_10_perQ', 'recall_100_perQ']):
            axes[i].scatter(df_ours_cumulative[x_axis_column], df_ours_cumulative[y_axis_column], s=20)
            rolling_mean = df_ours_cumulative[y_axis_column].rolling(window=50, min_periods=2).mean() # pick windo large enough to have smooth plot
            # print(rolling_mean)
            axes[i].plot(df_ours_cumulative[x_axis_column], rolling_mean, color='red', linewidth=1)
            axes[i].set_title(f'Scatter plot of {y_axis_column} with Running Mean')
            axes[i].set_xlabel("query sequence")
            axes[i].set_ylabel(y_axis_column)
            axes[i].legend(["Data points", "Running Mean"])
            axes[i].set_ylim(0, 1)
            axes[i].set_xscale("log")
            axes[i].set_yticks(np.arange(0, 1.05, 0.1))
        plt.tight_layout()
        save_plot(plt, plot_output_dir, f"{x_axis_column}_{y_axis_column}")

    target_num_queries = 1e6
    plot_cumulative_time(df_ours_cumulative, df_baselines_filtered, target_num_queries, xscale=xscale, yscale=yscale)

    # Total Time Breakdown charts for CrackIVF
    y_metrics = [
        "total_ms_mean", "overhead_ms_mean", "search_time_ms_mean", 
        # "crack_time_ms_mean", "refine_time_ms_mean", "get_local_region_time_ms_mean", "reorg_time_ms_mean", 
        # "inner_point_assignments_ms_mean", "update_invlists_ms_mean"
    ]
    plot_bar_charts(df_ours_cumulative, y_metrics, plot_output_dir)


# Step 3: QPS vs Recall Plot - Plotting Function
def plot_qps(df_for_plot, min_recall_to_show = 0.5):
    # Split the dataset by 'nthreads' to create different plots for each thread count
    unique_nthreads = df_for_plot['nthreads'].unique()
    recall = "recall_10"

    # print(df_for_plot[recall].max())
    # print(df_for_plot[recall].min())

    # Create a plot for each unique value of 'nthreads'
    for nthread in unique_nthreads:
        # Filter data for the current 'nthreads' value
        subset = df_for_plot[(df_for_plot['nthreads'] == nthread) & (df_for_plot[recall] > min_recall_to_show)].copy()
        print(subset['index_name'])
        subset = subset.query('nlist == 100 or nlist == 1e3 or nlist == 5e3 or nlist == 1e4 or nlist==1.6e4 or index_name == "BruteForce" or index_name == "CrackIVF-END"')

        # Initialize the plot
        # plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot QPS vs Recall_100 for each index
        for index in np.sort(subset['index_identifier'].unique()):
            index_data = subset[subset['index_identifier'] == index]
            index_data = index_data.sort_values(by=recall, ascending=True)  # Change to `False` for descending

            # Check if 'CrackIVF' is part of the index identifier
            if 'CrackIVF' in index and not "START" in index:
                # Make the line distinct for 'CrackIVF' by changing the marker and line width
                ax.plot(
                    index_data[recall], index_data['QPS'],
                    marker='D',  # Diamond markers for CrackIVF
                    label=str(index), 
                    linestyle='-', 
                    markersize=8,  # Larger markers for emphasis
                    linewidth=6  # Thicker line for distinction
                )
            elif 'BruteForce' in index:
                # Default plot style for other indexes (maintaining your original style)
                ax.plot(
                    index_data[recall], index_data['QPS'],
                    marker='o',  # Circle markers for other indices
                    label=str(index), 
                    linestyle='-', 
                    markersize=10,  # Regular marker size for others
                    linewidth = 3
                )
            else:
                # Default plot style for other indexes (maintaining your original style)
                ax.plot(
                    index_data[recall], index_data['QPS'],
                    marker='x',  # Circle markers for other indices
                    label=str(index), 
                    linestyle='-.', 
                    markersize=8,  # Regular marker size for others
                    linewidth = 3
                )

        # Set the plot title and labels
        # ax.set_title(f"{df_for_plot.dataset_name.iloc[0]} - batch_size= {subset.batch_size.iloc[0]} - nthreads={nthread}")
        ax.set_xlabel(recall.replace("_", "@"))
        ax.set_ylabel("Queries Per Second (QPS)")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_ylim(0, df_for_plot['QPS'].max() * 1.25)  # Set the y-axis limit to 110% of the maximum QPS value
        ax.yaxis.get_offset_text().set_fontsize(18)
        # ax.set_yscale("log")  # Using logarithmic scale for QPS

        # Show grid, legend, and layout
        ax.grid(True)
        # ax.legend(title="Index Name", bbox_to_anchor=(1.05, 1), loc='upper left')
        change_font_size(ax, 34)  # Change font size for better readability
        ax.legend(title="", loc='upper center', ncols=2, fontsize=20, frameon=False)
        fig.tight_layout()

        # Show the plot
        # plt.show()
        save_plot(plt, PLOT_OUTPUT_DIR, "qps_recall")


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
        "--desc",
        type=str,
        default=None,
        help="Optional description"
    )
    parser.add_argument(
        "--skew",
        type=str,
        required=True,
        help="Skew value (e.g., default, high, low)"
    )
    parser.add_argument(
        "--plotid",
        type=str,
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
    parser.add_argument(
        "--nthreads",
        type=int,
        required=False,
        default=16,
        help="Number of threads used in the experiment (default: 16)"
    )

    args = parser.parse_args()

    global dbname
    dbname = args.dbname

    # Folder and file paths
    baseline_folder = f'./results/{args.dbname}_{args.skew}_BASELINES/'
    our_folder = f'./results/{args.dbname}_{args.skew}_OURS/'
    

    baseline_file = os.path.join(baseline_folder, 'baseline_results.csv')
    our_file_summary = os.path.join(our_folder, 'summary_results.csv')
    our_file_summary_qps = os.path.join(our_folder, 'qps_summary_results.csv')
    our_file_detailed = os.path.join(our_folder, 'super_detailed_results.csv')

    if args.desc:
        our_file_summary = our_file_summary.split(".csv")[0] + f"_{args.desc}.csv"
        our_file_summary_qps = our_file_summary_qps.split(".csv")[0] + f"_{args.desc}.csv"
        our_file_detailed = our_file_detailed.split(".csv")[0] + f"_{args.desc}.csv"

    # Recall filter
    FILTER = "0.89 < recall_10 < 0.96 and nthreads == 16 and niter == 10 or index_name == 'CrackIVF' or (index_name == 'BruteForce' and nthreads ==16)" # T2I
    if args.plotid is None:
        PLOT_OUTPUT_DIR = f'./results/{args.dbname}_{args.skew}_OURS/plots'
    else:
        PLOT_OUTPUT_DIR = f'./results/{args.dbname}_{args.skew}_OURS/plots_{args.plotid}'
    create_folder_if_not_exists(PLOT_OUTPUT_DIR)
    
    # Call the main function
    main(our_file_summary_qps, baseline_file, our_file_summary, our_file_detailed, FILTER, PLOT_OUTPUT_DIR, xscale=args.xscale, yscale=args.yscale)