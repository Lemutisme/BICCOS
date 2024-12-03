import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_seq_items', None)

def venus_log_read(log_file_path):
    statuses = []
    times = []
    cut_generation = []
    cuts_gen_list = []

    with open(log_file_path, 'r') as file:
        for line in file:
             
            if 'Added ideal cuts, #cuts: ' in line:
                parts = line.split()
                # Remove the comma and convert to float before converting to int
                cut_value_str = parts[-3].replace(',', '')
                try:
                    cut_value = int(float(cut_value_str))
                    cut_generation.append(cut_value)
                except ValueError as e:
                    print(f"Error converting value: {cut_value_str} - {e}")

            # Process lines that start with 'Result: '
            if 'venus.verification.verifier : Result:' in line:

                cuts_gen_list.append(sum_of_stage_maxima(cut_generation))
                cut_generation = []

                parts = line.split()
                # Handle lines with '(timed out)'
                status = parts[-1]
                statuses.append(status)
            
            if line.startswith('Time: '):
                parts = line.split()
                time = float(parts[-1])
                times.append(time)
    
    return pd.DataFrame({
        'Status': statuses,
        'Time': times,
        #'Cut Generation': cuts_gen_list,
    })

def log_read(log_file_path, cut_num=False):
    statuses = []
    times = []
    
    bab_round_counts = []
    bab_round_count = 0
    domain_v_list = []
    domain_visited = []
    cut_generation = []
    cuts_gen_list = []
    unsat_nodes = []
    unsat_nodes_list = []

    lb_rhs = None  # To store the current (lb-rhs) value
    list_of_lb_rhs = []

    with open(log_file_path, 'r') as file:
        for line in file:
            # Check for "BaB round" occurrences
            if 'BaB round' in line:
                bab_round_count += 1
            
             # Capture the Current (lb-rhs) value
            if 'Current (lb-rhs):' in line:
                parts = line.split('Current (lb-rhs):')
                if len(parts) > 1:
                    lb_rhs = float(re.findall(r"[-+]?\d*\.\d+|\d+", parts[1].strip())[0])

            if 'domains visited' in line:
                parts = line.split()
                domain_visited.append(int(parts[0]))
            
            if 'Number of Verified Splits: ' in line:
                parts = line.split()
                unsat_nodes.append(int(parts[-3]))
            
            if 'Total number of valid cuts:' in line:
                parts = line.split()
                cut_generation.append(int(float(parts[5])))

            # Process lines that start with 'Result: '
            if line.startswith('Result: '):
                bab_round_counts.append(bab_round_count)
                bab_round_count = 0  # Reset the count for the next round

                domain_v_list.append(sum_of_stage_maxima(domain_visited))
                domain_visited = []

                cuts_gen_list.append(sum_of_stage_maxima(cut_generation))
                cut_generation = []

                unsat_nodes_list.append(sum_of_stage_maxima(unsat_nodes))
                unsat_nodes = []

                parts = line.split()
                # Handle lines with '(timed out)'
                if '(timed' in parts:
                    status = ' '.join(parts[1:4])  # Combine 'safe (timed out)'
                    time = float(parts[5])
                else:
                    status = parts[1]
                    time = float(parts[3])

                statuses.append(status)
                times.append(time)

                # Only record lb_rhs for 'unknown' results
                if status == 'unknown':
                    list_of_lb_rhs.append(lb_rhs)
                else:
                    list_of_lb_rhs.append(None)  # Append None for results other than unknown
    if cut_num:
        return pd.DataFrame({
            'Status': statuses,
            'Time': times,
            'Domain Visited': domain_v_list,
            'Cut Generation': cuts_gen_list,
            'Unsat Nodes': unsat_nodes_list,
            'BaB Round Count': bab_round_counts,
            'BaB_lb_rhs': list_of_lb_rhs
        })
    else:
        return pd.DataFrame({
            'Status': statuses,
            'Time': times,
            'Domain Visited': domain_v_list,
            'BaB Round Count': bab_round_counts,
            'BaB_lb_rhs': list_of_lb_rhs
        })

def sum_of_stage_maxima(lst):
    if not lst:
        return 0
    
    total_sum = 0
    current_max = lst[0]
    
    for i in range(1, len(lst)):
        if lst[i] < current_max:
            total_sum += current_max
            current_max = lst[i]
        else:
            current_max = max(current_max, lst[i])
    
    total_sum += current_max  # Add the last subsequence maximum
    
    return total_sum


def plot_time_vs_instances(df, cut=False):
    # Step 1: Filter the DataFrame
    if df.loc[(df['Status'] == 'safe')].shape[0] > 0:
        df_filtered = df.loc[(df['Status'] == 'safe')]
    else:
        df_filtered = df.loc[(df['Status'] == 'safe-incomplete') | (df['Status'] == 'safe-mip')]
    
    # Step 2: Sort the DataFrame by Time
    df_filtered = df_filtered.sort_values(by='Time')
    
    # Step 3: Calculate Cumulative Count
    df_filtered['Cumulative Count'] = range(1, len(df_filtered) + 1)
    
    # Step 4: Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Cumulative Count vs. Time
    ax1.plot(df_filtered['Cumulative Count'], df_filtered['Time'], marker='o', linestyle='-', label='Time')
    ax1.set_xlabel('Number of Instances Verified')
    ax1.set_ylabel('Time')
    ax1.set_title('Number of Instances Verified vs. Time')
    ax1.grid(True)
    
    # Set the y-axis limit
    max_time = max(df_filtered['Time'].max(), 200)
    ax1.set_ylim(0, max_time)
    
    # Add a horizontal line at y=200 to emphasize the time limitation
    if max_time > 200:
        ax1.axhline(y=200, color='red', linestyle='--', linewidth=2, label='Time Limit (200)')
    
    # Check if 'Cut Generation' column exists and plot it
    if cut and 'Cut Generation' in df.columns:
        ax2 = ax1.twinx()  # Create a second y-axis
        ax2.bar(df_filtered['Cumulative Count'], df_filtered['Cut Generation'], color='gray', alpha=0.6, label='Cut Generation')
        ax2.set_ylabel('Cut Generation')
    
    # Adding legends
    ax1.legend(loc='upper left')
    if cut and 'Cut Generation' in df.columns:
        ax2.legend(loc='upper right')
    
    plt.show()

def print_numbers_separated_by_space(numbers):
    # Convert each number to a string and join them with a space
    output = ' '.join(map(str, numbers))
    # Print the output
    return output

def main(log_file_path, to_execl=False, visualize=True, venus=False, cut=True):
    if venus:
        df = venus_log_read(log_file_path)
    else:
        df = log_read(log_file_path, cut)
    if to_execl:
        df.to_excel(f'{log_file_path}.xlsx', index=True)
    print(df.loc[(df['Status'] == 'safe') | (df['Status'] == 'safe-mip')])
    print(df.loc[df['Status'] == 'unknown'])
    print("All:\n", df)
    if venus:
        aggregated_data = df.groupby('Status').agg({
            'Time': ['count', 'mean'],
        })
    elif cut:
        aggregated_data = df.groupby('Status').agg({
            'Time': ['count', 'mean'],
            'Domain Visited': ['mean'],
            'Cut Generation': ['mean'],
            'Unsat Nodes': ['mean'],
            'BaB Round Count': ['mean']
        })
        print('#Cuts', df['Cut Generation'].sum())
    else:
        aggregated_data = df.groupby('Status').agg({
            'Time': ['count', 'mean'],
            'Domain Visited': ['mean'],
            'BaB Round Count': ['mean']
        })
    print(aggregated_data)
    safe_num = aggregated_data.iloc[0][0]
    safe_inc_num = aggregated_data.iloc[1][0]
    safe_time = aggregated_data.iloc[0][1]
    safe_inc_time = aggregated_data.iloc[1][1]
    safe_domain = aggregated_data.iloc[0][2]
    safe_inc_domain = aggregated_data.iloc[1][2]
    safe_cut = aggregated_data.iloc[0][3]
    safe_inc_cut = aggregated_data.iloc[1][3]
    def ave_ct(safe, safe_inc):
        return (safe * safe_num + safe_inc * safe_inc_num) / (safe_num + safe_inc_num)

    print('safe number', safe_num + safe_inc_num)
    print('safe time', ave_ct(safe_time, safe_inc_time))
    print('safe domain', ave_ct(safe_domain, safe_inc_domain))
    print('safe cut', ave_ct(safe_cut, safe_inc_cut))

    return safe_num + safe_inc_num, ave_ct(safe_time, safe_inc_time), safe_domain, safe_cut
    # Example usage
    #results = calculate_custom_metrics(aggregated_data)
    #print(results)


    # Get the index of rows where Status is either 'safe', 'safe-mip', or 'unknown'
    index_of_safe_unknown = df.loc[(df['Status'] == 'safe') | 
                               (df['Status'] == 'safe-mip') | 
                               (df['Status'] == 'unknown')].index

    # Print the index
    print(index_of_safe_unknown)
    if visualize and (df['Status'] == 'safe').empty:
        plot_time_vs_instances(df)



def mip_main(dataset, visualize=False):
    log_file_path = f'/home/duo/log//{dataset}_mip.log'
    biccos_log_file_path = f'/home/duo/log/Old/{dataset}_biccos_base20.log'
    df = log_read(log_file_path)
    df_biccos = log_read(biccos_log_file_path)
    df['Cut Generation'] = df_biccos['Cut Generation']

    print(df.loc[(df['Status'] == 'safe') | (df['Status'] == 'safe-mip')])
    print(df.loc[df['Status'] == 'unknown'])
    print("All:\n", df)
    aggregated_data = df.groupby('Status').agg({
        'Time': ['count', 'mean'],
        'Domain Visited': ['mean'],
        #'Cut Generation': ['mean'],
        'BaB Round Count': ['mean']
    })

    if visualize:
        plot_time_vs_instances(df)
    return df

def plot_multiple_dfs(dfs, labels, venus=False, dataset=None, Cut=False, time_limit=200):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Define a color map for different dataframes
    colors = plt.cm.get_cmap('tab10', len(dfs))
    
    # Check if any dataframe has 'Cut Generation' with non-zero values
    if Cut:
        has_cut_generation = any('Cut Generation' in df.columns and df['Cut Generation'].sum() > 0 for df in dfs)
    else:
        has_cut_generation = False
    ax2 = None
    for i, (df, label) in enumerate(zip(dfs, labels)):
        # Filter out rows where Status is 'safe'
        if venus:
            df_filtered = df.loc[(df['Status'] == 'safe-incomplete') | (df['Status'] == 'safe') | (df['Status'] == 'safe-mip') | (df['Status'] == 'safe-mip (timed out)')]
        else:
            df_filtered = df.loc[(df['Status'] == 'safe') | (df['Status'] == 'safe-mip') | (df['Status'] == 'safe-mip (timed out)')]
        
        # Sort the dataframe by Time
        df_filtered = df_filtered.sort_values(by='Time')
        
        # Compute the cumulative count of verified instances
        df_filtered['Cumulative Count'] = range(1, len(df_filtered) + 1)
        
        # Plot Cumulative Count vs. Time
        ax1.plot(df_filtered['Cumulative Count'], df_filtered['Time'], marker='o', linestyle='-', color=colors(i), label=label)
        
        # If 'Cut Generation' exists and has non-zero values, plot it as well
        if Cut:
            if has_cut_generation and 'Cut Generation' in df.columns and df['Cut Generation'].sum() > 0:
                if ax2 is None:
                    ax2 = ax1.twinx()  # Create a second y-axis if needed
                ax2.bar(df_filtered['Cumulative Count'], df_filtered['Cut Generation'], color=colors(i), alpha=0.3, label=f'{label} Cut Generation')
                ax2.set_ylabel('Cut Generation', fontsize=20)

    ax1.set_xlabel(f'Number of Instances Verified in {dataset}', fontsize=20)
    ax1.set_ylabel('Time', fontsize=20)
    ax1.set_title('Number of Instances Verified vs. Time', fontsize=20)
    ax1.grid(True)

    # Set the y-axis limit
    max_time = max(df_filtered['Time'].max(), time_limit)
    ax1.set_ylim(0, max_time + 5)
    
    # Add a horizontal line at y=200 to emphasize the time limitation
    if max_time + 5 > time_limit:
        ax1.axhline(y=time_limit, color='red', linestyle='--', linewidth=2, label='Time Limit (200)')
    
    # Combine legends from both y-axes if 'Cut Generation' was plotted
    lines, labels = ax1.get_legend_handles_labels()
    if has_cut_generation:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, fontsize=20)
    else:
        ax1.legend(lines, labels, fontsize=20)
    
    plt.show()

def command_gen(method, version, mts_param=(50,200,3), hard=False, extra_cmd=''):
    SDP_datasets = ['cifar_cnn_a_mix', 'cifar_cnn_a_mix4', 'cifar_cnn_a_adv', 'cifar_cnn_a_adv4', 'cifar_cnn_b_adv', 'cifar_cnn_b_adv4', 'mnist_cnn_a_adv', 'oval22']
    large_datasets = ['resnet_A', 'resnet_B', 'cifar100_small_2022','cifar100_med_2022','cifar100_large_2022','cifar100_super_2022','tinyimagenet_2022', 'cifar10_resnet', 'cifar100', 'tinyimagenet']
    hard_dict = {
        'cifar_cnn_a_mix': [  0,   2,   7,  11,  13,  18,  19,  24,  26,  35,  39,  40,  42,  52,
        54,  55,  59,  60,  64,  67,  70,  75,  76,  83,  97, 101, 102, 103,
       106, 107, 111, 112, 115, 116, 135, 136, 139, 142, 143, 150, 152, 160,
       162, 166, 172, 174, 178, 185, 187, 188, 189, 191, 192, 194, 199],
        'cifar_cnn_a_mix4': [2,  13,  19,  24,  38,  48,  54,  57,  59,  64,  66,  67,  76,  77,
        83,  93,  97, 104, 111, 120, 127, 132, 135, 137, 154, 158, 160, 162,
       172, 181, 185, 187, 189, 191, 198, 199],
        'cifar_cnn_a_adv': [ 11,  24,  38,  42,  54,  55,  60,  70,  84,  89,  95,  97, 103, 107,
       112, 116, 132, 139, 158, 160, 167, 169, 172, 183, 187, 195, 198],
        'cifar_cnn_a_adv4': [24, 52, 57, 64, 67, 97, 123, 128, 137, 159, 169, 178, 181, 183, 188],
        'cifar_cnn_b_adv': [  2,   6,   7,  11,  14,  18,  19,  35,  37,  38,  39,  42,  48,  49,
        51,  52,  53,  54,  55,  56,  59,  62,  64,  67,  70,  72,  73,  75,
        76,  79,  80,  81,  83,  84,  85,  93,  95,  97,  99, 101, 102, 103,
       104, 106, 107, 111, 112, 115, 120, 122, 124, 127, 131, 132, 135, 136,
       139, 142, 144, 148, 149, 150, 154, 159, 160, 162, 163, 164, 166, 172,
       174, 181, 184, 185, 186, 187, 188, 189, 190, 191, 194, 195, 197, 199],
        'cifar_cnn_b_adv4': [  6,  11,  13,  18,  19,  21,  24,  37,  48,  54,  56,  64,  72,  75,
        76,  83,  91,  93,  95,  97,  98, 113, 122, 127, 134, 139, 144, 149,
       162, 174, 175, 185, 187, 189, 191, 192, 199],
        'mnist_cnn_a_adv': [  1,   2,   3,   4,   8,   9,  10,  11,  15,  16,  17,  18,  19,  20,
        21,  22,  23,  24,  25,  26,  27,  29,  33,  35,  36,  37,  39,  41,
        42,  44,  48,  49,  50,  51,  52,  53,  57,  58,  59,  62,  63,  66,
        67,  68,  69,  74,  75,  76,  79,  81,  82,  85,  87,  88,  89,  90,
        94,  95,  96,  98,  99, 100, 101, 102, 103, 104, 106, 107, 108, 109,
       110, 111, 114, 116, 117, 119, 123, 125, 127, 128, 129, 131, 133, 135,
       136, 137, 138, 141, 143, 145, 146, 147, 149, 150, 155, 156, 159, 160,
       162, 163, 164, 166, 168, 169, 170, 176, 178, 179, 183, 186, 189, 190,
       191, 192, 194, 195, 196, 197, 198],
       'oval22': [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 15, 16, 17, 19,
       20, 21, 22, 23, 24, 25, 26, 29], 
       'cifar10_resnet': [ 0,  1,  6, 10, 14, 16, 20, 22, 27, 28, 31, 32, 33, 35, 36, 38, 43, 44,
       47, 48, 49, 52, 56, 59, 61, 62, 63, 64, 65, 66, 68, 69], 
       'cifar100': [  1,   2,   3,   4,   5,   7,   8,  10,  12,  13,  15,  17,  18,  19,
        20,  21,  22,  23,  24,  27,  29,  31,  32,  33,  35,  36,  38,  39,
        40,  41,  42,  43,  44,  45,  47,  48,  50,  52,  53,  54,  55,  56,
        57,  58,  59,  60,  61,  64,  65,  66,  67,  68,  69,  70,  71,  72,
        73,  74,  75,  77,  79,  81,  82,  83,  84,  85,  86,  88,  89,  90,
        92,  93,  94,  95,  97,  98,  99, 100, 101, 102, 103, 104, 105, 108,
       109, 110, 111, 112, 113, 114, 115, 116, 119, 120, 122, 124, 125, 126,
       127, 128, 129, 130, 131, 133, 134, 136, 137, 138, 140, 141, 142, 143,
       144, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 160,
       161, 162, 164, 166, 168, 169, 170, 171, 172, 173, 175, 177, 179, 182,
       183, 184, 186, 188, 193, 194, 196, 197, 198, 199], 
       'tinyimagenet': [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  13,  14,
        15,  16,  17,  18,  20,  21,  22,  23,  24,  25,  26,  27,  29,  30,
        31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
        45,  46,  47,  48,  49,  50,  52,  53,  54,  55,  57,  58,  59,  60,
        61,  62,  63,  64,  66,  67,  68,  69,  71,  72,  73,  75,  77,  78,
        79,  80,  81,  82,  83,  84,  85,  86,  87,  89,  90,  92,  93,  94,
        95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
       109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
       124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
       138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
       152, 153, 154, 155, 156, 157, 158, 159, 162, 163, 164, 165, 166, 167,
       168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
       182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
       196, 198, 199], 
    }
    if method == 'mts':
        command = f' --no_interm_transfer --multi_tree_branching --multi_tree_branching_keep_n_best_domains {mts_param[0]} --multi_tree_branching_target_batch_size {mts_param[1]} --multi_tree_branching_iterations {mts_param[2]}'
        #command = f'--number_cuts 300 --multi_tree_branching --multi_tree_branching_restore_best_tree --multi_tree_branching_keep_n_best_domains {mts_param[0]} --multi_tree_branching_target_batch_size {mts_param[1]} --multi_tree_branching_iterations {mts_param[2]}'
    elif method == 'mts_pure':
        command = f'--no_interm_transfer --no_biccos_constraint_strengthening --multi_tree_branching --multi_tree_branching_restore_best_tree --multi_tree_branching_keep_n_best_domains {mts_param[0]} --multi_tree_branching_target_batch_size {mts_param[1]} --multi_tree_branching_iterations {mts_param[2]}'
    elif method == 'combine':
        command = '--disable_auto_biccos_param --cplex_cuts --initial_max_domains 1'
    elif method == 'combine_mts':
        command = f' --cplex_cuts --initial_max_domains 1 --no_interm_transfer --multi_tree_branching --multi_tree_branching_keep_n_best_domains {mts_param[0]} --multi_tree_branching_target_batch_size {mts_param[1]} --multi_tree_branching_iterations {mts_param[2]}'
    elif method == 'cplex_outputcut':
        command = ' --cplex_cuts --initial_max_domains 1 --mip_add_output_cut --no_biccos_constraint_strengthening'# --tree_traversal depth_first --branching_reduceop min'
    elif method == 'cplex':
        command = ' --cplex_cuts --initial_max_domains 1 --no_biccos_constraint_strengthening --biccos_dropping_heuristics None'# --tree_traversal depth_first --branching_reduceop min'
    elif method == 'sc':
        command = f' --sanity_check Worst --no_interm_transfer --multi_tree_branching --multi_tree_branching_keep_n_best_domains {mts_param[0]} --multi_tree_branching_target_batch_size {mts_param[1]} --multi_tree_branching_iterations {mts_param[2]}' #--biccos_dropping_heuristics None'
    elif method == 'beta':
        command = '--number_cuts 60 --no_biccos_constraint_strengthening'
    elif method == 'auto':
        command = ' '
    else:
        command = '--disable_auto_biccos_param --no_interm_transfer --number_cuts 300' # --disable_pruning_in_iteration '
    datasets = SDP_datasets + large_datasets
    if hard:
        for dataset in datasets:
            instances = hard_dict[dataset] if dataset in hard_dict else []
            print(f'python abcrown.py --config exp_configs/BICCOS/{dataset}.yaml {command} --select_instance {print_numbers_separated_by_space(instances)} &> ../../log/exp/{dataset}_{method}_{version}.log')
    else:
        for dataset in datasets:
            print(f'python abcrown.py --config exp_configs/BICCOS/{dataset}.yaml {command} {extra_cmd} &> ../../log/exp/{dataset}_{method}_{version}.log')
    print('')
    for dataset in datasets:
        print(f'main(\'../log/exp/{dataset}_{method}_{version}.log\')')