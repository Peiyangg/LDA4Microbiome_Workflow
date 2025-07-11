import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import little_mallet_wrapper as lmw
    import json
    import umap
    import hdbscan
    from scipy.spatial.distance import jensenshannon
    from scipy.spatial import distance_matrix

    # from glob import glob
    import glob
    from collections import defaultdict
    import re
    from StripeSankey import StripeSankeyInline
    import xml.etree.ElementTree as ET
    return (
        ET,
        StripeSankeyInline,
        defaultdict,
        distance_matrix,
        glob,
        hdbscan,
        jensenshannon,
        json,
        lmw,
        np,
        os,
        pd,
        plt,
        re,
        sns,
        umap,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""# OLD (find semantic clusters)""")
    return


@app.cell(hide_code=True)
def _(jensenshannon, np, pd):
    # Functions
    def compute_jsd_matrix_rows(df):
        m = df.shape[0]  # Number of rows
        jsd_matrix = np.zeros((m, m))

        for i in range(m):
            for j in range(m):
                if i != j:
                    # Calculate JSD between rows i and j
                    jsd_matrix[i, j] = jensenshannon(df.iloc[i, :], df.iloc[j, :])**2

        return pd.DataFrame(jsd_matrix, columns=df.index, index=df.index)
    return (compute_jsd_matrix_rows,)


@app.cell
def _():
    # Inputs
    return


@app.cell
def _():
    # MC_range = range(2, 21)
    # base_directory = 'example'
    # lda_directory = os.path.join(base_directory, 'lda_loop')
    # output_directory_path = lda_directory
    return


@app.cell
def _(output_directory_path):
    path_to_all_df_probabilities_rel = output_directory_path + '/all_MC_probabilities_rel_2_20.csv'
    path_to_all_metrics = output_directory_path + '/all_MC_metrics_2_20.csv'
    return path_to_all_df_probabilities_rel, path_to_all_metrics


@app.cell
def _(path_to_all_df_probabilities_rel, path_to_all_metrics, pd):
    all_df_probabilities_rel = pd.read_csv(path_to_all_df_probabilities_rel, index_col=0)
    all_metrics = pd.read_csv(path_to_all_metrics, index_col=0)
    return all_df_probabilities_rel, all_metrics


@app.cell(hide_code=True)
def _(MC_range, all_metrics, np, output_directory_path):
    import xml.etree.ElementTree as ET
    def calculate_mean_coherence(xml_file_path):
        # Potential error sources and improvements
        try:
            # Parse the XML file
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # Potential error sources:
            # 1. Make sure the XML structure matches your expected path
            # 2. Verify the 'coherence' attribute exists
            coherence_scores = []

            # More robust way to extract coherence values
            for topic in root.findall('.//topic'):
                # Error might occur if:
                # - 'coherence' attribute is missing
                # - 'coherence' value cannot be converted to float
                try:
                    coherence_value = float(topic.get('coherence', 'NaN'))
                    # Optional: Skip NaN or invalid values
                    if not np.isnan(coherence_value):
                        coherence_scores.append(coherence_value)
                except (TypeError, ValueError) as e:
                    print(f"Error processing topic: {e}")

            # Check if any valid scores were found
            if not coherence_scores:
                raise ValueError("No valid coherence scores found in the XML")

            return np.mean(coherence_scores)

        except ET.ParseError:
            print("Error parsing XML file. Check the file format.")
            return None
        except FileNotFoundError:
            print(f"File not found: {xml_file_path}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


    mean_coherence=[]
    for numbers in MC_range:
        diagnosis_file_path=output_directory_path + f"/mallet.diagnostics.{numbers}.xml"
        mean_value=calculate_mean_coherence(diagnosis_file_path)
        mean_coherence.append(mean_value)

    all_metrics['Coherence']=mean_coherence
    return (
        ET,
        calculate_mean_coherence,
        diagnosis_file_path,
        mean_coherence,
        mean_value,
        numbers,
    )


@app.cell
def _(MC_range, pd):
    index_names = []
    num_dcs = []
    dcs = []

    # Generate the patterns for range(2, 21)
    for num_topics_index in MC_range:
        for dc in range(1, num_topics_index + 1):
            index_names.append(f"{num_topics_index}_{dc}")
            num_dcs.append(num_topics_index)
            dcs.append(dc)

    # Create DataFrame
    metadata = pd.DataFrame({
        'index_name': index_names,
        'number_of_DC': num_dcs,
        'DC': dcs
    })
    return dc, dcs, index_names, metadata, num_dcs, num_topics_index


@app.cell
def _(all_df_probabilities_rel, compute_jsd_matrix_rows, hdbscan, umap):
    jsd_matrix = compute_jsd_matrix_rows(all_df_probabilities_rel)
    jsd_matrix_filled = jsd_matrix.fillna(0)
    print("Jensen-Shannon Divergence Matrix done")

    # JSD Matrix Processing
    umap_10d_jsd = umap.UMAP(n_components=10, random_state=42,  metric='precomputed')
    data_10d_jsd = umap_10d_jsd.fit_transform(jsd_matrix_filled)

    hdbscan_clusterer_jsd = hdbscan.HDBSCAN(min_cluster_size=7, cluster_selection_method='leaf') 
    cluster_labels_jsd = hdbscan_clusterer_jsd.fit_predict(data_10d_jsd)

    umap_2d_jsd = umap.UMAP(n_components=2, random_state=42, metric='precomputed')
    data_2d_jsd = umap_2d_jsd.fit_transform(jsd_matrix_filled)
    return (
        cluster_labels_jsd,
        data_10d_jsd,
        data_2d_jsd,
        hdbscan_clusterer_jsd,
        jsd_matrix,
        jsd_matrix_filled,
        umap_10d_jsd,
        umap_2d_jsd,
    )


@app.cell
def _(base_directory, cluster_labels_jsd, data_2d_jsd, os, plt, sns):
    # Plot for JSD matrix
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=data_2d_jsd[:, 0], 
        y=data_2d_jsd[:, 1], 
        hue=cluster_labels_jsd, 
        palette='Spectral', 
        s=100,
        legend='full'
    )

    plt.title("UMAP 2D projection (JSD) colored by HDBSCAN clusters")
    # Hide tick labels and tick marks but keep the border
    plt.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    # Add border/stroke
    plt.box('on')

    # Adjust legend position and size
    plt.legend(
        title='Cluster',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0,
        ncol=1
    )

    plt.tight_layout()

    output_path_2 = os.path.join(base_directory, 'umap_jsd_clusters.png')
    plt.savefig(output_path_2, 
                dpi=600,              # Very high DPI for publication quality
                bbox_inches='tight',   # Ensure the entire plot is saved
                format='png',         # Save as PNG for better quality
                facecolor='white',    # Ensure white background
                edgecolor='none',     # No edge color
                transparent=False)    # Non-transparent background
    return (output_path_2,)


@app.cell
def _(cluster_labels_jsd, metadata):
    jsd=cluster_labels_jsd.tolist()
    metadata['jsd_hdbscan']=jsd

    distinct_clusters = metadata[metadata['jsd_hdbscan'] != -1].groupby('number_of_DC')['jsd_hdbscan'].nunique()
    df_distinct_clusters = distinct_clusters.reset_index()
    df_distinct_clusters.columns = ['Number of DC', 'Distinct hdbscan Clusters']
    return df_distinct_clusters, distinct_clusters, jsd


@app.cell
def _(
    all_metrics,
    base_directory,
    df_distinct_clusters,
    os,
    output_directory_path,
    plt,
):
    # Create figure with a specific size and high DPI
    fig = plt.figure(figsize=(20, 6), dpi=300)
    # Create a grid for subplots - 1 row, 3 columns
    gs = fig.add_gridspec(1, 3)

    # Plot perplexity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(all_metrics['Num_MCs'], all_metrics['Perplexity'], marker='o', label='Perplexity', linewidth=2)
    ax1.set_title('Perplexity', fontsize=12, pad=10)
    ax1.set_xlabel('Number of DC', fontsize=10)
    ax1.set_ylabel('Perplexity', fontsize=10)
    # Remove grid
    ax1.grid(False)
    # Set integer ticks for x-axis
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Plot coherence
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(all_metrics['Num_MCs'], all_metrics['Coherence'], marker='o', color='orange', label='Coherence', linewidth=2)
    ax2.set_title('Coherence Score', fontsize=12, pad=10)
    ax2.set_xlabel('Number of DC', fontsize=10)
    ax2.set_ylabel('Coherence Score', fontsize=10)
    # Remove grid
    ax2.grid(False)  # Fixed: was ax1.grid(False)
    # Set integer ticks for x-axis
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Fixed: was ax1.xaxis

    # Add table
    ax3 = fig.add_subplot(gs[0, 2])
    # Hide axes for table subplot
    ax3.axis('off')
    # Create table
    table_data = df_distinct_clusters.values
    column_labels = df_distinct_clusters.columns
    table = ax3.table(cellText=table_data,
                     colLabels=column_labels,
                     cellLoc='center',
                     loc='center')
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.39)
    # Add title for table
    ax3.set_title('Clustering of Jensen-Shannon Divergence', fontsize=12, pad=10)

    plt.tight_layout()

    # Save plot with high DPI to output directory
    output_path = os.path.join(output_directory_path, 'metrics_comparison.png')
    plt.savefig(output_path, 
                dpi=600,              # Very high DPI for publication quality
                bbox_inches='tight',   # Ensure the entire plot is saved
                format='png',         # Save as PNG for better quality
                facecolor='white',    # Ensure white background
                edgecolor='none',     # No edge color
                transparent=False,     # Non-transparent background
                pad_inches=0.1)       # Add small padding around the plot

    # Also save as PDF for vector graphics
    pdf_path = os.path.join(base_directory, 'metrics_comparison.pdf')
    plt.savefig(pdf_path,
                bbox_inches='tight',
                format='pdf',
                facecolor='white',
                edgecolor='none',
                transparent=False,
                pad_inches=0.1)

    plt.show()
    return (
        ax1,
        ax2,
        ax3,
        column_labels,
        fig,
        gs,
        output_path,
        pdf_path,
        table,
        table_data,
    )


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# NEW (StripeSankey)""")
    return


@app.cell(hide_code=True)
def _():
    # class StripeSankeyDataProcessor:
    #     def __init__(self, sample_mc_folder, mc_feature_folder,k_range, high_threshold,medium_threshold):
    #         self.sample_mc_folder = sample_mc_folder
    #         self.mc_feature_folder = mc_feature_folder
    #         self.k_range = k_range  # K from 2 to 10

    #         # Thresholds for representation levels
    #         self.high_threshold = high_threshold
    #         self.medium_threshold = medium_threshold

    #     def load_sample_mc_data(self):
    #         """Load all sample-MC probability files"""
    #         sample_mc_data = {}

    #         for k in self.k_range:
    #             filename = f'MC_Sample_Probabilities{k}.csv'
    #             filepath = os.path.join(self.sample_mc_folder, filename)

    #             if os.path.exists(filepath):
    #                 df = pd.read_csv(filepath, index_col=0)  # First column is MC names, rows=MCs, cols=samples
    #                 sample_mc_data[k] = df
    #                 print(f"Loaded K={k}: {df.shape[0]} topics (MCs), {df.shape[1]} samples")

    #                 # Sanity check - K=k should have exactly k topics (rows)
    #                 if df.shape[0] != k:
    #                     print(f"WARNING: K={k} has {df.shape[0]} topics, expected {k}")
    #             else:
    #                 print(f"File not found: {filename}")

    #         return sample_mc_data

    #     def categorize_sample_assignments(self, sample_mc_data):
    #         """
    #         For each topic at each K, categorize samples into high/medium representation levels
    #         Data structure: rows=MCs, columns=samples
    #         """
    #         categorized_data = {}

    #         for k, df in sample_mc_data.items():
    #             k_data = {
    #                 'nodes': {},  # topic_id -> {high_samples: [], medium_samples: [], high_count: int, medium_count: int, total_prob: float}
    #                 'sample_assignments': {}  # sample_id -> {assigned_topic: str, probability: float, level: str}
    #             }

    #             # Process each topic (row in the DataFrame)
    #             for topic_idx in range(df.shape[0]):
    #                 topic_name = f"K{k}_MC{topic_idx}"

    #                 # Get all sample probabilities for this topic (row across all samples)
    #                 topic_probs = df.iloc[topic_idx, :]  # This topic's probabilities across all samples

    #                 # Categorize samples by their probability for this topic
    #                 high_samples = []
    #                 medium_samples = []
    #                 total_prob = 0

    #                 for sample_name, prob in topic_probs.items():
    #                     if prob >= self.high_threshold:
    #                         high_samples.append((sample_name, prob))
    #                         total_prob += prob
    #                     elif prob >= self.medium_threshold:
    #                         medium_samples.append((sample_name, prob))
    #                         total_prob += prob

    #                 k_data['nodes'][topic_name] = {
    #                     'high_samples': high_samples,
    #                     'medium_samples': medium_samples,
    #                     'high_count': len(high_samples),
    #                     'medium_count': len(medium_samples),
    #                     'total_probability': total_prob
    #                 }

    #             # For each sample, find its PRIMARY topic assignment (highest probability above threshold)
    #             for sample_idx, sample_name in enumerate(df.columns):
    #                 sample_column = df.iloc[:, sample_idx]  # All topic probabilities for this sample

    #                 # Find the topic with highest probability above minimum threshold
    #                 max_prob = 0
    #                 assigned_topic = None
    #                 assignment_level = None

    #                 for topic_idx, prob in enumerate(sample_column):
    #                     if prob >= self.medium_threshold and prob > max_prob:
    #                         max_prob = prob
    #                         assigned_topic = f"K{k}_MC{topic_idx}"
    #                         assignment_level = 'high' if prob >= self.high_threshold else 'medium'

    #                 if assigned_topic:
    #                     k_data['sample_assignments'][sample_name] = {
    #                         'assigned_topic': assigned_topic,
    #                         'probability': max_prob,
    #                         'level': assignment_level
    #                     }

    #             categorized_data[k] = k_data
    #             print(f"K={k}: {len(k_data['sample_assignments'])} samples assigned to topics")

    #         return categorized_data

    #     def calculate_flows(self, categorized_data):
    #         """Calculate flows between consecutive K values based on sample reassignments"""
    #         flows = []

    #         k_values = sorted(categorized_data.keys())

    #         for i in range(len(k_values) - 1):
    #             source_k = k_values[i]
    #             target_k = k_values[i + 1]

    #             source_assignments = categorized_data[source_k]['sample_assignments']
    #             target_assignments = categorized_data[target_k]['sample_assignments']

    #             # Track flows between specific segments (topic + level combinations)
    #             flow_counts = defaultdict(lambda: defaultdict(int))
    #             flow_samples = defaultdict(lambda: defaultdict(list))

    #             # Find samples that have assignments in both K values
    #             common_samples = set(source_assignments.keys()) & set(target_assignments.keys())
    #             print(f"K{source_k}→K{target_k}: {len(common_samples)} samples to track")

    #             for sample in common_samples:
    #                 source_info = source_assignments[sample]
    #                 target_info = target_assignments[sample]

    #                 # Create segment identifiers (topic + representation level)
    #                 source_segment = f"{source_info['assigned_topic']}_{source_info['level']}"
    #                 target_segment = f"{target_info['assigned_topic']}_{target_info['level']}"

    #                 flow_counts[source_segment][target_segment] += 1
    #                 flow_samples[source_segment][target_segment].append({
    #                     'sample': sample,
    #                     'source_prob': source_info['probability'],
    #                     'target_prob': target_info['probability']
    #                 })

    #             # Convert to flow records
    #             for source_segment, targets in flow_counts.items():
    #                 for target_segment, count in targets.items():
    #                     if count > 0:  # Only include actual flows
    #                         avg_prob = np.mean([
    #                             (s['source_prob'] + s['target_prob']) / 2 
    #                             for s in flow_samples[source_segment][target_segment]
    #                         ])

    #                         flows.append({
    #                             'source_k': source_k,
    #                             'target_k': target_k,
    #                             'source_segment': source_segment,
    #                             'target_segment': target_segment,
    #                             'sample_count': count,
    #                             'average_probability': avg_prob,
    #                             'samples': flow_samples[source_segment][target_segment]
    #                         })

    #         print(f"Total flows calculated: {len(flows)}")
    #         return flows

    #     def prepare_sankey_data(self):
    #         """Main function to prepare all data for Sankey diagram"""
    #         print("Loading sample-MC data...")
    #         sample_mc_data = self.load_sample_mc_data()

    #         if not sample_mc_data:
    #             print("❌ No data loaded. Check your file paths and naming.")
    #             return None, None

    #         print("\nCategorizing sample assignments...")
    #         categorized_data = self.categorize_sample_assignments(sample_mc_data)

    #         print("\nCalculating flows...")
    #         flows = self.calculate_flows(categorized_data)

    #         # Prepare final data structure for StripeSankey
    #         sankey_data = {
    #             'nodes': {},
    #             'flows': flows,
    #             'k_range': list(sample_mc_data.keys()),  # Only include K values we actually have
    #             'thresholds': {
    #                 'high': self.high_threshold,
    #                 'medium': self.medium_threshold
    #             },
    #             'metadata': {
    #                 'total_samples': sample_mc_data[list(sample_mc_data.keys())[0]].shape[1] if sample_mc_data else 0,  # columns = samples
    #                 'k_values_processed': list(sample_mc_data.keys())
    #             }
    #         }

    #         # Collect all node data
    #         for k, k_data in categorized_data.items():
    #             for topic_name, node_data in k_data['nodes'].items():
    #                 sankey_data['nodes'][topic_name] = node_data

    #         print(f"\n✅ Data processing complete!")
    #         print(f"📊 Summary:")
    #         print(f"   - K values: {sankey_data['k_range']}")
    #         print(f"   - Total nodes: {len(sankey_data['nodes'])}")
    #         print(f"   - Total flows: {len(flows)}")
    #         print(f"   - Samples tracked: {sankey_data['metadata']['total_samples']}")

    #         # Show node summary by K
    #         for k in sankey_data['k_range']:
    #             k_nodes = [name for name in sankey_data['nodes'].keys() if name.startswith(f'K{k}_')]
    #             total_assigned = sum(data['high_count'] + data['medium_count'] 
    #                                for name, data in sankey_data['nodes'].items() 
    #                                if name.startswith(f'K{k}_'))
    #             print(f"   - K={k}: {len(k_nodes)} topics, {total_assigned} total sample assignments")

    #         return sankey_data, categorized_data

    #     def save_processed_data(self, sankey_data, output_path='sankey_data.json'):
    #         """Save processed data to JSON file"""
    #         if sankey_data is None:
    #             print("❌ No data to save")
    #             return

    #         # Convert numpy types to native Python types for JSON serialization
    #         def convert_numpy(obj):
    #             if isinstance(obj, np.integer):
    #                 return int(obj)
    #             elif isinstance(obj, np.floating):
    #                 return float(obj)
    #             elif isinstance(obj, np.ndarray):
    #                 return obj.tolist()
    #             return obj

    #         def deep_convert(data):
    #             if isinstance(data, dict):
    #                 return {k: deep_convert(v) for k, v in data.items()}
    #             elif isinstance(data, list):
    #                 return [deep_convert(item) for item in data]
    #             else:
    #                 return convert_numpy(data)

    #         converted_data = deep_convert(sankey_data)

    #         with open(output_path, 'w') as f:
    #             json.dump(converted_data, f, indent=2)

    #         print(f"💾 Data saved to {output_path}")



    #     def get_high_representation_samples(self, topic_id, categorized_data=None):
    #         """
    #         Get sample IDs that have high representation (>0.67) for a specific topic

    #         Args:
    #             topic_id (str): Global topic ID in format 'K{k}_MC{topic_idx}' (e.g., 'K4_MC2')
    #             categorized_data (dict, optional): Pre-computed categorized data. If None, will process data first.

    #         Returns:
    #             dict: {
    #                 'topic_id': str,
    #                 'high_samples': list of sample IDs,
    #                 'high_count': int,
    #                 'sample_details': list of tuples (sample_id, probability)
    #             }
    #         """
    #         # If categorized_data not provided, process the data first
    #         if categorized_data is None:
    #             print("Processing data to get categorized assignments...")
    #             sample_mc_data = self.load_sample_mc_data()
    #             if not sample_mc_data:
    #                 print("❌ No data could be loaded")
    #                 return None
    #             categorized_data = self.categorize_sample_assignments(sample_mc_data)

    #         # Parse the topic_id to extract K value and topic index
    #         try:
    #             # Expected format: K{k}_MC{topic_idx}
    #             parts = topic_id.split('_')
    #             if len(parts) != 2 or not parts[0].startswith('K') or not parts[1].startswith('MC'):
    #                 raise ValueError(f"Invalid topic_id format. Expected 'K{{k}}_MC{{topic_idx}}', got '{topic_id}'")

    #             k_value = int(parts[0][1:])  # Remove 'K' and convert to int
    #             topic_idx = int(parts[1][2:])  # Remove 'MC' and convert to int

    #         except (ValueError, IndexError) as e:
    #             print(f"❌ Error parsing topic_id '{topic_id}': {e}")
    #             return None

    #         # Check if K value exists in our data
    #         if k_value not in categorized_data:
    #             print(f"❌ K value {k_value} not found in data. Available K values: {list(categorized_data.keys())}")
    #             return None

    #         # Check if topic exists in the K value data
    #         if topic_id not in categorized_data[k_value]['nodes']:
    #             available_topics = [name for name in categorized_data[k_value]['nodes'].keys() 
    #                               if name.startswith(f'K{k_value}_')]
    #             print(f"❌ Topic '{topic_id}' not found. Available topics for K={k_value}: {available_topics}")
    #             return None

    #         # Get the topic data
    #         topic_data = categorized_data[k_value]['nodes'][topic_id]

    #         # Extract high representation samples
    #         high_samples_list = topic_data['high_samples']  # List of tuples (sample_id, probability)
    #         high_sample_ids = [sample_id for sample_id, prob in high_samples_list]

    #         result = {
    #             'topic_id': topic_id,
    #             'k_value': k_value,
    #             'topic_index': topic_idx,
    #             'high_samples': high_sample_ids,
    #             'high_count': len(high_sample_ids),
    #             'sample_details': high_samples_list,
    #             'threshold_used': self.high_threshold
    #         }

    #         print(f"✅ Found {len(high_sample_ids)} samples with high representation (>{self.high_threshold}) for topic '{topic_id}'")

    #         return result


    #     def extract_topic_coherence(self, xml_file_path):
    #         """Extract topic coherence data from a single MALLET diagnostic XML file"""
    #         try:
    #             # Parse the XML file
    #             tree = ET.parse(xml_file_path)
    #             root = tree.getroot()

    #             # List to store topic data
    #             topics_data = []

    #             # Extract data for each topic
    #             for topic in root.findall('topic'):
    #                 # Debug: print what we're getting from XML
    #                 topic_id_raw = topic.get('id')
    #                 print(f"Debug: Found topic with id='{topic_id_raw}'")

    #                 try:
    #                     topic_data = {
    #                         'topic_id': int(topic.get('id')),
    #                         'tokens': float(topic.get('tokens')) if topic.get('tokens') else 0.0,
    #                         'document_entropy': float(topic.get('document_entropy')) if topic.get('document_entropy') else 0.0,
    #                         'word_length': float(topic.get('word-length')) if topic.get('word-length') else 0.0,
    #                         'coherence': float(topic.get('coherence')) if topic.get('coherence') else 0.0,
    #                         'uniform_dist': float(topic.get('uniform_dist')) if topic.get('uniform_dist') else 0.0,
    #                         'corpus_dist': float(topic.get('corpus_dist')) if topic.get('corpus_dist') else 0.0,
    #                         'eff_num_words': float(topic.get('eff_num_words')) if topic.get('eff_num_words') else 0.0,
    #                         'token_doc_diff': float(topic.get('token-doc-diff')) if topic.get('token-doc-diff') else 0.0,
    #                         'rank_1_docs': float(topic.get('rank_1_docs')) if topic.get('rank_1_docs') else 0.0,
    #                         'allocation_ratio': float(topic.get('allocation_ratio')) if topic.get('allocation_ratio') else 0.0,
    #                         'allocation_count': float(topic.get('allocation_count')) if topic.get('allocation_count') else 0.0,
    #                         'exclusivity': float(topic.get('exclusivity')) if topic.get('exclusivity') else 0.0
    #                     }
    #                     topics_data.append(topic_data)

    #                 except (ValueError, TypeError) as e:
    #                     print(f"Warning: Could not parse topic {topic_id_raw}: {e}")
    #                     continue

    #             # Create DataFrame
    #             df = pd.DataFrame(topics_data)
    #             print(f"Extracted {len(df)} topics from {xml_file_path}")
    #             if not df.empty:
    #                 print(f"Topic IDs range: {df['topic_id'].min()} to {df['topic_id'].max()}")

    #             return df

    #         except ET.ParseError as e:
    #             print(f"❌ XML parsing error in {xml_file_path}: {e}")
    #             return pd.DataFrame()
    #         except Exception as e:
    #             print(f"❌ Unexpected error processing {xml_file_path}: {e}")
    #             return pd.DataFrame()

    #     def load_all_mallet_diagnostics_fixed(self, mallet_folder_path):
    #         """
    #         Load all MALLET diagnostic files from a folder and create global topic IDs
    #         Fixed version with better error handling and debugging

    #         Args:
    #             mallet_folder_path (str): Path to folder containing MALLET diagnostic XML files
    #                                      Files should be named like 'mallet.diagnostics.10.xml'

    #         Returns:
    #             pd.DataFrame: Combined dataframe with all topics and global IDs
    #         """
    #         # Find all MALLET diagnostic files
    #         pattern = os.path.join(mallet_folder_path, 'mallet.diagnostics.*.xml')
    #         xml_files = glob(pattern)

    #         if not xml_files:
    #             print(f"❌ No MALLET diagnostic files found in {mallet_folder_path}")
    #             print(f"Expected pattern: mallet.diagnostics.{{k}}.xml")
    #             return pd.DataFrame()

    #         print(f"Found {len(xml_files)} MALLET diagnostic files")

    #         all_topics_data = []

    #         for xml_file in sorted(xml_files):
    #             # Extract K value from filename
    #             filename = os.path.basename(xml_file)
    #             k_match = re.search(r'mallet\.diagnostics\.(\d+)\.xml', filename)

    #             if not k_match:
    #                 print(f"⚠️ Warning: Could not extract K value from filename {filename}")
    #                 continue

    #             k_value = int(k_match.group(1))
    #             print(f"\nProcessing {filename} -> K={k_value}")

    #             # Load topic data for this K value
    #             topics_df = self.extract_topic_coherence(xml_file)

    #             if topics_df.empty:
    #                 print(f"❌ No valid topics extracted from {filename}")
    #                 continue

    #             # Add K value and create global topic ID using the same logic as Sankey
    #             topics_df['k_value'] = k_value
    #             topics_df['global_topic_id'] = topics_df.apply(
    #                 lambda row: f"K{k_value}_MC{int(row['topic_id'])}", axis=1
    #             )

    #             # Add source filename for reference
    #             topics_df['source_file'] = filename

    #             all_topics_data.append(topics_df)

    #             # Show what global IDs we created
    #             sample_ids = topics_df['global_topic_id'].head(3).tolist()
    #             print(f"✅ Loaded K={k_value}: {len(topics_df)} topics")
    #             print(f"   Sample global IDs: {sample_ids}")

    #         if not all_topics_data:
    #             print("❌ No valid data could be loaded from any files")
    #             return pd.DataFrame()

    #         # Combine all data
    #         combined_df = pd.concat(all_topics_data, ignore_index=True)

    #         # Reorder columns to put global_topic_id first
    #         cols = ['global_topic_id', 'k_value', 'topic_id'] + [col for col in combined_df.columns 
    #                                                               if col not in ['global_topic_id', 'k_value', 'topic_id']]
    #         combined_df = combined_df[cols]

    #         print(f"\n✅ Combined MALLET diagnostics loaded successfully!")
    #         print(f"📊 Total topics: {len(combined_df)}")
    #         print(f"📊 K values: {sorted(combined_df['k_value'].unique())}")

    #         # Show sample of final global IDs
    #         sample_global_ids = combined_df['global_topic_id'].head(10).tolist()
    #         print(f"📊 Sample global IDs: {sample_global_ids}")

    #         return combined_df

    #     def integrate_mallet_diagnostics_fixed(self, sankey_data, mallet_folder_path):
    #         """
    #         Fixed version of MALLET integration with better debugging
    #         """
    #         print("🔧 Using FIXED MALLET integration...")

    #         # Load MALLET diagnostics using the fixed function
    #         mallet_df = self.load_all_mallet_diagnostics_fixed(mallet_folder_path)

    #         if mallet_df.empty:
    #             print("❌ No MALLET data to integrate")
    #             return sankey_data

    #         print(f"\n🔍 INTEGRATION DEBUG:")
    #         print(f"Sankey topics (first 5): {list(sankey_data['nodes'].keys())[:5]}")
    #         print(f"MALLET topics (first 5): {mallet_df['global_topic_id'].head().tolist()}")

    #         # Create a dictionary for fast lookup
    #         mallet_dict = mallet_df.set_index('global_topic_id').to_dict('index')

    #         # Track integration statistics
    #         integrated_count = 0
    #         missing_count = 0
    #         missing_topics = []

    #         # Integrate MALLET data into existing sankey nodes
    #         for topic_id, node_data in sankey_data['nodes'].items():
    #             if topic_id in mallet_dict:
    #                 # Add all MALLET diagnostic metrics to the node
    #                 mallet_data = mallet_dict[topic_id]

    #                 # Add MALLET metrics to node data
    #                 node_data['mallet_diagnostics'] = {
    #                     'coherence': mallet_data['coherence'],
    #                     'tokens': mallet_data['tokens'],
    #                     'document_entropy': mallet_data['document_entropy'],
    #                     'word_length': mallet_data['word_length'],
    #                     'uniform_dist': mallet_data['uniform_dist'],
    #                     'corpus_dist': mallet_data['corpus_dist'],
    #                     'eff_num_words': mallet_data['eff_num_words'],
    #                     'token_doc_diff': mallet_data['token_doc_diff'],
    #                     'rank_1_docs': mallet_data['rank_1_docs'],
    #                     'allocation_ratio': mallet_data['allocation_ratio'],
    #                     'allocation_count': mallet_data['allocation_count'],
    #                     'exclusivity': mallet_data['exclusivity']
    #                 }

    #                 integrated_count += 1
    #             else:
    #                 missing_count += 1
    #                 missing_topics.append(topic_id)

    #         # Update metadata
    #         sankey_data['metadata']['mallet_integration'] = {
    #             'integrated_topics': integrated_count,
    #             'missing_topics': missing_count,
    #             'total_mallet_topics': len(mallet_df),
    #             'integration_date': pd.Timestamp.now().isoformat(),
    #             'mallet_folder': mallet_folder_path
    #         }

    #         print(f"\n✅ MALLET integration complete!")
    #         print(f"📊 Integration summary:")
    #         print(f"   - Topics with MALLET data: {integrated_count}")
    #         print(f"   - Topics missing MALLET data: {missing_count}")
    #         print(f"   - Total MALLET topics available: {len(mallet_df)}")

    #         if missing_topics and len(missing_topics) <= 10:
    #             print(f"   - Missing topics: {missing_topics}")
    #         elif missing_topics:
    #             print(f"   - Missing topics (first 10): {missing_topics[:10]}...")

    #         # If still no matches, show some debugging info
    #         if integrated_count == 0:
    #             print("\n🔍 NO MATCHES FOUND - DEBUG INFO:")
    #             print("Available MALLET topic IDs:")
    #             for topic_id in sorted(mallet_dict.keys())[:10]:
    #                 print(f"   '{topic_id}'")
    #             print("\nAvailable Sankey topic IDs:")
    #             for topic_id in sorted(sankey_data['nodes'].keys())[:10]:
    #                 print(f"   '{topic_id}'")

    #         return sankey_data


    #     def load_perplexity_data(self, csv_file_path):
    #         """
    #         Load perplexity data from CSV file

    #         Args:
    #             csv_file_path (str): Path to CSV file containing Num_MCs and Perplexity columns

    #         Returns:
    #             dict: Dictionary mapping K values to perplexity scores
    #         """
    #         try:
    #             # Load the CSV file
    #             df = pd.read_csv(csv_file_path)

    #             # Clean up column names (remove any unnamed columns)
    #             df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    #             print(f"📊 Loaded perplexity data from {csv_file_path}")
    #             print(f"   Columns: {list(df.columns)}")
    #             print(f"   Rows: {len(df)}")

    #             # Check required columns
    #             if 'Num_MCs' not in df.columns:
    #                 print("❌ 'Num_MCs' column not found in CSV")
    #                 return {}

    #             if 'Perplexity' not in df.columns:
    #                 print("❌ 'Perplexity' column not found in CSV")
    #                 return {}

    #             # Create dictionary mapping K values to perplexity scores
    #             perplexity_dict = {}

    #             for _, row in df.iterrows():
    #                 k_value = int(row['Num_MCs'])
    #                 perplexity = float(row['Perplexity'])
    #                 perplexity_dict[k_value] = perplexity

    #             print(f"✅ Loaded perplexity data for K values: {sorted(perplexity_dict.keys())}")
    #             print(f"   Sample data: K=2 -> {perplexity_dict.get(2, 'N/A'):.4f}")

    #             return perplexity_dict

    #         except Exception as e:
    #             print(f"❌ Error loading perplexity data: {e}")
    #             return {}

    #     def integrate_perplexity_data(self, sankey_data, csv_file_path):
    #         """
    #         Integrate perplexity data with existing Sankey data
    #         All topics under the same K value get the same perplexity score

    #         Args:
    #             sankey_data (dict): Existing sankey data structure
    #             csv_file_path (str): Path to CSV file with perplexity data

    #         Returns:
    #             dict: Updated sankey_data with perplexity information
    #         """
    #         # Load perplexity data
    #         perplexity_dict = self.load_perplexity_data(csv_file_path)

    #         if not perplexity_dict:
    #             print("❌ No perplexity data to integrate")
    #             return sankey_data

    #         # Track integration statistics
    #         integrated_count = 0
    #         missing_count = 0
    #         k_values_found = set()
    #         k_values_missing = set()

    #         # Integrate perplexity data into existing sankey nodes
    #         for topic_id, node_data in sankey_data['nodes'].items():
    #             # Extract K value from topic_id (format: K{k}_MC{topic_idx})
    #             try:
    #                 k_part = topic_id.split('_')[0]
    #                 if k_part.startswith('K'):
    #                     k_value = int(k_part[1:])

    #                     if k_value in perplexity_dict:
    #                         # Add perplexity data to node
    #                         if 'model_metrics' not in node_data:
    #                             node_data['model_metrics'] = {}

    #                         node_data['model_metrics']['perplexity'] = perplexity_dict[k_value]
    #                         node_data['model_metrics']['k_value'] = k_value

    #                         integrated_count += 1
    #                         k_values_found.add(k_value)
    #                     else:
    #                         missing_count += 1
    #                         k_values_missing.add(k_value)

    #             except (ValueError, IndexError) as e:
    #                 print(f"⚠️ Warning: Could not parse K value from topic_id '{topic_id}': {e}")
    #                 missing_count += 1

    #         # Update metadata
    #         if 'model_integration' not in sankey_data['metadata']:
    #             sankey_data['metadata']['model_integration'] = {}

    #         sankey_data['metadata']['model_integration']['perplexity'] = {
    #             'integrated_topics': integrated_count,
    #             'missing_topics': missing_count,
    #             'k_values_with_perplexity': sorted(k_values_found),
    #             'k_values_missing_perplexity': sorted(k_values_missing),
    #             'total_k_values_available': len(perplexity_dict),
    #             'integration_date': pd.Timestamp.now().isoformat(),
    #             'source_file': csv_file_path
    #         }

    #         print(f"\n✅ Perplexity integration complete!")
    #         print(f"📊 Integration summary:")
    #         print(f"   - Topics with perplexity data: {integrated_count}")
    #         print(f"   - Topics missing perplexity data: {missing_count}")
    #         print(f"   - K values with perplexity: {sorted(k_values_found)}")

    #         if k_values_missing:
    #             print(f"   - K values missing perplexity: {sorted(k_values_missing)}")

    #         return sankey_data

    #     def integrate_all_model_data(self, sankey_data, mallet_folder_path, perplexity_csv_path):
    #         """
    #         Comprehensive function to integrate both MALLET diagnostics and perplexity data

    #         Args:
    #             sankey_data (dict): Existing sankey data structure
    #             mallet_folder_path (str): Path to folder containing MALLET diagnostic files
    #             perplexity_csv_path (str): Path to CSV file with perplexity data

    #         Returns:
    #             dict: Updated sankey_data with all model metrics integrated
    #         """
    #         print("🔧 Integrating all model data...")
    #         print("=" * 50)

    #         # First integrate MALLET diagnostics
    #         print("1️⃣ Integrating MALLET diagnostics...")
    #         sankey_data = self.integrate_mallet_diagnostics_fixed(sankey_data, mallet_folder_path)

    #         # Then integrate perplexity data
    #         print("\n2️⃣ Integrating perplexity data...")
    #         sankey_data = self.integrate_perplexity_data(sankey_data, perplexity_csv_path)

    #         print("\n✅ All model data integration complete!")

    #         # Show summary of what's available for a sample topic
    #         sample_topic = list(sankey_data['nodes'].keys())[0]
    #         sample_node = sankey_data['nodes'][sample_topic]

    #         print(f"\n📊 Sample topic '{sample_topic}' now contains:")
    #         print(f"   - Sample data: high_count={sample_node.get('high_count', 0)}, medium_count={sample_node.get('medium_count', 0)}")

    #         if 'mallet_diagnostics' in sample_node:
    #             mallet_data = sample_node['mallet_diagnostics']
    #             print(f"   - MALLET data: coherence={mallet_data.get('coherence', 'N/A'):.4f}, exclusivity={mallet_data.get('exclusivity', 'N/A'):.4f}")
    #         else:
    #             print("   - MALLET data: Not available")

    #         if 'model_metrics' in sample_node:
    #             model_data = sample_node['model_metrics']
    #             print(f"   - Model data: perplexity={model_data.get('perplexity', 'N/A'):.4f}")
    #         else:
    #             print("   - Model data: Not available")

    #         return sankey_data

    #     def get_topic_all_metrics(self, topic_id, sankey_data):
    #         """
    #         Get all available metrics for a specific topic (samples, MALLET, perplexity)

    #         Args:
    #             topic_id (str): Global topic ID (e.g., 'K4_MC2')
    #             sankey_data (dict): Sankey data with integrated metrics

    #         Returns:
    #             dict: All available metrics for the topic
    #         """
    #         if topic_id not in sankey_data['nodes']:
    #             print(f"❌ Topic '{topic_id}' not found in sankey data")
    #             return None

    #         node_data = sankey_data['nodes'][topic_id]

    #         # Compile all metrics
    #         all_metrics = {
    #             'topic_id': topic_id,
    #             'sample_metrics': {
    #                 'high_count': node_data.get('high_count', 0),
    #                 'medium_count': node_data.get('medium_count', 0),
    #                 'total_probability': node_data.get('total_probability', 0),
    #                 'high_samples': [sample_id for sample_id, prob in node_data.get('high_samples', [])],
    #                 'medium_samples': [sample_id for sample_id, prob in node_data.get('medium_samples', [])]
    #             }
    #         }

    #         # Add MALLET diagnostics if available
    #         if 'mallet_diagnostics' in node_data:
    #             all_metrics['mallet_diagnostics'] = node_data['mallet_diagnostics']

    #         # Add model metrics if available
    #         if 'model_metrics' in node_data:
    #             all_metrics['model_metrics'] = node_data['model_metrics']

    #         return all_metrics

    #     def export_comprehensive_topic_summary(self, sankey_data, output_path='topic_comprehensive_summary.csv'):
    #         """
    #         Export a comprehensive CSV with all metrics for all topics

    #         Args:
    #             sankey_data (dict): Sankey data with all integrated metrics
    #             output_path (str): Path for output CSV file

    #         Returns:
    #             pd.DataFrame: Summary dataframe
    #         """
    #         summary_data = []

    #         for topic_id, node_data in sankey_data['nodes'].items():
    #             # Parse topic info
    #             try:
    #                 parts = topic_id.split('_')
    #                 k_value = int(parts[0][1:])  # Remove 'K'
    #                 topic_idx = int(parts[1][2:])  # Remove 'MC'
    #             except:
    #                 k_value = None
    #                 topic_idx = None

    #             # Base metrics
    #             row = {
    #                 'topic_id': topic_id,
    #                 'k_value': k_value,
    #                 'topic_index': topic_idx,
    #                 'high_count': node_data.get('high_count', 0),
    #                 'medium_count': node_data.get('medium_count', 0),
    #                 'total_probability': node_data.get('total_probability', 0)
    #             }

    #             # Add MALLET diagnostics
    #             if 'mallet_diagnostics' in node_data:
    #                 mallet_data = node_data['mallet_diagnostics']
    #                 for key, value in mallet_data.items():
    #                     row[f'mallet_{key}'] = value

    #             # Add model metrics
    #             if 'model_metrics' in node_data:
    #                 model_data = node_data['model_metrics']
    #                 for key, value in model_data.items():
    #                     row[f'model_{key}'] = value

    #             summary_data.append(row)

    #         # Create DataFrame
    #         summary_df = pd.DataFrame(summary_data)

    #         # Sort by K value and topic index
    #         if 'k_value' in summary_df.columns and 'topic_index' in summary_df.columns:
    #             summary_df = summary_df.sort_values(['k_value', 'topic_index'])

    #         # Save to CSV
    #         summary_df.to_csv(output_path, index=False)

    #         print(f"✅ Comprehensive topic summary exported to {output_path}")
    #         print(f"📊 Summary: {len(summary_df)} topics with {len(summary_df.columns)} metrics each")

    #         return summary_df
    return


@app.cell
def _(os):
    base_directory          = 'example_test'
    lda_directory = os.path.join(base_directory, 'lda_results')
    MC_sample_directory    =   os.path.join(lda_directory, 'MC_Sample')
    MC_feature_directory    =   os.path.join(lda_directory, 'MC_Feature')
    MALLET_diagnostics_directory = os.path.join(lda_directory, 'Diagnostics')
    MC_range=range(2,5)
    range_str = f"{MC_range.start}_{MC_range.stop-1}"
    return (
        MALLET_diagnostics_directory,
        MC_feature_directory,
        MC_range,
        MC_sample_directory,
        base_directory,
        lda_directory,
        range_str,
    )


@app.cell(hide_code=True)
def _(ET, defaultdict, glob, json, np, os, pd, re):
    class StripeSankeyDataProcessor:
        def __init__(self, sample_mc_folder, mc_feature_folder, k_range, high_threshold, medium_threshold):
            self.sample_mc_folder = sample_mc_folder
            self.mc_feature_folder = mc_feature_folder
            self.k_range = k_range
            self.high_threshold = high_threshold
            self.medium_threshold = medium_threshold

        def load_sample_mc_data(self):
            """Load all sample-MC probability files"""
            sample_mc_data = {}

            for k in self.k_range:
                filename = f'MC_Sample_Probabilities{k}.csv'
                filepath = os.path.join(self.sample_mc_folder, filename)

                if os.path.exists(filepath):
                    df = pd.read_csv(filepath, index_col=0)
                    sample_mc_data[k] = df
                    print(f"Loaded K={k}: {df.shape[0]} topics (MCs), {df.shape[1]} samples")

                    if df.shape[0] != k:
                        print(f"WARNING: K={k} has {df.shape[0]} topics, expected {k}")
                else:
                    print(f"File not found: {filename}")

            return sample_mc_data

        def categorize_sample_assignments(self, sample_mc_data):
            """Categorize samples into high/medium representation levels"""
            categorized_data = {}

            for k, df in sample_mc_data.items():
                k_data = {
                    'nodes': {},
                    'sample_assignments': {}
                }

                # Process each topic
                for topic_idx in range(df.shape[0]):
                    topic_name = f"K{k}_MC{topic_idx}"
                    topic_probs = df.iloc[topic_idx, :]

                    high_samples = []
                    medium_samples = []
                    total_prob = 0

                    for sample_name, prob in topic_probs.items():
                        if prob >= self.high_threshold:
                            high_samples.append((sample_name, prob))
                            total_prob += prob
                        elif prob >= self.medium_threshold:
                            medium_samples.append((sample_name, prob))
                            total_prob += prob

                    k_data['nodes'][topic_name] = {
                        'high_samples': high_samples,
                        'medium_samples': medium_samples,
                        'high_count': len(high_samples),
                        'medium_count': len(medium_samples),
                        'total_probability': total_prob
                    }

                # Find primary topic assignments
                for sample_idx, sample_name in enumerate(df.columns):
                    sample_column = df.iloc[:, sample_idx]
                    max_prob = 0
                    assigned_topic = None
                    assignment_level = None

                    for topic_idx, prob in enumerate(sample_column):
                        if prob >= self.medium_threshold and prob > max_prob:
                            max_prob = prob
                            assigned_topic = f"K{k}_MC{topic_idx}"
                            assignment_level = 'high' if prob >= self.high_threshold else 'medium'

                    if assigned_topic:
                        k_data['sample_assignments'][sample_name] = {
                            'assigned_topic': assigned_topic,
                            'probability': max_prob,
                            'level': assignment_level
                        }

                categorized_data[k] = k_data
                print(f"K={k}: {len(k_data['sample_assignments'])} samples assigned to topics")

            return categorized_data

        def calculate_flows(self, categorized_data):
            """Calculate flows between consecutive K values"""
            flows = []
            k_values = sorted(categorized_data.keys())

            for i in range(len(k_values) - 1):
                source_k = k_values[i]
                target_k = k_values[i + 1]

                source_assignments = categorized_data[source_k]['sample_assignments']
                target_assignments = categorized_data[target_k]['sample_assignments']

                flow_counts = defaultdict(lambda: defaultdict(int))
                flow_samples = defaultdict(lambda: defaultdict(list))

                common_samples = set(source_assignments.keys()) & set(target_assignments.keys())
                print(f"K{source_k}→K{target_k}: {len(common_samples)} samples to track")

                for sample in common_samples:
                    source_info = source_assignments[sample]
                    target_info = target_assignments[sample]

                    source_segment = f"{source_info['assigned_topic']}_{source_info['level']}"
                    target_segment = f"{target_info['assigned_topic']}_{target_info['level']}"

                    flow_counts[source_segment][target_segment] += 1
                    flow_samples[source_segment][target_segment].append({
                        'sample': sample,
                        'source_prob': source_info['probability'],
                        'target_prob': target_info['probability']
                    })

                for source_segment, targets in flow_counts.items():
                    for target_segment, count in targets.items():
                        if count > 0:
                            avg_prob = np.mean([
                                (s['source_prob'] + s['target_prob']) / 2 
                                for s in flow_samples[source_segment][target_segment]
                            ])

                            flows.append({
                                'source_k': source_k,
                                'target_k': target_k,
                                'source_segment': source_segment,
                                'target_segment': target_segment,
                                'sample_count': count,
                                'average_probability': avg_prob,
                                'samples': flow_samples[source_segment][target_segment]
                            })

            print(f"Total flows calculated: {len(flows)}")
            return flows

        def extract_topic_coherence(self, xml_file_path):
            """Extract topic coherence data from a single MALLET diagnostic XML file"""
            try:
                tree = ET.parse(xml_file_path)
                root = tree.getroot()
                topics_data = []

                for topic in root.findall('topic'):
                    topic_id_raw = topic.get('id')
                    try:
                        topic_data = {
                            'topic_id': int(topic.get('id')),
                            'tokens': float(topic.get('tokens')) if topic.get('tokens') else 0.0,
                            'document_entropy': float(topic.get('document_entropy')) if topic.get('document_entropy') else 0.0,
                            'word_length': float(topic.get('word-length')) if topic.get('word-length') else 0.0,
                            'coherence': float(topic.get('coherence')) if topic.get('coherence') else 0.0,
                            'uniform_dist': float(topic.get('uniform_dist')) if topic.get('uniform_dist') else 0.0,
                            'corpus_dist': float(topic.get('corpus_dist')) if topic.get('corpus_dist') else 0.0,
                            'eff_num_words': float(topic.get('eff_num_words')) if topic.get('eff_num_words') else 0.0,
                            'token_doc_diff': float(topic.get('token-doc-diff')) if topic.get('token-doc-diff') else 0.0,
                            'rank_1_docs': float(topic.get('rank_1_docs')) if topic.get('rank_1_docs') else 0.0,
                            'allocation_ratio': float(topic.get('allocation_ratio')) if topic.get('allocation_ratio') else 0.0,
                            'allocation_count': float(topic.get('allocation_count')) if topic.get('allocation_count') else 0.0,
                            'exclusivity': float(topic.get('exclusivity')) if topic.get('exclusivity') else 0.0
                        }
                        topics_data.append(topic_data)
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not parse topic {topic_id_raw}: {e}")
                        continue

                df = pd.DataFrame(topics_data)
                return df

            except ET.ParseError as e:
                print(f"❌ XML parsing error in {xml_file_path}: {e}")
                return pd.DataFrame()
            except Exception as e:
                print(f"❌ Unexpected error processing {xml_file_path}: {e}")
                return pd.DataFrame()

        def load_all_mallet_diagnostics(self, mallet_folder_path):
            """Load all MALLET diagnostic files"""
            pattern = os.path.join(mallet_folder_path, 'mallet.diagnostics.*.xml')
            xml_files = glob.glob(pattern)

            if not xml_files:
                print(f"❌ No MALLET diagnostic files found in {mallet_folder_path}")
                return pd.DataFrame()

            print(f"Found {len(xml_files)} MALLET diagnostic files")
            all_topics_data = []

            for xml_file in sorted(xml_files):
                filename = os.path.basename(xml_file)
                k_match = re.search(r'mallet\.diagnostics\.(\d+)\.xml', filename)

                if not k_match:
                    print(f"⚠️ Warning: Could not extract K value from filename {filename}")
                    continue

                k_value = int(k_match.group(1))
                topics_df = self.extract_topic_coherence(xml_file)

                if topics_df.empty:
                    continue

                topics_df['k_value'] = k_value
                topics_df['global_topic_id'] = topics_df.apply(
                    lambda row: f"K{k_value}_MC{int(row['topic_id'])}", axis=1
                )

                all_topics_data.append(topics_df)

            if not all_topics_data:
                print("❌ No valid MALLET data could be loaded")
                return pd.DataFrame()

            combined_df = pd.concat(all_topics_data, ignore_index=True)
            print(f"✅ Combined MALLET diagnostics: {len(combined_df)} topics")
            return combined_df

        def load_perplexity_data(self, csv_file_path):
            """Load perplexity data from CSV file"""
            try:
                df = pd.read_csv(csv_file_path)
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

                if 'Num_MCs' not in df.columns or 'Perplexity' not in df.columns:
                    print("❌ Required columns not found in perplexity CSV")
                    return {}

                perplexity_dict = {}
                for _, row in df.iterrows():
                    k_value = int(row['Num_MCs'])
                    perplexity = float(row['Perplexity'])
                    perplexity_dict[k_value] = perplexity

                print(f"✅ Loaded perplexity data for K values: {sorted(perplexity_dict.keys())}")
                return perplexity_dict

            except Exception as e:
                print(f"❌ Error loading perplexity data: {e}")
                return {}

        def save_processed_data(self, sankey_data, output_path='sankey_data.json'):
            """Save processed data to JSON file"""
            if sankey_data is None:
                print("❌ No data to save")
                return

            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            def deep_convert(data):
                if isinstance(data, dict):
                    return {k: deep_convert(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [deep_convert(item) for item in data]
                else:
                    return convert_numpy(data)

            converted_data = deep_convert(sankey_data)

            with open(output_path, 'w') as f:
                json.dump(converted_data, f, indent=2)

            print(f"💾 Data saved to {output_path}")

        def process_all_data(self, mallet_folder_path, perplexity_csv_path, output_path='fully_integrated_sankey_data.json'):
            """
            🚀 SINGLE METHOD TO GET FULLY INTEGRATED DATA

            This method does everything in one call:
            1. Loads sample-MC data
            2. Categorizes assignments
            3. Calculates flows
            4. Integrates MALLET diagnostics
            5. Integrates perplexity data
            6. Saves the result

            Returns the fully integrated data ready for use.
            """
            print("🚀 Starting complete data processing pipeline...")
            print("=" * 60)

            # Step 1: Load and process base data
            print("1️⃣ Loading sample-MC data...")
            sample_mc_data = self.load_sample_mc_data()

            if not sample_mc_data:
                print("❌ No sample data loaded. Stopping process.")
                return None

            print("\n2️⃣ Categorizing sample assignments...")
            categorized_data = self.categorize_sample_assignments(sample_mc_data)

            print("\n3️⃣ Calculating flows...")
            flows = self.calculate_flows(categorized_data)

            # Step 2: Build base sankey structure
            print("\n4️⃣ Building base sankey structure...")
            sankey_data = {
                'nodes': {},
                'flows': flows,
                'k_range': list(sample_mc_data.keys()),
                'thresholds': {
                    'high': self.high_threshold,
                    'medium': self.medium_threshold
                },
                'metadata': {
                    'total_samples': sample_mc_data[list(sample_mc_data.keys())[0]].shape[1] if sample_mc_data else 0,
                    'k_values_processed': list(sample_mc_data.keys()),
                    'processing_timestamp': pd.Timestamp.now().isoformat()
                }
            }

            # Collect all node data
            for k, k_data in categorized_data.items():
                for topic_name, node_data in k_data['nodes'].items():
                    sankey_data['nodes'][topic_name] = node_data

            print(f"✅ Base structure created: {len(sankey_data['nodes'])} nodes, {len(flows)} flows")

            # Step 3: Integrate MALLET diagnostics
            print("\n5️⃣ Integrating MALLET diagnostics...")
            mallet_df = self.load_all_mallet_diagnostics(mallet_folder_path)

            if not mallet_df.empty:
                mallet_dict = mallet_df.set_index('global_topic_id').to_dict('index')
                integrated_count = 0

                for topic_id, node_data in sankey_data['nodes'].items():
                    if topic_id in mallet_dict:
                        mallet_data = mallet_dict[topic_id]
                        node_data['mallet_diagnostics'] = {
                            'coherence': mallet_data['coherence'],
                            'tokens': mallet_data['tokens'],
                            'document_entropy': mallet_data['document_entropy'],
                            'word_length': mallet_data['word_length'],
                            'uniform_dist': mallet_data['uniform_dist'],
                            'corpus_dist': mallet_data['corpus_dist'],
                            'eff_num_words': mallet_data['eff_num_words'],
                            'token_doc_diff': mallet_data['token_doc_diff'],
                            'rank_1_docs': mallet_data['rank_1_docs'],
                            'allocation_ratio': mallet_data['allocation_ratio'],
                            'allocation_count': mallet_data['allocation_count'],
                            'exclusivity': mallet_data['exclusivity']
                        }
                        integrated_count += 1

                sankey_data['metadata']['mallet_integration'] = {
                    'integrated_topics': integrated_count,
                    'total_mallet_topics': len(mallet_df)
                }
                print(f"✅ MALLET integration: {integrated_count} topics enriched")
            else:
                print("⚠️ No MALLET data available")

            # Step 4: Integrate perplexity data
            print("\n6️⃣ Integrating perplexity data...")
            perplexity_dict = self.load_perplexity_data(perplexity_csv_path)

            if perplexity_dict:
                integrated_count = 0

                for topic_id, node_data in sankey_data['nodes'].items():
                    try:
                        k_part = topic_id.split('_')[0]
                        if k_part.startswith('K'):
                            k_value = int(k_part[1:])

                            if k_value in perplexity_dict:
                                if 'model_metrics' not in node_data:
                                    node_data['model_metrics'] = {}

                                node_data['model_metrics']['perplexity'] = perplexity_dict[k_value]
                                node_data['model_metrics']['k_value'] = k_value
                                integrated_count += 1

                    except (ValueError, IndexError):
                        continue

                sankey_data['metadata']['perplexity_integration'] = {
                    'integrated_topics': integrated_count,
                    'total_k_values_available': len(perplexity_dict)
                }
                print(f"✅ Perplexity integration: {integrated_count} topics enriched")
            else:
                print("⚠️ No perplexity data available")

            # Step 5: Save the final result
            print("\n7️⃣ Saving fully integrated data...")
            self.save_processed_data(sankey_data, output_path)

            # Final summary
            print("\n" + "=" * 60)
            print("🎉 COMPLETE DATA PROCESSING FINISHED!")
            print(f"📊 Final Summary:")
            print(f"   - K values: {sankey_data['k_range']}")
            print(f"   - Total topics: {len(sankey_data['nodes'])}")
            print(f"   - Total flows: {len(sankey_data['flows'])}")
            print(f"   - Samples tracked: {sankey_data['metadata']['total_samples']}")

            if 'mallet_integration' in sankey_data['metadata']:
                print(f"   - MALLET enriched topics: {sankey_data['metadata']['mallet_integration']['integrated_topics']}")

            if 'perplexity_integration' in sankey_data['metadata']:
                print(f"   - Perplexity enriched topics: {sankey_data['metadata']['perplexity_integration']['integrated_topics']}")

            print(f"   - Data saved to: {output_path}")
            print("=" * 60)

            return sankey_data
    return (StripeSankeyDataProcessor,)


@app.cell
def _(
    MALLET_diagnostics_directory,
    MC_feature_directory,
    MC_range,
    MC_sample_directory,
    StripeSankeyDataProcessor,
    lda_directory,
    range_str,
):
    # Usage example:
    if __name__ == "__main__":
        MC_feature_folder = MC_feature_directory # Replace with actual path
        Sample_MC_folder = MC_sample_directory  
        mallet_folder = MALLET_diagnostics_directory
        preplexity_path = lda_directory + f'/all_MC_metrics_{range_str}.csv'
        # Simple one-call usage
        processor = StripeSankeyDataProcessor(
            sample_mc_folder=Sample_MC_folder,
            mc_feature_folder=MC_feature_folder,
            k_range=MC_range,
            high_threshold=0.67,
            medium_threshold=0.33
        )

        # 🚀 SINGLE CALL TO GET EVERYTHING
        fully_integrated_data = processor.process_all_data(
            mallet_folder_path=mallet_folder,
            perplexity_csv_path=preplexity_path
        )
    return (
        MC_feature_folder,
        Sample_MC_folder,
        fully_integrated_data,
        mallet_folder,
        preplexity_path,
        processor,
    )


@app.cell
def _(StripeSankeyInline, fully_integrated_data):
    widget = StripeSankeyInline(sankey_data=fully_integrated_data, mode='metric')
    widget
    return (widget,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
