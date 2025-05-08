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

    return (
        distance_matrix,
        hdbscan,
        jensenshannon,
        json,
        lmw,
        np,
        os,
        pd,
        plt,
        sns,
        umap,
        warnings,
    )


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
def _(os):
    MC_range = range(2, 21)
    base_directory = 'example'
    lda_directory = os.path.join(base_directory, 'lda_loop')
    output_directory_path = lda_directory
    return MC_range, base_directory, lda_directory, output_directory_path


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


if __name__ == "__main__":
    app.run()
