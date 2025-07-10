import marimo

__generated_with = "0.11.8"
app = marimo.App(width="full")


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.colors as mcolors
    import random
    import string
    import subprocess
    import os
    from collections import defaultdict
    from gensim import corpora
    from gensim.models import LdaModel, CoherenceModel
    import gensim
    from gensim.corpora import Dictionary

    import little_mallet_wrapper as lmw
    # Path to Mallet binary
    path_to_mallet = 'Mallet-202108/bin/mallet'

    from pathlib import Path
    return (
        CoherenceModel,
        Dictionary,
        LdaModel,
        Path,
        corpora,
        defaultdict,
        gensim,
        lmw,
        mcolors,
        np,
        os,
        path_to_mallet,
        pd,
        plt,
        random,
        sns,
        string,
        subprocess,
        warnings,
    )


@app.cell
def _():
    # get a cut of table just 10, and remove sample ID, and Shuffle numbers
    return


@app.cell(hide_code=True)
def _(
    CoherenceModel,
    Dictionary,
    defaultdict,
    jensenshannon,
    lmw,
    np,
    pd,
    random,
    string,
):
    # functions

    taxa_counts = {}
    def assign_id(genus_based):
        # Initialize count if this is the first time seeing this genus
        if genus_based not in taxa_counts:
            taxa_counts[genus_based] = 0
        else:
            # Increment count for subsequent occurrences
            taxa_counts[genus_based] += 1

        # Return genus name with count suffix
        return f"{genus_based}_{taxa_counts[genus_based]}"

    def generate_single_id(min_length=5, existing_ids=None):
        if existing_ids is None:
            existing_ids = set()

        # Generate a random string with the minimal length
        new_id = ''.join(random.choices(string.ascii_lowercase, k=min_length))

        # If the ID already exists, increase the length or regenerate until we get a unique ID
        while new_id in existing_ids:
            new_id = ''.join(random.choices(string.ascii_lowercase, k=min_length))

        existing_ids.add(new_id)
        return new_id


    def load_mallet_model_output(topic_distributions_path, word_weights_path):
        # Load topic distributions
        topic_distributions = lmw.load_topic_distributions(topic_distributions_path)

        # Load word weights
        word_topics = []
        with open(word_weights_path, 'r') as f:
            for line in f:
                parts = line.split()
                try:
                    if len(parts) < 2:
                        raise ValueError("Line does not have enough parts")

                    word = parts[1]
                    topic_freq_pairs = parts[2:]

                    for pair in topic_freq_pairs:
                        topic_id, frequency = pair.split(':')
                        word_topics.append((int(topic_id), word, int(frequency)))

                except ValueError as e:
                    # Log or print the problematic line for debugging
                    print(f"Skipping line due to format issues: {line} - Error: {e}")

        return topic_distributions, word_topics



    def calculate_perplexity(topic_distributions, epsilon=1e-10):

        perplexities = []

        for distribution in topic_distributions:
            # Ensure the distribution doesn't have zero values by clipping
            distribution = np.clip(distribution, epsilon, 1.0)
            # Calculate the entropy for this distribution
            entropy = -np.sum(np.log(distribution) * distribution)
            # Calculate perplexity and store it
            perplexities.append(np.exp(entropy))

        # Return the average perplexity over all samples
        return np.mean(perplexities)


    def calculate_coherence(word_topics, texts):
        """Calculate coherence score using gensim's CoherenceModel."""
        # Ensure texts are in the correct format
        processed_texts = []
        for text in texts:
            if isinstance(text, str):
                processed_texts.append(text.split())
            elif isinstance(text, list):
                processed_texts.append(text)
            else:
                raise ValueError(f"Unexpected input type: {type(text)}. Expected string or list.")


        # Prepare the data for coherence calculation
        id2word = Dictionary(processed_texts)
        corpus = [id2word.doc2bow(text_1) for text_1 in processed_texts]

        # Group word_topics by topic number
        topics_dict = defaultdict(list)
        for topic_num_1, word_1, freq_1 in word_topics:
            topics_dict[topic_num_1].append((word_1, freq_1))

        # Extract top 10 words for each topic
        topics = []
        for topic_num_2, word_freqs in topics_dict.items():
            # Sort words by frequency and take the top 10
            top_words = [word_2 for word_2, freq_2 in sorted(word_freqs, key=lambda x: x[1], reverse=True)[:10]]
            topics.append(top_words)

        # Calculate coherence
        coherence_model = CoherenceModel(topics=topics, texts=corpus, dictionary=id2word, coherence='c_v')
        return coherence_model.get_coherence()


    def update_genus(row, unknown_count):
        if row['Genus'] == 'uncultured':
            if pd.notna(row['Family']) and row['Family'] != 'uncultured':
                return f"{row['Family']}_uncultured"
            else:
                unknown_count[0] += 1  # Update the count in a mutable list to keep state
                return f"unknown_{unknown_count[0]}_rockwool"
        elif pd.isna(row['Genus']):
            if pd.notna(row['Family']) and row['Family'] != 'uncultured':
                return f"{row['Family']}_unknown"
            else:
                unknown_count[0] += 1  # Update the count in a mutable list to keep state
                return f"unknown_{unknown_count[0]}_rockwool"
        return row['Genus']

    def update_genus_new(row, unknown_count):
        # Case 1: If Genus is 'g__uncultured' or NaN
        if row['Genus'] == 'g__uncultured' or pd.isna(row['Genus']):
            # Try Family first
            if pd.notna(row['Family']) and row['Family'] != 'f__uncultured':
                return f"{row['Family']}"
            # Try Order
            elif pd.notna(row['Order']) and row['Order'] != 'o__uncultured':
                return f"{row['Order']}"
            # Try Class
            elif pd.notna(row['Class']) and row['Class'] != 'c__uncultured':
                return f"{row['Class']}"
            # Try Phylum
            elif pd.notna(row['Phylum']) and row['Phylum'] != 'p__uncultured':
                return f"{row['Phylum']}"
            # If all taxonomic levels are uncultured or NaN, use unknown_count
            else:
                unknown_count[0] += 1
                return f"unknown_{unknown_count[0]}"
        # Case 2: Return original Genus if it's valid
        return row['Genus']

    unknown_count = [0]

    # def assign_unknown_labels_rockwool(df, column):
    #     unknown_count = 0
    #     def label_unknown(value):
    #         nonlocal unknown_count
    #         if pd.isna(value):
    #             unknown_count += 1
    #             return f'unknown_{unknown_count}'
    #         return value

    #     df[column] = df[column].apply(label_unknown)
    #     return df

    def compute_jsd_matrix_rows(df):
        m = df.shape[0]  # Number of rows
        jsd_matrix = np.zeros((m, m))

        for i in range(m):
            for j in range(m):
                if i != j:
                    # Calculate JSD between rows i and j
                    jsd_matrix[i, j] = jensenshannon(df.iloc[i, :], df.iloc[j, :])**2

        return pd.DataFrame(jsd_matrix, columns=df.index, index=df.index)
    return (
        assign_id,
        calculate_coherence,
        calculate_perplexity,
        compute_jsd_matrix_rows,
        generate_single_id,
        load_mallet_model_output,
        taxa_counts,
        unknown_count,
        update_genus,
        update_genus_new,
    )


@app.cell
def _(mo):
    mo.md("""# Inputs:""")
    return


@app.cell
def _():
    asvtable_path           ='example/data/seqtab.pigs.csv'
    taxonomy_path           ='example/data/taxonomy.csv'
    # a directory to story all results and plots
    base_directory          = 'example'
    # model selection range
    MC_range = range(2, 21)
    return MC_range, asvtable_path, base_directory, taxonomy_path


@app.cell
def _(mo):
    mo.md("""# Data preprocessing for MALLET""")
    return


@app.cell
def _(base_directory, os):
    # Create new directory for LDA analysis
    intermediate_directory  =   os.path.join(base_directory, 'intermediate')
    loop_directory          =   os.path.join(base_directory, 'lda_loop')
    lda_directory           =   os.path.join(base_directory, 'lda_results')

    os.makedirs(intermediate_directory, exist_ok=True)
    os.makedirs(loop_directory, exist_ok=True)
    os.makedirs(lda_directory, exist_ok=True)


    # Now you can use lda_directory as your output path
    loop_output_directory_path = loop_directory
    Loop_2tables_directory_path = lda_directory

    path_to_training_data           = loop_output_directory_path + '/training.txt'
    path_to_formatted_training_data = loop_output_directory_path + '/mallet.training'
    return (
        Loop_2tables_directory_path,
        intermediate_directory,
        lda_directory,
        loop_directory,
        loop_output_directory_path,
        path_to_formatted_training_data,
        path_to_training_data,
    )


@app.cell
def _(asvtable_path, pd, taxonomy_path):
    sampletable=pd.read_csv(asvtable_path, index_col=0)
    sampletable = sampletable.iloc[:, :-1]
    asvtable = sampletable.T
    asv_id = pd.read_csv(taxonomy_path, index_col=0)
    return asv_id, asvtable, sampletable


@app.cell
def _(
    assign_id,
    generate_single_id,
    intermediate_directory,
    pd,
    taxonomy_path,
    unknown_count,
    update_genus_new,
):
    # assign taxonomy
    taxa_split=pd.read_csv(taxonomy_path, index_col=0)
    taxa_split['Genus_based'] = taxa_split.apply(lambda row: update_genus_new(row, unknown_count), axis=1)
    taxa_split['genus_ID'] = taxa_split['Genus_based'].apply(lambda x: assign_id(x))

    all_generated_ids = set()
    taxa_split['randomID'] = taxa_split.apply(
        lambda row: generate_single_id(min_length=5, existing_ids=all_generated_ids), 
        axis=1
    )
    taxa_split.to_csv(intermediate_directory+'/new_taxa.csv',index=True)
    return all_generated_ids, taxa_split


@app.cell
def _(intermediate_directory, path_to_training_data, sampletable, taxa_split):
    # Create Mallet input documents
    mapping_dict = taxa_split['randomID'].to_dict()
    sampletable_genusid = sampletable.copy()
    new_columns = {}
    for col in sampletable.columns:
        if col in mapping_dict:
            new_columns[col] = mapping_dict[col]
        else:
            new_columns[col] = col
    sampletable_genusid = sampletable_genusid.rename(columns=new_columns)
    sampletable_genusid.to_csv(intermediate_directory + '/annotaed_randomid.csv', index=True)

    # Convert it like a texture data
    doc_list=[]
    # Each sample becomes a document where ASVs are repeated based on their abundance
    for index, row in sampletable_genusid.iterrows():
        doc = []
        for asvs_id1, abundance in row.items():  # Using items() instead of iteritems()
            if abundance > 0:
                doc.extend([str(asvs_id1)] * int(abundance))
        doc_list.append(doc)

    flattened_nested_list = [' '.join(sublist) for sublist in doc_list]

    with open(path_to_training_data, 'w') as f:
        for document in flattened_nested_list:
            f.write(document + '\n')
    return (
        abundance,
        asvs_id1,
        col,
        doc,
        doc_list,
        document,
        f,
        flattened_nested_list,
        index,
        mapping_dict,
        new_columns,
        row,
        sampletable_genusid,
    )


@app.cell
def _(mo):
    mo.md("""# RUN loop""")
    return


@app.cell
def _(
    Loop_2tables_directory_path,
    MC_range,
    calculate_coherence,
    calculate_perplexity,
    flattened_nested_list,
    lmw,
    load_mallet_model_output,
    loop_output_directory_path,
    path_to_formatted_training_data,
    path_to_mallet,
    path_to_training_data,
    pd,
    sampletable_genusid,
    subprocess,
):
    # Initialize empty DataFrames to store all results
    all_df_pivot_rel = pd.DataFrame()
    all_df_probabilities_rel = pd.DataFrame()
    all_metrics = pd.DataFrame(columns=['Num_MCs', 'Perplexity', 'Coherence'])

    # Define the range of topics
    for num_topics in MC_range:  # Loop from 2 to 20 topics
        # Define file paths based on the current number of topics
        path_to_model = loop_output_directory_path + f'/mallet.model.{num_topics}'
        path_to_topic_keys = loop_output_directory_path + f'/mallet.topic_keys.{num_topics}'
        path_to_topic_distributions = loop_output_directory_path + f'/mallet.topic_distributions.{num_topics}'
        path_to_word_weights = loop_output_directory_path + f'/mallet.word_weights.{num_topics}'
        path_to_diagnostics = loop_output_directory_path + f'/mallet.diagnostics.{num_topics}.xml'

        # Define paths for individual model results
        path_to_DirichletComponentProbabilities = Loop_2tables_directory_path + f'/DirichletComponentProbabilities_{num_topics}.csv'
        path_to_TaxaProbabilities = Loop_2tables_directory_path + f'/TaxaProbabilities_{num_topics}.csv'
        path_to_ASVProbabilities = Loop_2tables_directory_path + f'/ASVProbabilities_{num_topics}.csv'

        # Training model with optimizing
        lmw.import_data(path_to_mallet,
                        path_to_training_data,
                        path_to_formatted_training_data,
                        flattened_nested_list)

        # Construct the MALLET command
        mallet_command = [
            path_to_mallet,
            'train-topics',
            '--input', path_to_formatted_training_data,
            '--num-topics', str(num_topics),  # Change number of topics as needed
            '--output-state', path_to_model,
            '--output-topic-keys', path_to_topic_keys,
            '--output-doc-topics', path_to_topic_distributions,
            '--word-topic-counts-file', path_to_word_weights,
            '--diagnostics-file', path_to_diagnostics,
            '--optimize-interval', '10',  # Enable alpha optimization every 10 iterations
            '--num-iterations', '1000',  # Number of iterations, adjust as needed
            '--random-seed', '43'
        ]

        # Run the MALLET command
        print(f"Running MALLET for {num_topics} microbial components...")
        subprocess.run(mallet_command, check=True)
        print(f"Completed MALLET for {num_topics} microbial components.")

        # Create index names for the topics
        topic_index = []
        for a in range(1, num_topics + 1):
            topic_index.append(str(num_topics) + '_' + str(a))

        # Load the MALLET model output
        topic_distributions, word_topics = load_mallet_model_output(path_to_topic_distributions, path_to_word_weights)

        # Map term IDs to genus names and create a new list of renamed word topic

        # rename_asv_topics = []
        # for topic_asv, term_asv, freq_asv in word_topics:
        #     new_term_asv = asvid_dict.get(term_asv, term_asv)  # Use term_asv to map term IDs to ASV
        #     rename_asv_topics.append((topic_asv, new_term_asv, freq_asv))

        # Convert the renamed word topics into a DataFrame
        df_asv= pd.DataFrame(word_topics, columns=['MC', 'Term', 'Frequency'])

        # print(rename_asv_topics[:10])

        # Pivot the DataFrame to get the desired format (Term columns, Topic rows)
        df_asv_pivot = df_asv.pivot_table(index='MC', columns='Term', values='Frequency', fill_value=0)

        # Merge columns with the same header by summing them
        # df_pivot_grouped = df_pivot.groupby(level=0, axis=1).sum()

        # Normalize the DataFrame to get probabilities
        # df_probabilities = df_pivot_grouped.div(df_pivot_grouped.sum(axis=1), axis=0)
        df_asv_probabilities = df_asv_pivot.div(df_asv_pivot.sum(axis=1), axis=0)

        # Rename the index with the generated index names
        # df_pivot_grouped.index = topic_index
        # df_probabilities.index = topic_index
        df_asv_probabilities.index = topic_index

        # Create and save the topic distribution DataFrame for this specific model
        df_topic_dist = pd.DataFrame(
            topic_distributions,  # Your nested list of topic probabilities
            index=sampletable_genusid.index,  # Use sampletable's index
            columns=topic_index  # Your list of topic names
        )

        # Save individual model results
        df_topic_dist.to_csv(path_to_DirichletComponentProbabilities, index=True)
        # df_probabilities.to_csv(path_to_TaxaProbabilities, index=True)
        df_asv_probabilities.to_csv(path_to_ASVProbabilities, index=True)
        print(f"Saved individual model results for {num_topics} topics.")

        # Concatenate the results to the overall DataFrames (as in your original code)
        # all_df_pivot_rel = pd.concat([all_df_pivot_rel, df_pivot_grouped])
        all_df_probabilities_rel = pd.concat([all_df_probabilities_rel, df_asv_probabilities])

        perplexity = calculate_perplexity(topic_distributions)
        coherence = calculate_coherence(word_topics, flattened_nested_list)

        new_row = pd.DataFrame([{
            'Num_MCs': num_topics,
            'Perplexity': perplexity,
            'Coherence': coherence
        }])
        all_metrics = pd.concat([all_metrics, new_row], ignore_index=True)

        print(f"Processed and appended results for {num_topics} MCs.")

    # After the loop, save the final combined DataFrames to CSV files
    all_df_pivot_rel.to_csv(loop_output_directory_path + '/all_MC_pivots_rel_2_20.csv')
    all_df_probabilities_rel.to_csv(loop_output_directory_path + '/all_MC_probabilities_rel_2_20.csv')
    all_metrics.to_csv(loop_output_directory_path + '/all_MC_metrics_2_20.csv')

    print("Saved final combined DataFrames.")
    return (
        a,
        all_df_pivot_rel,
        all_df_probabilities_rel,
        all_metrics,
        coherence,
        df_asv,
        df_asv_pivot,
        df_asv_probabilities,
        df_topic_dist,
        mallet_command,
        new_row,
        num_topics,
        path_to_ASVProbabilities,
        path_to_DirichletComponentProbabilities,
        path_to_TaxaProbabilities,
        path_to_diagnostics,
        path_to_model,
        path_to_topic_distributions,
        path_to_topic_keys,
        path_to_word_weights,
        perplexity,
        topic_distributions,
        topic_index,
        word_topics,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
