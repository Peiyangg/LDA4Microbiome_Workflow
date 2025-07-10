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


@app.cell(hide_code=True)
def _(CoherenceModel, Dictionary, defaultdict, lmw, np):
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


    def calculate_coherence(word_topics, texts, top_n=10):

        try:
            # Ensure texts are in the correct format (list of lists of words)
            processed_texts = []
            for text in texts:
                if isinstance(text, str):
                    # Split by whitespace and filter out empty strings
                    words = [word.strip() for word in text.split() if word.strip()]
                    processed_texts.append(words)
                elif isinstance(text, list):
                    # Already a list, but ensure all elements are strings
                    words = [str(word).strip() for word in text if str(word).strip()]
                    processed_texts.append(words)
                else:
                    print(f"Warning: Unexpected input type: {type(text)}. Skipping.")
                    continue
        
            if not processed_texts:
                raise ValueError("No valid texts found after processing")
        
            # Create dictionary from processed texts
            id2word = Dictionary(processed_texts)
        
            # Group word_topics by topic number
            topics_dict = defaultdict(list)
            for topic_num, word, freq in word_topics:
                # Ensure word is a string and exists in dictionary
                word_str = str(word).strip()
                if word_str in id2word.token2id:
                    topics_dict[topic_num].append((word_str, float(freq)))
        
            if not topics_dict:
                raise ValueError("No valid topics found in word_topics")
        
            # Extract top N words for each topic
            topics = []
            for topic_num, word_freqs in topics_dict.items():
                # Sort words by frequency (descending) and take top N
                top_words = [word for word, freq in sorted(word_freqs, key=lambda x: x[1], reverse=True)[:top_n]]
                if top_words:  # Only add non-empty topics
                    topics.append(top_words)
        
            if not topics:
                raise ValueError("No valid topics extracted")
        
            # Calculate coherence using 'c_v' measure with texts (not corpus)
            coherence_model = CoherenceModel(
                topics=topics, 
                texts=processed_texts,  # Use texts, not corpus
                dictionary=id2word, 
                coherence='c_v'
            )
        
            coherence_score = coherence_model.get_coherence()
        
            print(f"Coherence calculated successfully: {coherence_score:.4f}")
            return coherence_score
        
        except Exception as e:
            print(f"Error calculating coherence: {str(e)}")
            # Return a default value or re-raise depending on your needs
            return 0.0
    return calculate_coherence, calculate_perplexity, load_mallet_model_output


@app.cell
def _():
    base_directory          = 'example_test'
    # model selection range
    MC_range = range(2, 5)
    range_str = f"{MC_range.start}_{MC_range.stop-1}"
    return MC_range, base_directory, range_str


@app.cell
def _(base_directory, os):
    # Create new directory for LDA analysis
    intermediate_directory  =   os.path.join(base_directory, 'intermediate')
    loop_directory          =   os.path.join(base_directory, 'lda_loop')
    lda_directory           =   os.path.join(base_directory, 'lda_results')
    MC_sample_directory    =   os.path.join(lda_directory, 'MC_Sample')
    MC_feature_directory    =   os.path.join(lda_directory, 'MC_Feature')
    MALLET_diagnostics_directory = os.path.join(lda_directory, 'Diagnostics')

    os.makedirs(intermediate_directory, exist_ok=True)
    os.makedirs(loop_directory, exist_ok=True)
    os.makedirs(lda_directory, exist_ok=True)
    os.makedirs(MC_sample_directory, exist_ok=True)
    os.makedirs(MC_feature_directory, exist_ok=True)
    os.makedirs(MALLET_diagnostics_directory, exist_ok=True)

    # Now you can use lda_directory as your output path
    loop_output_directory_path = loop_directory
    # Loop_2tables_directory_path = lda_directory
    Loop_MC_sample_directory_path = MC_sample_directory
    Loop_MC_feature_directory_directory_path = MC_feature_directory

    path_to_training_data           = loop_output_directory_path + '/training.txt'
    path_to_formatted_training_data = loop_output_directory_path + '/mallet.training'
    return (
        Loop_MC_feature_directory_directory_path,
        Loop_MC_sample_directory_path,
        MALLET_diagnostics_directory,
        MC_feature_directory,
        MC_sample_directory,
        intermediate_directory,
        lda_directory,
        loop_directory,
        loop_output_directory_path,
        path_to_formatted_training_data,
        path_to_training_data,
    )


@app.cell
def _(intermediate_directory, path_to_training_data, pd):
    sampletable_genusid=pd.read_csv(intermediate_directory + '/annotaed_randomid.csv', index_col=0)

    with open(path_to_training_data, 'r') as f:
        flattened_nested_list = [line.strip() for line in f]
    return f, flattened_nested_list, sampletable_genusid


@app.cell
def _(
    Loop_MC_feature_directory_directory_path,
    Loop_MC_sample_directory_path,
    MALLET_diagnostics_directory,
    MC_range,
    calculate_coherence,
    calculate_perplexity,
    flattened_nested_list,
    lda_directory,
    lmw,
    load_mallet_model_output,
    loop_output_directory_path,
    path_to_formatted_training_data,
    path_to_mallet,
    path_to_training_data,
    pd,
    range_str,
    sampletable_genusid,
    subprocess,
):
    # Initialize empty DataFrames to store all results
    # all_df_pivot_rel = pd.DataFrame()
    all_df_probabilities_rel = pd.DataFrame()
    all_metrics = pd.DataFrame(columns=['Num_MCs', 'Perplexity', 'Coherence'])

    # Define the range of topics
    for num_topics in MC_range:  # Loop from 2 to 20 topics
        # Define file paths based on the current number of topics
        path_to_model = loop_output_directory_path + f'/mallet.model.{num_topics}'
        path_to_topic_keys = loop_output_directory_path + f'/mallet.topic_keys.{num_topics}'
        path_to_topic_distributions = loop_output_directory_path + f'/mallet.topic_distributions.{num_topics}'
        path_to_word_weights = loop_output_directory_path + f'/mallet.word_weights.{num_topics}'
        path_to_diagnostics = MALLET_diagnostics_directory + f'/mallet.diagnostics.{num_topics}.xml'

        # Define paths for individual model results
        path_to_DirichletComponentProbabilities = Loop_MC_sample_directory_path + f'/MC_Sample_probabilities{num_topics}.csv'
        # path_to_TaxaProbabilities = Loop_2tables_directory_path + f'/TaxaProbabilities_{num_topics}.csv'
        path_to_ASVProbabilities = Loop_MC_feature_directory_directory_path + f'/MC_Feature_Probabilities_{num_topics}.csv'

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
        df_topic_dist_wide=df_topic_dist.T

        # Save individual model results
        df_topic_dist_wide.to_csv(path_to_DirichletComponentProbabilities, index=True)
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
    # all_df_pivot_rel.to_csv(f'{loop_output_directory_path}/all_MC_pivots_rel_{range_str}.csv')
    all_df_probabilities_rel.to_csv(loop_output_directory_path + f'/all_MC_probabilities_rel_{range_str}.csv')
    all_metrics.to_csv(lda_directory + f'/all_MC_metrics_{range_str}.csv')

    print("Saved final combined DataFrames.")
    return (
        a,
        all_df_probabilities_rel,
        all_metrics,
        coherence,
        df_asv,
        df_asv_pivot,
        df_asv_probabilities,
        df_topic_dist,
        df_topic_dist_wide,
        mallet_command,
        new_row,
        num_topics,
        path_to_ASVProbabilities,
        path_to_DirichletComponentProbabilities,
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


if __name__ == "__main__":
    app.run()
