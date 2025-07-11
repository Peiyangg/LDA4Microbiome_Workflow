import marimo

__generated_with = "0.11.8"
app = marimo.App(width="full")


@app.cell
def _():
    # step 0
    import marimo as mo
    import random
    import string
    import os
    import pandas as pd

    # step 1
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import subprocess
    from pathlib import Path
    from collections import defaultdict
    from gensim import corpora
    from gensim.models import LdaModel, CoherenceModel
    import gensim
    from gensim.corpora import Dictionary
    import little_mallet_wrapper as lmw
    from typing import List, Dict, Tuple, Any

    # step 2
    import json
    import glob
    import re
    import xml.etree.ElementTree as ET
    from StripeSankey import StripeSankeyInline

    # step 3
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from scipy.cluster import hierarchy
    from scipy.spatial import distance
    import matplotlib.patches as mpatches
    return (
        Any,
        CoherenceModel,
        Dict,
        Dictionary,
        ET,
        LdaModel,
        LinearSegmentedColormap,
        List,
        ListedColormap,
        Path,
        StripeSankeyInline,
        Tuple,
        corpora,
        defaultdict,
        distance,
        gensim,
        glob,
        hierarchy,
        json,
        lmw,
        mcolors,
        mo,
        mpatches,
        np,
        os,
        pd,
        plt,
        random,
        re,
        sns,
        string,
        subprocess,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""## Input""")
    return


@app.cell
def _():
    path_to_mallet          = 'Mallet-202108/bin/mallet'
    base_directory          = 'example_test'
    SampleASVtable_path     = 'example_data/data/sampled_data.csv'
    taxonomy_path           = 'example_data/data/shuffled_taxonomy.csv'
    metadata_path           = 'example_data/data/shuffled_metadata.csv'
    return (
        SampleASVtable_path,
        base_directory,
        metadata_path,
        path_to_mallet,
        taxonomy_path,
    )


@app.cell
def _(mo):
    mo.md(r"""## Integrated""")
    return


@app.cell
def _(
    LDATrainer,
    SampleASVtable_path,
    SankeyDataProcessor,
    StripeSankeyInline,
    TaxonomyProcessor,
    base_directory,
    path_to_mallet,
    taxonomy_path,
):
    # Complete workflow
    # Step 1: Process taxonomy
    _tax_processor = TaxonomyProcessor(
            asvtable_path=SampleASVtable_path,
            taxonomy_path=taxonomy_path,
            base_directory=base_directory)
    _tax_results = _tax_processor.process_all()

    # Step 2: Train LDA models  
    _trainer = LDATrainer(base_directory, path_to_mallet)
    _lda_results = _trainer.train_models(range(2, 6))

    _processor_sankey = SankeyDataProcessor.from_lda_trainer(_trainer)
    _sankey_data = _processor_sankey.process_all_data()
    _widget = StripeSankeyInline(sankey_data=_sankey_data, mode='metric')
    _widget
    return


@app.cell
def _(
    LDAModelVisualizer,
    TopicFeatureProcessor,
    base_directory,
    metadata_path,
):
    # Step 3: Visualize sample-topic relationships
    _visualizer = LDAModelVisualizer(
        base_directory=base_directory,
        k_value=3,
        metadata_path=metadata_path,
        universal_headers=["Group", "Batch"],
        continuous_headers=["VHCD", "IgG", "Neutrophils", "Lymphocytes", "MCV"]
    )

    _visualizer.configure_colors(
            custom_colors={
                'Group': {
                    'Compromised': '#965454',
                    'Optimal': '#858b6f'
                }
            },
            continuous_cmaps={
                'VHCD': 'coolwarm',
                'Neutrophils': 'coolwarm'
            }
        )
    _sample_plots = _visualizer.create_all_visualizations()

    # Step 4: Analyze topic-feature relationships
    _feature_processor = TopicFeatureProcessor(base_directory, k_value=3)
    _genus_data, _genus_top = _feature_processor.process_feature_level('genus_ID')
    _feature_plots = _feature_processor.create_feature_heatmap('genus_ID')
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## STEP 0: Data processing""")
    return


@app.cell(hide_code=True)
def _(os, pd, random, string):
    class TaxonomyProcessor:
        """
        A class for processing taxonomic data and preparing it for LDA analysis.

        This class handles:
        - Reading and processing ASV tables and taxonomy data
        - Updating genus names based on taxonomic hierarchy
        - Assigning unique IDs to taxa
        - Creating directory structure for analysis
        - Preparing Mallet input documents
        """

        def __init__(self, asvtable_path, taxonomy_path, base_directory):
            """
            Initialize the TaxonomyProcessor.

            Args:
                asvtable_path (str): Path to the ASV table CSV file
                taxonomy_path (str): Path to the taxonomy CSV file
                base_directory (str): Base directory for storing results
            """
            self.asvtable_path = asvtable_path
            self.taxonomy_path = taxonomy_path
            self.base_directory = base_directory

            # Initialize counters and tracking variables
            self.unknown_count = [0]
            self.taxa_counts = {}
            self.all_generated_ids = set()

            # Initialize directory paths
            self._setup_directories()

            # Initialize data containers
            self.asvtable = None
            self.taxa_split = None
            self.sampletable_randomID = None

        def _setup_directories(self):
            """Create necessary directories for analysis."""
            self.intermediate_directory = os.path.join(self.base_directory, 'intermediate')
            self.loop_directory = os.path.join(self.base_directory, 'lda_loop')
            self.lda_directory = os.path.join(self.base_directory, 'lda_results')
            self.MC_sample_directory = os.path.join(self.lda_directory, 'MC_Sample')
            self.MC_feature_directory = os.path.join(self.lda_directory, 'MC_Feature')

            # Create directories and print status
            print("Creating directory structure...")
            created_dirs = []
            for directory in [self.intermediate_directory, self.loop_directory, 
                             self.lda_directory, self.MC_sample_directory, 
                             self.MC_feature_directory]:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    created_dirs.append(directory)
                    print(f"  ✓ Created: {directory}")
                else:
                    print(f"  ✓ Already exists: {directory}")

            # Set up path variables
            self.loop_output_directory_path = self.loop_directory
            self.Loop_2tables_directory_path = self.lda_directory
            self.Loop_MC_sample_directory_path = self.MC_sample_directory
            self.Loop_MC_feature_directory_directory_path = self.MC_feature_directory
            self.path_to_training_data = os.path.join(self.loop_output_directory_path, 'training.txt')
            self.path_to_formatted_training_data = os.path.join(self.loop_output_directory_path, 'mallet.training')

        def update_genus_new(self, row):
            """
            Update genus names based on taxonomic hierarchy.

            Args:
                row: DataFrame row containing taxonomic information

            Returns:
                str: Updated genus name
            """
            # Case 1: If Genus is 'g__uncultured' or NaN or 'd__Bacteria'
            if row['Genus'] == 'g__uncultured' or pd.isna(row['Genus']) or row['Genus'] == 'd__Bacteria':
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
                    self.unknown_count[0] += 1
                    return f"unknown_{self.unknown_count[0]}"
            # Case 2: Return original Genus if it's valid
            return row['Genus']

        def assign_id(self, genus_based):
            """
            Assign sequential IDs to genus names.

            Args:
                genus_based (str): Genus name to assign ID to

            Returns:
                str: Genus name with count suffix
            """
            # Initialize count if this is the first time seeing this genus
            if genus_based not in self.taxa_counts:
                self.taxa_counts[genus_based] = 0
            else:
                # Increment count for subsequent occurrences
                self.taxa_counts[genus_based] += 1
            # Return genus name with count suffix
            return f"{genus_based}_{self.taxa_counts[genus_based]}"

        def generate_single_id(self, min_length=5):
            """
            Generate a unique random ID.

            Args:
                min_length (int): Minimum length of the ID

            Returns:
                str: Unique random ID
            """
            # Generate a random string with the minimal length
            new_id = ''.join(random.choices(string.ascii_lowercase, k=min_length))
            # If the ID already exists, regenerate until we get a unique ID
            while new_id in self.all_generated_ids:
                new_id = ''.join(random.choices(string.ascii_lowercase, k=min_length))
            self.all_generated_ids.add(new_id)
            return new_id

        def load_and_process_data(self):
            """Load and process ASV table and taxonomy data."""
            print("\nLoading and processing data...")

            # Load ASV table
            print(f"  Loading ASV table from: {self.asvtable_path}")
            sampletable = pd.read_csv(self.asvtable_path, index_col=0)
            self.asvtable = sampletable.T
            print(f"  ✓ ASV table loaded: {self.asvtable.shape[0]} samples, {self.asvtable.shape[1]} ASVs")

            # Load and process taxonomy
            print(f"  Loading taxonomy from: {self.taxonomy_path}")
            self.taxa_split = pd.read_csv(self.taxonomy_path, index_col=0)
            print(f"  ✓ Taxonomy loaded: {len(self.taxa_split)} taxa")

            print("  Processing taxonomy...")
            self.taxa_split['Genus_based'] = self.taxa_split.apply(lambda row: self.update_genus_new(row), axis=1)
            self.taxa_split['genus_ID'] = self.taxa_split['Genus_based'].apply(lambda x: self.assign_id(x))
            self.taxa_split['randomID'] = self.taxa_split.apply(
                lambda row: self.generate_single_id(min_length=5), axis=1
            )
            print(f"  ✓ Taxonomy processing complete")

            # Save intermediate taxonomy file
            taxonomy_file = os.path.join(self.intermediate_directory, 'intermediate_taxa.csv')
            self.taxa_split.to_csv(taxonomy_file, index=True)
            print(f"  ✓ Saved intermediate taxonomy: {taxonomy_file}")

            # Create sample table with random IDs
            print("  Creating sample table with random IDs...")
            mapping_dict = self.taxa_split['randomID'].to_dict()
            self.sampletable_randomID = sampletable.copy()
            new_columns = {}
            for col in sampletable.columns:
                if col in mapping_dict:
                    new_columns[col] = mapping_dict[col]
                else:
                    new_columns[col] = col
            self.sampletable_randomID = self.sampletable_randomID.rename(columns=new_columns)

            # Save sample table with random IDs
            randomid_file = os.path.join(self.intermediate_directory, 'annotaed_randomid.csv')
            self.sampletable_randomID.to_csv(randomid_file, index=True)
            print(f"  ✓ Saved sample table with random IDs: {randomid_file}")

        def create_mallet_input(self):
            """Create Mallet input documents from processed data."""
            if self.sampletable_randomID is None:
                raise ValueError("Data must be loaded and processed first. Call load_and_process_data().")

            print("\nCreating Mallet input documents...")
            doc_list = []
            # Each sample becomes a document where ASVs are repeated based on their abundance
            for index, row in self.sampletable_randomID.iterrows():
                doc = []
                for asvs_id1, abundance in row.items():
                    if abundance > 0:
                        doc.extend([str(asvs_id1)] * int(abundance))
                doc_list.append(doc)

            flattened_nested_list = [' '.join(sublist) for sublist in doc_list]

            with open(self.path_to_training_data, 'w') as f:
                for document in flattened_nested_list:
                    f.write(document + '\n')

            print(f"  ✓ Mallet training data saved: {self.path_to_training_data}")
            print(f"  ✓ Created {len(doc_list)} documents for training")

        def process_all(self):
            """
            Run the complete processing pipeline.

            Returns:
                dict: Dictionary containing processed data and file paths
            """
            print("="*60)
            print("Starting TaxonomyProcessor pipeline...")
            print("="*60)

            self.load_and_process_data()
            self.create_mallet_input()

            print("\n" + "="*60)
            print("Processing complete! Summary:")
            print("="*60)
            print(f"Base directory: {self.base_directory}")
            print(f"Samples processed: {self.asvtable.shape[0] if self.asvtable is not None else 0}")
            print(f"Taxa processed: {len(self.taxa_split) if self.taxa_split is not None else 0}")
            print(f"Unknown taxa assigned: {self.unknown_count[0]}")

            print("\nDirectories created:")
            for directory in [self.intermediate_directory, self.loop_directory, 
                             self.lda_directory, self.MC_sample_directory, 
                             self.MC_feature_directory]:
                print(f"  • {directory}")

            print("\nFiles saved:")
            print(f"  • {os.path.join(self.intermediate_directory, 'intermediate_taxa.csv')}")
            print(f"  • {os.path.join(self.intermediate_directory, 'annotaed_randomid.csv')}")
            print(f"  • {self.path_to_training_data}")

            print("\nReady for LDA analysis!")
            print("="*60)

            return {
                'asvtable': self.asvtable,
                'taxa_split': self.taxa_split,
                'sampletable_randomID': self.sampletable_randomID,
                'paths': {
                    'intermediate_directory': self.intermediate_directory,
                    'loop_directory': self.loop_directory,
                    'lda_directory': self.lda_directory,
                    'MC_sample_directory': self.MC_sample_directory,
                    'MC_feature_directory': self.MC_feature_directory,
                    'path_to_training_data': self.path_to_training_data,
                    'path_to_formatted_training_data': self.path_to_formatted_training_data
                }
            }
    return (TaxonomyProcessor,)


@app.cell
def _():
    # Manual code
    # processor = TaxonomyProcessor(
    #         asvtable_path=SampleASVtable_path,
    #         taxonomy_path=taxonomy_path,
    #         base_directory=base_directory
    #     )

    # processed_data = processor.process_all()
    return


@app.cell
def _(mo):
    mo.md(r"""## STEP 1: Create Models""")
    return


@app.cell(hide_code=True)
def _(
    Any,
    CoherenceModel,
    Dict,
    Dictionary,
    List,
    Tuple,
    defaultdict,
    lmw,
    np,
    os,
    pd,
    subprocess,
):
    class LDATrainer:
        """
        A class for training LDA models across multiple topic numbers and managing results.

        This class handles:
        - Setting up directory structure for LDA analysis
        - Training MALLET LDA models for different numbers of topics
        - Processing and saving model outputs
        - Calculating model metrics (perplexity, coherence)
        - Aggregating results across all models
        """

        def __init__(self, base_directory: str, path_to_mallet: str):
            """
            Initialize the LDA Trainer.

            Args:
                base_directory (str): Base directory for storing all results
                path_to_mallet (str): Path to MALLET executable
            """
            self.base_directory = base_directory
            self.path_to_mallet = path_to_mallet

            # Set up directory structure
            self.paths = self._setup_directories()

            # Initialize result storage
            self.all_df_probabilities_rel = pd.DataFrame()
            self.all_metrics = pd.DataFrame(columns=['Num_MCs', 'Perplexity', 'Coherence'])

            # Data containers (to be set later)
            self.flattened_nested_list = None
            self.sampletable_genusid = None

        def _setup_directories(self) -> Dict[str, str]:
            """Set up directory structure and return path dictionary."""
            intermediate_directory = os.path.join(self.base_directory, 'intermediate')
            loop_directory = os.path.join(self.base_directory, 'lda_loop')
            lda_directory = os.path.join(self.base_directory, 'lda_results')
            MC_sample_directory = os.path.join(lda_directory, 'MC_Sample')
            MC_feature_directory = os.path.join(lda_directory, 'MC_Feature')
            MALLET_diagnostics_directory = os.path.join(lda_directory, 'Diagnostics')

            # Create all directories
            directories = [
                intermediate_directory, loop_directory, lda_directory,
                MC_sample_directory, MC_feature_directory, MALLET_diagnostics_directory
            ]

            print(f"Setting up LDA directory structure in: {self.base_directory}")
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                print(f"  ✓ Created/verified: {directory}")

            return {
                'intermediate_directory': intermediate_directory,
                'loop_directory': loop_directory,
                'lda_directory': lda_directory,
                'MC_sample_directory': MC_sample_directory,
                'MC_feature_directory': MC_feature_directory,
                'MALLET_diagnostics_directory': MALLET_diagnostics_directory,
                'path_to_training_data': os.path.join(loop_directory, 'training.txt'),
                'path_to_formatted_training_data': os.path.join(loop_directory, 'mallet.training')
            }

        def load_training_data(self):
            """
            Load training data from files created by TaxonomyProcessor.

            This method automatically loads:
            - flattened_nested_list from training.txt
            - sampletable_genusid from annotated_randomid.csv
            """
            # Load the sample table with random IDs
            sampletable_path = os.path.join(self.paths['intermediate_directory'], 'annotaed_randomid.csv')
            if not os.path.exists(sampletable_path):
                raise FileNotFoundError(f"Sample table not found: {sampletable_path}. Run TaxonomyProcessor first.")

            self.sampletable_genusid = pd.read_csv(sampletable_path, index_col=0)
            print(f"  ✓ Loaded sample table: {self.sampletable_genusid.shape}")

            # Load the flattened nested list from training data
            training_data_path = self.paths['path_to_training_data']
            if not os.path.exists(training_data_path):
                raise FileNotFoundError(f"Training data not found: {training_data_path}. Run TaxonomyProcessor first.")

            with open(training_data_path, 'r') as f:
                self.flattened_nested_list = [line.strip() for line in f]

            print(f"  ✓ Loaded training documents: {len(self.flattened_nested_list)} documents")
            print("Training data loaded successfully.")

        def _generate_file_paths(self, num_topics: int) -> Dict[str, str]:
            """Generate all file paths for a specific number of topics."""
            loop_path = self.paths['loop_directory']
            diagnostics_path = self.paths['MALLET_diagnostics_directory']
            mc_sample_path = self.paths['MC_sample_directory']
            mc_feature_path = self.paths['MC_feature_directory']

            return {
                'model': os.path.join(loop_path, f'mallet.model.{num_topics}'),
                'topic_keys': os.path.join(loop_path, f'mallet.topic_keys.{num_topics}'),
                'topic_distributions': os.path.join(loop_path, f'mallet.topic_distributions.{num_topics}'),
                'word_weights': os.path.join(loop_path, f'mallet.word_weights.{num_topics}'),
                'diagnostics': os.path.join(diagnostics_path, f'mallet.diagnostics.{num_topics}.xml'),
                'sample_probs': os.path.join(mc_sample_path, f'MC_Sample_probabilities{num_topics}.csv'),
                'feature_probs': os.path.join(mc_feature_path, f'MC_Feature_Probabilities_{num_topics}.csv')
            }

        def _create_topic_index(self, num_topics: int) -> List[str]:
            """Create index names for topics."""
            return [f"{num_topics}_{i}" for i in range(1, num_topics + 1)]

        def _train_single_model(self, num_topics: int, file_paths: Dict[str, str]) -> None:
            """Train a single MALLET LDA model."""

            lmw.import_data(
                self.path_to_mallet,
                self.paths['path_to_training_data'],
                self.paths['path_to_formatted_training_data'],
                self.flattened_nested_list
            )

            # Construct MALLET command
            mallet_command = [
                self.path_to_mallet,
                'train-topics',
                '--input', self.paths['path_to_formatted_training_data'],
                '--num-topics', str(num_topics),
                '--output-state', file_paths['model'],
                '--output-topic-keys', file_paths['topic_keys'],
                '--output-doc-topics', file_paths['topic_distributions'],
                '--word-topic-counts-file', file_paths['word_weights'],
                '--diagnostics-file', file_paths['diagnostics'],
                '--optimize-interval', '10',
                '--num-iterations', '1000',
                '--random-seed', '43'
            ]

            # Run MALLET
            print(f"Running MALLET for {num_topics} microbial components...")
            subprocess.run(mallet_command, check=True)
            print(f"Completed MALLET for {num_topics} microbial components.")

        def _process_model_output(self, num_topics: int, file_paths: Dict[str, str]) -> Tuple[pd.DataFrame, List]:
            """Process MALLET model output and save individual results."""
            # Load model output
            topic_distributions, word_topics = self._load_mallet_model_output(
                file_paths['topic_distributions'], 
                file_paths['word_weights']
            )

            # Create topic index
            topic_index = self._create_topic_index(num_topics)

            # Process ASV data
            df_asv = pd.DataFrame(word_topics, columns=['MC', 'Term', 'Frequency'])
            df_asv_pivot = df_asv.pivot_table(index='MC', columns='Term', values='Frequency', fill_value=0)
            df_asv_probabilities = df_asv_pivot.div(df_asv_pivot.sum(axis=1), axis=0)
            df_asv_probabilities.index = topic_index

            # Create topic distribution DataFrame
            df_topic_dist = pd.DataFrame(
                topic_distributions,
                index=self.sampletable_genusid.index,
                columns=topic_index
            )
            df_topic_dist_wide = df_topic_dist.T

            # Save individual results
            df_topic_dist_wide.to_csv(file_paths['sample_probs'], index=True)
            df_asv_probabilities.to_csv(file_paths['feature_probs'], index=True)
            print(f"Saved individual model results for {num_topics} topics.")

            return df_asv_probabilities, topic_distributions, word_topics

        def _load_mallet_model_output(self, topic_distributions_path: str, word_weights_path: str) -> Tuple[List, List]:
            """
            Load MALLET model output files.

            Args:
                topic_distributions_path: Path to topic distributions file
                word_weights_path: Path to word weights file

            Returns:
                Tuple of (topic_distributions, word_topics)
            """
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

        def _calculate_perplexity(self, topic_distributions: List, epsilon: float = 1e-10) -> float:
            """
            Calculate perplexity for topic distributions.

            Args:
                topic_distributions: List of topic probability distributions
                epsilon: Small value to avoid log(0)

            Returns:
                Average perplexity across all samples
            """
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

        def _calculate_coherence(self, word_topics: List, texts: List, top_n: int = 10) -> float:
            """
            Calculate coherence score for topics.

            Args:
                word_topics: List of (topic_id, word, frequency) tuples
                texts: List of document texts
                top_n: Number of top words to use for coherence calculation

            Returns:
                Coherence score
            """
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

        def train_models(self, MC_range: List[int], range_str: str = None) -> Dict[str, Any]:
            """
            Train LDA models for a range of topic numbers.

            Args:
                MC_range: List of numbers of topics to train
                range_str: String representation of range for file naming

            Returns:
                Dictionary containing all results and metrics
            """
            # Load training data automatically
            print("Loading training data...")
            self.load_training_data()

            if range_str is None:
                range_str = f"{min(MC_range)}-{max(MC_range)}"

            print("="*60)
            print(f"Starting LDA training for {len(MC_range)} different topic numbers...")
            print(f"Topic range: {MC_range}")
            print("="*60)

            for num_topics in MC_range:
                print(f"\n--- Processing {num_topics} topics ---")

                # Generate file paths
                file_paths = self._generate_file_paths(num_topics)

                # Train model
                self._train_single_model(num_topics, file_paths)

                # Process output
                df_asv_probabilities, topic_distributions, word_topics = self._process_model_output(num_topics, file_paths)

                # Calculate metrics
                perplexity = self._calculate_perplexity(topic_distributions)
                coherence = self._calculate_coherence(word_topics, self.flattened_nested_list)

                # Store results
                self.all_df_probabilities_rel = pd.concat([self.all_df_probabilities_rel, df_asv_probabilities])

                new_row = pd.DataFrame([{
                    'Num_MCs': num_topics,
                    'Perplexity': perplexity,
                    'Coherence': coherence
                }])
                self.all_metrics = pd.concat([self.all_metrics, new_row], ignore_index=True)

                print(f"Processed and appended results for {num_topics} MCs.")

            # Save final results
            self._save_final_results(range_str)

            return {
                'probabilities': self.all_df_probabilities_rel,
                'metrics': self.all_metrics,
                'paths': self.paths
            }

        def _save_final_results(self, range_str: str):
            """Save final combined results."""
            prob_path = os.path.join(self.paths['loop_directory'], f'all_MC_probabilities_rel_{range_str}.csv')
            metrics_path = os.path.join(self.paths['lda_directory'], f'all_MC_metrics_{range_str}.csv')

            self.all_df_probabilities_rel.to_csv(prob_path)
            self.all_metrics.to_csv(metrics_path)

            print("="*60)
            print("Training complete! Final results saved:")
            print(f"  • Probabilities: {prob_path}")
            print(f"  • Metrics: {metrics_path}")
            print("="*60)
    return (LDATrainer,)


@app.cell
def _():
    # Manual code
    # trainer = LDATrainer(
    #         base_directory='example_test',  # Same base directory as TaxonomyProcessor
    #         path_to_mallet=path_to_mallet
    #     )

    # MC_range = list(range(2, 4))  # 2 to 20 topics
    # results = trainer.train_models(MC_range)
    return


@app.cell
def _(mo):
    mo.md(r"""## STEP 2: Selection""")
    return


@app.cell(hide_code=True)
def _(ET, List, Optional, defaultdict, glob, json, np, os, pd, re):
    class SankeyDataProcessor:
        """
        A class for processing LDA results into Sankey diagram data.

        This class integrates with LDATrainer to automatically load and process:
        - Sample-MC probability data
        - MALLET diagnostic data
        - Perplexity metrics
        - Flow calculations between different K values
        """

        def __init__(self, base_directory: str, MC_range: List[int], 
                     high_threshold: float = 0.67, medium_threshold: float = 0.33,
                     range_str: Optional[str] = None):
            """
            Initialize the Sankey Data Processor.

            Args:
                base_directory: Base directory where LDATrainer saved results
                MC_range: List of K values (number of topics) to process
                high_threshold: Threshold for high representation (default 0.67)
                medium_threshold: Threshold for medium representation (default 0.33)
                range_str: String representation of range for file naming
            """
            self.base_directory = base_directory
            self.MC_range = MC_range
            self.high_threshold = high_threshold
            self.medium_threshold = medium_threshold

            if range_str is None:
                self.range_str = f"{min(MC_range)}-{max(MC_range)}"
            else:
                self.range_str = range_str

            # Set up paths automatically based on base_directory
            self._setup_paths()

            print(f"SankeyDataProcessor initialized:")
            print(f"  Base directory: {self.base_directory}")
            print(f"  K range: {self.MC_range}")
            print(f"  Thresholds: High={self.high_threshold}, Medium={self.medium_threshold}")

        def _setup_paths(self):
            """Set up all necessary paths based on base directory."""
            self.sample_mc_folder = os.path.join(self.base_directory, 'lda_results', 'MC_Sample')
            self.mc_feature_folder = os.path.join(self.base_directory, 'lda_results', 'MC_Feature')
            self.mallet_folder = os.path.join(self.base_directory, 'lda_results', 'Diagnostics')
            self.perplexity_path = os.path.join(
                self.base_directory, 'lda_results', f'all_MC_metrics_{self.range_str}.csv'
            )

            print(f"  Sample MC folder: {self.sample_mc_folder}")
            print(f"  MC feature folder: {self.mc_feature_folder}")
            print(f"  MALLET diagnostics: {self.mallet_folder}")
            print(f"  Perplexity file: {self.perplexity_path}")

        @classmethod
        def from_lda_trainer(cls, lda_trainer, high_threshold: float = 0.67, 
                            medium_threshold: float = 0.33):
            """
            Create SankeyDataProcessor from an LDATrainer instance.

            Args:
                lda_trainer: LDATrainer instance that has completed training
                high_threshold: Threshold for high representation
                medium_threshold: Threshold for medium representation

            Returns:
                SankeyDataProcessor instance
            """
            if not hasattr(lda_trainer, 'all_metrics') or lda_trainer.all_metrics.empty:
                raise ValueError("LDATrainer must have completed training before creating SankeyDataProcessor")

            # Get MC_range from the trainer's results
            MC_range = sorted(lda_trainer.all_metrics['Num_MCs'].unique().astype(int).tolist())

            return cls(
                base_directory=lda_trainer.base_directory,
                MC_range=MC_range,
                high_threshold=high_threshold,
                medium_threshold=medium_threshold
            )

        def load_sample_mc_data(self):
            """Load all sample-MC probability files"""
            sample_mc_data = {}

            for k in self.MC_range:
                filename = f'MC_Sample_probabilities{k}.csv'
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

        def load_all_mallet_diagnostics(self):
            """Load all MALLET diagnostic files"""
            pattern = os.path.join(self.mallet_folder, 'mallet.diagnostics.*.xml')
            xml_files = glob.glob(pattern)

            if not xml_files:
                print(f"❌ No MALLET diagnostic files found in {self.mallet_folder}")
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

        def load_perplexity_data(self):
            """Load perplexity data from CSV file"""
            try:
                df = pd.read_csv(self.perplexity_path)
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

        def save_processed_data(self, sankey_data, output_filename='sankey_data.json'):
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

            # Save to the LDA results directory
            output_path = os.path.join(self.base_directory, 'lda_results', output_filename)

            with open(output_path, 'w') as f:
                json.dump(converted_data, f, indent=2)

            print(f"💾 Data saved to {output_path}")
            return output_path

        def process_all_data(self, output_filename='fully_integrated_sankey_data.json'):
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
            mallet_df = self.load_all_mallet_diagnostics()

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
            perplexity_dict = self.load_perplexity_data()

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
            output_path = self.save_processed_data(sankey_data, output_filename)

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
    return (SankeyDataProcessor,)


@app.cell
def _():
    # Manual code
    # # method 1
    # processor_sankey_1 = SankeyDataProcessor.from_lda_trainer(trainer)
    # sankey_data_1 = processor_sankey_1.process_all_data()

    # # method 2
    # processor_sankey2 = SankeyDataProcessor(
    #         base_directory='example_test',
    #         MC_range=MC_range,
    #         high_threshold=0.67,
    #         medium_threshold=0.33
    #     )

    # sankey_data_2 = processor_sankey2.process_all_data()
    return


@app.cell
def _():
    # widget = StripeSankeyInline(sankey_data=sankey_data_1, mode='metric')
    # widget
    return


@app.cell
def _(mo):
    mo.md(r"""## STEP 3: Matrix""")
    return


@app.cell(hide_code=True)
def _(
    Any,
    Dict,
    LinearSegmentedColormap,
    List,
    ListedColormap,
    Optional,
    TopicFeatureProcessor,
    Tuple,
    distance,
    hierarchy,
    mcolors,
    mpatches,
    np,
    os,
    pd,
    plt,
    sns,
):
    class LDAModelVisualizer:
        """
        A class for visualizing LDA model results with customizable metadata and color schemes.
    
        This class handles:
        - Loading model results for a specific K value
        - Creating clustered heatmaps with metadata annotations
        - Topic-taxon distribution visualizations
        - Customizable color schemes for categorical and continuous variables
        """
    
        def __init__(self, base_directory: str, k_value: int, metadata_path: str,
                     universal_headers: List[str], continuous_headers: List[str] = None,
                     top_asv_count: int = 7, id_column: str = 'ID'):
            """
            Initialize the LDA Model Visualizer.
        
            Args:
                base_directory: Base directory where LDA results are stored
                k_value: Number of topics (K) to visualize
                metadata_path: Path to metadata CSV file
                universal_headers: List of categorical metadata columns to include (required)
                continuous_headers: List of continuous metadata columns to include (optional)
                top_asv_count: Number of top ASVs to show in heatmaps (default: 7)
                id_column: Column name to use for sample IDs (default: 'ID')
            """
            self.base_directory = base_directory
            self.k_value = k_value
            self.metadata_path = metadata_path
        
            # Set up paths
            self._setup_paths()
        
            # Initialize configuration
            self.config = {
                'universal_headers': universal_headers,
                'continuous_headers': continuous_headers or [],
                'top_asv_count': top_asv_count,
                'id_column': id_column,
                'custom_colors': {},  # Will be set with defaults or user values
                'continuous_cmaps': {},  # Will be set with defaults or user values
                'figsize': (14, 10),
                'heatmap_vmin': 0,
                'heatmap_vmax': 0.4,
                'cbar_ticks': [0, 0.02, 0.05, 0.1, 0.3]
            }
        
            # Set default colors
            self._set_default_colors()
        
            # Load data
            self._load_data()
        
            print(f"LDAModelVisualizer initialized for K={k_value}")
            print(f"  Base directory: {self.base_directory}")
            print(f"  Visualization directory: {self.viz_directory}")
            print(f"  Universal headers: {universal_headers}")
            print(f"  Continuous headers: {continuous_headers or 'None'}")
    
        def _setup_paths(self):
            """Set up all necessary paths."""
            self.loop_directory = os.path.join(self.base_directory, 'lda_loop')
            self.inter_directory = os.path.join(self.base_directory, 'intermediate')
            self.viz_directory = os.path.join(self.base_directory, 'lda_visualization')
            self.lda_directory = os.path.join(self.base_directory, 'lda_results')
            self.MC_sample_directory = os.path.join(self.lda_directory, 'MC_Sample')
            self.MC_feature_directory = os.path.join(self.lda_directory, 'MC_Feature')
        
            # Create visualization directory
            os.makedirs(self.viz_directory, exist_ok=True)
        
            # Set up file paths for this specific K value
            self.path_to_DirichletComponentProbabilities = os.path.join(
                self.MC_sample_directory, f"MC_Sample_probabilities{self.k_value}.csv"
            )
            self.path_to_ASVProbabilities = os.path.join(
                self.MC_feature_directory, f"MC_Feature_Probabilities_{self.k_value}.csv"
            )
            self.path_to_new_taxa = os.path.join(self.inter_directory, "intermediate_taxa.csv")
    
        def _set_default_colors(self):
            """Set up empty color configurations - will use seaborn/matplotlib defaults."""
            # Initialize empty - will use library defaults when colors are generated
            self.config['custom_colors'] = {}
            self.config['continuous_cmaps'] = {}
        
            print("✓ Color configuration initialized - will use library defaults")
    
        def _load_data(self):
            """Load all necessary data files."""
            try:
                # Load Dirichlet Component Probabilities (sample-topic distributions)
                DMP = pd.read_csv(self.path_to_DirichletComponentProbabilities, index_col=0).T
                self.DM_distributions = DMP.values.tolist()
                print(f"✓ Loaded sample-topic distributions: {DMP.shape}")
            
                # Load ASV probabilities (topic-feature distributions)
                self.ASV_probabilities = pd.read_csv(self.path_to_ASVProbabilities, index_col=0)
                print(f"✓ Loaded topic-ASV distributions: {self.ASV_probabilities.shape}")
            
                # Load metadata
                self.metadata = pd.read_csv(self.metadata_path, index_col=0)
                print(f"✓ Loaded metadata: {self.metadata.shape}")
            
                # Load taxonomy data if available
                if os.path.exists(self.path_to_new_taxa):
                    self.taxa_data = pd.read_csv(self.path_to_new_taxa, index_col=0)
                    print(f"✓ Loaded taxonomy data: {self.taxa_data.shape}")
                else:
                    self.taxa_data = None
                    print("! Taxonomy data not found")
                
            except Exception as e:
                raise FileNotFoundError(f"Error loading data: {e}")
    
        def configure_colors(self, 
                           custom_colors: Optional[Dict] = None,
                           continuous_cmaps: Optional[Dict] = None,
                           **kwargs):
            """
            Configure color schemes for visualization.
        
            Args:
                custom_colors: Dictionary of custom colors for categorical variables
                continuous_cmaps: Dictionary of colormaps for continuous variables
                **kwargs: Additional configuration parameters (figsize, heatmap_vmin, etc.)
            """
            if custom_colors is not None:
                self.config['custom_colors'].update(custom_colors)
            if continuous_cmaps is not None:
                self.config['continuous_cmaps'].update(continuous_cmaps)
            
            # Update any additional parameters
            for key, value in kwargs.items():
                if key in self.config:
                    self.config[key] = value
        
            print("✓ Color configuration updated")
    
        def rgb_to_hex(self, r: int, g: int, b: int) -> str:
            """Convert RGB values to hex code."""
            return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    
        def prepare_heatmap_data(self, headers_to_include: Optional[List[str]] = None) -> pd.DataFrame:
            """
            Create a multi-index heatmap table from distributions and metadata.
        
            Args:
                headers_to_include: List of column names to include in the multi-index header
            
            Returns:
                DataFrame with multi-index columns
            """
            # Use configured headers if none provided
            if headers_to_include is None:
                headers_to_include = self.config['universal_headers'] + self.config['continuous_headers']
        
            # Convert nested list to DataFrame and transpose
            distributions_df = pd.DataFrame(self.DM_distributions)
            multiheader = distributions_df.T

            # Fill NaN values in metadata with 0
            metadata_df_filled = self.metadata.fillna(0)

            # Get unique IDs from the metadata that match our samples
            sample_ids = metadata_df_filled[self.config['id_column']].values[:len(self.DM_distributions)]

            # Create a list of tuples for the multi-index
            header_tuples = []
            for idx, id_val in enumerate(sample_ids):
                # Get the matching metadata row
                metadata_row = metadata_df_filled[metadata_df_filled[self.config['id_column']] == id_val]

                if not metadata_row.empty:
                    # Extract values for each header
                    tuple_values = [metadata_row[col].values[0] for col in headers_to_include]
                    # Add the ID as the last element
                    tuple_values.append(id_val)
                    header_tuples.append(tuple(tuple_values))

            # Create column names for the multi-index (headers_to_include + id_column)
            multi_columns = headers_to_include + [self.config['id_column']]

            # Create a DataFrame for the multi-index
            header_df = pd.DataFrame(header_tuples, columns=multi_columns)

            # Create the MultiIndex
            multi_index = pd.MultiIndex.from_frame(header_df)

            # Set the multi-index on the columns
            multiheader.columns = multi_index

            return multiheader
    
        def create_clustered_heatmap(self, multiheader: pd.DataFrame, 
                                   custom_filename: Optional[str] = None,
                                   show_dendrograms: bool = False,
                                   figsize: Optional[Tuple[int, int]] = None) -> Tuple[Any, Any]:
            """
            Create a clustermap with color annotations for specified headers.
        
            Args:
                multiheader: DataFrame with multi-index columns to visualize
                custom_filename: Custom filename for output (without extension)
                show_dendrograms: Whether to show dendrograms
                figsize: Figure size tuple
            
            Returns:
                Tuple of (ClusterGrid object, Legend figure object)
            """
            if figsize is None:
                figsize = (8.27, 11.69)  # A4 size
        
            # Get headers to color
            headers_to_color = self.config['universal_headers'] + self.config['continuous_headers']
        
            # Set up file paths
            if custom_filename is None:
                base_filename = f"clustered_heatmap_K{self.k_value}"
            else:
                base_filename = custom_filename
            
            output_path = os.path.join(self.viz_directory, f"{base_filename}.png")
            legend_path = os.path.join(self.viz_directory, f"{base_filename}_legend.png")
        
            # Create the clustered heatmap using the integrated function
            g, legend_fig = self._create_clustered_heatmap_internal(
                multiheader=multiheader,
                id_column=self.config['id_column'],
                headers_to_color=headers_to_color,
                custom_colors=self.config['custom_colors'],
                continuous_headers=self.config['continuous_headers'],
                figsize=figsize,
                output_path=output_path,
                legend_path=legend_path,
                show_dendrograms=show_dendrograms,
                continuous_cmaps=self.config['continuous_cmaps']
            )
        
            print(f"✓ Clustered heatmap saved: {output_path}")
            if legend_fig is not None:
                print(f"✓ Legend saved: {legend_path}")
        
            return g, legend_fig
    
        def _create_clustered_heatmap_internal(self, multiheader, id_column=None, headers_to_color=None, 
                                             custom_colors=None, continuous_headers=None, figsize=(8.27, 11.69), 
                                             output_path=None, legend_path=None, show_dendrograms=False, 
                                             continuous_cmaps=None, continuous_colors=None):
            """Internal method for creating clustered heatmap - integrates your original function."""
            # Set default values if not provided
            if id_column is None:
                id_column = multiheader.columns.names[-1]

            if headers_to_color is None:
                headers_to_color = [name for name in multiheader.columns.names if name != id_column]

            if continuous_headers is None:
                continuous_headers = []

            if continuous_cmaps is None:
                continuous_cmaps = {}

            if continuous_colors is None:
                continuous_colors = {}

            # Get unique values for each header and create color palettes
            color_maps = {}
            colors_dict = {}

            # Define gray color for missing values
            missing_color = '#D3D3D3'  # Light gray

            # Create a unique palette for each header
            for header in headers_to_color:
                # Get unique values
                header_values = multiheader.columns.get_level_values(header)

                # Check if this header should use a continuous color scale
                if header in continuous_headers:
                    # Filter out non-numeric, zero, and missing values for finding min/max
                    numeric_values = pd.to_numeric(header_values, errors='coerce')
                    valid_mask = ~np.isnan(numeric_values) & (numeric_values != 0)

                    if not any(valid_mask):
                        # If no valid numeric values, fall back to categorical
                        print(f"Warning: Header '{header}' specified as continuous but contains no valid numeric values. Using categorical colors.")
                        is_continuous = False
                    else:
                        is_continuous = True
                        # Get min and max for normalization
                        vmin = np.min(numeric_values[valid_mask])
                        vmax = np.max(numeric_values[valid_mask])

                        # Create a normalization function
                        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

                        # Determine which colormap to use for this header
                        if header in continuous_colors:
                            # User provided custom colors to create a colormap
                            color_pair = continuous_colors[header]
                            cmap_name = f"custom_{header.replace(' ', '_')}"

                            # Create a custom colormap from the provided colors
                            if isinstance(color_pair, (list, tuple)) and len(color_pair) >= 2:
                                cmap = LinearSegmentedColormap.from_list(cmap_name, color_pair)
                            else:
                                print(f"Warning: Invalid color pair for '{header}'. Expected [color1, color2]. Using default colormap.")
                                cmap = plt.cm.viridis

                        elif header in continuous_cmaps:
                            # User specified a specific colormap
                            cmap_name = continuous_cmaps[header]
                            if isinstance(cmap_name, str):
                                try:
                                    # Try to get a matplotlib colormap
                                    cmap = plt.get_cmap(cmap_name)
                                except:
                                    print(f"Warning: Colormap '{cmap_name}' not found. Using default.")
                                    cmap = plt.cm.viridis
                            else:
                                # Assume it's already a colormap object
                                cmap = cmap_name
                        else:
                            # Use default colormap
                            cmap = plt.cm.viridis

                        # Store info for legend creation
                        color_maps[header] = {
                            'type': 'continuous',
                            'cmap': cmap,
                            'norm': norm,
                            'vmin': vmin,
                            'vmax': vmax
                        }

                        # Map values to colors
                        colors = []
                        for val in header_values:
                            try:
                                num_val = float(val)
                                if np.isnan(num_val) or num_val == 0:  # Treat zero as missing value
                                    colors.append(missing_color)
                                else:
                                    colors.append(cmap(norm(num_val)))
                            except (ValueError, TypeError):
                                colors.append(missing_color)

                        colors_dict[header] = pd.Series(colors, index=multiheader.columns)
                        continue  # Skip the categorical color assignment
                else:
                    is_continuous = False

                # Categorical coloring (for non-continuous headers)
                if not is_continuous:
                    # Filter out None, NaN, and empty string values for color assignment
                    unique_values = header_values.unique()
                    valid_values = [v for v in unique_values if pd.notna(v) and v != '']

                    if pd.api.types.is_numeric_dtype(np.array(valid_values, dtype=object)):
                        valid_values = sorted(valid_values)

                    # Use custom colors if provided, otherwise generate a palette
                    if custom_colors and header in custom_colors:
                        # Use custom color dictionary for this header
                        lut = custom_colors[header].copy()  # Make a copy to avoid modifying the original
                    else:
                        # Generate default palette
                        palette = sns.color_palette("deep", len(valid_values))
                        lut = dict(zip(valid_values, palette))

                    # Add a color for missing values (None, NaN, or empty string)
                    lut[None] = missing_color
                    lut[np.nan] = missing_color
                    lut[''] = missing_color

                    # Store the color lookup table
                    color_maps[header] = {
                        'type': 'categorical',
                        'lut': lut
                    }

                    # Map colors to columns, handling missing values
                    colors = []
                    for val in header_values:
                        if pd.isna(val) or val == '':
                            colors.append(missing_color)
                        elif val in lut:
                            colors.append(lut[val])
                        else:
                            colors.append(missing_color)  # If value not in lut for some reason

                    colors_dict[header] = pd.Series(colors, index=multiheader.columns)

            # Create a DataFrame of colors
            multi_colors = pd.DataFrame(colors_dict)

            # Create the clustermap
            g = sns.clustermap(
                multiheader, 
                center=0, 
                cmap="vlag",
                col_colors=multi_colors,
                dendrogram_ratio=(.1, .2),
                cbar_pos=(-.08, .50, .03, .2),
                linewidths=.75, 
                figsize=figsize,
                col_cluster=True, 
                row_cluster=True
            )

            # Get the specified ID column values for x-tick labels
            new_labels = [multiheader.columns.get_level_values(id_column)[i] for i in g.dendrogram_col.reordered_ind]

            # Set the x-tick positions and labels
            g.ax_heatmap.set_xticks(np.arange(len(new_labels)) + 0.5)
            g.ax_heatmap.set_xticklabels(new_labels, fontsize=4, rotation=45, ha='right')
            g.ax_heatmap.tick_params(axis='x', which='major', length=3, width=0.5, bottom=True)
            g.ax_heatmap.xaxis.set_tick_params(labeltop=False, top=False)

            # Show/hide dendrograms based on parameter
            g.ax_row_dendrogram.set_visible(show_dendrograms)
            g.ax_col_dendrogram.set_visible(show_dendrograms)

            # Save the figure if path is provided
            if output_path:
                g.savefig(output_path, dpi=300, format='png')

            # Create separate legend file if path is provided
            legend_fig = None
            if legend_path:
                legend_fig = self._create_legend_file(
                    color_maps=color_maps,
                    headers_to_color=headers_to_color,
                    continuous_headers=continuous_headers,
                    missing_color=missing_color,
                    output_path=legend_path
                )

            return g, legend_fig
    
        def _create_legend_file(self, color_maps, headers_to_color, continuous_headers, missing_color, output_path=None):
            """Create a separate legend file with vertical organization."""
            from matplotlib.cm import ScalarMappable
        
            # Create figure for the legends
            fig_height = 1 + 0.8 * len(headers_to_color)  # Dynamic height based on number of headers
            fig, ax = plt.subplots(figsize=(5, fig_height))
            ax.axis('off')  # Hide the axes

            # Configure background
            fig.patch.set_facecolor('white')

            # Vertical spacing parameters
            y_start = 0.95
            y_step = 0.9 / len(headers_to_color)

            legends = []

            # Track headers we've already seen to handle duplicates
            seen_headers = {}

            # Create legends in vertical stack
            for i, header in enumerate(headers_to_color):
                # Handle duplicate header names by creating unique titles
                if header in seen_headers:
                    seen_headers[header] += 1
                    display_title = f"{header} ({seen_headers[header]})"
                else:
                    seen_headers[header] = 1
                    display_title = header

                # Calculate y position
                y_pos = y_start - i * y_step

                # Check if this is a continuous or categorical header
                if header in continuous_headers and color_maps[header]['type'] == 'continuous':
                    # Get colormap info
                    cmap_info = color_maps[header]
                    cmap = cmap_info['cmap']
                    norm = cmap_info['norm']

                    # Create a new axis for the colorbar
                    cax_height = 0.02
                    cax_width = 0.3
                    cax = fig.add_axes([0.35, y_pos - cax_height/2, cax_width, cax_height])

                    # Create the colorbar
                    sm = ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')

                    # Add title
                    cbar.set_label(display_title, fontsize=10, labelpad=8)
                    cbar.ax.tick_params(labelsize=8)

                    legends.append(cbar)
                else:
                    # Categorical legend
                    lut = color_maps[header]['lut']

                    # Filter out None/NaN keys for the legend
                    filtered_lut = {k: v for k, v in lut.items() if k is not None and not (isinstance(k, float) and np.isnan(k)) and k != ''}

                    # Add a "Missing" entry if there were missing values
                    has_missing = None in lut or np.nan in lut or '' in lut
                    if has_missing:
                        filtered_lut["Missing"] = missing_color

                    # Create handles for the legend
                    handles = [plt.Rectangle((0,0), 1.5, 1.5, color=color, ec="k") for label, color in filtered_lut.items()]
                    labels = list(filtered_lut.keys())

                    # Add legend
                    num_items = len(filtered_lut)

                    # Determine number of columns based on number of items
                    legend_ncol = 1
                    if num_items > 6:
                        legend_ncol = 2
                    if num_items > 12:
                        legend_ncol = 3

                    leg = ax.legend(
                        handles, 
                        labels, 
                        title=display_title,
                        loc="center", 
                        bbox_to_anchor=(0.5, y_pos),
                        ncol=legend_ncol,
                        frameon=True, 
                        fontsize=8,
                        title_fontsize=10
                    )

                    # Need to manually add the legend
                    ax.add_artist(leg)
                    legends.append(leg)

            plt.tight_layout()

            # Save figure if path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

            return fig
    
        def get_top_tokens(self, top_n: Optional[int] = None) -> pd.DataFrame:
            """
            Get top tokens for each topic.
        
            Args:
                top_n: Number of top tokens to extract per topic
            
            Returns:
                DataFrame with top tokens for each topic
            """
            if top_n is None:
                top_n = self.config['top_asv_count']
        
            top_tokens_df = self.ASV_probabilities.apply(
                lambda row: row.nlargest(top_n), axis=1
            )
        
            return top_tokens_df
    
        def create_topic_taxon_heatmap(self, 
                                     highlight_taxa_dict: Optional[Dict] = None,
                                     custom_filename: Optional[str] = None,
                                     threshold: float = 0,
                                     cmap_below: str = 'Blues',
                                     cmap_above: str = 'Reds') -> Any:
            """
            Create a topic-taxon heatmap with different color scales.
        
            Args:
                highlight_taxa_dict: Dictionary with colors as keys and lists of taxa names as values
                custom_filename: Custom filename for output
                threshold: Boundary value that separates the two color scales
                cmap_below: Colormap for values below threshold
                cmap_above: Colormap for values above threshold
            
            Returns:
                Matplotlib axes object
            """
            if custom_filename is None:
                output_file = os.path.join(self.viz_directory, f"topic_taxon_heatmap_K{self.k_value}.png")
            else:
                output_file = os.path.join(self.viz_directory, f"{custom_filename}.png")
        
            # Use the ASV probabilities data
            ax = self._create_topic_taxon_heatmap_internal(
                data_matrix=self.ASV_probabilities,
                output_file=output_file,
                highlight_taxa_dict=highlight_taxa_dict,
                threshold=threshold,
                cmap_below=cmap_below,
                cmap_above=cmap_above,
                taxa_already_as_columns=True,
                vmin=self.config['heatmap_vmin'],
                vmax=self.config['heatmap_vmax']
            )
        
            print(f"✓ Topic-taxon heatmap saved: {output_file}")
            return ax
    
        def _create_topic_taxon_heatmap_internal(self, data_matrix, output_file=None,
                                               highlight_taxa_dict=None, threshold=0,
                                               cmap_below='Blues', cmap_above='Reds',
                                               taxa_already_as_columns=False,
                                               vmin=None, vmax=None):
            """Internal method for creating topic-taxon heatmap."""
            # Create a copy to avoid modifying the original
            df = data_matrix.copy()

            # Get the dataframe in the right orientation: taxa as columns, topics as rows
            if not taxa_already_as_columns:
                df = df.T

            # Determine color scale boundaries
            if vmin is None:
                vmin = df.values.min()
            if vmax is None:
                vmax = df.values.max()

            # Ensure the threshold is within the data range and create valid ordering
            threshold = max(vmin, min(vmax, threshold))
        
            # For TwoSlopeNorm, we need vmin < vcenter < vmax
            # If threshold equals vmin or vmax, adjust slightly
            if threshold == vmin:
                threshold = vmin + (vmax - vmin) * 0.01  # Move threshold slightly above vmin
            elif threshold == vmax:
                threshold = vmax - (vmax - vmin) * 0.01  # Move threshold slightly below vmax
        
            # Double-check the ordering
            if not (vmin < threshold < vmax):
                # If we still don't have proper ordering, use a simple colormap instead
                print(f"Warning: Cannot create TwoSlopeNorm with vmin={vmin:.6f}, threshold={threshold:.6f}, vmax={vmax:.6f}")
                print("Using simple colormap instead")
            
                # Use a simple colormap
                # Create figure with appropriate size based on number of taxa columns
                # Limit the maximum width to prevent oversized figures
                max_width = 50  # Maximum figure width in inches
                calculated_width = max(12, len(df.columns) * 0.25)
                figure_width = min(calculated_width, max_width)
            
                if calculated_width > max_width:
                    print(f"Warning: Calculated figure width ({calculated_width:.1f}) exceeds maximum ({max_width})")
                    print(f"Using maximum width of {max_width} inches. Consider filtering to fewer taxa.")
            
                plt.figure(figsize=(figure_width, 10))
                ax = sns.heatmap(df, cmap='Blues', linewidths=0.5, linecolor='white',
                                 cbar_kws={'label': 'Probability'})
            else:
                # For LDA word weights (all positive values), use TwoSlopeNorm
                if vmin >= 0:  # If all data is positive (LDA word weights)
                    # Create colormaps
                    cmap_below_obj = plt.get_cmap(cmap_below)
                    cmap_above_obj = plt.get_cmap(cmap_above)

                    # Create a colormap that changes at the threshold
                    below_colors = cmap_below_obj(np.linspace(0, 1, 128))
                    above_colors = cmap_above_obj(np.linspace(0, 1, 128))

                    # Create a new colormap
                    all_colors = np.vstack((below_colors, above_colors))
                    custom_cmap = ListedColormap(all_colors)

                    # Use TwoSlopeNorm for positive data with a threshold
                    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=threshold, vmax=vmax)
                
                    # Create figure with appropriate size based on number of taxa columns
                    # Limit the maximum width to prevent oversized figures
                    max_width = 50  # Maximum figure width in inches
                    calculated_width = max(12, len(df.columns) * 0.25)
                    figure_width = min(calculated_width, max_width)
                
                    if calculated_width > max_width:
                        print(f"Warning: Calculated figure width ({calculated_width:.1f}) exceeds maximum ({max_width})")
                        print(f"Using maximum width of {max_width} inches. Consider filtering to fewer taxa.")
                
                    plt.figure(figsize=(figure_width, 10))

                    # Create the heatmap with the appropriate colormap and norm
                    ax = sns.heatmap(df, cmap=custom_cmap, norm=norm, linewidths=0.5, linecolor='white',
                                     cbar_kws={'label': 'Probability'})
                else:
                    # For data that can be negative, use a custom diverging norm
                    # Define a custom normalization class
                    class DivergingNorm(mcolors.Normalize):
                        def __init__(self, vmin=None, vmax=None, threshold=0, clip=False):
                            self.threshold = threshold
                            mcolors.Normalize.__init__(self, vmin, vmax, clip)

                        def __call__(self, value, clip=None):
                            # Normalize values to [0, 1] for each segment separately
                            x = np.ma.array(value, copy=True)

                            # For values below or equal to threshold
                            mask_below = x <= self.threshold
                            if np.any(mask_below):
                                if self.threshold > self.vmin:
                                    x_below = (self.threshold - x[mask_below]) / (self.threshold - self.vmin)
                                else:
                                    x_below = np.zeros_like(x[mask_below])
                                x[mask_below] = x_below

                            # For values above threshold
                            mask_above = x > self.threshold
                            if np.any(mask_above):
                                if self.vmax > self.threshold:
                                    x_above = (x[mask_above] - self.threshold) / (self.vmax - self.threshold)
                                else:
                                    x_above = np.ones_like(x[mask_above])
                                x[mask_above] = x_above

                            return np.ma.array(x, mask=np.ma.getmask(value))

                    # Create the custom colormap and norm
                    cmap_below_obj = plt.get_cmap(cmap_below)
                    cmap_above_obj = plt.get_cmap(cmap_above)
                    below_colors = cmap_below_obj(np.linspace(0, 1, 128))
                    above_colors = cmap_above_obj(np.linspace(0, 1, 128))
                    all_colors = np.vstack((below_colors, above_colors))
                    custom_cmap = ListedColormap(all_colors)

                    # Create the custom diverging norm
                    norm = DivergingNorm(vmin=vmin, vmax=vmax, threshold=threshold)
                
                    # Create figure with appropriate size based on number of taxa columns
                    plt.figure(figsize=(max(12, len(df.columns) * 0.25), 10))

                    # Create the heatmap with the appropriate colormap and norm
                    ax = sns.heatmap(df, cmap=custom_cmap, norm=norm, linewidths=0.5, linecolor='white',
                                     cbar_kws={'label': 'Probability'})

            # Set the title
            plt.title(f'Topic-Taxon Distribution (K={self.k_value})', fontsize=16)

            # Rotate x-tick labels for better readability
            plt.xticks(rotation=90)

            # Now apply highlighting based on the actual column names
            if highlight_taxa_dict:
                # Get all tick labels
                tick_labels = df.columns.tolist()

                # Create a mapping from column names to positions
                column_to_position = {col: pos for pos, col in enumerate(tick_labels)}

                # Highlight the taxa in the dictionary
                for color, taxa_list in highlight_taxa_dict.items():
                    for taxon_name in taxa_list:
                        # Check if this taxon is in our columns
                        if taxon_name in column_to_position:
                            position = column_to_position[taxon_name]

                            # Get the x-tick label at this position
                            x_tick_labels = ax.get_xticklabels()
                            if position < len(x_tick_labels):
                                label = x_tick_labels[position]

                                # Style the label
                                label.set_color(color)
                                label.set_fontweight('bold')

            # Tight layout to ensure all elements are visible
            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, bbox_inches='tight', dpi=300)

            return ax
    
        def create_clustered_taxa_heatmap(self, 
                                        highlight_dict: Optional[Dict] = None,
                                        custom_filename: Optional[str] = None,
                                        rename_columns: Optional[Dict] = None) -> Tuple[Any, Any]:
            """
            Create a clustered heatmap from token probability data with optional highlighting.
        
            Args:
                highlight_dict: Dictionary with colors as keys and lists of taxa names as values
                custom_filename: Custom filename for output
                rename_columns: Dictionary to rename specific columns
            
            Returns:
                Tuple of (figure, axes)
            """
            if custom_filename is None:
                output_path = os.path.join(self.viz_directory, f"clustered_taxa_heatmap_K{self.k_value}.png")
            else:
                output_path = os.path.join(self.viz_directory, f"{custom_filename}.png")
        
            # Get top tokens
            top_tokens_df = self.get_top_tokens()
        
            # Create the clustered heatmap
            fig, ax = self._create_clustered_heatmap_taxa_internal(
                top_tokens_df=top_tokens_df,
                output_path=output_path,
                highlight_dict=highlight_dict,
                vmin=self.config['heatmap_vmin'],
                vmax=self.config['heatmap_vmax'],
                figsize=self.config['figsize'],
                cbar_ticks=self.config['cbar_ticks'],
                rename_columns=rename_columns
            )
        
            print(f"✓ Clustered taxa heatmap saved: {output_path}")
            return fig, ax
    
        def _create_clustered_heatmap_taxa_internal(self, top_tokens_df, output_path=None, highlight_dict=None, 
                                                  vmin=0, vmax=0.4, figsize=(14, 10), 
                                                  cbar_ticks=[0, 0.02, 0.05, 0.1, 0.3],
                                                  rename_columns=None):
            """Internal method for creating clustered taxa heatmap."""
            # Make a copy to avoid modifying the original dataframe
            top_tokens_df_customized = top_tokens_df.copy()

            # Rename columns if specified
            if rename_columns:
                top_tokens_df_customized = top_tokens_df_customized.rename(columns=rename_columns)

            # Fill any NaN values with 0 to avoid distance computation issues
            df_for_clustering = top_tokens_df_customized.fillna(0)

            # Add a small epsilon to any zero rows to avoid distance computation issues
            epsilon = 1e-10

            # Add epsilon to rows that are all zeros
            if (df_for_clustering.sum(axis=1) == 0).any():
                zero_rows = df_for_clustering.sum(axis=1) == 0
                df_for_clustering.loc[zero_rows] = epsilon

            # Add epsilon to columns that are all zeros
            if (df_for_clustering.sum(axis=0) == 0).any():
                zero_cols = df_for_clustering.sum(axis=0) == 0
                df_for_clustering.loc[:, zero_cols] = epsilon

            # Compute clustering with the cleaned data
            row_linkage = hierarchy.linkage(distance.pdist(df_for_clustering.values), method='ward')
            row_order = hierarchy.dendrogram(row_linkage, no_plot=True)['leaves']

            # Compute the clustering for columns (taxa)
            col_linkage = hierarchy.linkage(distance.pdist(df_for_clustering.values.T), method='ward')
            col_order = hierarchy.dendrogram(col_linkage, no_plot=True)['leaves']

            # Reorder the original dataframe according to the clustering
            df_clustered = top_tokens_df_customized.iloc[row_order, col_order]

            # Fill NaN values with zeros for visualization purposes
            df_clustered_filled = df_clustered.fillna(0)

            # Set up the plot
            fig, ax = plt.subplots(figsize=figsize)

            # Create a continuous colormap using PuBu
            cmap = plt.cm.PuBu

            # Create a custom normalization
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # Create the heatmap with annotations and continuous colors
            heatmap = sns.heatmap(df_clustered_filled, 
                          cmap=cmap,
                          norm=norm,
                          cbar_kws={'label': 'Probability',
                                    'ticks': cbar_ticks,
                                    'shrink': 0.5,  # Make the colorbar shorter
                                    'fraction': 0.046,  # Adjust width
                                    'pad': 0.04,    # Adjust distance from plot
                                    'aspect': 20},  # Make it thinner
                          annot=True,
                          fmt='.2f',
                          annot_kws={'size': 8},
                          square=True,
                          mask=pd.isna(df_clustered),
                          ax=ax)

            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)

            # Apply highlighting if provided
            if highlight_dict:
                # Get the current tick labels
                xlabels = [label.get_text() for label in ax.get_xticklabels()]

                # Create a list to hold legend handles
                legend_handles = []

                # Process each color in the highlight dict
                for color, taxa_list in highlight_dict.items():
                    # Find which taxa in the list are actually in the x-axis labels
                    for i, label_text in enumerate(xlabels):
                        if label_text in taxa_list:
                            # Get the current label
                            label = ax.get_xticklabels()[i]
                            # Set its color
                            label.set_color(color)
                            # Make it bold
                            label.set_fontweight('bold')

                    # Add to legend only if at least one taxon with this color exists in the plot
                    if any(taxon in xlabels for taxon in taxa_list):
                        patch = mpatches.Patch(color=color, label=f"{color.capitalize()} highlighted taxa")
                        legend_handles.append(patch)

                # Add legend if there are handles
                if legend_handles:
                    plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1), 
                              borderaxespad=0.)

            # Set title
            plt.title(f'Clustered Taxa Heatmap (K={self.k_value}, Top {self.config["top_asv_count"]} ASVs)', fontsize=14)

            # Tight layout to ensure all elements are visible
            plt.tight_layout()

            # Save the figure if a path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')

            return fig, ax
    
        def create_all_visualizations(self, 
                                    custom_prefix: Optional[str] = None,
                                    highlight_taxa_dict: Optional[Dict] = None,
                                    include_feature_analysis: bool = True) -> Dict[str, Any]:
            """
            Create all standard visualizations for the model.
        
            Args:
                custom_prefix: Custom prefix for all output files
                highlight_taxa_dict: Dictionary for highlighting specific taxa
                include_feature_analysis: Whether to include topic-feature analysis
            
            Returns:
                Dictionary containing all visualization objects
            """
            print(f"Creating all visualizations for K={self.k_value}...")
            print("=" * 50)
        
            results = {}
        
            # 1. Prepare heatmap data
            print("1️⃣ Preparing heatmap data...")
            multiheader = self.prepare_heatmap_data()
            results['multiheader'] = multiheader
        
            # 2. Create clustered heatmap with metadata
            print("\n2️⃣ Creating clustered heatmap with metadata...")
            filename = f"{custom_prefix}_clustered_metadata" if custom_prefix else None
            g, legend_fig = self.create_clustered_heatmap(
                multiheader=multiheader,
                custom_filename=filename
            )
            results['clustered_heatmap'] = g
            results['legend_figure'] = legend_fig
        
            # 3. Create clustered taxa heatmap (topic-feature analysis)
            if include_feature_analysis:
                print("\n3️⃣ Creating topic-feature analysis...")
                try:
                    # Import and use TopicFeatureProcessor
                    feature_processor = TopicFeatureProcessor(self.base_directory, self.k_value)
                
                    # Create the default clustered taxa heatmap
                    filename = f"{custom_prefix}_clustered_taxa" if custom_prefix else None
                    ax_taxa = feature_processor.create_default_visualization(
                        top_n=self.config.get('top_asv_count', 10),
                        custom_filename=filename,
                        highlight_features=highlight_taxa_dict
                    )
                    results['clustered_taxa_heatmap'] = ax_taxa
                    results['feature_processor'] = feature_processor
                
                except Exception as e:
                    print(f"Warning: Could not create topic-feature analysis: {e}")
                    results['clustered_taxa_heatmap'] = None
        
            print("\n" + "=" * 50)
            print("🎉 All visualizations created successfully!")
            print(f"📁 Outputs saved to: {self.viz_directory}")
            if include_feature_analysis:
                print("📊 Topic-feature analysis uses genus_ID level by default")
            print("=" * 50)
        
            return results
    
        def get_summary(self) -> Dict[str, Any]:
            """
            Get a summary of the loaded model and configuration.
        
            Returns:
                Dictionary containing model summary information
            """
            return {
                'k_value': self.k_value,
                'base_directory': self.base_directory,
                'num_samples': len(self.DM_distributions),
                'num_topics': len(self.DM_distributions[0]) if self.DM_distributions else 0,
                'num_asvs': self.ASV_probabilities.shape[1] if hasattr(self, 'ASV_probabilities') else 0,
                'metadata_columns': list(self.metadata.columns) if hasattr(self, 'metadata') else [],
                'configured_headers': {
                    'universal': self.config['universal_headers'],
                    'continuous': self.config['continuous_headers']
                },
                'visualization_config': {
                    'top_asv_count': self.config['top_asv_count'],
                    'custom_colors': list(self.config['custom_colors'].keys()),
                    'continuous_cmaps': list(self.config['continuous_cmaps'].keys())
                }
            }
    return (LDAModelVisualizer,)


@app.cell(hide_code=True)
def _():
    # Manual code
    # MC_sample_visualizer = LDAModelVisualizer(
    #         base_directory='example_test',
    #         k_value=3,
    #         metadata_path='example_data/data/shuffled_metadata.csv',
    #         universal_headers=["Group", "Batch"],  # REQUIRED
    #         continuous_headers=["VHCD", "IgG", "Neutrophils", "Lymphocytes", "MCV"],  # Optional
    #         top_asv_count=7,  # Optional (default: 7)
    #         id_column='ID'    # Optional (default: 'ID')
    #     )

    # MC_sample_visualizer.configure_colors(
    #         custom_colors={
    #             'Group': {
    #                 'Compromised': '#965454',
    #                 'Optimal': '#858b6f'
    #             }
    #         },
    #         continuous_cmaps={
    #             'VHCD': 'coolwarm',
    #             'Neutrophils': 'coolwarm'
    #         }
    #     )
    return


@app.cell(hide_code=True)
def _(
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    feature_level,
    os,
    output_dir,
    pd,
    plt,
    sns,
):
    class TopicFeatureProcessor:
        """
        A class for processing topic-feature matrices by mapping ASVs to taxonomic levels.
    
        This class handles:
        - Loading ASV probabilities and taxonomic data
        - Mapping ASVs to specified taxonomic levels
        - Grouping and summing probabilities by taxonomic level
        - Creating visualizations of topic-feature matrices
        - Integration with LDAModelVisualizer workflow
        """
    
        def __init__(self, base_directory: str, k_value: int):
            """
            Initialize the Topic Feature Processor.
        
            Args:
                base_directory: Base directory where LDA results are stored
                k_value: Number of topics (K) to process
            """
            self.base_directory = base_directory
            self.k_value = k_value
        
            # Set up paths
            self._setup_paths()
        
            # Initialize data containers
            self.new_taxa = None
            self.asv_probabilities = None
            self.processed_data = {}
        
            print(f"TopicFeatureProcessor initialized for K={k_value}")
            print(f"  Base directory: {self.base_directory}")
    
        def _setup_paths(self):
            """Set up all necessary paths."""
            self.inter_directory = os.path.join(self.base_directory, 'intermediate')
            self.lda_directory = os.path.join(self.base_directory, 'lda_results')
            self.MC_feature_directory = os.path.join(self.lda_directory, 'MC_Feature')
            self.viz_directory = os.path.join(self.base_directory, 'lda_visualization')
        
            # Create visualization directory
            os.makedirs(self.viz_directory, exist_ok=True)
        
            # Set up file paths
            self.path_to_new_taxa = os.path.join(self.inter_directory, "intermediate_taxa.csv")
            self.path_to_ASVProbabilities = os.path.join(
                self.MC_feature_directory, f"MC_Feature_Probabilities_{self.k_value}.csv"
            )
    
        def load_data(self):
            """Load taxonomic and ASV probability data."""
            try:
                # Load taxonomic data
                print(f"Reading taxonomic data from: {self.path_to_new_taxa}")
                self.new_taxa = pd.read_csv(self.path_to_new_taxa, index_col=0)
                print(f"✓ Taxonomic data loaded: {self.new_taxa.shape}")
            
                # Load ASV probabilities
                print(f"Reading ASV probabilities from: {self.path_to_ASVProbabilities}")
                self.asv_probabilities = pd.read_csv(self.path_to_ASVProbabilities, index_col=0)
                print(f"✓ ASV probabilities loaded: {self.asv_probabilities.shape}")
            
                # Display available taxonomic levels
                available_levels = [col for col in self.new_taxa.columns if col not in ['randomID']]
                print(f"✓ Available taxonomic levels: {available_levels}")
            
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Required data file not found: {e}")
            except Exception as e:
                raise Exception(f"Error loading data: {e}")
    
        def process_feature_level(self, 
                                feature_level: str = 'genus_ID', 
                                top_n: int = 10,
                                get_top_tokens_func: Optional[Callable] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
            """
            Process ASV probabilities at a specified taxonomic level.
        
            Args:
                feature_level: Taxonomic level to group by ('Genus', 'Family', 'Order', etc.)
                top_n: Number of top features to extract per topic
                get_top_tokens_func: Custom function to get top tokens
            
            Returns:
                Tuple of (processed_probabilities_df, top_tokens_df)
            """
            # Load data if not already loaded
            if self.new_taxa is None or self.asv_probabilities is None:
                self.load_data()
        
            # Validate feature level
            if feature_level not in self.new_taxa.columns:
                available_cols = [col for col in self.new_taxa.columns if col not in ['randomID']]
                raise ValueError(f"Feature level '{feature_level}' not found. Available: {available_cols}")
        
            print(f"\nProcessing feature level: {feature_level}")
            print("=" * 50)
        
            # Create mapping dictionary
            print(f"Creating mapping dictionary using column: {feature_level}")
            mapping_dict = dict(zip(self.new_taxa['randomID'], self.new_taxa[feature_level]))
        
            # Create MC labels
            MC_list = [f"MC{i}" for i in range(self.k_value)]
            print(f"Created {len(MC_list)} MC labels: {MC_list}")
        
            # Prepare ASV probabilities
            AP = self.asv_probabilities.copy()
            AP = AP.reset_index(drop=True)
            AP.index = MC_list
        
            print(f"Original ASV probabilities shape: {AP.shape}")
        
            # Map column names using the dictionary
            print("Mapping ASV IDs to taxonomic names...")
            original_columns = AP.columns.tolist()
            AP.columns = [mapping_dict.get(col, col) for col in AP.columns]
        
            # Count mapped columns
            mapped_count = sum(1 for orig_col in original_columns if orig_col in mapping_dict)
            print(f"Mapped {mapped_count}/{len(original_columns)} columns to taxonomic names")
        
            # Group by taxonomic level and sum probabilities
            print(f"Grouping by {feature_level} and summing probabilities...")
            grouped_AP = AP.groupby(level=0, axis=1).sum()
        
            print(f"✓ Grouped probabilities shape: {grouped_AP.shape}")
            print(f"✓ Unique {feature_level} features: {len(grouped_AP.columns)}")
        
            # Calculate top tokens
            top_tokens_df = None
            if get_top_tokens_func is None:
                # Default function
                def default_top_tokens(row):
                    return row.nlargest(top_n)
                get_top_tokens_func = default_top_tokens
        
            try:
                print(f"Calculating top {top_n} tokens for each MC...")
                top_tokens_df = grouped_AP.apply(get_top_tokens_func, axis=1)
                print(f"✓ Top tokens calculated: {top_tokens_df.shape}")
            except Exception as e:
                print(f"Warning: Could not calculate top tokens - {str(e)}")
                top_tokens_df = None
        
            # Store processed data
            self.processed_data[feature_level] = {
                'grouped_probabilities': grouped_AP,
                'top_tokens': top_tokens_df,
                'mapping_count': mapped_count,
                'total_features': len(grouped_AP.columns)
            }
        
            # Print summary
            print("\n=== PROCESSING SUMMARY ===")
            print(f"Feature level: {feature_level}")
            print(f"Number of MCs: {len(MC_list)}")
            print(f"Original ASVs: {len(original_columns)}")
            print(f"Mapped ASVs: {mapped_count}")
            print(f"Unique {feature_level} features: {len(grouped_AP.columns)}")
            print("=" * 50)
        
            return grouped_AP, top_tokens_df
    
        def create_feature_heatmap(self, 
                                 feature_level: str = 'genus_ID',
                                 use_top_tokens: bool = True,
                                 top_n: int = 10,
                                 figsize: Tuple[int, int] = (16, 8),  # Increased default width
                                 custom_filename: Optional[str] = None,
                                 highlight_features: Optional[Dict] = None,
                                 **heatmap_kwargs) -> plt.Axes:
            """
            Create a heatmap of topic-feature probabilities.
        
            Args:
                feature_level: Taxonomic level to visualize
                use_top_tokens: Whether to use only top tokens or all features
                top_n: Number of top tokens to show (if use_top_tokens=True)
                figsize: Figure size
                custom_filename: Custom filename for output
                highlight_features: Dict with colors as keys and feature lists as values
                **heatmap_kwargs: Additional arguments for seaborn heatmap
            
            Returns:
                matplotlib Axes object
            """
            # Process data if not already done
            if feature_level not in self.processed_data:
                self.process_feature_level(feature_level, top_n=top_n)
        
            # Get data to plot
            if use_top_tokens and self.processed_data[feature_level]['top_tokens'] is not None:
                data_to_plot = self.processed_data[feature_level]['top_tokens']
                plot_title = f'Top {top_n} {feature_level} Features by Topic (K={self.k_value})'
            else:
                data_to_plot = self.processed_data[feature_level]['grouped_probabilities']
                plot_title = f'All {feature_level} Features by Topic (K={self.k_value})'
        
            # Set up the plot
            plt.figure(figsize=figsize)
        
            # Default heatmap parameters
            default_kwargs = {
                'cmap': 'Blues',
                'annot': True,
                'fmt': '.3f',
                'annot_kws': {'size': 9, 'weight': 'bold'},  # Larger, bolder text
                'cbar_kws': {'label': 'Probability'},
                'linewidths': 0.5,
                'square': False  # Allow rectangular cells for better text fit
            }
            default_kwargs.update(heatmap_kwargs)
        
            # Create heatmap
            ax = sns.heatmap(data_to_plot, **default_kwargs)
        
            # Set aspect ratio to allow wider cells
            ax.set_aspect('auto')
        
            # Adjust subplot parameters to ensure everything fits
            plt.subplots_adjust(bottom=0.25, left=0.1, right=0.9, top=0.85)
        
            # Set title
            plt.title(plot_title, fontsize=14, pad=20)
        
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
            # Apply highlighting if provided
            if highlight_features:
                self._apply_feature_highlighting(ax, highlight_features)
        
            # Adjust layout
            plt.tight_layout()
        
            # Save if filename provided
            if custom_filename is None:
                filename = f"topic_{feature_level}_heatmap_K{self.k_value}"
                if use_top_tokens:
                    filename += f"_top{top_n}"
            else:
                filename = custom_filename
        
            output_path = os.path.join(self.viz_directory, f"{filename}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature heatmap saved: {output_path}")
        
            return ax
    
        def _apply_feature_highlighting(self, ax: plt.Axes, highlight_features: Dict):
            """Apply color highlighting to specific features in the heatmap."""
            # Get current tick labels
            xlabels = [label.get_text() for label in ax.get_xticklabels()]
        
            # Apply highlighting
            for color, feature_list in highlight_features.items():
                for i, label_text in enumerate(xlabels):
                    if label_text in feature_list:
                        # Get the current label and style it
                        label = ax.get_xticklabels()[i]
                        label.set_color(color)
                        label.set_fontweight('bold')
    
        def create_feature_comparison(self, 
                                    feature_levels: List[str] = None,
                                    top_n: int = 10,
                                    figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
            """
            Create a comparison plot of multiple taxonomic levels.
        
            Args:
                feature_levels: List of taxonomic levels to compare (defaults to common levels)
                top_n: Number of top features to show for each level
                figsize: Figure size
            
            Returns:
                matplotlib Figure object
            """
            # Use default feature levels if none provided
            if feature_levels is None:
                feature_levels = ['genus_ID', 'Genus', 'Family']
            # Process all feature levels
            for level in feature_levels:
                if level not in self.processed_data:
                    self.process_feature_level(level, top_n=top_n)
        
            # Create subplots
            n_levels = len(feature_levels)
            fig, axes = plt.subplots(1, n_levels, figsize=figsize)
        
            if n_levels == 1:
                axes = [axes]
        
            # Create heatmap for each level
            for i, level in enumerate(feature_levels):
                data = self.processed_data[level]['top_tokens']
                if data is not None:
                    sns.heatmap(
                        data, 
                        ax=axes[i],
                        cmap='Blues',
                        annot=True,
                        fmt='.3f',
                        cbar=True,
                        xticklabels=True,
                        yticklabels=True if i == 0 else False
                    )
                    axes[i].set_title(f'{level}\n(Top {top_n})', fontsize=12)
                    axes[i].tick_params(axis='x', rotation=45)
        
            plt.suptitle(f'Topic-Feature Comparison (K={self.k_value})', fontsize=16)
            plt.tight_layout()
        
            # Save comparison plot
            output_path = os.path.join(self.viz_directory, f"feature_comparison_K{self.k_value}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature comparison saved: {output_path}")
        
            return fig
    
        def get_feature_summary(self, feature_level: str) -> Dict[str, Any]:
            """
            Get summary statistics for a processed feature level.
        
            Args:
                feature_level: Taxonomic level to summarize
            
            Returns:
                Dictionary with summary statistics
            """
            if feature_level not in self.processed_data:
                raise ValueError(f"Feature level '{feature_level}' not processed yet")
        
            data = self.processed_data[feature_level]
            grouped_probs = data['grouped_probabilities']
        
            # Calculate statistics
            summary = {
                'feature_level': feature_level,
                'k_value': self.k_value,
                'num_topics': len(grouped_probs),
                'num_features': len(grouped_probs.columns),
                'mapped_asvs': data['mapping_count'],
                'total_probability_mass': grouped_probs.sum().sum(),
                'avg_probability_per_feature': grouped_probs.mean(axis=0).mean(),
                'max_probability': grouped_probs.max().max(),
                'min_probability': grouped_probs.min().min(),
                'sparsity': (grouped_probs == 0).sum().sum() / grouped_probs.size,
                'top_features_per_topic': {}
            }
        
            # Get top feature for each topic
            for topic in grouped_probs.index:
                top_feature = grouped_probs.loc[topic].idxmax()
                top_prob = grouped_probs.loc[topic].max()
                summary['top_features_per_topic'][topic] = {
                    'feature': top_feature,
                    'probability': top_prob
                }
        
            return summary
    
        def create_default_visualization(self,
                                       top_n: int = 10,
                                       custom_filename: Optional[str] = None,
                                       highlight_features: Optional[Dict] = None) -> plt.Axes:
            """
            Create the default clustered taxa heatmap using genus_ID.
        
            Args:
                top_n: Number of top features to show
                custom_filename: Custom filename for output
                highlight_features: Dict with colors as keys and feature lists as values
            
            Returns:
                matplotlib Axes object
            """
            # Process genus_ID level if not already done
            if 'genus_ID' not in self.processed_data:
                self.process_feature_level('genus_ID', top_n=top_n)
        
            # Create the heatmap with default filename
            if custom_filename is None:
                custom_filename = f"clustered_taxa_heatmap_K{self.k_value}"
        
            ax = self.create_feature_heatmap(
                feature_level='genus_ID',
                use_top_tokens=True,
                top_n=top_n,
                custom_filename=custom_filename,
                highlight_features=highlight_features
            )
        
            return ax
            """
            Save processed data to CSV files.
        
            Args:
                feature_level: Taxonomic level to save
                output_dir: Output directory (uses viz_directory if None)
            """
            if feature_level not in self.processed_data:
                raise ValueError(f"Feature level '{feature_level}' not processed yet")
        
            if output_dir is None:
                output_dir = self.viz_directory
        
            data = self.processed_data[feature_level]
        
            # Save grouped probabilities
            grouped_file = os.path.join(output_dir, f"topic_{feature_level}_probabilities_K{self.k_value}.csv")
            data['grouped_probabilities'].to_csv(grouped_file)
            print(f"✓ Grouped probabilities saved: {grouped_file}")
        
            # Save top tokens if available
            if data['top_tokens'] is not None:
                top_tokens_file = os.path.join(output_dir, f"topic_{feature_level}_top_tokens_K{self.k_value}.csv")
                data['top_tokens'].to_csv(top_tokens_file)
                print(f"✓ Top tokens saved: {top_tokens_file}")
    return (TopicFeatureProcessor,)


@app.cell
def _():
    # MC_Feature_processor = TopicFeatureProcessor(
    #         base_directory='example_test',
    #         k_value=3
    #     )
    # # Process different taxonomic levels
    # genus_data, genus_top = MC_Feature_processor.process_feature_level('Genus', top_n=10)
    # family_data, family_top = MC_Feature_processor.process_feature_level('Family', top_n=8)

    # # Plot different taxnomic levels data
    # ax1 = MC_Feature_processor.create_feature_heatmap('Genus', use_top_tokens=True, top_n=10)
    # ax2 = MC_Feature_processor.create_feature_heatmap('Family', use_top_tokens=False)
    return


if __name__ == "__main__":
    app.run()
