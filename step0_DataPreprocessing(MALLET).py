import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import random
    import string
    import os
    import pandas as pd
    return mo, os, pd, random, string


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

            # Create directories
            for directory in [self.intermediate_directory, self.loop_directory, 
                             self.lda_directory, self.MC_sample_directory, 
                             self.MC_feature_directory]:
                os.makedirs(directory, exist_ok=True)

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
            # Load ASV table
            sampletable = pd.read_csv(self.asvtable_path, index_col=0)
            self.asvtable = sampletable.T

            # Load and process taxonomy
            self.taxa_split = pd.read_csv(self.taxonomy_path, index_col=0)
            self.taxa_split['Genus_based'] = self.taxa_split.apply(lambda row: self.update_genus_new(row), axis=1)
            self.taxa_split['genus_ID'] = self.taxa_split['Genus_based'].apply(lambda x: self.assign_id(x))
            self.taxa_split['randomID'] = self.taxa_split.apply(
                lambda row: self.generate_single_id(min_length=5), axis=1
            )

            # Save intermediate taxonomy file
            self.taxa_split.to_csv(os.path.join(self.intermediate_directory, 'intermediate_taxa.csv'), index=True)

            # Create sample table with random IDs
            mapping_dict = self.taxa_split['randomID'].to_dict()
            self.sampletable_randomID = sampletable.copy()
            new_columns = {}
            for col in sampletable.columns:
                if col in mapping_dict:
                    new_columns[col] = mapping_dict[col]
                else:
                    new_columns[col] = col
            self.sampletable_randomID = self.sampletable_randomID.rename(columns=new_columns)
            self.sampletable_randomID.to_csv(os.path.join(self.intermediate_directory, 'annotaed_randomid.csv'), index=True)

        def create_mallet_input(self):
            """Create Mallet input documents from processed data."""
            if self.sampletable_randomID is None:
                raise ValueError("Data must be loaded and processed first. Call load_and_process_data().")

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

        def process_all(self):
            """
            Run the complete processing pipeline.

            Returns:
                dict: Dictionary containing processed data and file paths
            """
            self.load_and_process_data()
            self.create_mallet_input()

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
def _(TaxonomyProcessor):
    if __name__ == "__main__":
        # Initialize the processor
        processor = TaxonomyProcessor(
            asvtable_path='example/data/sampled_data.csv',
            taxonomy_path='example/data/shuffled_taxonomy.csv',
            base_directory='example_test'
        )

        # Process all data
        results = processor.process_all()

        # Access processed data
        print("Processing complete!")
    return processor, results


if __name__ == "__main__":
    app.run()
