import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import os
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import json
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.cluster import hierarchy
    from scipy.spatial import distance
    import xml.etree.ElementTree as ET
    import matplotlib.patches as mpatches
    return (
        ET,
        LinearSegmentedColormap,
        distance,
        hierarchy,
        json,
        mcolors,
        mo,
        mpatches,
        np,
        os,
        pd,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(
    LinearSegmentedColormap,
    Top_ASV_of_each_MC,
    distance,
    hierarchy,
    mcolors,
    mpatches,
    np,
    pd,
    plt,
    sns,
):
    # Functions

    def rgb_to_hex(r, g, b):
        """Convert RGB values to hex code"""
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    def prepare_heatmap_data(DM_distributions, metadata_df, id_column='averageID', headers_to_include=None):
        """
        Create a multi-index heatmap table from distributions and metadata

        Parameters:
        -----------
        DM_distributions : list of lists
            Nested list where each child list is one sample's topic distribution
        metadata_df : DataFrame
            DataFrame containing metadata for each sample
        id_column : str, default='averageID'
            Column name to use for matching/joining data
        headers_to_include : list, optional
            List of column names to include in the multi-index header
            If None, all columns in metadata_df will be used

        Returns:
        --------
        DataFrame with multi-index columns
        """
        # Convert nested list to DataFrame and transpose
        distributions_df = pd.DataFrame(DM_distributions)
        multiheader = distributions_df.T

        # If no headers specified, use all columns except id_column
        if headers_to_include is None:
            headers_to_include = [col for col in metadata_df.columns if col != id_column]

        # Fill NaN values in metadata with 0
        metadata_df_filled = metadata_df.fillna(0)

        # Get unique IDs from the metadata that match our samples
        sample_ids = metadata_df_filled[id_column].values[:len(DM_distributions)]

        # Create a list of tuples for the multi-index
        header_tuples = []
        for idx, id_val in enumerate(sample_ids):
            # Get the matching metadata row
            metadata_row = metadata_df_filled[metadata_df_filled[id_column] == id_val]

            if not metadata_row.empty:
                # Extract values for each header
                tuple_values = [metadata_row[col].values[0] for col in headers_to_include]
                # Add the ID as the last element
                tuple_values.append(id_val)
                header_tuples.append(tuple(tuple_values))

        # Create column names for the multi-index (headers_to_include + id_column)
        multi_columns = headers_to_include + [id_column]

        # Create a DataFrame for the multi-index
        header_df = pd.DataFrame(header_tuples, columns=multi_columns)

        # Create the MultiIndex
        multi_index = pd.MultiIndex.from_frame(header_df)

        # Set the multi-index on the columns
        multiheader.columns = multi_index

        return multiheader

    def create_clustered_heatmap(multiheader, id_column=None, headers_to_color=None, custom_colors=None, 
                                 continuous_headers=None, figsize=(8.27, 11.69), output_path=None, 
                                 legend_path=None, show_dendrograms=False, continuous_cmaps=None,
                                 continuous_colors=None):
        """
        Create a clustermap with color annotations for specified headers, supporting both categorical
        and continuous color scales. Saves the legend to a separate file.

        Parameters:
        -----------
        multiheader : DataFrame
            DataFrame with multi-index columns to visualize
        id_column : str, optional
            Column name to use for x-axis labels
            If None, uses the last level of the MultiIndex
        headers_to_color : list, optional
            List of column headers to use for color annotations
            If None, uses all headers except the id_column
        custom_colors : dict, optional
            Dictionary mapping header names to dictionaries of value-color pairs
            Example: {'Diagnosis': {'healthy': '#7B8B6F', 'infested': '#965454'}}
        continuous_headers : list, optional
            List of headers that should use continuous color scales instead of categorical
            These headers should contain numeric data
        figsize : tuple, default=(8.27, 11.69)
            Figure size in inches (default is A4)
        output_path : str, optional
            Path to save the figure, if None, figure is not saved
        legend_path : str, optional
            Path to save the separate legend file, if None, legend is not saved
        show_dendrograms : bool, default=False
            Whether to show the dendrograms
        continuous_cmaps : dict, optional
            Dictionary mapping header names to specific colormap names or custom colormaps
            Example: {'Temperature': 'viridis', 'pH': 'coolwarm'}
        continuous_colors : dict, optional
            Dictionary mapping header names to pairs of colors for creating custom colormaps
            Example: {'Temperature': ['white', 'red'], 'pH': ['blue', 'yellow']}

        Returns:
        --------
        tuple
            (ClusterGrid object, Legend figure object)
        """
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

        # Store any custom colormaps created during execution
        created_colormaps = {}

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
                            created_colormaps[cmap_name] = cmap
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
                        # No specific colormap provided, use a default or generated one
                        # First, check if a custom color gradient is appropriate based on the header name
                        header_lower = header.lower()

                        # Use a sensible default based on common header names
                        if 'temperature' in header_lower or 'temp' in header_lower:
                            cmap = plt.cm.hot
                        elif 'ph' in header_lower or 'acid' in header_lower:
                            cmap = plt.cm.PiYG
                        elif 'time' in header_lower or 'date' in header_lower:
                            cmap = plt.cm.Blues
                        elif 'concentration' in header_lower or 'density' in header_lower:
                            cmap = plt.cm.Greens
                        elif 'pressure' in header_lower:
                            cmap = plt.cm.Oranges
                        elif any(x in header_lower for x in ['weight', 'mass']):
                            cmap = plt.cm.YlOrBr
                        elif any(x in header_lower for x in ['height', 'length', 'width']):
                            cmap = plt.cm.BuPu
                        else:
                            # Use a default colormap from matplotlib's built-in options
                            standard_cmaps = [plt.cm.viridis, plt.cm.plasma, plt.cm.inferno, 
                                             plt.cm.magma, plt.cm.cividis, plt.cm.cool, 
                                             plt.cm.YlGnBu, plt.cm.YlOrRd]

                            # Assign based on position in continuous_headers list
                            cmap_index = continuous_headers.index(header) % len(standard_cmaps)
                            cmap = standard_cmaps[cmap_index]

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
                # Choose appropriate color palette based on header type
                header_lower = header.lower()

                # Start with some default palette options from seaborn
                palette_options = {
                    "categorical": "deep",
                    "sequential": "Blues",
                    "diverging": "RdBu",
                    "qualitative": "Set1"
                }

                # Try to find a sensible default palette based on common header naming patterns
                if 'diagnosis' in header_lower or 'disease' in header_lower:
                    palette = sns.color_palette("Set1", len(valid_values))
                elif 'time' in header_lower or 'date' in header_lower:
                    palette = sns.color_palette("Blues", len(valid_values))
                elif 'treatment' in header_lower or 'water' in header_lower:
                    palette = sns.color_palette("pastel", len(valid_values))
                elif 'stock' in header_lower or 'source' in header_lower:
                    palette = sns.color_palette("bright", len(valid_values))
                elif 'group' in header_lower or 'category' in header_lower:
                    palette = sns.color_palette("husl", len(valid_values))
                elif 'status' in header_lower or 'state' in header_lower:
                    palette = sns.color_palette("Set2", len(valid_values))
                elif 'location' in header_lower or 'site' in header_lower:
                    palette = sns.color_palette("Paired", len(valid_values))
                elif 'gender' in header_lower or 'sex' in header_lower:
                    palette = sns.color_palette("RdBu", len(valid_values))
                elif 'age' in header_lower or 'year' in header_lower:
                    palette = sns.color_palette("YlGnBu", len(valid_values))
                else:
                    # Use a default palette
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

        # Then set the labels for these positions
        g.ax_heatmap.set_xticklabels(new_labels, fontsize=4, rotation=45, ha='right')

        # Make tick marks thinner and shorter
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
            legend_fig = create_legend_file(
                color_maps=color_maps,
                headers_to_color=headers_to_color,
                continuous_headers=continuous_headers,
                missing_color=missing_color,
                output_path=legend_path
            )

        # Return the clustermap and legend figure
        return g, legend_fig

    def create_legend_file(color_maps, headers_to_color, continuous_headers, missing_color, output_path=None):
        """
        Create a separate legend file with vertical organization

        Parameters:
        -----------
        color_maps : dict
            Dictionary containing color mapping information
        headers_to_color : list
            List of header names to create legends for
        continuous_headers : list
            List of headers that use continuous color scales
        missing_color : str
            Hex color code for missing values
        output_path : str, optional
            Path to save the legend file, if None, figure is displayed but not saved

        Returns:
        --------
        matplotlib.figure.Figure
            The legend figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
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


    def create_topic_taxon_heatmap(data_matrix, output_file=None,
                               highlight_taxa_dict=None, threshold=0,
                               cmap_below='Blues', cmap_above='Reds',
                               taxa_already_as_columns=False,
                               vmin=None, vmax=None):
        """
        Create a topic-taxon heatmap with different color scales for values above and below a threshold.
        Specifically designed for LDA (Latent Dirichlet Allocation) word weights, which are always positive.

        Parameters:
        -----------
        data_matrix : pandas.DataFrame
            DataFrame containing topic-taxon distribution data (LDA word weights)
        output_file : str, optional
            Path to save the figure
        highlight_taxa_dict : dict, optional
            Dictionary with colors as keys and lists of taxa names to highlight as values
        threshold : float, optional
            Boundary value that separates the two color scales. Default is 0.
            For LDA weights, set this to a meaningful value within your data range (e.g., 0.01).
        cmap_below : str, optional
            Matplotlib colormap name for values below or equal to the threshold
        cmap_above : str, optional
            Matplotlib colormap name for values above the threshold
        taxa_already_as_columns : bool, optional
            Whether taxa are already represented as columns in the data_matrix
        vmin : float, optional
            Minimum value for the color scale. If None, uses the minimum value in the data
        vmax : float, optional
            Maximum value for the color scale. If None, uses the maximum value in the data
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import seaborn as sns
        from matplotlib.colors import ListedColormap

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

        # Ensure the threshold is within the data range
        threshold = max(vmin, min(vmax, threshold))

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
        plt.title('Topic-Taxon Distribution', fontsize=16)

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

    def get_top_tokens(row, top_n=Top_ASV_of_each_MC):
        return row.nlargest(top_n)

    def create_clustered_heatmap_taxa(top_tokens_df, output_path=None, highlight_dict=None, 
                                vmin=0, vmax=0.4, figsize=(14, 10), 
                                cbar_ticks=[0, 0.02, 0.05, 0.1, 0.3],
                                rename_columns=None):
        """
        Create a clustered heatmap from token probability data with optional highlighting.

        Parameters:
        -----------
        top_tokens_df : pandas.DataFrame
            DataFrame containing token probability data
        output_path : str, optional
            Path to save the figure
        highlight_dict : dict, optional
            Dictionary with colors as keys and lists of taxa names to highlight as values
        vmin : float, default=0
            Minimum value for color scale
        vmax : float, default=0.4
            Maximum value for color scale
        figsize : tuple, default=(14, 10)
            Figure size as (width, height)
        cbar_ticks : list, default=[0, 0.02, 0.05, 0.1, 0.3]
            Ticks for the colorbar
        rename_columns : dict, optional
            Dictionary to rename specific columns, e.g. {'old_name': 'new_name'}

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        """
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

        # Adjust layout
        plt.tight_layout()

        # Save the figure if a path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig, ax
    return (
        create_clustered_heatmap,
        create_clustered_heatmap_taxa,
        create_legend_file,
        create_topic_taxon_heatmap,
        get_top_tokens,
        prepare_heatmap_data,
        rgb_to_hex,
    )


@app.cell
def _(mo):
    mo.md("""# Inputs""")
    return


@app.cell
def _():
    Number_of_Topic='7'
    Top_ASV_of_each_MC = 7
    base_directory = '/Users/huopeiyang/Library/CloudStorage/OneDrive-KULeuven/py_working/LDA_workflow/Luke_test'
    metadata_path = '/Users/huopeiyang/Library/CloudStorage/OneDrive-KULeuven/py_working/Luke_data/metadata_cleaned.csv'
    return Number_of_Topic, Top_ASV_of_each_MC, base_directory, metadata_path


@app.cell
def _(rgb_to_hex):
    # list metadata you want to add in the 
    univesal_headers = ["Group","Batch"]
    continue_list = ["VHCD","IgG","Neutrophils","Lymphocytes","MCV"]
    metadata_list = univesal_headers+continue_list

    # Create custom colors dictionary
    group_colors = {
        'Group': {
            'Compromised': rgb_to_hex(150, 84, 84),  # #965454
            'Optimal': rgb_to_hex(133, 139, 111)     # #858b6f
        }
    }

    customed_dic = {'VHCD': 'coolwarm', 'Neutrophils': 'coolwarm','Lymphocytes': 'coolwarm','IgG':'coolwarm','MCV':'coolwarm'}
    return (
        continue_list,
        customed_dic,
        group_colors,
        metadata_list,
        univesal_headers,
    )


@app.cell
def _(Number_of_Topic, base_directory, os):
    loop_directory = os.path.join(base_directory, 'lda_loop')
    inter_directory = os.path.join(base_directory, 'intermediate')
    viz_directory = os.path.join(base_directory, 'lda_visualization')
    results_directory = os.path.join(base_directory, 'lda_results')
    os.makedirs(viz_directory, exist_ok=True)

    output_directory_path = base_directory + '/lda_onenumber'
    path_to_DirichletComponentProbabilities             = results_directory + f"/DirichletComponentProbabilities_{Number_of_Topic}.csv"
    path_to_TaxaProbabilities                          = results_directory + f"/ASVProbabilities_{Number_of_Topic}.csv"
    path_to_TaxaProbabilities_json                     = viz_directory + '/ASVProbability.json'
    path_to_ASVProbabilities                           = results_directory + f"/ASVProbabilities_{Number_of_Topic}.csv"
    path_to_mc_taxa_heatmap                            = viz_directory + "/mc-taxa-heatmap.png"
    path_to_lda_diagnosis                              = loop_directory + f"/mallet.diagnostics.{Number_of_Topic}.xml"
    path_to_new_taxa                                   = inter_directory + "/new_taxa.csv"
    return (
        inter_directory,
        loop_directory,
        output_directory_path,
        path_to_ASVProbabilities,
        path_to_DirichletComponentProbabilities,
        path_to_TaxaProbabilities,
        path_to_TaxaProbabilities_json,
        path_to_lda_diagnosis,
        path_to_mc_taxa_heatmap,
        path_to_new_taxa,
        results_directory,
        viz_directory,
    )


@app.cell
def _(mo):
    mo.md("""# Visualization: Sample vs Dirichlet Component""")
    return


@app.cell
def _(
    metadata_path,
    path_to_DirichletComponentProbabilities,
    path_to_TaxaProbabilities,
    pd,
):
    noises=[]

    DMP = pd.read_csv(
        path_to_DirichletComponentProbabilities, 
        index_col=0
    ).drop(noises, errors='ignore') 
    DM_distributions = DMP.values.tolist()
    TP = pd.read_csv(path_to_TaxaProbabilities, index_col=0)
    metadata = pd.read_csv(metadata_path,index_col=0)
    return DMP, DM_distributions, TP, metadata, noises


@app.cell
def _(mo):
    mo.md("""# Plot""")
    return


@app.cell
def _(DM_distributions, metadata, metadata_list, prepare_heatmap_data):
    multiheader = prepare_heatmap_data(
        DM_distributions=DM_distributions,
        metadata_df=metadata,
        id_column='ID',
        headers_to_include=metadata_list
    )
    return (multiheader,)


@app.cell
def _(
    continue_list,
    create_clustered_heatmap,
    customed_dic,
    group_colors,
    metadata_list,
    multiheader,
    os,
    viz_directory,
):
    g, leg = create_clustered_heatmap(multiheader, 
                                    headers_to_color=metadata_list,
                                    continuous_headers=continue_list,
                                    continuous_cmaps=customed_dic,
                                    custom_colors = group_colors,
                                    output_path=os.path.join(viz_directory, "clustered-selected_headers_divergingColor.png"),
                                    legend_path=os.path.join(viz_directory, "clustered-selected_headers-legend_divergingColor.png")
                                     )
    return g, leg


@app.cell
def _(mo):
    mo.md("""### Visualization: Dirichlet Component vs Taxa""")
    return


@app.cell
def _(path_to_new_taxa, pd):
    new_taxa=pd.read_csv(path_to_new_taxa, index_col=0)
    new_taxa_dict = dict(zip(new_taxa['randomID'], new_taxa['genus_ID']))
    return new_taxa, new_taxa_dict


@app.cell
def _(Number_of_Topic, path_to_ASVProbabilities, pd):
    MC_list=["MC"+str(a) for a in range(int(Number_of_Topic))]
    AP = pd.read_csv(path_to_ASVProbabilities, index_col=0)
    AP = AP.reset_index()
    AP = AP.drop(columns=['index'])
    AP.index = MC_list
    return AP, MC_list


@app.cell
def _(AP, get_top_tokens, new_taxa_dict):
    AP.columns = [new_taxa_dict.get(col, col) for col in AP.columns]
    top_tokens_df = AP.apply(get_top_tokens, axis=1)
    return (top_tokens_df,)


@app.cell
def _(
    create_clustered_heatmap_taxa,
    path_to_mc_taxa_heatmap,
    plt,
    top_tokens_df,
):
    fig, ax = create_clustered_heatmap_taxa(
        top_tokens_df, 
        output_path=path_to_mc_taxa_heatmap,
    )
    plt.show()
    return ax, fig


if __name__ == "__main__":
    app.run()
