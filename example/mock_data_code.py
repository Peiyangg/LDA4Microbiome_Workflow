import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from numpy.random import dirichlet
    return dirichlet, mo, np, pd


@app.cell
def _(np):
    def sample_rows_and_columns(df, n_row_samples=10, n_col_samples=5):
    
        n_rows, n_cols = df.shape
    
        # Ensure we don't sample more than available
        n_row_samples = min(n_row_samples, n_rows)
        n_col_samples = min(n_col_samples, n_cols)
    
        # Sample row indices
        row_indices = np.random.choice(n_rows, size=n_row_samples, replace=False)
    
        # Sample column indices
        col_indices = np.random.choice(n_cols, size=n_col_samples, replace=False)
    
        # Get sampled DataFrame
        sampled_df = df.iloc[row_indices, col_indices].copy()
    
        return sampled_df

    def shuffle_columns_independently(df):
        """
        Shuffle values within each column independently
    
        Parameters:
        df: pandas DataFrame
    
        Returns:
        shuffled_df: pandas DataFrame with shuffled values in each column
        """
    
        # Create a copy to avoid modifying original
        shuffled_df = df.copy()
    
        # Shuffle each column independently
        for col in shuffled_df.columns:
            shuffled_df[col] = np.random.permutation(shuffled_df[col].values)
    
        return shuffled_df
    return sample_rows_and_columns, shuffle_columns_independently


@app.cell
def _(
    sample_rows_and_columns,
    sampletable,
    shuffle_columns_independently,
    taxo,
):
    subdf = sample_rows_and_columns(sampletable, n_row_samples=30,  n_col_samples=5000)
    shuffled_df = shuffle_columns_independently(taxo)
    return shuffled_df, subdf


if __name__ == "__main__":
    app.run()
