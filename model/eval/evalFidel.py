import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from scipy.spatial.distance import jensenshannon

def evaluate_fidelity(real_df, synthetic_df, bins=10):
    """
    Evaluate fidelity between two datasets.

    Parameters:
    - real_df (pd.DataFrame): The real dataset.
    - synthetic_df (pd.DataFrame): The synthetic dataset.
    - bins (int): Number of bins to use for histogram-based metrics (JS divergence).

    Returns:
    - results (dict): A dictionary with fidelity metrics for each column.
    """
    results = {}

    for col in real_df.columns:
        col_results = {}
        
        # Check if column is numeric or categorical
        if pd.api.types.is_numeric_dtype(real_df[col]):
            # Drop missing values for numerical tests.
            data_real = real_df[col].dropna()
            data_synth = synthetic_df[col].dropna()
            
            # KS Test: compares the empirical distributions.
            ks_stat, ks_p = ks_2samp(data_real, data_synth)
            
            # Wasserstein Distance: a measure of the distance between two distributions.
            w_distance = wasserstein_distance(data_real, data_synth)
            
            # Jensen-Shannon Divergence:
            # Compute histograms over the same bin edges for both datasets.
            hist_real, bin_edges = np.histogram(data_real, bins=bins, density=True)
            hist_synth, _ = np.histogram(data_synth, bins=bin_edges, density=True)
            # Avoid zero probabilities by adding a small constant.
            hist_real = hist_real + 1e-8
            hist_synth = hist_synth + 1e-8
            # Normalize histograms to get probability distributions.
            hist_real = hist_real / hist_real.sum()
            hist_synth = hist_synth / hist_synth.sum()
            js_div = jensenshannon(hist_real, hist_synth)

            col_results['ks_stat'] = ks_stat
            col_results['ks_p'] = ks_p
            col_results['wasserstein_distance'] = w_distance
            col_results['js_divergence'] = js_div

        else:
            # For categorical columns, compare frequency distributions.
            counts_real = real_df[col].value_counts().sort_index()
            counts_synth = synthetic_df[col].value_counts().sort_index()
            
            # Get the union of categories.
            all_categories = sorted(set(counts_real.index).union(set(counts_synth.index)))
            counts_real = counts_real.reindex(all_categories, fill_value=0)
            counts_synth = counts_synth.reindex(all_categories, fill_value=0)
            
            # Build a contingency table: rows represent datasets, columns represent categories.
            contingency = np.array([counts_real.values, counts_synth.values])
            
            # Chi-squared test for independence.
            chi2, chi2_p, dof, expected = chi2_contingency(contingency)
            
            col_results['chi2_stat'] = chi2
            col_results['chi2_p'] = chi2_p
            col_results['degrees_of_freedom'] = dof
            col_results['contingency_table'] = contingency.tolist()
        
        results[col] = col_results

    return results

# Example usage:
if __name__ == "__main__":
    # Replace these with your actual data loading steps.
    # For demonstration, we'll create two simple DataFrames.
    real_data = pd.DataFrame({
        'age': np.random.normal(30, 5, 1000),
        'income': np.random.normal(50000, 10000, 1000),
        'gender': np.random.choice(['M', 'F'], 1000)
    })

    synthetic_data = pd.DataFrame({
        'age': np.random.normal(30, 6, 1000),
        'income': np.random.normal(50000, 12000, 1000),
        'gender': np.random.choice(['M', 'F'], 1000)
    })

    fidelity_results = evaluate_fidelity(real_data, synthetic_data)
    
    # Print the fidelity metrics for each column.
    for col, metrics in fidelity_results.items():
        print(f"\nColumn: {col}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
