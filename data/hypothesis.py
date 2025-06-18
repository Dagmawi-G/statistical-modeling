# File: C:\Users\hp\Desktop\Week Three\hypothesis_testing.py
import pandas as pd
import scipy.stats as stats
import numpy as np
import os

# Pull dataset from DVC
os.system('dvc pull')  # Ensure insurance.csv is available

# Load dataset
try:
    data = pd.read_csv('insurance.csv')
except FileNotFoundError:
    # Placeholder dataset
    print("Warning: Using placeholder dataset. Obtain insurance.csv from facilitators.")
    data = pd.DataFrame({
        'PolicyID': range(1, 1001),
        'Province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal'], 1000),
        'PostalCode': np.random.choice(['2000', '8000', '4000'], 1000),
        'Gender': np.random.choice(['Male', 'Female'], 1000),
        'TotalPremium': np.random.uniform(500, 2000, 1000),
        'TotalClaims': np.random.choice([0, np.random.uniform(100, 10000)], 1000, p=[0.8, 0.2])
    })

# Calculate metrics
data['ClaimOccurred'] = data['TotalClaims'] > 0
data['Margin'] = data['TotalPremium'] - data['TotalClaims']

# Hypothesis testing function
def hypothesis_test(data, group_col, metric, metric_name):
    groups = data[group_col].value_counts().index[:2]  # Select top 2 categories
    group_a = data[data[group_col] == groups[0]]
    group_b = data[data[group_col] == groups[1]]
    
    result = {'Group A': groups[0], 'Group B': groups[1], 'Metric': metric_name, 'p-value': None, 'Decision': None}
    
    if metric_name == 'Claim Frequency':
        contingency_table = pd.crosstab(data[group_col], data['ClaimOccurred'])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        result['p-value'] = p_value
        result['Test'] = 'Chi-squared'
        
    elif metric_name in ['Claim Severity', 'Margin']:
        group_a_metric = group_a[group_a['ClaimOccurred']][metric] if metric_name == 'Claim Severity' else group_a[metric]
        group_b_metric = group_b[group_b['ClaimOccurred']][metric] if metric_name == 'Claim Severity' else group_b[metric]
        u_stat, p_value = stats.mannwhitneyu(group_a_metric, group_b_metric)
        result['p-value'] = p_value
        result['Test'] = 'Mann-Whitney U'
        result['Mean Difference'] = group_a_metric.mean() - group_b_metric.mean()
    
    result['Decision'] = 'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'
    return result

# Run tests
results = []
results.append(hypothesis_test(data, 'Province', 'ClaimOccurred', 'Claim Frequency'))
results.append(hypothesis_test(data, 'Province', 'TotalClaims', 'Claim Severity'))
results.append(hypothesis_test(data, 'PostalCode', 'ClaimOccurred', 'Claim Frequency'))
results.append(hypothesis_test(data, 'PostalCode', 'TotalClaims', 'Claim Severity'))
results.append(hypothesis_test(data, 'PostalCode', 'Margin', 'Margin'))
results.append(hypothesis_test(data, 'Gender', 'ClaimOccurred', 'Claim Frequency'))
results.append(hypothesis_test(data, 'Gender', 'TotalClaims', 'Claim Severity'))

# Save results
with open('hypothesis_test_results.txt', 'w') as f:
    f.write("Task 3: A/B Hypothesis Testing Results\n\n")
    for r in results:
        f.write(f"Hypothesis: {r['Metric']} by {r['Group A']} vs {r['Group B']}\n")
        f.write(f"Test: {r['Test']}\n")
        f.write(f"p-value: {r['p-value']:.4f}\n")
        f.write(f"Decision: {r['Decision']}\n")
        if 'Mean Difference' in r:
            f.write(f"Mean Difference: {r['Mean Difference']:.2f}\n")
        f.write("-" * 50 + "\n")

print("Hypothesis testing completed. See 'hypothesis_test_results.txt' for details.")