# Import necessary libraries

import pandas as pd
import numpy as np
import skbio
from skbio.stats.composition import clr
import re

# Load the dataset
genus = pd.read_csv("/content/drive/My Drive/GSE113690_Autism_16S_rRNA_OTU_assignment_and_abundance.csv")

# Display the dataset
genus

# Identify columns with all zero values
zero_cols = genus.columns[(genus == 0).all()]
print(zero_cols)

# Extract OTU and taxonomy information
taxa = genus[['OTU', 'taxonomy']].set_index('OTU')
# Transpose the dataset and set OTU as index
genus_T = genus.drop('taxonomy', axis=1).set_index('OTU').transpose()

# Create binary target variable based on sample names
target = genus_T.index.to_list()
binary_target = np.array([1 if t.startswith('A') else 0 for t in target ])

# Add 1 to all values to avoid division by zero
genus_T = genus_T + 1
# Calculate total species count for each sample
total_species = genus_T.sum(axis=1)
# Define absolute abundance value
abs_abundance = 31757
# Calculate relative abundance
pd_rel_abundance = genus_T.div(total_species, axis=0)

# Add Diagnosis column to the relative abundance dataframe
pd_rel_abundance['Diagnosis'] = binary_target

# Save the relative abundance dataframe to a CSV file
pd_rel_abundance.to_csv("/content/drive/My Drive/data_autismo_relative.csv")

# Apply CLR transformation to the relative abundance dataframe
df_clr = pd.DataFrame(clr(pd_rel_abundance), columns=pd_rel_abundance.columns)

# Display the CLR-transformed dataframe
df_clr

# Add Diagnosis column to the CLR-transformed dataframe
df_clr['Diagnosis'] = binary_target

# Save the CLR-transformed dataframe to a CSV file
df_clr.to_csv('/content/drive/My Drive/df_autismo_clr')

# Display the CLR-transformed dataframe
df_clr

