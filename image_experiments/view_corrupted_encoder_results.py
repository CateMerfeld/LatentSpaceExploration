import torch

import pandas as pd

data = torch.load('/home/cmdunham/scratch/latent_space_exploration/mnist_6_8_embeddings.pt')

# column names for the 10-dim output vector
cols = [f"dim_{i}" for i in range(10)]

# --- test set ---
test_df = pd.DataFrame(data['test_encoded'].numpy(), columns=cols)
test_df['true_label'] = data['test_true_labels'].numpy()

# print(test_df.head())

# sum of all dim_* columns for the first 30 rows
dim_cols = [col for col in test_df.columns if 'dim' in col]
# print(test_df.loc[:29, dim_cols].sum(axis=1))

totals = test_df.loc[:, dim_cols].sum(axis=1)

# print(totals[:5])

print('------------------------')


# # find all values in the dim_* columns greater than 0.99
# mask = test_df[dim_cols] > 0.99
# # list of tuples: (row_index, column, value)
# high_certainty = [ (int(idx), col, float(test_df.at[idx, col]))
# 				   for col in dim_cols
# 				   for idx in test_df.index[mask[col]] ]

# print('Found', len(high_certainty), 'values > 0.99')
# if len(high_certainty) > 0:
# 	print(high_certainty[:50])

# find all values in the dim_* columns = 1
mask = test_df[dim_cols] == 1.0
# list of tuples: (row_index, column, value)
high_certainty = [ (int(idx), col, float(test_df.at[idx, col]))
				   for col in dim_cols
				   for idx in test_df.index[mask[col]] ]

print('Found', len(high_certainty), 'values == 1.0')
if len(high_certainty) > 0:
	print(high_certainty[:50])

# less_than_one = [i for i in totals if i < 1]
# print('There were:', len(less_than_one), 'totals less than 1' )
# print(less_than_one[:10])
# print('------------------------')

# greater_than_one = [i for i in totals if i > 1]
# print('There were:', len(greater_than_one), 'totals greater than 1' )
# print(greater_than_one[:10])
# print('------------------------')

# less_than_one = [i for i in totals if i < .9]
# print('There were:', len(less_than_one), 'totals less than 1' )
# if len(less_than_one) > 0:
#     print(less_than_one[:10])
# print('------------------------')

# greater_than_one = [i for i in totals if i > 1.1]
# print('There were:', len(greater_than_one), 'totals greater than 1' )
# if len(greater_than_one) > 0:
#     print(greater_than_one[:10])
# print('------------------------')