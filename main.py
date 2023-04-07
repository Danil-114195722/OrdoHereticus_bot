import numpy as np
import pandas as pd


csv_df = pd.read_csv('spam.csv')

print(csv_df, '\n')
print(csv_df.loc(csv_df['v1']))
# print(csv_df)
