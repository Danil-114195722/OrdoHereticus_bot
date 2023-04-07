import numpy as np
import pandas as pd
from googletrans import Translator


csv_df = pd.read_csv('spam.csv')

print(csv_df, '\n')

first_line = csv_df.loc[0]
print(first_line['message'])


