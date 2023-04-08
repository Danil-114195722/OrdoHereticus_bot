import pandas as pd

from data.constants import PROJECT_PATH


csv_from = f"{PROJECT_PATH}/dataset/enron.csv"
csv_to = f"{PROJECT_PATH}/dataset/shuffle_enron.csv"

df = pd.read_csv(csv_from)

print(df, '\n\n')
df = df.sample(frac=1).reset_index(drop=True)
print(df)

df.to_csv(csv_to, index=False)
