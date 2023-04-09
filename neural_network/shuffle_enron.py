import pandas as pd
from pathlib import Path


# путь до папки с проектом "OrdoHereticus_bot"
PROJECT_PATH = Path(__file__).resolve().parent.parent

csv_from = f"{PROJECT_PATH}/dataset/enron.csv"
csv_to = f"{PROJECT_PATH}/dataset/shuffle_enron.csv"

df = pd.read_csv(csv_from)

print(df, '\n\n')
df = df.sample(frac=1).reset_index(drop=True)
print(df)

df.to_csv(csv_to, index=False)
