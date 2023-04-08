import pandas as pd

# чтение исходного файла
df = pd.read_csv('spam.csv', sep=';')

# запись нового файла
df.to_csv('new_spam.csv', index=False)