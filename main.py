import pandas as pd
from requests.exceptions import ConnectionError
from translate import Translator


translator = Translator(to_lang="ru", from_lang="en")
print(translator.translate(text="What's Up?"), '\n\n')

csv_en_df = pd.read_csv('spam_en.csv')

print(csv_en_df, '\n\n')

csv_ru_df = csv_en_df.copy()

# csv_ru_df.loc[csv_ru_df['message']] = translator.translate(csv_ru_df['message'])
# csv_ru_df = csv_ru_df.replace({'message': {str(csv_ru_df['message']): translator.translate(str(csv_ru_df['message']))}})

csv_ru_df['message'] = csv_ru_df['message'].apply(translator.translate)

print(csv_ru_df)
# print(first_line)


