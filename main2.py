from translate import Translator
import pandas as pd
import warnings


warnings.filterwarnings('ignore')
translator = Translator(to_lang="ru", from_lang="en")

en_file_path = 'spam_en.csv'
ru_file_path = 'spam_ru.csv'

ru_df = pd.DataFrame({'label': ['label'], 'message': ['message']})

with open(en_file_path, 'r') as csv_en_file:
    # with open(ru_file_path, 'w') as csv_ru_file:
    for num, line in enumerate(csv_en_file.readlines()[1:5], 0):

        en_message = ','.join(line.split(',')[1:])
        label = line[0]

        ru_message = translator.translate(en_message)

        ru_df = ru_df.append({'label': label, 'message': ru_message}, ignore_index=True)

ru_df = ru_df.iloc[1:]
ru_df.to_csv(ru_file_path)

print(ru_df)
