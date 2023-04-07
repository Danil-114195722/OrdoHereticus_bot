from translate import Translator

translator = Translator(to_lang='Russian')

result = translator.translate(text="What's Up?")

print(result)
