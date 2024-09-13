from openai import OpenAI
import os
import pandas as pd
import csv
os.environ["OPENAI_API_KEY"] = ''

def refine_translation(prompt):
    client = OpenAI()
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo-0125",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    return completion.choices[0].message.content

def translate_sentences(input_file, output_file):

    df = pd.read_csv("Gemma_scores_GPT35_EN.csv", sep=',', header=0)

    with open(output_file, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['sent_id', 'generator', 'evaluator', 'refined_translation'])

        for index, row in df.iterrows():
            print(f"********* Processing row {index} ****************")
            sentence_id = row['sent_id'] 
            english_sentence = row['English_Sentence']
            translation = row['Translation']
            explanation = row['explanation']
            score = row['score']
            prompt = f"""A week ago, you provided an German translation for the following English sentence:
                    \n English Sentence: {english_sentence}
                    \n German Translation: {translation}
                    \n To assess the quality of the translation, another person reviewed it and gave it a score of {score} out of 5. Here is the explanation: ```{explanation}```
                    \n Would you reconsider your translation of "{english_sentence}" and provide a better one?
                    """
            refined_translation = refine_translation(prompt)
            writer.writerow([sentence_id, "gpt-3.5-turbo-0125", "google/gemma-1.1-7b-it", refined_translation])

input_file = "Gemma_scores_GPT35_EN.csv"
output_file = "refinedTranslations_Gemma_neutral_GPT35_EN.csv"
translate_sentences(input_file, output_file)