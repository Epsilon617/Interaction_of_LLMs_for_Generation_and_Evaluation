from openai import OpenAI
import os
import pandas as pd
import csv
os.environ["OPENAI_API_KEY"] = ''

def score(prompt):
    client = OpenAI()
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo-0125",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    return completion.choices[0].message.content

def score_sentences(input_file, output_file):

    df = pd.read_csv(input_file, sep=',')

    with open(output_file, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['sent_id', 'generator', 'evaluator', 'score'])

        for index, row in df.iterrows():
            print(f"********* Processing row {index} ****************")
            prompt = f"""     German source:\n
                ```{row['german_sentence']}```\n
                English translation:\n
                ```{row['Translation']}```\n
                \n
                Based on the source segment and machine translation surrounded with triple backticks, identify
                error types in the translation and classify them. The categories of errors are: accuracy
                (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar,
                inconsistency, punctuation, register, spelling),
                style (awkward), terminology (inappropriate for context, inconsistent use), non-translation,
                other, or no-error.\n
                Each error is classified as one of three categories: critical, major, and minor.
                Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what
                the text is trying to say is still understandable. Minor errors are technically errors,
                but do not disrupt the flow or hinder comprehension.
                \n Number of critical errors: \n Number of major errors: \n Number of minor errors:
            """
            score_value = score(prompt)
            writer.writerow([row["id"], row["Model"], "gpt-3.5-turbo-0125", score_value])
            
input_file = "Translations_withSRC.csv"
output_file = "evaluation_results_GPT35_SecondPrompt.csv"
score_sentences(input_file, output_file)