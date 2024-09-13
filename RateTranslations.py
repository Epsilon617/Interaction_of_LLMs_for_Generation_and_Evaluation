from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import csv
import pandas as pd

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ''

models = {
    "google/gemma-1.1-7b-it": HuggingFaceEndpoint(
        repo_id="google/gemma-1.1-7b-it", 
        max_length=128, 
        temperature=0.6,
        max_new_tokens = 400
    ),
    "mistralai/Mistral-7B-Instruct-v0.2": HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
        max_length=128, 
        temperature=0.6,
        max_new_tokens = 400
    ),
    "meta-llama/Llama-2-7b-chat-hf": HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-2-7b-chat-hf", 
        max_length=128, 
        temperature=0.6,
        max_new_tokens = 400
    ),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
        max_length=128, 
        temperature=0.6,
        max_new_tokens = 400
    )
}

def get_score(llm, german_sentence, english_sentence):
    question = f"""     German source:\n
                ```{german_sentence}```\n
                English translation:\n
                ```{english_sentence}```\n
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
    template = """Question: {question} 
            """
    prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    output = llm_chain.invoke(question)
    return output['text']

csv_file_path = 'evaluation_results1.csv'
file_exists = os.path.isfile(csv_file_path)

with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['sent_id', 'generator', 'evaluator', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if not file_exists:
        writer.writeheader()

    df_source = pd.read_csv("Translations_withSRC.csv", sep=',', header=0)
    for index, row in df_source.iterrows():
        print(f"********* Processing row {index} ****************")
        sentence_id = row['id']
        model = row['Model']
        en_sent = row['Translation']
        de_sent = row['german_sentence']
     
        for model_name in ['google/gemma-1.1-7b-it', 'mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mixtral-8x7B-Instruct-v0.1']:
            llm = models.get(model_name)
            score_value = get_score(llm, de_sent, en_sent)
            writer.writerow({'sent_id': sentence_id, 'generator': model, 'evaluator': model_name, 'score': score_value})
            csvfile.flush()  
               
print("CSV file has been updated after each entry.")