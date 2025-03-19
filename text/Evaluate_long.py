import os
import pandas as pd
import json
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import sacrebleu
import statistics
import warnings
warnings.filterwarnings("ignore")
import evaluate  

sacrebleu = evaluate.load("sacrebleu")
rouge = evaluate.load('rouge')



from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity




import torch
import json
import random
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os

import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv


def response(query):
    llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment="gpt-4o",temperature=0  )



    messages = [
                    ("system", syst_prompt ),
                    ("human",query),
                ]
    answer = llm.invoke(messages).content

    return answer

embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')


#load benchmark
data_long = pd.read_csv("/home/anshul/mm-telco/Long_multi_BENCH_data.csv")

syst_prompt = """
Please provide the correct answer to the following telecommunications-related long form question. 


Input Example :

        Question: How does the 5G system's user plane architecture support differentiated service requirements, such as low latency and high throughput?
  

Output Example :

        The 5G system's user plane architecture supports differentiated service requirements through the use of Service Hosting Environments located within the operator's network. 
        These environments enable services to be offered closer to end-users, meeting localization requirements such as low latency and reduced bandwidth pressure.

"""




def LLM_as_judge(rubric , answer):
    llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment="gpt-4o-mini",temperature=0  )


    syst_prompt_judge = ''' You are an objective and meticulous evaluator specializing in telecom 3GPP. For each evaluation task, you will receive two inputs: a Rubric that outlines the expected criteria and Answer to Evaluate . Your task is to evaluate the generated answer solely based on the provided rubric. 

    Score the answer on a scale from 0 to 100, where 100 indicates that the answer perfectly satisfies all rubric criteria, and 0 indicates that it meets none of the criteria. Your evaluation should consider completeness, correctness, and relevance to the rubric's requirements.

    Output must be a valid JSON object with a single key "score" containing the numerical score. Do not include any additional text, explanation, or formatting outside of the JSON.

    Example output:
    {"score": 85} '''

    messages = [
                    ("system", syst_prompt_judge ),
                    ("human", f"Rubrics: {rubric} \n\n , Answer to Evaluate:{answer}"),
                ]
    score = json.loads(llm.invoke(messages).content)['score']

    return score




# Function to calculate the scores for each candidate answer
def calculate_metrics(predicted_answer, candidate_answers,Rubric):
    metrics = {"ROUGE 1": [],"ROUGE 2": [],"ROUGE L": [], "SEM Score": [], "BLEU": [],"LLM Judge":[]}

    # Initialize ROUGE scorer
    # rouge_scorer_inst = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'], use_stemmer=True)

    # ROUGE score
    ref_answer = candidate_answers 
    print("rougue")
    rouge_scores = rouge.compute(predictions=[predicted_answer],references=[ref_answer])
    # rouge_score_combined = (rouge_scores["rouge1"].fmeasure + rouge_scores["rougeL"].fmeasure) / 2



    print("Semscore")
    embeddings = embed_model.encode([predicted_answer,ref_answer], normalize_embeddings=True)
    sem_score = cosine_similarity(embeddings[0].reshape(1, -1),embeddings[1].reshape(1, -1))[0][0]

    print("LLM Judge")


    llm_score = LLM_as_judge(Rubric,predicted_answer)




    # BLEU score
    print("bleu")
    bleu_score = sacrebleu.compute(predictions=[predicted_answer],references= [ref_answer], lowercase=True )["score"]
    # Store the scores
    metrics["ROUGE 1"].append(rouge_scores["rouge1"])
    metrics["ROUGE 2"].append(rouge_scores["rouge2"])
    metrics["ROUGE L"].append(rouge_scores["rougeL"])
    metrics["SEM Score"].append(sem_score)

    metrics["BLEU"].append(bleu_score)
    metrics["LLM Judge"].append(llm_score)

    return metrics

# Function to evaluate long-answer questions and save the best one
def evaluate_long_answers(model):
    print(f"Evaluating model: {model}")
    save_path_csv = os.path.join('/home/anshul/mm-telco/long_results/'+model + "_LONG_answers.csv")  

    
    if os.path.exists(save_path_csv):
        responses = pd.read_csv(save_path_csv)
        start = len(responses)
    else:
        start = 1

    print(f"Starting from {start} ....")

    for idx, row in tqdm(data_long.iloc[start-1:].iterrows()):
        attempt = 0
        success = False

        while attempt < 2 and not success:
            try:
                user_prompt = f"""
                Question: {row['Question']}\n
    
                """
                
                # LLM to predict the correct answer
         
                predicted_answer  = response( user_prompt)

                candidate_answers = row['Answer']
                rubric = row['Proof']

                # Calculate the metrics for each candidate answer
                metrics = calculate_metrics(predicted_answer , candidate_answers,rubric)



                # Select the best answer based on the highest average of all metrics
                rougue1 = metrics["ROUGE 1"][0]
                rougue2 = metrics["ROUGE 2"][0]  
                rouguel = metrics["ROUGE L"][0]  
                bleu = metrics["BLEU"][0] 
                SemScore = metrics["SEM Score"][0] 
                llm_score= metrics["LLM Judge"][0]


 
                # Save the  answer with its corresponding metrics
                output = {
                    "Question id": idx,
                    "Question": row["Question"],
                    "Answer": row['Answer'] ,
                 
                    "Predicted Answer": predicted_answer ,
                    "ROUGE 1": rougue1,
                    "ROUGE 2": rougue2,
                    "ROUGE L": rouguel,
                    "SEM Score": SemScore ,
                    "LLM Judge Score":llm_score ,
       
                    "BLEU": bleu ,

                }

                output_df = pd.DataFrame([output])
                output_df.to_csv(save_path_csv, mode='a', header=not pd.io.common.file_exists(save_path_csv), index=False)

                success = True
            except Exception as e:
                attempt += 1
                print(f"Error type: {type(e).__name__} on attempt {attempt}")
                print(f"Error message: {e}")
                if attempt == 2:
                    print(f"Failed after 3 attempts for Question id {idx}. Skipping to next.")

if __name__ == "__main__":
    models = ["gpt4o"]
    for model in models:
        evaluate_long_answers(model)





