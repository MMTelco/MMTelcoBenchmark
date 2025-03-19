import os
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func  


os.environ['HF_TOKEN'] = "hf_token"


device_id = 0  
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

print("Loading Qwen2.5-VL-7B-Instruct Model for QA evaluation...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map={"": device_id}
)
print(f"Model is loaded on: {next(model.parameters()).device}")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


syst_prompt = (
    "You are an expert in telecommunications. Provide a concise and accurate answer "
    "to the following question."
)

def generate_answer(query, image_path):
    """
    Generate an answer for a given question and its associated image.
    """
    try:
        messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": syst_prompt}]},
                {"role": "user", "content": [
                    {"type": "text", "text": query},
                    {"type": "image", "image": Image.open(image_path)}
                ]}
            ]
        ]
      
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=text, images=image_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(device)
        
        # Generate answer
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text[0].strip()
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return "ERROR"

def compute_similarity_metrics(pred, actual):
    """
    Compute BLEU, ROUGE-L, and Semantic (BERTScore F1) scores 
    between the predicted and actual answers.
    """
    bleu_score = sentence_bleu([actual.split()], pred.split())
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_score_val = rouge.score(actual, pred)["rougeL"].fmeasure
    P, R, F1 = bert_score_func([pred], [actual], lang="en", verbose=False)
    sem_score = F1[0].item()
    return bleu_score, rouge_score_val, sem_score

def evaluate_qa():
    print("Loading QA dataset...")
    dataset_path = "/DATAY/telecom/ImageRetrieval/benchmark_qa.csv"
    df = pd.read_csv(dataset_path)

    required_columns = ["Image Path", "Query", "Answer"]
    df = df.dropna(subset=required_columns)

    bleu_scores = []
    rouge_scores = []
    sem_scores = []
    

    generated_answers_list = []
    
    print(f"Starting evaluation on {len(df)} entries...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        attempt = 0
        success = False
        while attempt < 3 and not success:
            try:
                query = row["Query"]
                ground_truth = row["Answer"]
                image_path = row["Image Path"]
                
                generated_answer = generate_answer(query, image_path)
                # Save the generated answer with associated data
                generated_answers_list.append({
                    "Index": idx,
                    "Query": query,
                    "Ground Truth": ground_truth,
                    "Image Path": image_path,
                    "Generated Answer": generated_answer
                })
                
                if generated_answer == "ERROR":
                    raise Exception("Generation error")
                
                bleu, rouge_metric, sem_score = compute_similarity_metrics(generated_answer, ground_truth)
                bleu_scores.append(bleu)
                rouge_scores.append(rouge_metric)
                sem_scores.append(sem_score)
                success = True
            except Exception as e:
                attempt += 1
                print(f"Error on attempt {attempt} for index {idx}: {e}")
                if attempt == 3:
                    print(f"Failed after 3 attempts for index {idx}. Skipping this entry.")
    
    if generated_answers_list:
        gen_df = pd.DataFrame(generated_answers_list)
        gen_csv = "qwen2.5vl_instruct_generated_answers.csv"
        gen_df.to_csv(gen_csv, index=False)
        print(f"Generated answers saved to {gen_csv}")
    
    if bleu_scores and rouge_scores and sem_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        avg_sem = sum(sem_scores) / len(sem_scores)
        
        print(f"Average BLEU Score: {avg_bleu}")
        print(f"Average ROUGE Score: {avg_rouge}")
        print(f"Average Semantic Score: {avg_sem}")
        
        summary_df = pd.DataFrame({
            "Average BLEU Score": [avg_bleu],
            "Average ROUGE Score": [avg_rouge],
            "Average Semantic Score": [avg_sem]
        })
        summary_csv = "qwen2.5vl_instruct_qa_evaluation_summary_only.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Evaluation completed! Summary saved to {summary_csv}")
    else:
        print("No valid scores to average.")

if __name__ == "__main__":
    evaluate_qa()
