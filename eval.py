# import pandas as pd
# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText
# from nltk.translate.bleu_score import sentence_bleu
# from rouge_score import rouge_scorer
# from tqdm import tqdm
# from PIL import Image

# # Load Qwen2.5-VL-7B-Instruct Model
# print("Loading Qwen2.5-VL-7B-Instruct Model...")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# # model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct").to(device)

# from transformers import AutoProcessor, AutoModelForCausalLM

# # # Load Qwen2.5-VL model
# # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# # Load the dataset
# file_path = "/DATAY/telecom/ImageRetrieval/query_data.csv"  # Change this to your dataset path
# df = pd.read_csv(file_path).head(10)

# # Ensure required columns exist
# required_columns = ["Image Path", "Query", "Answer"]
# df = df.dropna(subset=required_columns)  # Drop rows where any required field is missing

# # Function to generate answer using Qwen2.5-VL-7B-Instruct
# def generate_answer(image_path, query):
#     try:
#         # Load image
#         image = Image.open(image_path).convert("RGB")
        
#         # Preprocess input
#         inputs = processor(images=image, text=query, return_tensors="pt").to(device)
        
#         # Generate response
#         generated_ids = model.generate(**inputs, max_length=50)
#         generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#         return generated_answer
#     except Exception as e:
#         print(f"❌ Error processing {image_path}: {e}")
#         return "ERROR"

# # Function to compute BLEU and ROUGE scores
# def compute_similarity_metrics(pred, actual):
#     # BLEU Score
#     bleu_score = sentence_bleu([actual.split()], pred.split())

#     # ROUGE Score
#     rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
#     rouge_score = rouge.score(actual, pred)["rougeL"].fmeasure

#     return bleu_score, rouge_score

# # Evaluate dataset
# results = []
# print("Generating and Evaluating Answers...")
# for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
#     image_path = row["Image Path"]
#     query = row["Query"]
#     ground_truth = row["Answer"]

#     # Generate answer using Qwen2.5-VL-7B-Instruct
#     generated_answer = generate_answer(image_path, query)

#     # Compute BLEU and ROUGE scores
#     bleu, rouge = compute_similarity_metrics(generated_answer, ground_truth)

#     # Store results
#     results.append({
#         "Image Path": image_path,
#         "Query": query,
#         "Ground Truth Answer": ground_truth,
#         "Generated Answer": generated_answer,
#         "BLEU Score": bleu,
#         "ROUGE Score": rouge
#     })

# # Convert results to DataFrame
# results_df = pd.DataFrame(results)

# # Save evaluation results to CSV
# output_file = "qwen2.5vl_instruct_evaluation_bleu_rouge.csv"
# results_df.to_csv(output_file, index=False)

# print(f"✅ Evaluation completed! Results saved to {output_file}")

# import pandas as pd
# import torch
# import gc
# import nltk
# from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
# from nltk.translate.bleu_score import sentence_bleu
# from rouge_score import rouge_scorer
# from tqdm import tqdm
# from PIL import Image
# from huggingface_hub import login

# # Download NLTK resources
# nltk.download('punkt')

# # Memory optimization configuration
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16
# )

# # Authenticate with Hugging Face
# login("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")

# # Load components with explicit image token handling
# processor = AutoProcessor.from_pretrained(
#     "meta-llama/Llama-3.2-11B-Vision",
#     use_fast=False,
#     token=True
# )

# # Add image token as special token
# IMAGE_TOKEN = "<image>"
# if IMAGE_TOKEN not in processor.tokenizer.additional_special_tokens:
#     processor.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})

# model = AutoModelForImageTextToText.from_pretrained(
#     "meta-llama/Llama-3.2-11B-Vision",
#     quantization_config=bnb_config,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True
# ).eval()

# # Resize embeddings and configure model
# model.resize_token_embeddings(len(processor.tokenizer))
# model.config.image_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

# # Image processing parameters
# IMAGE_SIZE = 448
# MAX_SEQ_LENGTH = 256
# file_path = "/DATAY/telecom/ImageRetrieval/query_data.csv"
# df = pd.read_csv(file_path).head(10).dropna(subset=["Image Path", "Query", "Answer"])

# def compute_similarity_metrics(pred, actual):
#     """Calculate BLEU and ROUGE-L scores"""
#     try:
#         bleu = sentence_bleu([actual.split()], pred.split())
#         scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#         rouge = scorer.score(actual, pred)['rougeL'].fmeasure
#         return bleu, rouge
#     except Exception as e:
#         print(f"Metric calculation error: {str(e)}")
#         return 0.0, 0.0

# def generate_answer(image_path, query):
#     try:
#         # Memory management
#         torch.cuda.empty_cache()
#         gc.collect()

#         # Load and process image
#         image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        
#         # Create formatted prompt with explicit image token
#         formatted_prompt = f"{IMAGE_TOKEN}\nQuestion: {query}\nAnswer:"
        
#         # Process inputs with proper token mapping
#         inputs = processor(
#             text=formatted_prompt,
#             images=image,
#             return_tensors="pt",
#             max_length=MAX_SEQ_LENGTH,
#             truncation=True,
#             padding="max_length",
#             add_special_tokens=True
#         ).to(model.device)

#         # Verify image token presence
#         input_tokens = processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
#         if IMAGE_TOKEN not in input_tokens:
#             raise ValueError("Image token not found in processed inputs")

#         # Generate response
#         with torch.inference_mode():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=80,
#                 do_sample=False,
#                 temperature=0.7,
#                 top_k=20,
#                 repetition_penalty=1.25
#             )

#         # Cleanup
#         del inputs
#         torch.cuda.empty_cache()
        
#         return processor.decode(outputs[0], skip_special_tokens=True)

#     except Exception as e:
#         print(f"Error processing {image_path}: {str(e)[:200]}")
#         return "ERROR"

# # Processing loop with validation
# results = []
# print("Generating and Evaluating Answers...")
# for idx, row in tqdm(df.iterrows(), total=len(df)):
#     try:
#         # Validate data
#         if pd.isna(row["Image Path"]) or pd.isna(row["Query"]) or pd.isna(row["Answer"]):
#             continue
            
#         # Generate and evaluate
#         generated = generate_answer(row["Image Path"], row["Query"])
#         bleu, rouge = compute_similarity_metrics(generated, row["Answer"])
        
#         results.append({
#             "Image Path": row["Image Path"],
#             "Query": row["Query"],
#             "Ground Truth": row["Answer"],
#             "Generated Answer": generated,
#             "BLEU": bleu,
#             "ROUGE": rouge
#         })

#     except Exception as e:
#         print(f"Row {idx} error: {str(e)}")
    
#     # Memory cleanup
#     gc.collect()
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()

# # Save results
# results_df = pd.DataFrame(results)
# results_df.to_csv("llama3_vision_evaluation.csv", index=False)
# print("✅ Evaluation completed successfully!")






# import os
# import pandas as pd
# import torch
# from tqdm import tqdm
# from PIL import Image
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from nltk.translate.bleu_score import sentence_bleu
# from rouge_score import rouge_scorer

# # Set your Hugging Face token
# os.environ['HF_TOKEN'] = "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"

# # Setup device and load model/processor
# device_id = 0  # adjust if needed
# device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

# print("Loading Qwen2.5-VL-7B-Instruct Model for QA evaluation...")
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     device_map={"": device_id}
# )
# print(f"Model is loaded on: {next(model.parameters()).device}")

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# # Define a system prompt tailored for QA
# syst_prompt = (
#     "You are an expert in telecommunications. Provide a concise and accurate answer "
#     "to the following question."
# )

# def generate_answer(query, image_path):
#     """
#     Generate an answer for a given question and its associated image.
#     """
#     try:
#         messages = [
#             [
#                 {"role": "system", "content": [{"type": "text", "text": syst_prompt}]},
#                 {"role": "user", "content": [
#                     {"type": "text", "text": query},
#                     {"type": "image", "image": Image.open(image_path)}
#                 ]}
#             ]
#         ]
#         # Prepare input text using the chat template
#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, _ = process_vision_info(messages)
#         inputs = processor(text=text, images=image_inputs, padding=True, return_tensors="pt")
#         inputs = inputs.to(device)
        
#         # Generate answer
#         generated_ids = model.generate(**inputs, max_new_tokens=128)
#         generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
#         output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         return output_text[0].strip()
#     except Exception as e:
#         print(f"❌ Error processing {image_path}: {e}")
#         return "ERROR"

# def compute_similarity_metrics(pred, actual):
#     """
#     Compute BLEU and ROUGE-L scores between the predicted and actual answers.
#     """
#     bleu_score = sentence_bleu([actual.split()], pred.split())
#     rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
#     rouge_score = rouge.score(actual, pred)["rougeL"].fmeasure
#     return bleu_score, rouge_score

# def evaluate_qa():
#     print("Loading QA dataset...")
#     # Update the file path to your dataset; remove .head(10) to process the full dataset
#     dataset_path = "/DATAY/telecom/ImageRetrieval/query_data.csv"
#     df = pd.read_csv(dataset_path).head(10)
    
#     # Ensure required columns exist
#     required_columns = ["Image Path", "Query", "Answer"]
#     df = df.dropna(subset=required_columns)
    
#     # Lists to accumulate scores
#     bleu_scores = []
#     rouge_scores = []
    
#     print(f"Starting evaluation on {len(df)} entries...")
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
#         attempt = 0
#         success = False
#         while attempt < 3 and not success:
#             try:
#                 query = row["Query"]
#                 ground_truth = row["Answer"]
#                 image_path = row["Image Path"]
                
#                 generated_answer = generate_answer(query, image_path)
#                 # Skip if generation fails
#                 if generated_answer == "ERROR":
#                     raise Exception("Generation error")
                
#                 bleu, rouge_metric = compute_similarity_metrics(generated_answer, ground_truth)
#                 bleu_scores.append(bleu)
#                 rouge_scores.append(rouge_metric)
#                 success = True
#             except Exception as e:
#                 attempt += 1
#                 print(f"Error on attempt {attempt} for index {idx}: {e}")
#                 if attempt == 3:
#                     print(f"Failed after 3 attempts for index {idx}. Skipping this entry.")
    
#     if bleu_scores and rouge_scores:
#         avg_bleu = sum(bleu_scores) / len(bleu_scores)
#         avg_rouge = sum(rouge_scores) / len(rouge_scores)
        
#         print(f"Average BLEU Score: {avg_bleu}")
#         print(f"Average ROUGE Score: {avg_rouge}")
        
#         # Create a DataFrame with only the average scores and save to CSV
#         summary_df = pd.DataFrame({
#             "Average BLEU Score": [avg_bleu],
#             "Average ROUGE Score": [avg_rouge]
#         })
#         summary_csv = "qwen2.5vl_instruct_qa_evaluation_summary_only.csv"
#         summary_df.to_csv(summary_csv, index=False)
#         print(f"✅ Evaluation completed! Summary saved to {summary_csv}")
#     else:
#         print("No valid scores to average.")

# if __name__ == "__main__":
#     evaluate_qa()



import os
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func  # new import for semantic evaluation


os.environ['HF_TOKEN'] = "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"


device_id = 0  # adjust if needed
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
        # Prepare input text using the chat template
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
