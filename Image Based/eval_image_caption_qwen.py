import os
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import sacrebleu
from rouge_score import rouge_scorer

os.environ['HF_TOKEN'] = "hf_token"

device_id = 0  
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

print("Loading Qwen2.5-VL-7B-Instruct Model for image caption evaluation...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map={"": device_id}
)
print(f"Model is loaded on: {next(model.parameters()).device}")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

syst_prompt = (
    "You are an expert in telecommunications and 3GPP technical documents. "
    "Provide a concise, technically accurate caption for the given image that highlights its relevance to 3GPP standards."
)

def generate_caption(image_path):
    """
    Generate a caption for a given image using the Qwen model.
    """
    try:
      
        prompt = "Generate a technical caption for this image."
        messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": syst_prompt}]},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": Image.open(image_path)}
                ]}
            ]
        ]
     
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=text, images=image_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(device)
        

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
     
        torch.cuda.empty_cache()
        return output_text[0].strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        torch.cuda.empty_cache()
        return "ERROR"

def compute_similarity_metrics(pred, actual):
    """
    Compute sacreBLEU and ROUGE-L scores between the generated caption and the ground truth caption.
    """
    bleu_result = sacrebleu.sentence_bleu(pred, [actual])
    bleu_score = bleu_result.score  # usually a percentage value
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_score = rouge.score(actual, pred)["rougeL"].fmeasure
    return bleu_score, rouge_score

def evaluate_captions():
    print("Loading caption evaluation dataset...")
   
    dataset_path = "/DATAY/telecom/ImageRetrieval/benchmark_image_caption.csv"
    df = pd.read_csv(dataset_path) 
    
  
    required_columns = ["Image Path", "Image Caption"]
    df = df.dropna(subset=required_columns)
    
    bleu_scores = []
    rouge_scores = []
    results = [] 
    
    print(f"Starting evaluation on {len(df)} entries...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        attempt = 0
        success = False
        while attempt < 3 and not success:
            try:
                image_path = row["Image Path"]
                ground_truth_caption = row["Image Caption"]
                
                generated_caption = generate_caption(image_path)
                if generated_caption == "ERROR":
                    raise Exception("Generation error")
                
                bleu, rouge_metric = compute_similarity_metrics(generated_caption, ground_truth_caption)
                bleu_scores.append(bleu)
                rouge_scores.append(rouge_metric)
                
                result = row.to_dict()
                result.update({
                    "Generated Caption": generated_caption,
                    "sacreBLEU Score": bleu,
                    "ROUGE Score": rouge_metric
                })
                results.append(result)
                
                success = True
            except Exception as e:
                attempt += 1
                print(f"Error on attempt {attempt} for index {idx}: {e}")
                if attempt == 3:
                    print(f"Failed after 3 attempts for index {idx}. Skipping this entry.")
    
        torch.cuda.empty_cache()
    
    if results:
        results_df = pd.DataFrame(results)
        results_csv = "qwen2.5vl_instruct_caption_evaluation_results.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"Individual results saved to {results_csv}")
        
        if bleu_scores and rouge_scores:
            avg_bleu = sum(bleu_scores) / len(bleu_scores)
            avg_rouge = sum(rouge_scores) / len(rouge_scores)
            
            print(f"Average sacreBLEU Score: {avg_bleu}")
            print(f"Average ROUGE Score: {avg_rouge}")
            
            summary_df = pd.DataFrame({
                "Average sacreBLEU Score": [avg_bleu],
                "Average ROUGE Score": [avg_rouge]
            })
            summary_csv = "qwen2.5vl_instruct_caption_evaluation_summary.csv"
            summary_df.to_csv(summary_csv, index=False)
            print(f"Summary saved to {summary_csv}")
    else:
        print("No valid results to save.")

if __name__ == "__main__":
    evaluate_captions()
