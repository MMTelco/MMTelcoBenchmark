import os 
import pandas as pd
import json
from tqdm import tqdm


syst_prompt = """
Please provide the correct answer to the following telecommunications-related multiple-choice question. The answers must be in a JSON format as follows:

Outpur format Json : 


{
"correct_answer":"<answer>"
"correct_option": "<answer id>"

}

Input Example :

        Question id : 2
        Question: What is the typical end-to-end latency in process automation applications?
        Option 1: 10 ms
        Option 2: 50 ms
        Option 3: 100 ms
        Option 4: 500 ms


Output Example :
        { 
       "correct_answer":"50 ms",
        "correct_option": "Option 2" ,
        }


Make sure to use the format above for the answer the given question and include the correct option as 'correct_option' and answer as 'correct_answer' .
"""

#load the questions
questions = pd.read_csv("Evaluate/MCQ_TelcoBench_V2_10k.csv")

def evaluate(model):
    print(f"Evaluating model: {model}")

    save_path_csv = os.path.join(model + "_MCQ_answers_V2_10k.csv")  

    # for open a models replace with ChatOpenAI


    llm = ###
    
    if os.path.exists(save_path_csv):
        responses = pd.read_csv(save_path_csv)
        start = len(responses)
        
    else:
        start = 1
    print(f"Starting from {start} ....")

    for idx, row in tqdm(questions.iloc[start-1:].iterrows()):
        attempt = 0
        success = False

        while attempt < 3 and not success:
            try:
                user_prompt = f"""
                Question id {idx}\n
                Question: {row["Question"]}\n
                Option 1: {row["Option 1"]}\n
                Option 2: {row["Option 2"]}\n
                Option 3: {row["Option 3"]}\n
                Option 4: {row["Option 4"]}\n\n
                """

                # Get the correct answer and the correct option
                correct_answer = row[f"{row['Answer']}"]
                correct_option = row["Answer"]

                # Create the message list to pass into the LLM
                messages = [
                    ("system", syst_prompt),
                    ("human", user_prompt),
                ]

                # Invoke the LLM to get the answer in JSON format
                predicted_answer = llm.invoke(messages).content

                # Convert the predicted answer string into JSON
                predicted_json = json.loads(predicted_answer)

                # Extract the predicted answer and option from the JSON
                llm_answer = predicted_json["correct_answer"]
                llm_answer_option = predicted_json["correct_option"]

                # Check if the LLM's predicted answer matches the correct answer
                if llm_answer.lower() == correct_answer.lower() :
                    answered_status = "Right"
                elif llm_answer_option == correct_option:
                    answered_status = "Right"

                else:
                    answered_status = "Wrong"

                # Prepare the output to be saved
                output = {
                    "Question id": idx,
                    "Question": row["Question"],
                    "Option 1": row["Option 1"],
                    "Option 2": row["Option 2"],
                    "Option 3": row["Option 3"],
                    "Option 4": row["Option 4"],
                    "Correct Answer": correct_answer,
                    "Correct Option": correct_option,
                    "Generated Option":  llm_answer_option ,
                    "Generated Answer": llm_answer,
                    "Truth": answered_status,
                    "Working Group": row["Working Group"],
                    "TS Document": row["Source"],
                    "3GPP Series": row["Series"],
                }

                # Convert to DataFrame and append to CSV
                output_df = pd.DataFrame([output])
                output_df.to_csv(save_path_csv, mode='a', header=not pd.io.common.file_exists(save_path_csv), index=False)

                # If successful, exit the retry loop
                success = True
            except Exception as e:
                attempt += 1
                print(f"Error type: {type(e).__name__} on attempt {attempt}")
                print(f"Error message: {e}")
                if attempt == 3:
                    print(f"Failed after 3 attempts for Question id {idx}. Skipping to next.")



if __name__ == "__main__":
    models = [""]
    for model in models:
        evaluate(model)


       
