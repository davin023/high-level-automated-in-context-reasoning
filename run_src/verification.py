import os
import sys
sys.path.append(".")

import time
import math
import json
import random
import numpy as np
from tqdm import trange
from itertools import groupby
import argparse
from copy import deepcopy

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

try:
    from rapidfuzz import fuzz, process
except:
    pass

from eval_src.Evaluator import *
import ast
from collections import Counter


class AnswerExtraction:
    def __init__(self, k, root_dir, file_dir, evaluator, type, score_type, device, dataset_name):
        self.root_dir = root_dir
        self.file_dir = file_dir
        self.evaluator = evaluator
        self.dataset_name = dataset_name
        self.num = int(re.findall(r'\d+', root_dir)[0])
        if "SVAMP" in root_dir:
            self.num = 300
        elif "MATH" in root_dir:
            self.num = 500
        elif "GSM8K" in root_dir:
            self.num = 1319
        elif "STG" in root_dir:
            self.num = 687
        elif "AMC" in root_dir:
            self.num = 40
        elif "GPQA" in root_dir:
            self.num = 198
        self.k = k
        self.type = type
        self.score_type = score_type
        self.device = device
        self.output_dir = f"./run_outputs/structure/{self.root_dir}/"
        self.train_path_solutions_dir = os.path.join(self.output_dir, "train_path_solutions.json")
        self.train_path_questions_dir = os.path.join(self.output_dir, "train_path_questions.json")
        
        self.store = {}
        self.path_question = {}
    
       
    def process_sentence(self, a):
        a = re.sub(r"^Let's think step by step\.?", "", a).strip()
        
        answer_match = re.search(r"The answer is:?.*", a)
        if answer_match:
            answer = answer_match.group(0)
            a = a.replace(answer, "").strip()  
        else:
            answer = ""

        sentences = re.split(r'(?<=\.)\s+', a)  

        steps = []
        for i, sentence in enumerate(sentences, 1):
            steps.append(f"Step {i}: {sentence.strip()}")

        result = " ки\n".join(steps) + " ки"  
        if answer:
            result += f" {answer} ки"

        return result
    
    def obtain_model_completions(self, answers):
        all_model_completions = answers.get("all_model_completions", {})

        model_solutions = []
        for i in range(1, len(all_model_completions) + 1):
            rollout_key = f"rollout_{i}"
            rollout_value = all_model_completions.get(rollout_key, {})
            model_solution = rollout_value.get("model_solution")
            if model_solution:
                model_solutions.append(model_solution)
        
        return model_solutions
      
    def calculate_score(self, scores, score_type, input_id, step_tag_id):
        """Calculates the final score based on the selected score_type."""
        step_scores = scores[input_id == step_tag_id]
        
        if score_type == 'average':
            return step_scores.mean().to('cpu', dtype=torch.float32).item()  
        elif score_type == 'product':
            return step_scores.prod().to('cpu', dtype=torch.float32).item()  
        elif score_type == 'max':
            return step_scores.max().to('cpu', dtype=torch.float32).item()  
        elif score_type == 'min':
            return step_scores.min().to('cpu', dtype=torch.float32).item()  
        else:
            return step_scores.to('cpu', dtype=torch.float32).item()  
    
    def get_most_common_answer(self, scores_list, n):
        sorted_by_score = sorted(scores_list.items(), key=lambda x: x[1], reverse=True)
        top_scores = [item[0] for item in sorted_by_score[:n]]
        top_answers = [self.evaluator.extract_answer_from_model_completion(x) for x in top_scores]
        most_common = Counter(top_answers).most_common(2)
        
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # If there is a tie, consider the next n+1 items
            extended_scores = [item[0] for item in sorted_by_score[:n+1]]
            extended_answers = [self.evaluator.extract_answer_from_model_completion(x) for x in extended_scores]
            return Counter(extended_answers).most_common(1)[0][0]
        else:
            return most_common[0][0]
    
    
    def extract_answer(self):
        self.correct = 0
        self.correct_con = 0
        self.correct_min = 0
        self.correct_score_half = 0
        self.correct_score_half_min = 0
        self.correct_score_3 = 0
        self.correct_score_3_min = 0
        self.correct_score_5 = 0
        self.correct_score_5_min = 0
        self.correct_unlimited = 0
        self.total_correct = 0
        
        if self.type == "math-shepherd-mistral-7b-prm":
            good_token, bad_token = '+', '-'
            step_tag = 'ки'
            path = 'peiyi9979/math-shepherd-mistral-7b-prm'
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path).to(self.device).eval()
            candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]
            step_tag_id = tokenizer.encode(f"{step_tag}")[-1]
        
        elif self.type == "llama3.1-8b-prm-mistral-data" or self.type == "llama3.1-8b-orm-mistral-data":
            path = "RLHFlow/Llama3.1-8B-PRM-Mistral-Data" if "prm" in self.type else "RLHFlow/Llama3.1-8B-ORM-Mistral-Data"
            model = AutoModelForCausalLM.from_pretrained(path).to(self.device).eval()
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)  # 
            tokenizer.padding_side = "right"
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            plus_tag_id = tokenizer.encode('+')[-1]
            minus_tag_id = tokenizer.encode('-')[-1]
            candidate_tokens = [plus_tag_id,minus_tag_id]

        def load_dataset_mapping(file_path):
            """Load dataset mapping from a JSON file."""
            with open(file_path, "r") as f:
                return {item['id']: item for item in json.load(f)}

        if "MATH" in dataset_name:
            acc_subject = {
                "algebra": 0, "prealgebra": 0, "number theory": 0,
                "counting & probability": 0, "geometry": 0,
                "intermediate algebra": 0, "precalculus": 0
            }
            acc_level = {str(i): 0 for i in range(1, 6)}
            acc_num = {**acc_subject, **acc_level}
            id_map = load_dataset_mapping("./data/MATH/test_all.json")

        elif "GPQA" in dataset_name:
            acc_subjects = [
                "physics (general)", "chemistry (general)", "organic chemistry", "inorganic chemistry",
                "quantum mechanics", "electromagnetism and photonics", "high-energy particle physics",
                "genetics", "astrophysics", "molecular biology", "relativistic mechanics", 
                "optics and acoustics", "condensed matter physics"
            ]
            acc_subject = {subject: 0 for subject in acc_subjects}
            acc_num = acc_subject.copy()
            with open("./data/GPQA/test_all.json", "r") as f:
                raw_data_items = json.load(f)
            id_map = {k: item for k, item in enumerate(raw_data_items)}


        for i in range(self.num):
            print(i)
            self.total_correct += 1
            index = f"{i:04d}"
            answer_file = os.path.join(self.file_dir, f"answer_sheets/Question {index} - Answer.json")
            solution_file = os.path.join(self.file_dir, f"answer_sheets/Question {index} - Final Solutions.json")
                
            try:
                with open(answer_file, "r") as f:
                    answers = json.load(f)
                with open(solution_file, "r") as f:
                    solutions = json.load(f)
            except FileNotFoundError:
                continue
            
            # Extract problem details
            id = answers["id"]
            question = answers["problem"]
            gt_answer = answers["gold_answer"]

            # Extract model answers
            try:
                model_all_answers = answers.get("model_all_answer", [])
                if not model_all_answers:
                    model_all_answers = [
                        solution["trace"]["0"]["chain_of_thought"]["text"] 
                        if "chain_of_thought" in solution["trace"]["0"] 
                        else " ".join([str(s[1]) for s in solution["trace"]["0"]["answers"][1:]])
                        for solution in solutions
                    ]
            except Exception:
                model_all_answers = []
            
            
            # Filter and process model answers
            model_not_filtered_answers = [self.evaluator.extract_answer_from_model_completion(x) for x in model_all_answers]
            model_answers = [item for item in model_not_filtered_answers if item is not None]
            model_not_filtered_solutions = model_all_answers
            model_solutions = [item for item in model_not_filtered_solutions if item is not None]
            
            if "MATH" in self.dataset_name:
                item = id_map[id]
                level = item["extra_info"]["level"]
                subject = item["extra_info"]["subject"]
                acc_num[str(level)] += 1
                acc_num[subject.lower()] += 1
            elif "GPQA" in self.dataset_name:
                item = id_map[i]
                subject = item["subdomain"]
                acc_num[subject.lower()] += 1
            
            if self.type == "consistency":
                most_common_answer = Counter(model_answers).most_common(1)[0][0]
            elif "prm" in type.lower() or "orm" in type.lower():
                best_score, best_answer = None, None
                best_score_min, best_answer_min = None, None
                step_ans, scores_list, scores_list_min = {}, {}, {}
                for j, ans in enumerate(model_solutions):
                    try:
                        if len(self.evaluator.extract_answer_from_model_completion(ans)) > 100 or self.evaluator.extract_answer_from_model_completion(ans) == "unknown":
                            continue
                    except:
                        continue
                    try:
                        x = self.process_sentence(ans)
                    except:
                        continue
                    
                    step_ans[x] = ans
                    if self.type == "math-shepherd-mistral-7b-prm":
                        input_for_prm = f"{question} {x}"
                        input_id = torch.tensor([tokenizer.encode(input_for_prm)], device=device)  # , device=device
                        with torch.no_grad():
                            logits = model(input_id).logits[:,:,candidate_tokens]
                            scores = logits.softmax(dim=-1)[:,:,0] 
                            
                            result = self.calculate_score(scores, "product", input_id, step_tag_id)
                            result_min = self.calculate_score(scores, "min", input_id, step_tag_id)
                    
                    elif self.type == "llama3.1-8b-orm-mistral-data":
                        conversation = []
                        single_step_score = []
                        if "mistral" in self.type:
                            processed_ans = ans.replace(" ки","")
                            conversation.append({"content": question + " " + processed_ans,"role":"user"})
                        else:
                            conversation.append({"content": question + " " + ans,"role":"user"})
                        conversation.append({"content":"+","role":"assistant"})
                        input_ids = tokenizer.apply_chat_template(conversation,return_tensors="pt").to(device)
                        with torch.no_grad():
                            logits = model(input_ids).logits[:,-3,candidate_tokens] #simple version for llama3.1-instruct, the +/- is predicted by the '-3' position
                            scores = logits.softmax(dim=-1)[:,0] # 0 means the prob of + (1 mean -)
                        single_step_score.append(scores[0].detach().to('cpu', dtype=torch.float32).item())
                        result = single_step_score[0]
                        result_min = single_step_score[0]
                    
                    elif self.type == "llama3.1-8b-prm-mistral-data":
                        conversation = []
                        single_step_score = []
                        if "mistral" in self.type:
                            ans_list = x.split("ки\n")
                        else:
                            ans_list = x.split("\n\n")
                        ans_list = [p.strip("\n").strip() for p in ans_list if p != ""]
                        
                        for k in range(len(ans_list)):
                            if k == 0:
                                text = question + " " + ans_list[0]
                            else:
                                text = ans_list[k]
                            conversation.append({"content": text, "role": "user"})
                            conversation.append({"content": "+", "role": "assistant"})

                            input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
                            with torch.no_grad():
                                logits = model(input_ids).logits[:, -3, candidate_tokens]
                                scores = logits.softmax(dim=-1)[:, 0]
                                single_step_score.append(scores[0].detach().to('cpu', dtype=torch.float32).item())
                            
                        # result = math.prod(single_step_score)
                        result = sum(single_step_score)/len(single_step_score)
                        result_min = min(single_step_score)
                        
                    scores_list[ans] = result
                    scores_list_min[ans] = result_min
                    
                    if best_score is None or result > best_score:
                        best_score = result
                        best_answer = step_ans[x]
                    
                    if best_score_min is None or result_min > best_score_min:
                        best_score_min = result_min
                        best_answer_min = step_ans[x]
                
                most_consistency_answer = Counter(model_answers).most_common(1)[0][0]
                most_common_answer = self.evaluator.extract_answer_from_model_completion(best_answer)
                most_common_answer_min = self.evaluator.extract_answer_from_model_completion(best_answer_min)
                
                if len(list(scores_list.keys())) == 0:
                    continue
                
                elif len(list(scores_list.keys())) == 1:
                    assert len(list(scores_list_min.keys())) == 1
                    most_common_solution_half_con = list(scores_list.keys())[0]
                    most_common_solution_half_con_min = list(scores_list_min.keys())[0]
                    most_common_answer_half_con = self.evaluator.extract_answer_from_model_completion(most_common_solution_half_con)
                    most_common_answer_half_con_min = self.evaluator.extract_answer_from_model_completion(most_common_solution_half_con_min)
                    
                    most_common_answer_3_con = most_common_answer_half_con
                    most_common_answer_3_con_min = most_common_answer_half_con_min
                    
                    most_common_answer_5_con = most_common_answer_half_con
                    most_common_answer_5_con_min = most_common_answer_half_con_min
                else:
                    # For the first group of scores
                    most_common_answer_3_con = self.get_most_common_answer(scores_list, 3)
                    most_common_answer_5_con = self.get_most_common_answer(scores_list, 5)
                    half_len = len(sorted(scores_list.keys())) // 2
                    most_common_answer_half_con = self.get_most_common_answer(scores_list, half_len)

                    # For the second group of scores
                    most_common_answer_3_con_min = self.get_most_common_answer(scores_list_min, 3)
                    most_common_answer_5_con_min = self.get_most_common_answer(scores_list_min, 5)
                    half_min_len = len(sorted(scores_list_min.keys())) // 2
                    most_common_answer_half_con_min = self.get_most_common_answer(scores_list_min, half_min_len)
            
            # if i in []:
            #     print(i)
            if self.type == "consistency":
                if not self.evaluator.check_answers_equiv(most_common_answer, gt_answer):
                    try:
                        sorted_scores = dict(sorted(scores_list.items(), key=lambda x: x[1], reverse=True))
                        print(f"==> Question: {question}")
                        print(f"==> gt_answer: {gt_answer}")
                        print(f"==> Most common answer: {most_common_answer}")
                        print(f"==> Model answer: {model_answers}")
                        print(f"==> Score_answer:")
                        for ans, score in sorted_scores.items():
                            final_ans = self.evaluator.extract_answer_from_model_completion(ans)
                            print(f"{score}: {final_ans}")
                    except:
                        a = 1
            else:
                if not self.evaluator.check_answers_equiv(most_common_answer_3_con, gt_answer):
                    try:
                        sorted_scores = dict(sorted(scores_list_min.items(), key=lambda x: x[1], reverse=True))
                        print(f"==> Question: {question}")
                        print(f"==> gt_answer: {gt_answer}")
                        print(f"==> Most common answer: {most_common_answer_3_con}")
                        print(f"==> Model answer: {model_answers}")
                        print(f"==> Score_answer:")
                        for ans, score in sorted_scores.items():
                            final_ans = self.evaluator.extract_answer_from_model_completion(ans)
                            print(f"{score}: {final_ans}")
                    except:
                        a = 1
                
                
            try:
                if self.evaluator.check_answers_equiv(most_common_answer, gt_answer):
                    self.correct += 1
                    if "MATH" in dataset_name:
                        acc_subject[subject.lower()] += 1
                        acc_level[str(level)] += 1
                    elif "GPQA" in dataset_name:
                        acc_subject[subject.lower()] += 1
                if self.evaluator.check_answers_equiv(most_consistency_answer, gt_answer):
                    self.correct_con += 1
                if self.evaluator.check_answers_equiv(most_common_answer_min, gt_answer):
                    self.correct_min += 1
                if self.evaluator.check_answers_equiv(most_common_answer_half_con, gt_answer):
                    self.correct_score_half += 1
                if self.evaluator.check_answers_equiv(most_common_answer_half_con_min, gt_answer):
                    self.correct_score_half_min += 1
                
                if self.evaluator.check_answers_equiv(most_common_answer_3_con, gt_answer):
                    self.correct_score_3 += 1
                if self.evaluator.check_answers_equiv(most_common_answer_3_con_min, gt_answer):
                    self.correct_score_3_min += 1
                
                if self.evaluator.check_answers_equiv(most_common_answer_5_con, gt_answer):
                    self.correct_score_5 += 1
                if self.evaluator.check_answers_equiv(most_common_answer_5_con_min, gt_answer):
                    self.correct_score_5_min += 1
            except:
                a = 1
            
            flag = 0
            for ans in model_answers:
                if self.evaluator.check_answers_equiv(ans, gt_answer):
                    self.correct_unlimited += 1
                    flag = 1
                    break
            
            if flag == 0:
                print(f"gt_answer: {gt_answer}")
                print(f"ans: {model_answers}")
                a = 1
        
        acc = float(self.correct)/float(self.total_correct)
        acc_unlimited = float(self.correct_unlimited)/float(self.total_correct)
        try:
            acc_con = float(self.correct_con)/float(self.total_correct)
            acc_min = float(self.correct_min)/float(self.total_correct)
            acc_score_half = float(self.correct_score_half)/float(self.total_correct)
            acc_score_half_min = float(self.correct_score_half_min)/float(self.total_correct)
            acc_score_3 = float(self.correct_score_3)/float(self.total_correct)
            acc_score_3_min = float(self.correct_score_3_min)/float(self.total_correct)
            acc_score_5 = float(self.correct_score_5)/float(self.total_correct)
            acc_score_5_min = float(self.correct_score_5_min)/float(self.total_correct)
        except:
            a = 1
        
        if "MATH" in self.dataset_name:
            for key in acc_subject.keys():
                acc_type = float(1.0*acc_subject[key]) / float(1.0*acc_num[key]) if acc_num[key] != 0 else 0.0
                print(f"{key}: {acc_type:.4f}")
            for key in acc_level.keys():
                acc_type = float(1.0*acc_level[str(key)]) / float(1.0*acc_num[str(key)]) if acc_num[key] != 0 else 0
                print(f"{key}: {acc_type:.4f}")
        elif "GPQA" in self.dataset_name:
            for key in acc_subject.keys():
                acc_type = float(1.0*acc_subject[key]) / float(1.0*acc_num[key]) if acc_num[key] != 0 else 0.0
                print(f"{key}: {acc_type:.4f}")
        
        print(f"============={type}=============")
        if "prm" in type.lower():
            if "llama" in self.type:
                print(f"Correct consistency: {acc_con:.4f}")
                # print(f"Correct_ave: {acc:.4f}")
                # print(f"Correct_min: {acc_min:.4f}")
                # print(f"Correct_ave_score_3: {acc_score_3:.4f}")
                print(f"Correct_min_score_3: {acc_score_3_min:.4f}")
                # print(f"Correct_ave_score_5: {acc_score_5:.4f}")
                # print(f"Correct_min_score_5: {acc_score_5_min:.4f}")
                # print(f"Correct_ave_score_half: {acc_score_half:.4f}")
                # print(f"Correct_min_score_half: {acc_score_half_min:.4f}")
                acc_dict = {
                    "Correct consistency": acc_con,
                    "Correct_ave": acc,
                    "Correct_min": acc_min,
                    "Correct_ave_score_3": acc_score_3,
                    "Correct_min_score_3": acc_score_3_min,
                    "Correct_ave_score_5": acc_score_5,
                    "Correct_min_score_5": acc_score_5_min,
                    "Correct_ave_score_half": acc_score_half,
                    "Correct_min_score_half": acc_score_half_min
                }
            else:
                print(f"Correct consistency: {acc_con:.4f}")
                # print(f"Correct_product: {acc:.4f}")
                # print(f"Correct_min: {acc_min:.4f}")
                # print(f"Correct_product_score_3: {acc_score_3:.4f}")
                print(f"Correct_min_score_3: {acc_score_3_min:.4f}")
                # print(f"Correct_product_score_5: {acc_score_5:.4f}")
                # print(f"Correct_min_score_5: {acc_score_5_min:.4f}")
                # print(f"Correct_product_score_half: {acc_score_half:.4f}")
                # print(f"Correct_min_score_half: {acc_score_half_min:.4f}")
                acc_dict = {
                    "Correct consistency": acc_con,
                    "Correct_product": acc,
                    "Correct_min": acc_min,
                    "Correct_product_score_3": acc_score_3,
                    "Correct_min_score_3": acc_score_3_min,
                    "Correct_product_score_5": acc_score_5,
                    "Correct_min_score_5": acc_score_5_min,
                    "Correct_product_score_half": acc_score_half,
                    "Correct_min_score_half": acc_score_half_min
                }
        elif "orm" in type.lower():
            print(f"Correct consistency: {acc_con:.4f}")
            # print(f"Correct_max: {acc:.4f}")
            print(f"Correct_max_score_3: {acc_score_3:.4f}")
            # print(f"Correct_max_score_5: {acc_score_5:.4f}")
            # print(f"Correct_max_score_half: {acc_score_half:.4f}")
            acc_dict = {
                "Correct consistency": acc_con,
                "Correct_max": acc,
                "Correct_max_score_3": acc_score_3,
                "Correct_max_score_5": acc_score_5,
                "Correct_max_score_half": acc_score_half
            }
        else:
            print(f"Correct: {acc:.4f}")
            acc_dict = {"Correct": acc}
        
        print(f"Correct Unlimited: {acc_unlimited:.4f}")
        acc_dict["Correct Unlimited"] = acc_unlimited
        
        path = os.path.join("./run_outputs", root_dir, f"{self.type}_acc.json")
        with open(path, "w") as f:
            json.dump(acc_dict, f)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    root_dir = f"Your directory path containing reasoning path for test questions"
    file_dir = os.path.join("./run_outputs", root_dir, "05-15_00-00")

    parser = argparse.ArgumentParser(description="verification")
    parser.add_argument('--type', type=str, default="llama3.1-8b-prm-mistral-data", choices=["consistency", "llama3.1-8b-orm-mistral-data", "llama3.1-8b-prm-mistral-data", "math-shepherd-mistral-7b-prm"], help="verification type")
    parser.add_argument('--score_type', type=str, default="product", choices=["product", "min", "average", "max"], help="prm solution score type")
    parser.add_argument('--k', type=int, default=0.95, help='balance factor')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = root_dir.split("/")[0].strip("")
    model_name = root_dir.split("/")[1].split("/")[0].strip()
    if "GSM" in dataset_name and "HARD" not in dataset_name:
        evaluator = eval("GSM8KEvaluator()")
    elif "MATH" in dataset_name:
        evaluator = eval("MATHEvaluator()")
    else:
        evaluator = eval(f"{dataset_name}Evaluator()")
    
    print("================Extracting Answers================")
    score_type = args.score_type
    type = args.type
    k = args.k
    answer_extraction = AnswerExtraction(k, root_dir, file_dir, evaluator, type, score_type, device, dataset_name)
    answer_extraction.extract_answer()
    print("================Original================")
    with open(os.path.join(file_dir, "final_result.txt"), "r") as f:
        print(f.read())
    print(f"dataset_name: {dataset_name} | model_name: {model_name} | type: {type}")
    print(root_dir)