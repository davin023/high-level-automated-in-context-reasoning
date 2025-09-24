import os
import sys
sys.path.append(".")

import math
import json
import wandb
import random
import numpy as np
from tqdm import trange
from itertools import groupby
from typing import List, Dict, Tuple
from copy import deepcopy
from common.utils import fix_seeds
from FlagEmbedding import BGEM3FlagModel

try:
    from rapidfuzz import fuzz, process
except:
    pass

from models.IO_System import IO_System
from common.utils import read_txt, read_json, save_json
from eval_src.Evaluator import *
from common.arguments import get_parser
import ast


class ModelLoader:
    def __init__(self, args):
        self.args = args
        self.tokenizer, self.model = self.load_model()

    def load_model(self):
        if self.args.api == "huggingface":
            from models.HuggingFace_API import load_HF_model
            tokenizer, model = load_HF_model(self.args.model_ckpt)
        elif self.args.api == "vllm":
            from models.vLLM_API import load_vLLM_model
            tokenizer, model = load_vLLM_model(self.args.model_ckpt, self.args.seed, self.args.tensor_parallel_size, self.args.half_precision)
        elif self.args.api == "gpt3.5-turbo":
            from models.OpenAI_API import load_OpenAI_model
            tokenizer, model = load_OpenAI_model(self.args.model_ckpt)
        else:
            raise ValueError("Unsupported API type")
        
        return tokenizer, model


class PathExtractor:
    def __init__(self, k, structure_dir, root_dir, file_dir, evaluator):
        self.root_dir = root_dir
        self.file_dir = file_dir
        self.evaluator = evaluator
        self.num = int(re.findall(r'\d+', root_dir)[0])
        self.num = 200
        self.k = k
        self.output_dir = f"./run_outputs/{structure_dir}/{self.root_dir}/"
        self.train_path_solutions_dir = os.path.join(self.output_dir, "train_path_solutions.json")
        self.train_path_questions_dir = os.path.join(self.output_dir, "train_path_questions.json")
        
        self.store = {}
        self.path_question = {}
    
    def extract_path(self):
        self.correct = 0
        self.total_correct = 0
        for i in range(self.num):
            self.total_correct += 1
            if i < 10:
                index = f"000{i}"
            elif i < 100:
                index = f"00{i}"
            elif i < 1000:
                index = f"0{i}"
            answer_file = os.path.join(self.file_dir, f"answer_sheets/Question {index} - Answer.json")
            solution_file = os.path.join(self.file_dir, f"answer_sheets/Question {index} - Final Solutions.json")
            
            with open(answer_file, "r") as f:
                answers = json.load(f)

            try:
                with open(solution_file, "r") as f:
                    solutions = json.load(f)
                    sorted_solutions = self.sort_solutions(solutions)

                    selected_solution = self.find_valid_solution(sorted_solutions, answers)
                    if not selected_solution:
                        # print(i)
                        continue

                    select_trace = selected_solution["trace"]["0"]
                    confidence_flag, leaf_confidence = self.get_leaf_confidence(selected_solution)
                    path = tuple([x[0] for x in select_trace["path"]])

                    if path not in self.store:
                        self.path_question[path] = [answers["problem"]]
                        self.store[path] = [{
                            "id": 0,
                            "question": answers["problem"],
                            "gold_solution": answers["gold_solution"],
                            "gold_answer": answers["gold_answer"],
                            "model_solution": "Question: "+select_trace['answers'][0][1]+"\nAnswer:\n" + "\n".join([f"Step {j+1}: ({path[j+1].lower()}) "+x[1] for j, x in enumerate(select_trace['answers'][1:])]),
                            "model_answer": selected_solution["model_answer"],
                            "leaf_confidence": leaf_confidence
                        }]
                    else:
                        self.path_question[path].append(answers["problem"])
                        path_exist_len = len(self.store[path])
                        self.store[path].append({
                            "id": path_exist_len,
                            "question": answers["problem"],
                            "gold_solution": answers["gold_solution"],
                            "gold_answer": answers["gold_answer"],
                            "model_solution": "Question: "+select_trace['answers'][0][1]+"\nAnswer:\n" + "\n".join([f"Step {j+1}: ({path[j+1].lower()}) "+x[1] for j, x in enumerate(select_trace['answers'][1:])]),
                            "model_answer": selected_solution["model_answer"],
                            "leaf_confidence": leaf_confidence
                        })
            
            except:
                continue
            
        # acc = float(self.correct)/float(self.total_correct)
        # print(f"Correct: {acc:.2f}")
        
        self.store_str_keys = {str(key): value for key, value in self.store.items()}
        self.path_question_str_keys = {str(key): value for key, value in self.path_question.items()}
        if not os.path.exists(self.train_path_solutions_dir) and not os.path.exists(self.train_path_questions_dir):
            self.save_results()

    def sort_solutions(self, solutions):
        for x in solutions:
            if x["rollout_id"] is None:
                x["rollout_id"] = 0
        sorted_solutions = sorted(solutions, key=lambda x: x["rollout_id"], reverse=True)
        grouped_solutions = [
            list(group) for _, group in groupby(
                sorted_solutions,
                key=lambda x: x["rollout_id"]
            )
        ]

        final_sorted_solutions = []
        for group in grouped_solutions:
            if len(group) > 1:
                for x in group:
                    try:
                        x['value'] = x["trace"]["0"]["chain_of_thought"]["value"]
                    except:
                        len_keys = len(list(x["trace"].keys()))
                        try:
                            x['value'] = x["trace"][f'{len_keys-1}']["chain_of_thought"]["value"]
                        except:
                            try:
                                x['value'] = x["trace"][f'{len_keys-1}']['subanswer']['value']
                            except:
                                x['value'] = 0
                try:
                    group = sorted(
                        group,
                        key=lambda x: (
                            len(x["trace"]["0"]["path"])*self.k
                            -x['value']*(1-self.k)
                        )
                    )
                except:
                    group = sorted(
                        group,
                        key=lambda x: (
                            len(x["trace"]["0"]["path"])
                        )
                    )
            final_sorted_solutions.extend(group)

        return final_sorted_solutions

    def find_valid_solution(self, sorted_solutions, answers):
        ans = self.evaluator.extract_answer_from_model_completion(sorted_solutions[0]["trace"]["0"]['answers'][-1][-1])
        self.correct += int(self.evaluator.check_answers_equiv(ans, answers["gold_answer"]))
        for solution in sorted_solutions:
            select_trace = solution["trace"]["0"]
            model_answer = self.evaluator.extract_answer_from_model_completion(select_trace['answers'][-1][-1])
            if self.evaluator.check_answers_equiv(model_answer, answers["gold_answer"]):
                solution["model_answer"] = model_answer
                return solution
        return None
    
    def get_leaf_confidence(self, solution):
        confidence_flag = 0
        temp_trace = solution["trace"]
        select_trace = temp_trace["0"]
        try:
            leaf_confidence = select_trace["chain_of_thought"]["value"]
            confidence_flag = 1
        except:
            for j in range(1, len(list(temp_trace.keys()))):
                try:
                    leaf_confidence = temp_trace[str(j)]["chain_of_thought"]["value"]
                    confidence_flag = 1
                    break
                except:
                    continue
        if confidence_flag == 0:
            leaf_confidence = 0
        return confidence_flag, leaf_confidence
    
    def save_results(self):
        # store_str_keys = {str(key): value for key, value in self.store.items()}
        # path_question_str_keys = {str(key): value for key, value in self.path_question.items()}
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self.train_path_solutions_dir, "w") as f:
            json.dump(self.store_str_keys, f)
        with open(self.train_path_questions_dir, "w") as f:
            json.dump(self.path_question_str_keys, f)


class RephrasingHandler:
    def __init__(self, rephrasing_prompt_template: str, io_system: IO_System):
        self.rephrasing_prompt_template = rephrasing_prompt_template
        self.io = io_system
    
    def generate_sub_questions(self, questions: list):
        io_inputs = []
        for question in questions:
            io_input = self.rephrasing_prompt_template
            io_input += "\n\n"
            io_input += f"Question 6: {question}\nQuestion 6.1: "
            io_inputs.append(io_input)
        
        io_outputs = self.io.generate(model_input=io_inputs, max_tokens=512, num_return=1, stop_tokens=["Done", "\n\n"])
        results = []
        for io_output in io_outputs:
            # new added
            io_output = io_output[0].split("Done")[0]
            count_subquestion = io_output.count("Answer 6")
            
            results.append((io_output, count_subquestion))
        
        return results, count_subquestion
    
    def generate_rephrased_question(self, questions: list):
        io_inputs = []
        for question in questions:
            io_input = self.rephrasing_prompt_template
            io_input += "\n\n"
            io_input += "Original Question: " + question + "\n"
            io_input += "Rephrased Question: Given a list of conditions, please answer the question. Condition 1: "
            io_inputs.append(io_input)
        
        io_outputs = self.io.generate(model_input=io_inputs, max_tokens=512, num_return=1, stop_tokens=["\n", "\n\n"])
        results = []
        for io_output in io_outputs:
            # new added
            io_output = io_output[0].split("?")[0] + "?"
            count_conditions = io_output.count("Condition") + 1
            
            results.append((io_output, count_conditions))
        
        return results, count_conditions


class ProblemDecomposer:
    def __init__(self, model_loader: ModelLoader, rephrasing_prompt_template: str, structure_dir: str):
        self.tokenizer = model_loader.tokenizer
        self.model = model_loader.model
        self.io = IO_System(model_loader.args, self.tokenizer, self.model)
        self.rephrasing_handler = RephrasingHandler(rephrasing_prompt_template, self.io)
        self.structure_dir = structure_dir

    def decompose_problems(self, train_path_questions, type):  ### type: condition / subquestion
        train_decompose_list = {}
        train_difficulty_list = {}
        
        for p, q_list in train_path_questions.items():
            if type == "subquestion":
                io_outputs, _ = self.rephrasing_handler.generate_sub_questions(q_list)
            elif type == "condition":
                io_outputs, _ = self.rephrasing_handler.generate_rephrased_question(q_list)
            
            for i, (io_output, count_conditions) in enumerate(io_outputs):
                if i == 0:
                    train_decompose_list[p] = [{
                        "question": q_list[i],
                        "count_conditions": count_conditions,
                        "prompt": "Condition 1: " + io_output
                    }]
                    train_difficulty_list[p] = [count_conditions]
                else:
                    train_decompose_list[p].append({
                        "question": q_list[i],
                        "count_conditions": count_conditions,
                        "prompt": "Condition 1: " + io_output
                    })
                    train_difficulty_list[p].append(count_conditions)

        return train_decompose_list, train_difficulty_list


class ProblemDecompositionManager:  
    def __init__(self, k, structure_dir, dataset_name, root_dir, file_dir, args, evaluator, attribute_type):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.structure = structure_dir
        self.structure_dir = f"./run_outputs/{structure_dir}/{self.root_dir}/"
        self.args = args
        self.k = k
        self.file_dir = file_dir
        self.evaluator = evaluator
        self.attribute_type = attribute_type
        self.train_decompose_dir = os.path.join(self.structure_dir, f"{attribute_type}_train_decompose.json")
        self.train_path_difficulty_list_dir = os.path.join(self.structure_dir, f"{attribute_type}_train_path_difficulty_list.json")
        self.train_path_difficulty_count_dir = os.path.join(self.structure_dir, f"{attribute_type}_train_path_difficulty_count.json")
        self.store_path_difficulty_dir = f"./run_outputs/{structure_dir}/{attribute_type}_|_{dataset_name}_|_{model_name}_path_difficulty_count.json"
        self.test_set_path_dir = "./data/MATH/test_all.json"
        
        self.load_data()
        if attribute_type != "semantic":
            self.model_loader = ModelLoader(self.args)
            self.extractor = PathExtractor(self.k, structure_dir, root_dir, file_dir, evaluator)
            self.extractor.extract_path()
        self.train_path_solutions = self.extractor.store_str_keys
        self.train_path_questions = self.extractor.path_question_str_keys
    
        self.decomposer = ProblemDecomposer(self.model_loader, self.rephrasing_prompt_template, self.structure_dir)
        self.sim_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def load_data(self):
        if "GSM" in self.dataset_name and "HARD" not in self.dataset_name:
            dataset_name = "GSM8K"
        elif "MATH" in self.dataset_name:
            dataset_name = "MATH"
        else:
            dataset_name = self.dataset_name
        self.datasets = dataset_name
        rephrasing_prompt_template_path = os.path.join(f"./prompts/{dataset_name}", "list_conditions_prompt_template.txt")
        subquestion_prompt_template_path = os.path.join(f"./prompts/{dataset_name}", "list_subquestions_prompt_template.txt")
        if not os.path.exists(self.structure_dir):
            os.makedirs(self.structure_dir)
        
        if self.attribute_type == "condition":
            self.rephrasing_prompt_template = read_txt(rephrasing_prompt_template_path)
        elif self.attribute_type == "subquestion":
            self.rephrasing_prompt_template = read_txt(subquestion_prompt_template_path)
        elif self.attribute_type == "semantic":  
            self.rephrasing_prompt_template = read_txt(subquestion_prompt_template_path)
        elif self.attribute_type == "confidence":  
            self.rephrasing_prompt_template = read_txt(subquestion_prompt_template_path)

        # train_path_solutions_dir = os.path.join(self.structure_dir, "train_path_solutions.json")
        # train_path_questions_dir = os.path.join(self.structure_dir, "train_path_questions.json")
        # self.train_path_solutions = read_json(train_path_solutions_dir)
        # self.train_path_questions = read_json(train_path_questions_dir)
        
        self.testset = read_json(self.test_set_path_dir)
        
    def decompose(self):
        self.train_decompose_dict, self.train_difficulty_list_dict = self.decomposer.decompose_problems(self.train_path_questions, self.attribute_type)
        sorted_train_difficulty_list = dict(
            sorted(self.train_difficulty_list_dict.items(), key=lambda x: sum(x[1]) / len(x[1]))
        )
        
        self.train_path_difficulty_count_dict = {}
        for k, v in sorted_train_difficulty_list.items():
            self.train_path_difficulty_count_dict[k] = sum(v) / len(v)
            print(f"{k}: {sum(v) / len(v):.2f}")
        
        # self.train_path_difficulty_count_dict["('USER_QUESTION', 'CHAIN_OF_THOUGHT')"] = float(round(self.train_path_difficulty_count_dict["('USER_QUESTION', 'CHAIN_OF_THOUGHT')"]))
        self.save_results()
        
    def save_results(self):
        if not os.path.exists(self.train_decompose_dir):
            save_json(self.train_decompose_dict, self.train_decompose_dir)
        if not os.path.exists(self.train_path_difficulty_list_dir):
            save_json(self.train_difficulty_list_dict, self.train_path_difficulty_list_dir)
        if not os.path.exists(self.train_path_difficulty_count_dir):
            save_json(self.train_path_difficulty_count_dict, self.train_path_difficulty_count_dir)
        if not os.path.exists(self.store_path_difficulty_dir):
            save_json(self.train_path_difficulty_count_dict, self.store_path_difficulty_dir)
    
    def find_nearest_train_path(self, count_conditions, k):
        # Find k paths with the minimum distance to count_conditions
        distances = []
        for key_str in self.train_path_difficulty_count.keys():
            key_tuple = ast.literal_eval(key_str)  
            distance = abs(len(key_tuple) - count_conditions)
            distances.append((distance, key_str))

        # Sort distances and take the nearest k
        distances.sort(key=lambda x: x[0])
        nearest_keys_strs = [key_str for _, key_str in distances[:k]]

        # Convert the nearest keys (string representation) to lists
        nearest_keys_lists = [list(ast.literal_eval(key_str)) for key_str in nearest_keys_strs]
        return nearest_keys_lists    
        
    def select_questions(self, train_path_questions, num_questions=20):
            selected_questions = {}
            for path, questions in train_path_questions.items():
                if len(questions) >= num_questions:
                    selected_questions[path] = random.sample(questions, num_questions)
                else:
                    selected_questions[path] = questions
                    
            return selected_questions
    
    def obtain_path(self):
        k = 5
        self.sim_type = "average"  # average / max
        test_question_paths = {}
        test_set_name = self.test_set_path_dir.split("/")[-2]
        if self.attribute_type in ["condition", "subquestion"]:
            self.store_test_path_difficulty_dir = f"./run_outputs/{self.structure}/{test_set_name}_|_{attribute_type}_|_test_question_path_{model_name}.json"
            rephrasing_handler = self.decomposer.rephrasing_handler
            user_question_list = [item["problem"] for item in self.testset]
            if self.attribute_type == "condition":
                self.train_path_difficulty_count = read_json(self.store_path_difficulty_dir)
                io_outputs, _ = rephrasing_handler.generate_rephrased_question(user_question_list)
                
            elif self.attribute_type == "subquestion":
                self.train_path_difficulty_count = read_json(self.store_path_difficulty_dir)
                io_outputs, _ = rephrasing_handler.generate_sub_questions(user_question_list)

            for i, (io_output, count_condition) in enumerate(io_outputs):
                path = self.find_nearest_train_path(count_condition, k)
                test_question_paths[user_question_list[i]] = path
        
        elif self.attribute_type == "semantic":
            self.store_test_path_difficulty_dir = f"./run_outputs/{self.structure}/{test_set_name}_|_{attribute_type}_{self.sim_type}_|_test_question_path.json"
            self.train_path_questions = self.select_questions(self.train_path_questions)
            for item in self.testset:
                test_question = item['problem']
                similarities = {}

                for path, questions in self.train_path_questions.items():
                    if self.sim_model:  
                        path_similarities = []
                        for question in questions:
                            sentences = [test_question, question]
                            embeddings = self.sim_model.encode(sentences, batch_size=2)['dense_vecs']
                            similarity = embeddings[0] @ embeddings[1].T
                            path_similarities.append(similarity)
                        
                        if self.sim_type == "average":
                            avg_similarity = sum(path_similarities) / len(path_similarities) if path_similarities else 0
                            similarities[path] = avg_similarity
                        elif self.sim_type == "max":
                            max_similarity = max(path_similarities) if path_similarities else 0
                            similarities[path] = max_similarity

                top_k_paths = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
                test_question_paths[test_question] = [path for path, _ in top_k_paths]   
                print(len(list(test_question_paths.keys())))

        save_json(test_question_paths, self.store_test_path_difficulty_dir)
        return test_question_paths   
            

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    fix_seeds(args.seed)
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.model_ckpt = "meta-llama/Meta-Llama-3-8B-Instruct"
    k = 0.95                         # ratio of path to score
    attribute_type = "condition"    # 'condition', 'subquestion', 'confidence', 'semantic'      
    structure_dir = "structure"

    root_dir = f"Your directory path containing MCTS-generated solutions for the seed dataset"
    file_dir = os.path.join("./run_outputs", root_dir, "05-15_00-00")

    dataset_name = root_dir.split("/")[0].strip("")
    model_name = root_dir.split("/")[1].split("/")[0].strip()
    if "GSM" in dataset_name and "HARD" not in dataset_name:
        evaluator = eval("GSM8KEvaluator()")
    elif "MATH" in dataset_name:
        evaluator = eval("MATHEvaluator()")
    else:
        evaluator = eval(f"{dataset_name}Evaluator()")
    
    manager = ProblemDecompositionManager(k, structure_dir, dataset_name, root_dir, file_dir, args, evaluator, attribute_type=attribute_type)  
    manager.decompose()
    test_question_paths = manager.obtain_path()
    print(f"dataset_name: {dataset_name}, model_name: {model_name}, ratio: {k}, attribute_type: {attribute_type}")