# Licensed under the MIT license.
import sys
sys.path.append(".")
import re
import numpy as np, os, json
from tqdm import trange
from typing import List, Dict, Tuple
from copy import deepcopy
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from transformers import BartForSequenceClassification, BartTokenizer
import ast
try:
    from rapidfuzz import fuzz, process
except:
    pass

import random
from collections import Counter
from models.IO_System import IO_System
from common.utils import read_txt, read_json
from eval_src.Evaluator import Evaluator
from run_src.MCTS_backbone import MCTS_Searcher, MCTS_Node
from run_src.hiaricl_utils import (
    Node_Type,
    GeneratorError,
    reach_terminal_subquestion,
    reach_terminal_ost,
    concat_subqs_and_subas,
    concat_ost,
    concat_subqs_subas_as_ost,
    concat_all_parent_steps,
    make_hint,
    make_response_prefix,
    split_user_question,
    print_tree_from_root,
    find_valid_solution_nodes,
    stochastic_find_best_solution,
    filtered_output_list
)


def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        if "bge" in args.sim_model:
            self.sim_model_name = "bge"
            self.sim_model = BGEM3FlagModel(args.sim_model, use_fp16=True)
        elif "mpnet" in args.sim_model:
            self.sim_model_name = "mpnet"
            self.sim_model = SentenceTransformer(args.sim_model)
        
        self.if_entailment = args.if_entailment
        if args.if_entailment:
            self.entail_model = BartForSequenceClassification.from_pretrained(args.entailment_model)
            self.entail_tokenizer = BartTokenizer.from_pretrained(args.entailment_model)
        else:
            self.entail_model = None
            self.entail_tokenizer = None
            
        self.evaluator = evaluator

        self.if_ost_select = args.if_ost_select
        self.num_dc = args.num_dc
        self.num_ost = args.num_ost  
        self.num_dc_votes = args.num_dc_votes       
        self.max_tokens = args.max_tokens      
        self.dataset_name = args.dataset_name

        self.num_cot = args.num_cot   

        with open(args.decompose_template_path, "r") as f:
            decompose_template = json.load(f)
            self.question_index = decompose_template["index"]

        self.decompose_prompt = read_txt(args.decompose_prompt_path)   
        self.fewshot_cot_prompt = read_txt(args.fewshot_cot_prompt_path)
        self.fewshot_cot_config = read_json(args.fewshot_cot_config_path)
        self.fewshot_cot_check_prompt = read_txt(args.fewshot_cot_check_prompt_path)
        self.fewshot_cot_check_config = read_json(args.fewshot_cot_check_config_path)
        self.rephrasing_prompt_template = read_txt(args.rephrasing_prompt_template_path)
        
        
        if not args.disable_ost: 
            self.fewshot_ost_prompt = read_txt(args.fewshot_ost_prompt_path)
            self.fewshot_ost_config = read_json(args.fewshot_ost_config_path)
        
        if not args.disable_sa:  
            self.rephrasing_prompt_template = read_txt(args.rephrasing_prompt_template_path)      
            self.decompose_prompt_rephrased = read_txt(args.decompose_prompt_rephrased_path)      
            self.fewshot_cot_prompt_rephrased = read_txt(args.fewshot_cot_prompt_rephrased_path)
            self.fewshot_ost_prompt_rephrased = read_txt(args.fewshot_ost_prompt_rephrased_path)
        
        if not args.disable_srr:  
            self.fewshot_refine_sum_prompt = read_txt(args.fewshot_refine_sum_prompt_path)
            self.fewshot_refine_sum_config = read_json(args.fewshot_refine_sum_config_path)

    def _get_most_likely_answer(self, user_question, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:  
            _, most_confident_answer_full_completion, _, confidence = self.evaluator.find_most_confident_answer(
                user_question, io_output_list, self.fewshot_cot_check_config["prompt_template"], self.fewshot_cot_check_prompt, self.io, self.entail_model, self.entail_tokenizer
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence

    def _fewshot_cot_answer_question(self, question: str, paraphrased: bool, num_return: int, hint: str = None):
        fewshot_cot_prompt = self.fewshot_cot_prompt if not paraphrased else self.fewshot_cot_prompt_rephrased
        question += "\n\n" + hint if hint is not None else ""   
        io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt, instruction=question)
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
            stop_tokens=self.fewshot_cot_config["stop_tokens"],  
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  
        return io_input, cleaned_io_output_list

    def generate_chain_of_thought(self, user_question: str, paraphrased: bool, hint: str):
        chain_of_thought_list, value_list = [], []
        
        def find_first_with_digit_in_last_ten(strings, k=-5):
            for index, string in enumerate(strings):
                last_ten_chars = string[k:]
                if any(char.isdigit() for char in last_ten_chars):
                    return index
            return -1  

        #! few shot cot
        num_return = self.num_cot
        io_input, io_output_list = self._fewshot_cot_answer_question(
            question=user_question, paraphrased=paraphrased, num_return=num_return, hint=hint
        )
        
        cleaned_io_output_list = [
            io_output.replace("I hope it is correct", "").rstrip(".").strip()
            if 'answer is' in io_output.lower() and "I hope it is correct".lower() in io_output[-25:].lower()
            else io_output
            for io_output in io_output_list
        ]

        if len(cleaned_io_output_list) == 0:
            index = find_first_with_digit_in_last_ten(io_output_list, k=-7)
            if index == -1:
                return [], []
            else:
                cleaned_io_output_list = [io_output_list[index]]
                likelihood = [0.01]
                chain_of_thought_list.append(cleaned_io_output_list[0])
                value_list.append(likelihood[0])
                return chain_of_thought_list, value_list
            
            
        try:   
            most_likely_answer, likelihood = self._get_most_likely_answer(user_question, cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )
            

        chain_of_thought_list.append(most_likely_answer)
        value_list.append(likelihood)
 
        return chain_of_thought_list, value_list

    def generate_subquestions(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        subquestion_list, subanswer_list, value_list = [], [], []
        decompose_prompt = self.decompose_prompt if not paraphrased else self.decompose_prompt_rephrased

        #! generate subquestions
        existing_subquestions_and_subanswers, next_subquestion_id = concat_subqs_and_subas(
            solution_trace, self.question_index
        )
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Question {self.question_index}: {user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            + f"Question {self.question_index}.{next_subquestion_id}:"
        )
        io_output_list = self.io.generate(
            io_input,
            max_tokens=128,
            num_return=self.num_dc,
            stop_tokens=[
                "\n",
                "\n\n",
                "Answer",
                "Answer ",
                f"Answer {self.question_index}.{next_subquestion_id}",
                f"Answer {self.question_index}.{next_subquestion_id}:",
                f"Answer {self.question_index}.{next_subquestion_id}: ",
            ],
        )

        subquestion_list = [o.strip() for o in io_output_list]

        #! generate subanswers to the subquestions generated above
        io_input_list = []
        for subquestion in subquestion_list:
            io_input = (
                decompose_prompt
                + "\n\n"
                + f"Question {self.question_index}: {user_question}"
                + "\n"
                + existing_subquestions_and_subanswers
                + f"Question {self.question_index}.{next_subquestion_id}: "
                + subquestion
                + "\n"
                + f"Answer {self.question_index}.{next_subquestion_id}:"
            )
            io_input_list.append(io_input)

        if reach_terminal_subquestion(subquestion=subquestion, user_question=user_question):
            num_return = self.num_cot
        else:
            num_return = self.num_dc_votes

        io_output_list = self.io.generate(
            io_input_list,
            max_tokens=512,
            num_return=num_return,
            stop_tokens=[
                "\n",
                "\n\n",
                f"Question {self.question_index}.{next_subquestion_id + 1}",
            ],
        )
        cleaned_io_output_list = [
            [io_output.strip() for io_output in io_output_group] for io_output_group in io_output_list
        ]

        for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
            try:  
                most_likely_answer, likelihood = self._get_most_likely_answer(user_question, cleaned_io_output_group)
            except Exception as e:
                raise GeneratorError(
                    source="generate answer to subquestions",
                    io_input=io_input_list[i],
                    io_output_list=cleaned_io_output_group,
                )
            subanswer_list.append(most_likely_answer)
            value_list.append(likelihood)

        assert len(subquestion_list) == len(subanswer_list) == len(value_list)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        potential_answers_list = [None] * len(subquestion_list)

        return subquestion_list, subanswer_list, value_list, potential_answers_list

    def generate_rephrased_user_question(self, user_question: str):
        rephrased_user_question_list = []
        io_input = self.rephrasing_prompt_template
        io_input += "\n\n"
        io_input += "Original Question: " + user_question + "\n"
        io_input += "Rephrased Question: Given a list of conditions, please answer the question. Condition 1: "
        io_output = self.io.generate(model_input=io_input, max_tokens=512, num_return=1, stop_tokens=["\n", "\n\n"])[0]

        io_output = io_output.split("?")[0] + "?"
        io_output = "Given a list of conditions, please answer the question. Condition 1: " + io_output
        rephrased_user_question_list.append(io_output)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  
        potential_answers_list = [None] * len(rephrased_user_question_list)

        return rephrased_user_question_list, potential_answers_list

    def generate_self_reflection_and_refine(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
    ):
        self_reflection_and_refine_list = []
        existing_ost = concat_all_parent_steps(solution_trace)
        io_input = self.fewshot_refine_sum_config["prompt_template"].format(
            examples=self.fewshot_refine_sum_prompt,
            instruction=user_question,
            steps=existing_ost.strip(),
        )
        io_output_list = self.io.generate(
            model_input=io_input, max_tokens=512, num_return=3, stop_tokens=self.fewshot_refine_sum_config["stop_tokens"]  
        )
        output_list = []
        outputs = []
        key = "<CORRECT>"

        # import pdb
        # pdb.set_trace()
        for x in io_output_list:
            output = x.split(key)[-1].split("\n\n")[0]
            output = re.sub(r'^[\n:]+', '', output, count=3)
            if len(output) >= 10:
                output_list.append(output.strip())
            else:
                a = 1
                # print(output.strip())

        if output_list:
            self_reflection_and_refine_list, sim_list = filtered_output_list(output_list, user_question, self.sim_model, self.sim_model_name)
            outputs = [self_reflection_and_refine_list[sim_list.index(max(sim_list))]] if sim_list else []
        else:
            outputs = [existing_ost.strip()]
        
        potential_answers_list = [None] * len(outputs)
        return outputs, potential_answers_list
    
    def generate_ost(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
        parent_is_subquestion: bool,
        parent_is_self_reflection_and_refine=False,
    ):
        ost_list = []
        if parent_is_self_reflection_and_refine:
            existing_ost = solution_trace[0]['answers'][-1][1] + "\n"
            matches = re.findall(r'Step \d+:', existing_ost)
            next_ost_id = len(matches) + 1
        else:
            if parent_is_subquestion:   
                existing_ost, next_ost_id = concat_subqs_subas_as_ost(solution_trace)
            else:
                existing_ost, next_ost_id = concat_ost(solution_trace)
        io_input = (
            self.fewshot_ost_config["prompt_template"].format(
                examples=self.fewshot_ost_prompt if not paraphrased else self.fewshot_ost_prompt_rephrased,
                instruction=user_question,
            )
            + existing_ost
            + f"Step {next_ost_id}:"   
        )
        io_output_list = self.io.generate(
            model_input=io_input, max_tokens=256, num_return=self.num_ost, stop_tokens=["\n", "\n\n"]  
        )
        
        if not self.if_ost_select:
            for x in io_output_list:
                if x.strip() not in ost_list:
                    ost_list.append(x.strip())
        else:
            # similarity determination
            for io_output in io_output_list:
                sentences = [user_question, io_output]
                if self.sim_model_name == "bge":
                    embeddings = self.sim_model.encode(sentences, batch_size=2)['dense_vecs']
                    threshold = 0.5
                elif self.sim_model_name == "mpnet":
                    embeddings = self.sim_model.encode(sentences)
                    threshold = 0.4
                    if self.dataset_name == "GPQA":
                        threshold = 0.3
                similarity = embeddings[0] @ embeddings[1].T
                if similarity > threshold or "answer is" in io_output[-50:].lower():
                    if io_output.strip() not in ost_list:
                        ost_list.append(io_output.strip())
                else:
                    print("No similarity\nQuestion: " + user_question + "\nAnswer: " + io_output + "\nSimilarity: " + str(similarity))            

        # TODO: new added
        if len(ost_list) == 0:
            ost_list = [io_output_list[0].strip()]

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  
        potential_answers_list = [None] * len(ost_list)

        return ost_list, potential_answers_list


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        # --- For instantiating root node ---
        node_value: float = None,
        generator: Generator = None,
        user_question: str = None,
        max_depth_allowed: int = None,
        disable_sa: bool = None,
        disable_ost: bool = None,
        disable_cot: bool = None,
        disable_dc: bool = None,
        disable_srr: bool = None,
        # -----------------------------------
        # --- For instantiating SYSTEM_ANALYSIS node ---
        system_analysis: str = None,
        # ------------------------------------------------------
        expected_answer: str = None,
        # -----------------------------------
        # --- For instantiating CHAIN_OF_THOUGHT node ---
        chain_of_thought: str = None,
        # --------------------------------------------
        # --- For instantiating DIVIDE_AND_CONQUER node ---
        subquestion: str = None,
        subanswer: str = None,
        is_new_subquestion: bool = None,
        # -----------------------------------
        # --- For instantiating SELF_REFLECTION_AND_REFINE node ---
        reflection_and_refine: str = None,
        # -------------------------------------------
        # --- For instantiating OST node ---
        ost: str = None,
    ) -> None:
        super().__init__()

        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None   
            if node_value is not None:
                assert node_value > 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:   
                assert depth == 0 
                assert all(   
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        system_analysis,
                        chain_of_thought,
                        subquestion,
                        subanswer,
                        reflection_and_refine,
                        is_new_subquestion,
                        ost,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [generator, disable_sa, user_question, expected_answer, max_depth_allowed, disable_ost]
                )
            elif node_type is Node_Type.SYSTEM_ANALYSIS:      
                assert depth == 1
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_sa,
                        user_question,
                        expected_answer,
                        chain_of_thought,
                        subquestion,
                        subanswer,
                        reflection_and_refine,
                        is_new_subquestion,
                        ost,
                        max_depth_allowed,
                        disable_ost,
                    ]
                )
                assert all(attr is not None for attr in [parent, system_analysis])
            elif node_type is Node_Type.CHAIN_OF_THOUGHT:                
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_sa,
                        user_question,
                        expected_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        reflection_and_refine,
                        ost,
                        max_depth_allowed,
                        disable_ost,
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, chain_of_thought])
            elif node_type is Node_Type.SELF_REFLECTION_AND_REFINE:                
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        node_value,
                        disable_sa,
                        user_question,
                        expected_answer,
                        subquestion,
                        subanswer,
                        chain_of_thought,
                        is_new_subquestion,
                        ost,
                        max_depth_allowed,
                        disable_ost,
                    ]
                )
                assert all(attr is not None for attr in [parent, reflection_and_refine])
            elif node_type is Node_Type.DIVIDE_AND_CONQUER:                  
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_sa,
                        user_question,
                        expected_answer,
                        chain_of_thought,
                        reflection_and_refine,
                        ost,
                        max_depth_allowed,
                        disable_ost,
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, node_value, subquestion, subanswer, is_new_subquestion]
                )
            elif node_type is Node_Type.ONE_STEP_THOUGHT:                    
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_sa,
                        user_question,
                        system_analysis,
                        expected_answer,
                        chain_of_thought,
                        reflection_and_refine,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        max_depth_allowed,
                        disable_ost,
                    ]
                )
                assert all(attr is not None for attr in [parent, ost])
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.chain_of_thought = chain_of_thought
        self.reflection_and_refine = reflection_and_refine
        self.subquestion = subquestion
        self.subanswer = subanswer
        self.is_new_subquestion = is_new_subquestion
        self.ost = ost

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.expected_answer = expected_answer
            self.generator = generator
            self.disable_ost = disable_ost
            self.disable_cot = disable_cot
            self.disable_dc = disable_dc
            self.disable_sa = disable_sa
            self.disable_srr = disable_srr
            self.question_index = generator.question_index
            self.max_depth_allowed = max_depth_allowed
        else:  # inherit from parent   
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.expected_answer = parent.expected_answer
            self.generator = parent.generator
            self.question_index = parent.generator.question_index
            self.max_depth_allowed = parent.max_depth_allowed
            self.disable_ost = parent.disable_ost
            self.disable_cot = parent.disable_cot
            self.disable_dc = parent.disable_dc
            self.disable_sa = parent.disable_sa
            self.disable_srr = parent.disable_srr

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.SYSTEM_ANALYSIS:
            self.paraphrased = True
            self.user_question = system_analysis
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased

        #! record number of subquestions till now
        if parent is None:  # root
            self.subquestion_counter = 0
        else:
            if node_type is Node_Type.DIVIDE_AND_CONQUER and is_new_subquestion:
                self.subquestion_counter = parent.subquestion_counter + 1
            else:
                self.subquestion_counter = parent.subquestion_counter

        #! record number of one-step thought steps till now
        if parent is None:  # root
            self.ost_counter = 0
        else:
            if node_type is Node_Type.ONE_STEP_THOUGHT or node_type is Node_Type.SELF_REFLECTION_AND_REFINE:
                self.ost_counter = parent.ost_counter + 1
            else:
                self.ost_counter = parent.ost_counter

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "ost": {}, "reflection_and_refine": {}, "path": [(self.node_type.value, self.id)], "answers": [(0, user_question)]}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)
            self.solution_trace[0]['path'].append((self.node_type.value, self.id))
            answer_id = self.solution_trace[0]['answers'][-1][0] + 1

            if node_type is Node_Type.SYSTEM_ANALYSIS:
                self.solution_trace[0]["user_question"] = system_analysis
                self.solution_trace[0]['answers'].append((answer_id, system_analysis))
            elif node_type is Node_Type.CHAIN_OF_THOUGHT:
                assert self.subquestion_counter in self.solution_trace.keys()
                assert self.subquestion_counter == parent.subquestion_counter
                self.solution_trace[self.subquestion_counter]["chain_of_thought"] = {
                    "text": chain_of_thought,
                    "value": node_value,
                }
                self.solution_trace[0]['answers'].append((answer_id, chain_of_thought))
            elif node_type is Node_Type.DIVIDE_AND_CONQUER:
                assert is_new_subquestion and self.subquestion_counter == parent.subquestion_counter + 1
                self.solution_trace[self.subquestion_counter] = {
                    "subquestion": subquestion,
                    "subanswer": {"text": subanswer, "value": node_value},
                    "ost": {},
                    "reflection_and_refine": {},
                }
                self.solution_trace[0]['answers'].append((answer_id, subquestion + "\n" + subanswer))
            elif node_type is Node_Type.ONE_STEP_THOUGHT:
                assert "ost" in self.solution_trace[self.subquestion_counter].keys()
                self.solution_trace[self.subquestion_counter]["ost"][self.ost_counter] = ost
                self.solution_trace[0]['answers'].append((answer_id, ost))
            elif node_type is Node_Type.SELF_REFLECTION_AND_REFINE:
                assert "reflection_and_refine" in self.solution_trace[self.subquestion_counter].keys()
                self.solution_trace[self.subquestion_counter]["reflection_and_refine"][self.ost_counter] = reflection_and_refine
                self.solution_trace[0]['answers'].append((answer_id, reflection_and_refine))

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "UQ",
            Node_Type.SYSTEM_ANALYSIS: "SA",
            Node_Type.CHAIN_OF_THOUGHT: "COT",
            Node_Type.SELF_REFLECTION_AND_REFINE: "SRR",
            Node_Type.DIVIDE_AND_CONQUER: "DC",
            Node_Type.ONE_STEP_THOUGHT: "OST",
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def find_children(self, rollout_id: int):
        try:
            self.children, terminate = self.children or self._create_children()
        except:
            terminate = False
            self.children = self.children
        
        for child in self.children:
            child.set_rollout_id(rollout_id)
        
        return self.children, terminate
    
    def _create_children(self):  
        def do_action_perform_system_analysis():
            verbose_print(f"---- Performing system analysis for node {self.id}...", self.verbose)

            rephrased_user_question_list, potential_answers_list = self.generator.generate_rephrased_user_question(
                user_question=self.user_question
            )
            for rephrased_user_question, potential_answers in zip(rephrased_user_question_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.SYSTEM_ANALYSIS,
                        system_analysis=rephrased_user_question
                    )
                )

        def do_action_perform_one_step_thought(parent_is_subquestion=False):
            verbose_print(f"---- Performing one-step thought for node {self.id}...", self.verbose)

            parent_is_self_reflection_and_refine = True if self.node_type is Node_Type.SELF_REFLECTION_AND_REFINE else False
            ost_list, potential_answers_list = self.generator.generate_ost(
                user_question=self.user_question, 
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
                parent_is_subquestion=parent_is_subquestion,
                parent_is_self_reflection_and_refine=parent_is_self_reflection_and_refine
            )
            for ost, potential_answers in zip(ost_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.ONE_STEP_THOUGHT,
                        ost=ost
                    )
                )        
        
        def do_action_perform_chain_of_thought():
            verbose_print(f"---- Performing chain-of-thought for node {self.id}...", self.verbose)

            if (
                self.node_type is not Node_Type.USER_QUESTION
                and self.node_type is not Node_Type.SYSTEM_ANALYSIS
            ):
                hint = make_hint(self.solution_trace, self.node_type)
            else:
                hint = None

            (chain_of_thought_list, value_list) = self.generator.generate_chain_of_thought(
                user_question=self.user_question, paraphrased=self.paraphrased, hint=hint
            )
            if len(chain_of_thought_list) == 0:  
                return
            for chain_of_thought, value in zip(chain_of_thought_list, value_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()  # this should not happen
                if chain_of_thought is None:
                    continue
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.CHAIN_OF_THOUGHT,
                        node_value=value,
                        chain_of_thought=chain_of_thought,
                    )
                )
        
        def do_action_perform_divide_and_conquer():
            verbose_print(f"---- Performing divide and conquer for node {self.id}...", self.verbose)

            (subquestion_list, subanswer_list, value_list, potential_answers_list) = (
                self.generator.generate_subquestions(
                    user_question=self.user_question, solution_trace=self.solution_trace, paraphrased=self.paraphrased
                )
            )
            for subquestion, subanswer, value, potential_answers in zip(
                subquestion_list, subanswer_list, value_list, potential_answers_list
            ):
                if np.isnan(value) or value <= 0:
                    value = 0.01
                    # breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIVIDE_AND_CONQUER,
                        node_value=value,
                        subquestion=subquestion,
                        subanswer=subanswer,
                        is_new_subquestion=True
                    )
                )

        def do_action_perform_self_reflection_refine(parent_is_subquestion=False):
            verbose_print(f"---- Performing self-reflection and refinement for node {self.id}...", self.verbose)

            self_reflection_and_refine_list, potential_answers_list = self.generator.generate_self_reflection_and_refine(
                user_question=self.user_question, 
                solution_trace=self.solution_trace,
            )
            if len(self_reflection_and_refine_list) == 0:
                # import pdb
                # pdb.set_trace()
                return
            for self_reflection_and_refine, potential_answers in zip(self_reflection_and_refine_list, potential_answers_list):
                if self_reflection_and_refine is None:
                    continue
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.SELF_REFLECTION_AND_REFINE,
                        reflection_and_refine=self_reflection_and_refine
                    )
                )
        
        if not if_use_cards:
            if self.node_type is Node_Type.USER_QUESTION:
                terminate = False
                
                if not self.disable_ost:
                    do_action_perform_one_step_thought()

                if not self.disable_cot:
                    do_action_perform_chain_of_thought()
                    if self.children[-1].node_type.value=="CHAIN_OF_THOUGHT" and self.children[-1].node_value > 0.93:
                        terminate = True

                if not terminate:
                    if not self.disable_dc:
                        do_action_perform_divide_and_conquer()

                    if not self.disable_sa:
                        do_action_perform_system_analysis()
                
            elif self.node_type is Node_Type.SYSTEM_ANALYSIS:
                terminate = False
                
                if not self.disable_ost:
                    do_action_perform_one_step_thought()

                if not self.disable_cot:
                    do_action_perform_chain_of_thought()
                    if self.children[-1].node_type.value=="CHAIN_OF_THOUGHT" and self.children[-1].node_value > 0.95:
                        terminate = True

                if not terminate:
                    if not self.disable_dc:
                        do_action_perform_divide_and_conquer()
                    
            elif self.node_type is Node_Type.CHAIN_OF_THOUGHT:
                raise ValueError("Chain-of-Thought node cannot create children!!")
            elif self.node_type is Node_Type.DIVIDE_AND_CONQUER:
                terminate = False
                if not self.disable_ost:  
                    do_action_perform_one_step_thought(parent_is_subquestion=True)

                if not self.disable_cot:
                    do_action_perform_chain_of_thought()
                    if self.children[-1].node_type.value=="CHAIN_OF_THOUGHT" and self.children[-1].node_value > 0.95:
                        terminate = True

                if not terminate:
                    if not self.disable_dc:
                        do_action_perform_divide_and_conquer()  
                
                    if not self.disable_srr:
                        do_action_perform_self_reflection_refine()
                    
            elif self.node_type is Node_Type.ONE_STEP_THOUGHT:
                terminate = False
                
                if not self.disable_ost:
                    do_action_perform_one_step_thought()

                if not self.disable_cot:
                    do_action_perform_chain_of_thought()
                    if self.children[-1].node_type.value=="CHAIN_OF_THOUGHT" and self.children[-1].node_value > 0.93:
                        terminate = True
                
                if not terminate:
                    if not self.disable_srr:
                        do_action_perform_self_reflection_refine()

            elif self.node_type is Node_Type.SELF_REFLECTION_AND_REFINE:
                terminate = False
                if not self.disable_ost:
                    do_action_perform_one_step_thought()

                if not self.disable_cot:
                    do_action_perform_chain_of_thought()
                    if self.children[-1].node_type.value=="CHAIN_OF_THOUGHT" and self.children[-1].node_value > 0.93:
                        terminate = True

        else:
            depth = self.depth
            try:
                cur_action = train_path[depth+1]
            except:
                cur_action = Node_Type.CHAIN_OF_THOUGHT.value
            terminate = False
            if cur_action == Node_Type.ONE_STEP_THOUGHT.value:
                do_action_perform_one_step_thought()
            elif cur_action == Node_Type.SELF_REFLECTION_AND_REFINE.value:
                do_action_perform_self_reflection_refine()
            elif cur_action == Node_Type.CHAIN_OF_THOUGHT.value:
                do_action_perform_chain_of_thought()
                if self.children[-1].node_type.value=="CHAIN_OF_THOUGHT" and self.children[-1].node_value > 0.93:
                    terminate = True
            elif cur_action == Node_Type.DIVIDE_AND_CONQUER.value:
                do_action_perform_divide_and_conquer()
            elif cur_action == Node_Type.SYSTEM_ANALYSIS.value:
                do_action_perform_system_analysis()
            
        return self.children, terminate

    def is_valid_leaf_node(self):
        return (
            self.node_type is Node_Type.ONE_STEP_THOUGHT and reach_terminal_ost(self.ost)
        ) or (
            self.node_type is Node_Type.DIVIDE_AND_CONQUER and reach_terminal_subquestion(self.subquestion, self.user_question)
        ) or self.node_type is Node_Type.CHAIN_OF_THOUGHT

    def is_valid_solution_node(self):
        return (
            (
                self.node_type is Node_Type.DIVIDE_AND_CONQUER
                and reach_terminal_subquestion(self.subquestion, self.user_question)
            )
            or (self.node_type is Node_Type.ONE_STEP_THOUGHT and reach_terminal_ost(self.ost))
            or self.node_type is Node_Type.CHAIN_OF_THOUGHT
        )

    def set_potential_score(self, score: float):
        self.potential_score = score

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node(): 
            if self.node_value is None:
                return 0
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION or self.node_type is Node_Type.SYSTEM_ANALYSIS


def search_for_answers(args, evaluator, original_js, user_question: str, question_id: int, gt_answer: str, generator: Generator):
    verbose_print(f"********************* Searching for answers to question {question_id} ********************* ", args.verbose)
    global if_use_cards
    if_use_cards = args.if_use_cards
    if if_use_cards:
        global train_paths, train_path, pre_rephrase_output
        train_paths = args.reason_structure[user_question][:args.num_cards]
        train_paths = [list(ast.literal_eval(x)) if not isinstance(x, list) else x for x in train_paths]

    
    model_solutions = []        # record the best solution for each simulation      
    model_all_solutions = []    # record all solutions for each simulation
    model_rollout_nodes = []    # record the terminal node for each simulation
    model_best_path = []        # record the best node for each simulation
    if not if_use_cards:
        train_paths = [None]
    
    filter_k = 3
    print(f"len(train_paths): {len(train_paths)}")
    for j, train_path in enumerate(train_paths):
        print(j, "\t", train_path)
        
        if j != 0:   
            if if_use_cards:  
                try:
                    if chosen_node.node_type.value == "CHAIN_OF_THOUGHT" and chosen_node.node_value >= 0.93 or top_confidence / len(all_solutions) >= 0.8 and len(all_solutions) >= 5 or top_confidence / len(valid_solutions) >= 0.8 and len(valid_solutions) >= filter_k:
                        break
                except:
                    print(1)
            else:
                if chosen_node.node_type.value == "CHAIN_OF_THOUGHT" and chosen_node.node_value >= 0.93 or top_confidence / len(all_solutions) >= 0.8 and len(all_solutions) >= 5:
                    break
        
        if if_use_cards:
            args.max_depth_allowed = len(train_path)          
            args.num_rollouts = args.reuse_rollouts           

        #! build an MCTS searcher
        mcts_searcher = MCTS_Searcher(
            exploration_weight=args.mcts_exploration_weight,
            weight_scheduler=args.mcts_weight_scheduler,
            num_rollouts=args.num_rollouts,
            discount=args.mcts_discount_factor,
            verbose=args.verbose,
        )

        #! build the MCTS tree
        root_node = Reasoning_MCTS_Node(
            parent=None,
            depth=0,
            node_type=Node_Type.USER_QUESTION,  
            verbose=args.verbose,
            generator=generator,
            user_question=user_question,
            expected_answer=gt_answer,
            max_depth_allowed=args.max_depth_allowed,         
            disable_sa=args.disable_sa,   
            disable_ost=args.disable_ost,
            disable_cot=args.disable_cot,
            disable_dc=args.disable_dc,
            disable_srr=args.disable_srr,
        )

        for i in (pbar := trange(args.num_rollouts, disable=False, position=0)):
            # rollout_node = mcts_searcher.do_rollout(root_node, i)
            # model_rollout_nodes.append(rollout_node) 
            try:
                rollout_node = mcts_searcher.do_rollout(root_node, i)
                model_rollout_nodes.append(rollout_node) 
            except:
                # import pdb
                # pdb.set_trace()
                continue

            if not args.disable_answer_selection:
                if args.api == "debug":
                    best_solution, chosen_node, all_solution_nodes, all_solutions = "Debug: I don't know!", None, [], []
                else:  
                    _, best_solution, top_confidence, chosen_node, all_solution_nodes, all_solutions = stochastic_find_best_solution(
                        root_node, generator.evaluator, enable_potential_score=args.enable_potential_score
                    )
                    if best_solution == None:
                        continue
                    model_solutions.append(best_solution)
                    model_all_solutions.append(all_solutions)
                    model_best_path.append(chosen_node.solution_trace[0]['path'])
                    valid_solutions = [solution for solution in all_solutions if solution is not None]  # modify
                    if if_use_cards:
                        if chosen_node.node_type.value == "CHAIN_OF_THOUGHT" and chosen_node.node_value >= 0.93 or top_confidence / len(all_solutions) >= 0.8 and len(all_solutions) >= 5 or top_confidence / len(valid_solutions) >= 0.8 and len(valid_solutions) >= filter_k:
                            break
                    else:
                        if chosen_node.node_type.value == "CHAIN_OF_THOUGHT" and chosen_node.node_value >= 0.93 or top_confidence / len(all_solutions) >= 0.8 and len(all_solutions) >= 5:
                            break
            else:
                chosen_node = None
                all_solution_nodes = find_valid_solution_nodes(root_node)

            if args.save_tree:
                print_tree_from_root(
                    mcts_searcher=mcts_searcher,
                    rollout_id=i,
                    root_node=root_node,
                    chosen_node=chosen_node,
                    file=None,
                )

        if if_use_cards:
            if j != 0:
                try:
                    model_all_solutions[-1] = model_all_solutions[-1] + model_all_solutions[index]
                except:
                    print(1)
            index = len(model_all_solutions) - 1
    
    try:
        js = [{"rollout_id": node.rollout_id, "trace": node.solution_trace} for node in all_solution_nodes]
        with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Final Solutions.json"), "w") as f:
            json.dump(js, f)

        js2 = [{"rollout_id": i, "trace": node.solution_trace} for i, node in enumerate(model_rollout_nodes)]
        with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Rollout Solutions.json"), "w") as f:
            json.dump(js2, f)
            
        js3 = [{"rollout_id": i, "path": node.solution_trace[0]['path']} for i, node in enumerate(model_rollout_nodes)]
        with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Rollout Path.json"), "w") as f:
            json.dump(js3, f)
        
        js4 = []
        for i, node in enumerate(model_rollout_nodes):
            complete_solutions = node.solution_trace[0]['answers'][0][1] + "\n\n" + "\n".join(["Step "+str(x[0])+": "+x[1] for x in node.solution_trace[0]['answers']])
            js4.append({"rollout_id": i, "solutions": complete_solutions})
        with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Rollout Complete Solutions.json"), "w") as f:
            json.dump(js4, f)

    except:
        print(1)

    print(f"len(model_solutions): {len(model_solutions)}")
    print(f"len(model_all_solutions): {sum([len(x) for x in model_all_solutions])}")


    # import pdb
    # pdb.set_trace()
    if not args.disable_answer_selection:
        stopping_id = i
        assert len(model_solutions) == len(model_all_solutions)
        assert len(model_solutions) == len(model_best_path)
        stopping_id = min(stopping_id, len(model_solutions)-1)
        for rollout_id, (model_path, model_solution, model_all_solution) in enumerate(
            zip(model_best_path, model_solutions, model_all_solutions)  
        ):
            model_answer = evaluator.extract_answer_from_model_completion(model_solution)
            model_all_answers = [evaluator.extract_answer_from_model_completion(a) for a in model_all_solution]
            
            # total_correct_con
            print(f"=================================\nRollout {rollout_id}: {model_path}\n")

            correct = evaluator.check_answers_equiv(model_answer, gt_answer)
            correct_limit = any([evaluator.check_answers_equiv(a, gt_answer) for a in model_all_answers])
            most_common_answer = Counter(model_all_answers).most_common(1)[0][0]
            correct_con = evaluator.check_answers_equiv(most_common_answer, gt_answer)
            print(f"model_answer: {model_answer}\ngt_answer: {gt_answer}\ncorrect: {correct}")
            
            random_answer = random.choice(model_all_answers)
            correct_random = evaluator.check_answers_equiv(random_answer, gt_answer)

            if rollout_id == stopping_id:   
                correct = int(correct)
                correct_limit = int(correct_limit)
                correct_con = int(correct_con)
                correct_random = int(correct_random)
                original_js["model_completion"] = model_solution
                original_js["model_answer"] = model_answer
                original_js["model_all_answer"] = model_all_solution
            
            original_js["all_model_completions"][f"rollout_{rollout_id}"] = {
                "model_solution": model_solution,
                "model_answer": model_answer,
                "model_path": model_path,
                "correct": correct,
                "correct_limit": correct_limit,
            }

    return original_js, model_solutions, model_all_solutions, model_best_path, correct, correct_limit, correct_con, correct_random