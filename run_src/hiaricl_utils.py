# Licensed under the MIT license.

from enum import Enum, unique
import re
import math
from typing import Dict, Tuple
from colorama import Fore, Style
import math
from eval_src import Evaluator
import func_timeout
import torch
from collections import Counter


@unique   
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"                              # root node
    SYSTEM_ANALYSIS = "SYSTEM_ANALYSIS"                          # a1: systm analysis, sa
    ONE_STEP_THOUGHT = "ONE_STEP_THOUGHT"                        # a2: one-step thought, ost
    CHAIN_OF_THOUGHT = "CHAIN_OF_THOUGHT"                        # a3: chain-of-thought, cot
    DIVIDE_AND_CONQUER = "DIVIDE_AND_CONQUER"                    # a4: divide and conquer, dc
    SELF_REFLECTION_AND_REFINE = "SELF_REFLECTION_AND_REFINE"    # a5: self-reflection and refine, srr
    

class GeneratorError(Exception):
    def __init__(self, source, io_input, io_output_list) -> None:
        super().__init__()

        self.source = source
        self.io_input = io_input
        self.io_output_list = io_output_list


def split_user_question(user_question: str):
    user_question = user_question.strip().rstrip(".")
    last_period_id = user_question.rfind(".")
    assert last_period_id < len(user_question) - 1
    user_question_context = user_question[: last_period_id + 1].strip()
    user_question_problem = user_question[last_period_id + 1 :].strip()
    return user_question_context, user_question_problem


def reach_terminal_subquestion(subquestion: str, user_question: str):
    assert subquestion is not None

    if "Now we can answer" in subquestion:
        #! remember that: when the original question is answerable, please start the subquestion with "Now we can answer the question: "
        return True

    user_question_2nd_part = split_user_question(user_question)[1]
    if user_question_2nd_part.lower() in subquestion.lower() and len(user_question_2nd_part) > 5:
        return True

    return False


def reach_terminal_ost(ost: str):
    assert ost is not None

    return "answer is" in ost.lower()


def print_tree_from_root(mcts_searcher, rollout_id, root_node, chosen_node=None, file=None):
    color_print = False if file else True

    def my_print(text):
        if file:
            file.write(text + "\n")
        else:
            print(text)

    def print_tree(parent_node, node, file, rollout_id):
        to_print = ""

        num_indent = 4
        dash = "-" * num_indent * node.depth
        space = " " * num_indent * node.depth

        attributes = f"Q: {round(mcts_searcher.Q[node], 2)}" + "; " + f"N: {mcts_searcher.N[node]}" + "; "
        attributes += f"V: {round(node.node_value, 2)}" if node.node_value is not None else "V: None"

        uct_value = "UCT: " + str(
            round(mcts_searcher._compute_uct(parent_node=parent_node, node=node, rollout_id=rollout_id), 2)
        )
        attributes += "; " + uct_value

        solution_marker = "(T) " if node.is_valid_solution_node() else "" 

        node_info = "[" + solution_marker + node.__str__() + ": " + attributes + "]"
        if chosen_node and node == chosen_node:
            node_info = "[" + node_info + "]"
        node_info += " "

        if color_print and node.is_valid_solution_node():
            node_details = Fore.RED + Style.BRIGHT + node_info + Fore.RESET + Style.RESET_ALL
        else:
            node_details = node_info

        if node.node_type is Node_Type.USER_QUESTION:
            gt = node.expected_answer.replace("\n", " ")
            node_details += f"User: {node.user_question}" + "\n" + space + " " * len(node_info) + f"Ground truth: {gt}"
        elif node.node_type is Node_Type.SYSTEM_ANALYSIS:
            node_details += f"Reph-User: {node.user_question}"
        elif node.node_type is Node_Type.CHAIN_OF_THOUGHT:
            node_details += f"Ans: {node.chain_of_thought}"
        elif node.node_type is Node_Type.SELF_REFLECTION_AND_REFINE:
            node_details += f"Refine-Summary: {node.self_reflection_and_refine}"
        elif node.node_type is Node_Type.DIVIDE_AND_CONQUER:
            node_details += f"Q: {node.subquestion}" + "\n" + space + " " * len(node_info) + f"A: {node.subanswer}"
        elif node.node_type is Node_Type.ONE_STEP_THOUGHT:
            node_details += f"OST: {node.ost}"

        to_print += dash + node_details

        my_print(to_print)

        for child in node.children:
            print_tree(node, child, file, rollout_id)

        if node.depth == 0:
            my_print("\n" + "=" * 50 + "\n")

    print_tree(parent_node=None, node=root_node, file=file, rollout_id=rollout_id)


def concat_all_parent_steps(solution_trace: Dict[int, Dict[str, str]]) -> str:
    """Return: concatenated all parent steps"""
    solution_trace_str = ""
    for i, x in enumerate(solution_trace[0]['answers'][1:]):
        step_i = x[1].replace("\n", " ")
        solution_trace_str += f"Step {i+1}: " + step_i + "\n"

    return solution_trace_str


def concat_subqs_and_subas(solution_trace: Dict[int, Dict[str, str]], question_index: int) -> Tuple[str, int]:
    """Return: concatenated subqs and suba, next subquestion id"""
    solution_trace_str = ""

    for subquestion_id, solution_step in solution_trace.items():
        if subquestion_id == 0:
            continue

        assert subquestion_id > 0
        assert "subquestion" in solution_step.keys() and "subanswer" in solution_step.keys()

        solution_trace_str += f"Question {question_index}." + str(subquestion_id) + ": " + solution_step["subquestion"]
        solution_trace_str += "\n"
        solution_trace_str += (
            f"Answer {question_index}." + str(subquestion_id) + ": " + solution_step["subanswer"]["text"]
        )
        solution_trace_str += "\n"

    next_subquestion_id = int(sorted(solution_trace.keys())[-1]) + 1
    return solution_trace_str, next_subquestion_id


def concat_ost(solution_trace: Dict[int, Dict[str, str]]) -> Tuple[str, int]:
    """Return: concatenated one-step thought steps, next one-step thought step id"""
    last_tuple = list(solution_trace.items())[-1]
    last_tuple_id, last_tuple_recording = last_tuple[0], last_tuple[1]
    assert "ost" in last_tuple_recording.keys()
    if len(last_tuple_recording["ost"]) > 0:
        solution_trace_str = ""
        for step_id, step_text in last_tuple_recording["ost"].items():
            solution_trace_str += f"Step {step_id}: " + step_text + "\n"
        return solution_trace_str, step_id + 1
    else:
        # no one-step thought step yet
        return "", 1


def concat_subqs_subas_as_ost(solution_trace: Dict[int, Dict[str, str]]) -> Tuple[str, int]:
    """Return: concatenated subqs and subas as one-step thought steps, next one-step thought step id"""
    """Example solution trace (subq suba):
    {
        "0": {
            "user_question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "ost": {}
        },
        "1": {
            "subquestion": " How many eggs do the ducks lay each day?",
            "subanswer": {
                "text": "The ducks lay 16 eggs per day. The answer is 16.",
                "value": 1.0
            },
            "ost": {}
        },
        "2": {
            "subquestion": " How many eggs does Janet eat or use for baking muffins?",
            "subanswer": {
                "text": "Janet eats 3 eggs for breakfast and uses 4 eggs for baking muffins. That's a total of 3 + 4 = 7 eggs. The answer is 7.",
                "value": 1.0
            },
            "ost": {}
        },
        "3": {
            "subquestion": " Now we can answer the question: How much in dollars does she make every day at the farmers' market?",
            "subanswer": {
                "text": "Since the ducks lay 16 eggs per day and Janet eats/use 7 eggs, she has 16 - 7 = 9 eggs left to sell at the market. Each egg is sold for $2, so she makes 9 * 2 = 18 dollars. The answer is 18.",
                "value": 1.0
            },
            "ost": {}
        }
    },

    Expected output:
        subqs_subas_as_ost_str:

            Step 1: The ducks lay 16 eggs per day.
            Step 2: Janet eats 3 eggs for breakfast and uses 4 eggs for baking muffins. That's a total of 3 + 4 = 7 eggs.
            Step 3: Since the ducks lay 16 eggs per day and Janet eats/use 7 eggs, she has 16 - 7 = 9 eggs left to sell at the market. Each egg is sold for $2, so she makes 9 * 2 = 18 dollars.

        next_ost_id: 4
    """
    subqs_subas_as_ost_str = ""
    step_id = 1
    while step_id in solution_trace:
        if "subanswer" in solution_trace[step_id]:
            match = re.search(r"(.+?\.) The answer is", solution_trace[step_id]["subanswer"]["text"])
            if match:
                step_text = match.group(1).strip()
            else:
                step_text = solution_trace[step_id]["subanswer"]["text"].strip()
            subqs_subas_as_ost_str += f"Step {step_id}: " + step_text + "\n"
            step_id += 1
        else:
            # not subquestions yet
            return "", 1
    return subqs_subas_as_ost_str, step_id


def safe_execute(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            if keys is None:
                return locals_.get('ans', None)
            else:
                return [locals_.get(k, None) for k in keys]
        except Exception:
            return None
    try:
        ans = func_timeout.func_timeout(5, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans

def floatify_ans(ans):
    if ans is None:
        return None
    elif type(ans) == dict:
        ans = list(ans.values())[0]
    elif type(ans) == bool:
        ans = ans
    elif type(ans) in [list, tuple]:
        if not ans:
            return None
        else:
            try:
                ans = float(ans[0])
            except Exception:
                ans = str(ans[0])
    else:
        try:
            ans = float(ans)
        except Exception:
            ans = str(ans)
    return ans

def concat_solution_trace(solution_trace: Dict[int, Dict[str, str]]):
    """Note that the solution trace might be subqs-subas and also one-step thought steps."""
    solution_trace_str = ""
    final_step_str = ""
    end_node_type = None
    reward_value = 0.0

    for item_idx, (subq_id, solution_step) in enumerate(solution_trace.items()):
        if item_idx == 0:
            if len(solution_step["ost"]) == 0 and "chain_of_thought" in solution_step.keys():
                solution_trace_str += solution_step["chain_of_thought"]["text"].strip()
                final_step_str = solution_step["chain_of_thought"]["text"].strip()
                reward_value = (
                    solution_step["chain_of_thought"]["value"] if "value" in solution_step["chain_of_thought"] else 0.0
                )
                end_node_type = Node_Type.CHAIN_OF_THOUGHT
                break
            elif len(solution_step["ost"]) > 0 and "chain_of_thought" in solution_step.keys():
                for step_id, step_text in solution_step["ost"].items():
                    solution_trace_str += step_text.strip() + " "
                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["chain_of_thought"]["text"].strip()
                final_step_str = solution_step["chain_of_thought"]["text"].strip()
                reward_value = (
                    solution_step["chain_of_thought"]["value"] if "value" in solution_step["chain_of_thought"] else 0.0
                )
                end_node_type = Node_Type.CHAIN_OF_THOUGHT
                break
            elif len(solution_step["ost"]) > 0 and "chain_of_thought" not in solution_step.keys():
                final_step_str = None
                for i, (step_id, step_text) in enumerate(solution_step["ost"].items()):
                    solution_trace_str += step_text.strip() + " "
                    if i == len(solution_step["ost"].items()) - 1:
                        final_step_str = step_text.strip()
                        reward_value = 0.0
                solution_trace_str = solution_trace_str.strip()
                end_node_type = Node_Type.ONE_STEP_THOUGHT
            else:
                continue
        elif 0 < item_idx < len(solution_trace) - 1:
            intermediate_step = solution_step["subanswer"]["text"].split("The answer is")[0].strip()
            solution_trace_str += intermediate_step + " "
            # concat trace for one-step thought step after subquestion
            if len(solution_step["ost"]) > 0 and "chain_of_thought" in solution_step.keys():
                for step_id, step_text in solution_step["ost"].items():
                    solution_trace_str += step_text.strip() + " "
                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["chain_of_thought"]["text"].strip()
                final_step_str = solution_step["chain_of_thought"]["text"].strip()
                reward_value = (
                    solution_step["chain_of_thought"]["value"] if "value" in solution_step["chain_of_thought"] else 0.0
                )
                end_node_type = Node_Type.CHAIN_OF_THOUGHT
                break
            elif len(solution_step["ost"]) > 0 and "chain_of_thought" not in solution_step.keys():
                final_step_str = None
                for i, (step_id, step_text) in enumerate(solution_step["ost"].items()):
                    solution_trace_str += step_text.strip() + " "
                    if i == len(solution_step["ost"].items()) - 1:
                        final_step_str = step_text.strip()
                        reward_value = 0.0
                solution_trace_str = solution_trace_str.strip()
                end_node_type = Node_Type.ONE_STEP_THOUGHT
        elif item_idx == len(solution_trace) - 1:
            assert item_idx > 0
            if (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost"]) == 0
                and "chain_of_thought" not in solution_step.keys()
            ):
                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["subanswer"]["text"].strip()
                final_step_str = solution_step["subanswer"]["text"].strip()
                reward_value = solution_step["subanswer"]["value"] if "value" in solution_step["subanswer"] else 0.0
                end_node_type = Node_Type.DIVIDE_AND_CONQUER
                break
            elif (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost"]) > 0
                and "chain_of_thought" not in solution_step.keys()
            ):
                intermediate_step = solution_step["subanswer"]["text"].split("The answer is")[0].strip()
                solution_trace_str += intermediate_step + " "
                final_step_str = None
                for i, (step_id, step_text) in enumerate(solution_step["ost"].items()):
                    solution_trace_str += step_text.strip() + " "
                    if i == len(solution_step["ost"].items()) - 1:
                        final_step_str = step_text.strip()
                        reward_value = 0.0
                solution_trace_str = solution_trace_str.strip()
                end_node_type = Node_Type.ONE_STEP_THOUGHT
            elif (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost"]) > 0
                and "chain_of_thought" in solution_step.keys()
            ):
                intermediate_step = solution_step["subanswer"]["text"].split("The answer is")[0].strip()
                solution_trace_str += intermediate_step + " "
                for step_id, step_text in solution_step["ost"].items():
                    solution_trace_str += step_text.strip() + " "
                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["chain_of_thought"]["text"].strip()
                final_step_str = solution_step["chain_of_thought"]["text"].strip()
                reward_value = (
                    solution_step["chain_of_thought"]["value"] if "value" in solution_step["chain_of_thought"] else 0.0
                )
                end_node_type = Node_Type.CHAIN_OF_THOUGHT
                break
            elif (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost"]) == 0
                and "chain_of_thought" in solution_step.keys()
            ):
                intermediate_step = solution_step["subanswer"]["text"].split("The answer is")[0].strip()
                solution_trace_str += intermediate_step + " "
                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["chain_of_thought"]["text"].strip()
                final_step_str = solution_step["chain_of_thought"]["text"].strip()
                reward_value = (
                    solution_step["chain_of_thought"]["value"] if "value" in solution_step["chain_of_thought"] else 0.0
                )
                end_node_type = Node_Type.CHAIN_OF_THOUGHT
                break
            elif "chain_of_thought" in solution_step.keys():
                assert len(solution_step["ost"]) == 0
                assert "subanswer" not in solution_step.keys()

                solution_trace_str += "Now we can answer the question: "
                solution_trace_str += solution_step["chain_of_thought"]["text"].strip()
                final_step_str = solution_step["chain_of_thought"]["text"].strip()
                reward_value = (
                    solution_step["chain_of_thought"]["value"] if "value" in solution_step["chain_of_thought"] else 0.0
                )
                end_node_type = Node_Type.CHAIN_OF_THOUGHT
                break
            else:
                import pdb

                pdb.set_trace()

    solution_trace_str = solution_trace_str.replace("Let's think step by step. ", "")
    solution_trace_str = "Let's think step by step. " + solution_trace_str

    return solution_trace_str.strip(), final_step_str.strip(), end_node_type, min(0, reward_value) + 1


def concat_rap_solution_trace(solution_trace: str):
    solution_trace_list = solution_trace.split("\n")
    answer_list = []
    for item in solution_trace_list:
        if item.startswith("Answer"):
            item = re.sub(r"Answer \d+\.\d+: ", "", item)
            final_step = item
            item = re.sub(r" The answer is \d+\.", "", item)
            answer_list.append(item)
    return " ".join(answer_list).strip(), final_step


def concat_subq_suba_trace(solution_trace: Dict[int, Dict[str, str]]):
    """Note that the solution trace might be subqs-subas and also one-step thought steps."""
    solution_trace_str = ["Let's think step by step."]
    final_step_str = ""
    end_node_type = None

    for item_idx, (subq_id, solution_step) in enumerate(solution_trace.items()):
        if item_idx == 0:
            assert len(solution_step["ost"]) == 0
            assert "chain_of_thought" not in solution_step.keys()
        elif 0 < item_idx < len(solution_trace) - 1:
            assert len(solution_step["ost"]) == 0
            solution_trace_str.append(
                {"subq": solution_step["subquestion"], "suba": solution_step["subanswer"]["text"]}
            )
        elif item_idx == len(solution_trace) - 1:
            assert item_idx > 0
            if (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost"]) == 0
                and "chain_of_thought" not in solution_step.keys()
            ):
                solution_trace_str.append(
                    {"subq": solution_step["subquestion"], "suba": solution_step["subanswer"]["text"]}
                )
                final_step_str = solution_step["subanswer"]["text"].strip()
                end_node_type = Node_Type.DIVIDE_AND_CONQUER
                break
            elif (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost"]) > 0
                and "chain_of_thought" not in solution_step.keys()
            ):
                assert False
            elif (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost"]) > 0
                and "chain_of_thought" in solution_step.keys()
            ):
                assert False
            elif (
                "subanswer" in solution_step.keys()
                and len(solution_step["ost"]) == 0
                and "chain_of_thought" in solution_step.keys()
            ):
                solution_trace_str.append(
                    {"subq": solution_step["subquestion"], "suba": solution_step["subanswer"]["text"]}
                )
                solution_trace_str.append(solution_step["chain_of_thought"]["text"])
                final_step_str = solution_step["chain_of_thought"]["text"].strip()
                end_node_type = Node_Type.CHAIN_OF_THOUGHT
                break
            elif "chain_of_thought" in solution_step.keys():
                assert len(solution_step["ost"]) == 0
                assert "subanswer" not in solution_step.keys()

                solution_trace_str.append(
                    solution_step["chain_of_thought"]["text"].replace("Let's think step by step.", "")
                )
                final_step_str = solution_step["chain_of_thought"]["text"].strip()
                end_node_type = Node_Type.CHAIN_OF_THOUGHT
                break
            else:
                import pdb
                pdb.set_trace()

    return solution_trace_str, final_step_str.strip(), end_node_type


def mask_solution_trace(
    solution_trace_str: str, num_return: int, left_boundary: float, right_boundary: float
) -> list[str]:
    if num_return == 1:
        interval = 0
    else:
        assert num_return > 1
        assert right_boundary >= left_boundary, f"right_boundary: {right_boundary} < left_boundary: {left_boundary}"
        interval = (right_boundary - left_boundary) / (num_return - 1)

    words_in_solution_trace = solution_trace_str.split(" ")
    ost_len = len(words_in_solution_trace)
    # Mask the solution trace string from least to most
    masked_solution_traces = []
    for i in range(num_return):
        prefix_part_ratio = left_boundary + i * interval
        prefix_part_num_words = math.ceil(ost_len * prefix_part_ratio)  
        prefix_part_str = " ".join(words_in_solution_trace[:prefix_part_num_words])
        masked_solution_traces.append(prefix_part_str)

    return masked_solution_traces


def mask_subq_suba_trace(solution_trace_str: list, num_return: int, evaluator: Evaluator) -> list[str]:
    if num_return == 1:
        interval = 0
    else:
        assert num_return > 1

    # Mask the solution trace string from least to most
    masked_solution_traces = []
    for i in range(1, len(solution_trace_str)):
        if "subq" in solution_trace_str[i]:
            prefix_part_str = (
                "\n".join(
                    [
                        f"Subquestion: {item['subq']}\nSubanswer: {item['suba']}" if "subq" in item else item
                        for item in solution_trace_str[:i]
                    ]
                )
                + "\nSubquestion: "
                + solution_trace_str[i]["subq"]
                + "\nSubanswer: "
            )
            curr_answer = evaluator.extract_answer_from_model_completion(solution_trace_str[i]["suba"])
            masked_solution_traces.append([prefix_part_str, curr_answer])
        else:
            prefix_part_str = (
                "\n".join(
                    [
                        f"Subquestion: {item['subq']}\nSubanswer: {item['suba']}" if "subq" in item else item
                        for item in solution_trace_str[:i]
                    ]
                )
                + "\nSubquestion: Now we can answer the question:"
            )
            curr_answer = evaluator.extract_answer_from_model_completion(solution_trace_str[i])
            masked_solution_traces.append([prefix_part_str, curr_answer])
    return masked_solution_traces


def make_hint(
    solution_trace: Dict[int, Dict[str, str]], node_type: Node_Type, new_subq=None, new_suba=None, new_ost=None
) -> str:
    if node_type in [Node_Type.DIVIDE_AND_CONQUER]:
        hint = ""

        for subquestion_id, solution_step in solution_trace.items():
            if subquestion_id == 0:
                continue

            assert subquestion_id > 0
            assert "subquestion" in solution_step.keys() and "subanswer" in solution_step.keys()

            hint += f"Hint " + str(subquestion_id) + " (You can partially refer to these imcomplete steps): " + solution_step["subquestion"]    
            hint += " "
            hint += solution_step["subanswer"]["text"]
            hint += "\n"

        if new_subq is not None and new_suba is not None:
            hint += f"Hint {len(solution_trace)}: " + new_subq + " " + new_suba

        hint = hint.strip("\n")
    elif node_type is Node_Type.ONE_STEP_THOUGHT:
        hint = "Hint (You can partially refer to these imcomplete steps): "
        last_tuple = list(solution_trace.items())[-1]
        last_tuple_recording = last_tuple[1]
        assert last_tuple_recording["ost"]
        for step_id, step_text in last_tuple_recording["ost"].items():
            hint += step_text + " "

        if new_ost is not None:
            hint += new_ost

        hint = hint.strip(" ")
    
    elif node_type is Node_Type.SELF_REFLECTION_AND_REFINE:
        hint = "Hint (You can partially refer to these imcomplete steps):\n" + solution_trace[0]['answers'][-1][1]  
    else:
        raise ValueError(f"Invalid node type: {node_type}.")

    return hint


def make_response_prefix(
    solution_trace: Dict[int, Dict[str, str]], node_type: Node_Type, new_subq=None, new_suba=None, new_ost=None
) -> str:
    if node_type in [Node_Type.DIVIDE_AND_CONQUER]:
        response_prefix = ""
        answer_marker = "The answer is"

        for subquestion_id, solution_step in solution_trace.items():
            if subquestion_id == 0:
                continue

            assert subquestion_id > 0
            assert "subquestion" in solution_step.keys() and "subanswer" in solution_step.keys()

            response_prefix += solution_step["subanswer"]["text"].split(answer_marker)[0]
            response_prefix += " "

        if new_subq is not None and new_suba is not None:
            response_prefix += new_suba.split(answer_marker)[0]

        response_prefix = response_prefix.strip(" ")
    elif node_type is Node_Type.ONE_STEP_THOUGHT:
        response_prefix = ""

        last_tuple = list(solution_trace.items())[-1]
        last_tuple_recording = last_tuple[1]
        if "ost" in last_tuple_recording.keys():
            for step_id, step_text in last_tuple_recording["ost"].items():
                response_prefix += step_text + " "

        if new_ost is not None:
            response_prefix += new_ost

        response_prefix = response_prefix.strip(" ")
    elif node_type is None and solution_trace is None:
        response_prefix = ""
    else:
        raise ValueError(f"Invalid node type: {node_type}.")

    think = "Let's think step by step. "
    return think + response_prefix if think not in response_prefix else response_prefix


def find_valid_solution_nodes(root_node):  
    valid_solution_nodes = []

    def recursion(node):  
        if node.is_valid_solution_node(): 
            valid_solution_nodes.append(node)
            return

        if not node.children:  #! no children
            return

        try:
            for child in node.children:
                recursion(child)
        except:
            print(node.children, node.children.node_type, node.children.solution_trace[0]['path'])

    recursion(root_node)

    return valid_solution_nodes


def find_best_solution(root_node, evaluator, enable_potential_score=False):
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None

    def extract_solution_from_node(node):
        if node.node_type is Node_Type.DIVIDE_AND_CONQUER:
            return node.subanswer
        elif node.node_type is Node_Type.CHAIN_OF_THOUGHT:
            return node.chain_of_thought
        else:
            return None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]

    def calculate_potential_score_for_solution_node(node):
        model_answer = evaluator.extract_answer_from_model_completion(extract_solution_from_node(node))
        potential_answers_history = node.potential_answers_history  # {depth -> [potential answers]}
        assert potential_answers_history[node.depth] is None

        potential_score = 1
        for depth, depth_potential_answers in potential_answers_history.items():
            if depth < node.depth:
                depth_score = sum(
                    evaluator.check_answers_equiv(dpa, model_answer) for dpa in depth_potential_answers
                ) / len(depth_potential_answers)
                potential_score *= depth_score

        node.set_potential_score(potential_score)
        return potential_score

    prior_weights = (
        [calculate_potential_score_for_solution_node(node) for node in solution_nodes]
        if enable_potential_score
        else None
    )
    top_answer, top_completion, top_completion_id, top_confidence = evaluator.find_most_confident_answer(
        solutions, prior_weights
    )
    return top_answer, top_completion, top_confidence, solution_nodes[top_completion_id], solution_nodes


def stochastic_find_best_solution(
    root_node,
    evaluator,
    enable_potential_score,
):
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None, None, None, None, None

    def extract_solution_from_node(node):   
        if node.node_type is Node_Type.DIVIDE_AND_CONQUER:
            return node.subanswer
        elif node.node_type is Node_Type.CHAIN_OF_THOUGHT:
            return node.chain_of_thought
        elif node.node_type is Node_Type.ONE_STEP_THOUGHT and 'answer is:' in node.ost:
            return node.ost
        else:
            return None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]

    def calculate_potential_score_for_solution_node(node):
        model_answer = evaluator.extract_answer_from_model_completion(extract_solution_from_node(node))
        potential_answers_history = node.potential_answers_history  # {depth -> [potential answers]}
        assert potential_answers_history[node.depth] is None

        potential_score = 1
        for depth, depth_potential_answers in potential_answers_history.items():
            if depth < node.depth:
                depth_score = sum(
                    evaluator.check_answers_equiv(dpa, model_answer) for dpa in depth_potential_answers
                ) / len(depth_potential_answers)
                potential_score *= depth_score

        node.set_potential_score(potential_score)
        return potential_score

    prior_weights = (
        [calculate_potential_score_for_solution_node(node) for node in solution_nodes]
        if enable_potential_score
        else None
    )
    top_answer, top_completion, top_completion_id, top_confidence = evaluator.stochastic_find_most_confident_answer(
        completions=solutions, prior_weights=prior_weights
    )
    return top_answer, top_completion, top_confidence, solution_nodes[top_completion_id], solution_nodes, solutions



def filtered_output_list(io_output_list, user_question, sim_model, sim_model_name):
    filtered_by_sim = []
    sim_list = []
    
    for io_output in io_output_list:
        sentences = [user_question, io_output]
        
        if sim_model_name == "bge":
            embeddings = sim_model.encode(sentences, batch_size=2)['dense_vecs']
            threshold = 0.5
        elif sim_model_name == "mpnet":
            embeddings = sim_model.encode(sentences)
            threshold = 0.4
        
        similarity = embeddings[0] @ embeddings[1].T
        
        # Check if the condition is met, if so, add to filtered_by_sim
        if similarity > threshold or "answer is" in io_output[-50:].lower():
            if io_output.strip() not in filtered_by_sim:
                filtered_by_sim.append(io_output.strip())
                sim_list.append(similarity)
        else:
            print(f"No similarity\nQuestion: {user_question}\nAnswer: {io_output}\nSimilarity: {similarity}")
    
    return filtered_by_sim, sim_list


def count_steps(x):
    pattern = r'step \d+'
    matches = re.findall(pattern, x)
    return len(matches)



def process_sentence(a):
    a = re.sub(r"^Let's think step by step\.?", "", a).strip()
    
    answer_match = re.search(r"The answer is:?.*", a)
    if answer_match:
        answer = answer_match.group(0)
        a = a.replace(answer, "").strip()  # Remove the answer part from the original sentence
    else:
        answer = ""

    # Split by sentence
    sentences = re.split(r'(?<=\.)\s+', a)  

    # Generate each step one by one
    steps = []
    for i, sentence in enumerate(sentences, 1):
        steps.append(f"Step {i}: {sentence.strip()}")

    result = " ки\n".join(steps) + " ки"  

    if answer:
        result += f" {answer} ки"

    return result

def calculate_score(scores, score_type, input_id, step_tag_id):
    """
    Calculate the final score based on the given scores, selecting the method according to score_type
    """
    step_scores = scores[input_id == step_tag_id]
    
    if score_type == 'average':
        return step_scores.mean()  
    elif score_type == 'product':
        return step_scores.prod()  
    elif score_type == 'max':
        return step_scores.max() 
    elif score_type == 'min':
        return step_scores.min()  
    else:
        return step_scores  


def get_top_half_most_common(score_dict):
    """
    Select the top half of the scores from the dictionary, and return the key that appears most frequently among these keys.

    Parameters:
        score_dict (dict): A dictionary where the keys are strings and the values are scores (numeric type).

    Returns:
        str: The key that appears most frequently among the top half of the scores.
    """
    # Sort by score and select the top half
    sorted_by_score = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    half_length = len(sorted_by_score) // 2  
    scores_first_half = [item[0] for item in sorted_by_score[:half_length]]
    
    # Count the most frequent key
    most_common_solution_half_con = Counter(scores_first_half).most_common(1)[0][0]
    return most_common_solution_half_con


def prm_obtain_answer(type, question, model_solutions, model, tokenizer, candidate_tokens, step_tag_id, device, score_type):
    best_score = None
    best_answer = None
    best_score_min = None
    best_answer_min = None
    step_ans = {}
    scores_dict = {}
    scores_dict_min = {}
    for j, ans in enumerate(model_solutions):
        try:
            x = process_sentence(ans)
        except:
            continue
        step_ans[x] = ans
        input_for_prm = f"{question} {x}"
        input_id = torch.tensor([tokenizer.encode(input_for_prm)], device=device)
        with torch.no_grad():
            logits = model(input_id).logits[:,:,candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0] 
            
            result = calculate_score(scores, score_type, input_id, step_tag_id)  
            result_min = calculate_score(scores, "min", input_id, step_tag_id)
            scores_dict[ans] = result.cpu().item()
            scores_dict_min[ans] = result_min.cpu().item()
            
            if best_score is None or result > best_score:
                best_score = result.cpu().item()
                best_answer = step_ans[x]
            
            if best_score_min is None or result_min > best_score_min:
                best_score_min = result_min.cpu().item()
                best_answer_min = step_ans[x]

        
    if len(list(scores_dict.keys())) == 1:
        assert len(list(scores_dict_min.keys())) == 1
        score_half_product = list(scores_dict.keys())[0]
        score_half_min = list(scores_dict_min.keys())[0]
    else:
        score_half_product = get_top_half_most_common(scores_dict)
        score_half_min = get_top_half_most_common(scores_dict_min)
    
    return best_answer, best_answer_min, score_half_product, score_half_min