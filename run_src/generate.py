# Licensed under the MIT license.

import sys
import time
import os, json, time
from tqdm import tqdm

sys.path.append(".")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from common.utils import fix_seeds, setup_model_parallel, read_json
from common.arguments import get_parser, post_process_args, save_args
from MCTS_for_reasoning import Generator, search_for_answers
from eval_src.Evaluator import *
from collections import Counter
import concurrent.futures
import multiprocessing


def main(args):
    # import pdb
    # pdb.set_trace()
    max_threads = 32
    multiprocessing.set_start_method('spawn', force=True)
    fix_seeds(args.seed)
    args.local_rank, args.world_size = 0, 8
    args.tensor_parallel_size = 8

    test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist."
    data_item_list = read_json(test_file)
    
    # import pdb
    # pdb.set_trace()
    if "GSM" in args.dataset_name and "HARD" not in args.dataset_name:
        dataset_name = "GSM8K"
    elif "MATH" in args.dataset_name:
        dataset_name = "MATH"
    else:
        dataset_name = args.dataset_name
    
    evaluator = eval(f"{dataset_name}Evaluator()") 

    tokenizer, model = None, None
    if args.api == "huggingface":
        from models.HuggingFace_API import load_HF_model

        tokenizer, model = load_HF_model(args.model_ckpt)
    elif args.api == "vllm":
        from models.vLLM_API import load_vLLM_model

        tokenizer, model = load_vLLM_model(args.model_ckpt, args.seed, args.tensor_parallel_size, args.half_precision)
    elif args.api in ["gpt3.5-turbo", "gpt-4o"]:
        from models.OpenAI_API import load_OpenAI_model

        tokenizer, model = load_OpenAI_model(args.model_ckpt)
    generator = Generator(args, tokenizer, model, evaluator)
    
    if args.if_use_cards:
        args.reason_structure = read_json(args.reuse_dir)
    
    total_correct = 0
    total_correct_con = 0
    total_correct_random = 0
    total_correct_limit = 0
    num_tested = 0
    start_time = time.time()
    model_name = args.model_ckpt.split("/")[-1]
        
    def init_acc_subject_and_num(subjects, levels=None):
        acc_subject = {subject: 0 for subject in subjects}
        if levels is not None:
            acc_num = {subject: 0 for subject in subjects}
            for level in levels:
                acc_num[str(level)] = 0
        else:
            acc_num = {subject: 0 for subject in subjects}
        return acc_subject, acc_num

    acc_level, acc_subject, acc_num = {}, {}, {}
    if "MATH" in dataset_name:
        subjects = [
            "algebra", "prealgebra", "number theory", "counting & probability", 
            "geometry", "intermediate algebra", "precalculus"
        ]
        levels = [1, 2, 3, 4, 5]
        acc_subject, acc_num = init_acc_subject_and_num(subjects, levels)
        acc_level = {str(i): 0 for i in levels}

    elif "GPQA" in dataset_name:
        subjects = [
            "physics (general)", "chemistry (general)", "organic chemistry", "inorganic chemistry", 
            "quantum mechanics", "electromagnetism and photonics", "high-energy particle physics", 
            "genetics", "astrophysics", "molecular biology", "relativistic mechanics", 
            "optics and acoustics", "condensed matter physics"
        ]
        acc_subject, acc_num = init_acc_subject_and_num(subjects)
    
    # import pdb
    # pdb.set_trace()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i, data_item in enumerate(
            (pbar := tqdm(data_item_list, disable=args.local_rank > 0 or args.verbose, position=1)) 
        ):
            try:
                problem_id = data_item.get("id", 0)
                problem = data_item.get("problem") or data_item.get("question")
                gt_solution = data_item.get("solution") or data_item.get("answer")
            except KeyError:
                problem_id, problem, gt_solution = 0, None, None

            if problem is not None and gt_solution is not None:
                if "AMC" in dataset_name:
                    gt_answer = gt_solution
                else:
                    gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)
            else:
                gt_answer = None  


            js = {
                "id": problem_id,
                "problem": problem,
                "model_completion": None,
                "model_answer": None,
                "all_model_completions": {},
                "gold_solution": gt_solution,
                "gold_answer": gt_answer,
                "subacc": {},
                "level": None,
                "subject": None,
            }
            if "MATH" in dataset_name:
                level = data_item["extra_info"]["level"]
                subject = data_item["extra_info"]["subject"]
                js["level"] = level
                js["subject"] = subject
            elif "GPQA" in dataset_name:
                subject = data_item["subdomain"]
                js["subject"] = subject
            
            
            model_solutions, stopping_id, model_all_solutions = [], -1, []
            futures.append(executor.submit(search_for_answers, args, evaluator, js, problem, i, gt_answer, generator))

        # import pdb
        # pdb.set_trace()
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                js, model_solutions, model_all_solutions, model_best_path, correct, correct_limit, correct_con, correct_random = future.result()
            except:
                continue
            print("get result!")
            with open(os.path.join(args.answer_sheets_dir, f"Question {i:04d} - Answer.json"), "w") as f:
                json.dump(js, f)
            num_tested += 1
            total_correct += int(correct)
            total_correct_limit += int(correct_limit)
            total_correct_con += int(correct_con)
            total_correct_random += int(correct_random)
            if "MATH" in dataset_name:
                level = js["level"]
                subject = js['subject']
                acc_num[str(level)] += 1
                acc_num[subject.lower()] += 1
                acc_subject[subject.lower()] += int(correct)
                acc_level[str(level)] += int(correct)
            elif "GPQA" in dataset_name:
                subject = js['subject']
                acc_num[subject.lower()] += 1
                acc_subject[subject.lower()] += int(correct)

            if any(x in dataset_name for x in ("MATH", "GPQA")):
                for key in acc_subject:
                    js["subacc"][key] = acc_subject[key] / acc_num[key] if acc_num[key] else 0.0
                    print(f"{key}: {js['subacc'][key]:.4f}")
                
                if "MATH" in dataset_name:
                    for key in acc_level:
                        js["subacc"][str(key)] = acc_level[str(key)] / acc_num[str(key)] if acc_num[key] else 0.0
                        print(f"{key}: {js['subacc'][str(key)]:.4f}")
                
            
            print(f"accuracy: {total_correct/(num_tested):.4f}")
            print(f"consistency: {total_correct_con/(num_tested):.4f} | random: {total_correct_random/(num_tested):.4f}")
            print(f"limit accuracy: {total_correct_limit/(num_tested):.4f}")            
            if args.if_use_cards:
                print(f"model: {model_name} | dataset: {args.dataset_name} | file: {args.file} | difficulty: {args.attribute_type}\n")
            else:
                print(f"model: {model_name} | dataset: {args.dataset_name} | file: {args.file}\n")

            with open(os.path.join(args.run_outputs_dir, "intermediate_result.txt"), "w") as f:
                if not args.disable_answer_selection:
                    f.write(f"Num tested: {num_tested}\n")
                    f.write(f"Num correct: {total_correct}\n")
                    f.write(f"Acc: {total_correct/(num_tested)}\n")
            

    end_time = time.time()

    total_seconds = end_time - start_time
    avg_seconds = (end_time - start_time) / num_tested

    total_hours = int(total_seconds // 3600)
    total_minutes = int((total_seconds % 3600) // 60)
    total_seconds_remainder = total_seconds % 60

    avg_minutes = int(avg_seconds // 60)
    avg_seconds_remainder = avg_seconds % 60
    
    if not args.disable_answer_selection:
        print(f"==> Acc: {total_correct/(num_tested)}")
    print("====================================================")
    print(f"==> Accuracy: {total_correct/(num_tested):.4f} | All Accuracy: {total_correct_limit/(num_tested):.4f}")
    print(f"==> Consistency: {total_correct_con/(num_tested):.4f} | random: {total_correct_random/(num_tested):.4f}")
    print(f"==> Total time: {total_hours}h {total_minutes}m {total_seconds_remainder:.2f}s, Avg time: {avg_minutes}m {avg_seconds_remainder:.2f}s")
    print("----------------------------------------------------")
    print(f"model: {args.model_ckpt} | dataset: {args.dataset_name} | num_rollouts: {args.num_rollouts} | file: {args.file}")
    print(f"Output dir: {args.run_outputs_dir}")

    with open(os.path.join(args.run_outputs_dir, "final_result.txt"), "w") as f:
        if not args.disable_answer_selection:
            f.write(f"Num tested: {num_tested}\n")
            f.write(f"Num correct: {total_correct}\n")
            f.write(f"Acc: {total_correct/(num_tested)}\n")
            f.write(f"Acc: {total_correct_limit/(num_tested)}\n")
        f.write(f"Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s\n")


if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    #! ----------------------------------------------------------------------------
    if "GSM" in args.dataset_name and "HARD" not in args.dataset_name:
        prompts_dir = os.path.join(args.prompts_root, "GSM8K")
    elif "MATH" in args.dataset_name:
        prompts_dir = os.path.join(args.prompts_root, "MATH")
    else:
        prompts_dir = os.path.join(args.prompts_root, args.dataset_name)
    
    args.rephrasing_prompt_template_path = os.path.join(prompts_dir, "list_conditions_prompt_template.txt")
    args.fewshot_cot_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
    args.fewshot_cot_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_config.json")
    args.fewshot_cot_check_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_check_prompt.txt")
    args.fewshot_cot_check_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_check_config.json")

    args.fewshot_ost_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
    args.fewshot_ost_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_config.json")
    
    args.fewshot_pot_refine_prompt_path = os.path.join(prompts_dir, "fewshot_pot", "fewshot_pot_refine_prompt.txt")
    args.fewshot_pot_refine_config_path = os.path.join(prompts_dir, "fewshot_pot", "fewshot_pot_refine_config.json")
    
    args.fewshot_refine_sum_prompt_path = os.path.join(prompts_dir, "fewshot_thought_reprocess", "fewshot_refine_prompt.txt")
    args.fewshot_refine_sum_config_path = os.path.join(prompts_dir, "fewshot_thought_reprocess", "fewshot_refine_config.json")
        
    args.decompose_template_path = os.path.join(prompts_dir, "decompose", "decompose_template.json")
    args.decompose_prompt_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    if not args.disable_sa:
        args.rephrasing_prompt_template_path = os.path.join(prompts_dir, "rephrasing_prompt_template.txt")
        args.fewshot_cot_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
        args.fewshot_ost_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
        args.decompose_prompt_rephrased_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    args.mode = "run"
    args = post_process_args(args)
    print(args)
    save_args(args)
    main(args)
