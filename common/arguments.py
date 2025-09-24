# Licensed under the MIT license.

import os, json, torch, math
from argparse import ArgumentParser
from datetime import datetime

torch.cuda.empty_cache()


def get_parser():
    parser = ArgumentParser()
    allowed_apis = ["together", "huggingface", "llama", "vllm", "debug", "gpt3.5-turbo", "gpt-4o"]
    parser.add_argument("--api", type=str, choices=allowed_apis, default="vllm", help=f"API to use: Choose from {allowed_apis}.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--if_use_cards", type=lambda x: (str(x).lower() == 'true'), default=True)

    #! LLM settings
    parser.add_argument("--model_ckpt", default="meta-llama/Meta-Llama-3-8B-Instruct") 
    parser.add_argument("--sim_model", default="sentence-transformers/all-mpnet-base-v2") 
    parser.add_argument("--if_entailment", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--entailment_model", default="facebook/bart-large-mnli")
    parser.add_argument("--model_parallel", default=False)
    parser.add_argument("--half_precision", default=False)

    parser.add_argument("--max_tokens", type=int, default=1024, help="max_tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="temperature")
    parser.add_argument("--top_k", type=int, default=40, help="top_k")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p")
    parser.add_argument("--num_beams", type=int, default=1, help="num_beams")
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    parser.add_argument("--test_batch_size", type=int, default=1)  
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    #! prompt settings
    parser.add_argument("--prompts_root", default="prompts")

    #! dataset settings
    parser.add_argument("--data_root", default="data")
    allowed_dataset_names = ["MATH", "GSM8K", "GSM8KHARD", "STG", "SVAMP", "AMC", "GPQA"]
    parser.add_argument(
        "--dataset_name",
        default = "MATH",
        choices=allowed_dataset_names,
        help=f"Test dataset name: Choose from {allowed_dataset_names}.",
    )
    parser.add_argument("--test_json_filename", type=str, default="test_all", choices=["test_all", "train_all"])

    #! outputs settings
    parser.add_argument("--run_outputs_root", type=str, default="run_outputs")
    parser.add_argument("--eval_outputs_root", type=str, default="eval_outputs")
    
    
    # arguments in 'do_generate.py'
    # parser.add_argument("--gpu_id", type=str, default='0,3')
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--reuse_rollouts", type=int, default=1)
    parser.add_argument("--num_cards", type=int, default=5)
    parser.add_argument("--attribute_type", type=str, default='condition', choices=['condition', 'subquestion', 'confidence', 'semantic'])
    parser.add_argument("--similarity_type", type=str, default='max', choices=['average', 'max'])
    parser.add_argument("--max_depth_allowed", type=int, default=5)

    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)  
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const") 
    parser.add_argument("--save_tree", default=False)

    # action settings
    parser.add_argument("--if_ost_select", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--num_ost", type=int, default=3)
    parser.add_argument("--num_cot", type=int, default=32)
    parser.add_argument("--num_dc", type=int, default=3, help="Number of trials for 'divide and conquer'.")
    parser.add_argument("--num_dc_votes", type=int, default=10, help="Number of trials for subquestions of each question.")

    parser.add_argument("--disable_sa", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 1: system analysis")
    parser.add_argument("--disable_ost", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 2: one-step thought")
    parser.add_argument("--disable_cot", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 3: chain-of-thought")
    parser.add_argument("--disable_dc", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 4: divide and conquer")
    parser.add_argument("--disable_srr", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 5: self-reflection and refinement")

    parser.add_argument("--enable_potential_score", default=False)
    parser.add_argument("--disable_answer_selection", default=False)
    
    return parser


def post_process_args(args):
    model_name = args.model_ckpt.split("/")[-1]
    sim_model = args.sim_model.split("/")[-1]
    args.file = args.test_json_filename.split("_")[0]
    suffix = f"{args.file}_|_rolls_{args.num_rollouts}_|_reuse_train_{args.if_use_cards}_|_sa_{not args.disable_sa}_|_ost_{not args.disable_ost}_select_{args.if_ost_select}_n_{args.num_ost}_|_cot_{not args.disable_cot}_|_dc_{not args.disable_dc}_|_srr_{not args.disable_srr}_|_{sim_model}"
    
    if args.if_use_cards:
        structure_dir = "structure"

        if args.attribute_type in ["condition", "subquestion"]:
            suffix = f"{args.file}_|_rolls_{args.num_rollouts}_|_reuse_train_{args.if_use_cards}_|_reuse_rolls_{args.reuse_rollouts}_|_{args.attribute_type}_|_reuse_paths_{args.num_cards}_|_sa_{not args.disable_sa}_|_ost_{not args.disable_ost}_select_{args.if_ost_select}_n_{args.num_ost}_|_cot_{not args.disable_cot}_|_dc_{not args.disable_dc}_|_srr_{not args.disable_srr}"
            if "MATH" in args.dataset_name:
                args.reuse_dir = f"./run_outputs/{structure_dir}/MATH_|_{args.attribute_type}_|_test_question_path_llama-3-8b-instruct.json"
            elif "GSM" in args.dataset_name and "HARD" not in args.dataset_name:
                args.reuse_dir = f"./run_outputs/{structure_dir}/GSM8K_|_{args.attribute_type}_|_test_question_path_llama-3-8b-instruct.json"
            else:
                args.reuse_dir = f"./run_outputs/{structure_dir}/{args.dataset_name}_|_{args.attribute_type}_|_test_question_path_llama-3-8b-instruct.json"
        
        elif args.attribute_type == "semantic":
            suffix = f"{args.file}_|_rolls_{args.num_rollouts}_|_reuse_train_{args.if_use_cards}_|_reuse_rolls_{args.reuse_rollouts}_|_{args.attribute_type}_{args.similarity_type}_|_reuse_paths_{args.num_cards}_|_sa_{not args.disable_sa}_|_ost_{not args.disable_ost}_select_{args.if_ost_select}_n_{args.num_ost}_|_cot_{not args.disable_cot}_|_dc_{not args.disable_dc}_|_srr_{not args.disable_srr}"
            args.reuse_dir = f"./run_outputs/{structure_dir}/{args.dataset_name}_|_{args.attribute_type}_{args.similarity_type}_|_test_question_path.json"    
    
    
    if args.mode == "run":
        args.run_outputs_dir = os.path.join(
            args.run_outputs_root,
            args.dataset_name,
            model_name,
            suffix,
            f"{datetime.now().strftime('%m-%d_%H-%M')}"
        )
        os.makedirs(args.run_outputs_dir, exist_ok=True)
    elif args.mode == "eval":
        args.eval_outputs_dir = os.path.join(
            args.eval_outputs_root,
            args.dataset_name,
            model_name,
            suffix,
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        )
        os.makedirs(args.eval_outputs_dir, exist_ok=True)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    args.answer_sheets_dir = os.path.join(args.run_outputs_dir, "answer_sheets")
    os.makedirs(args.answer_sheets_dir, exist_ok=True)

    # Check GPU
    # import pdb
    # pdb.set_trace()
    num_gpus = torch.cuda.device_count()
    cuda_devices = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    assert len(cuda_devices) > 0, "No GPU available."
    args.cuda_0 = cuda_devices[0]
    args.cuda_1 = cuda_devices[1] if len(cuda_devices) > 1 else None
    args.cuda_2 = cuda_devices[2] if len(cuda_devices) > 2 else None
    args.cuda_3 = cuda_devices[3] if len(cuda_devices) > 3 else None

    if len(cuda_devices) == 1:
        if args.cuda_0 == "NVIDIA A100-SXM4-40GB" and not args.half_precision:
            print("Warning! A100-SXM4-40GB is used, but half_precision is not enabled.")
    
    os.environ["WORLD_SIZE"] = str(num_gpus)
    args.tensor_parallel_size = num_gpus
    if num_gpus > 1:
        os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'

    return args


def save_args(args):
    # Save args as json
    if args.mode == "run":
        with open(os.path.join(args.run_outputs_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    elif args.mode == "eval":
        with open(os.path.join(args.eval_outputs_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)