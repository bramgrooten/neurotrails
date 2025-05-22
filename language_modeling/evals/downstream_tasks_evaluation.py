import os
import argparse
import json
import torch
import torch.utils.data
import transformers
from transformers import AutoConfig, AutoTokenizer, modeling_utils
from loguru import logger
from peft_pretraining import training_utils, args_utils
from peft_pretraining.modeling_llama import LlamaForCausalLM
from evaluation_architectures import TreeNetLlama
from evaluation_architectures import FullEnsembleLlama
from lm_eval.models import huggingface
from lm_eval import simple_evaluate
from safetensors.torch import load_file
transformers.logging.set_verbosity_error()


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=2_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--no_save", action="store_true", help="Do not save the model")
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=1.0)   
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--single_gpu", default=False, action="store_true", help="Disable DDP and use single GPU")
    parser.add_argument("--console_log", type=str, default="default")
    parser.add_argument('--wandb_mode', type=str, default="online", choices=["online", "offline", "disabled"])
    
    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")

    # NeuroTrails args
    parser.add_argument('--num_ensemble', type=int, default=1, help='How many models/heads in the ensemble. Default=1')
    parser.add_argument('--blocks_in_head', type=int, default=5, help='How many backbone blocks to be part of each head. Default=5')
    parser.add_argument('--full_ensemble', default=False, action="store_true", help='Use full ensemble instead of tree ensemble')

    # Sparsity args
    parser.add_argument('--density', type=float, default=1.0, help='The pruning rate / death rate.')
    parser.add_argument('--fc_density', type=float, default=1.0, help='The density of the overall sparse network.')
    parser.add_argument('--ddt', action='store_true', default=False, help='Enable dynamic dense training. Default: False.')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between mask update')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold, CS_death.')
    parser.add_argument('--reinit', type=str, default='no', help='Weight reinitialization mode. Choose from: no, zero.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death_rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--decay_schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--init_prune_epoch', type=int, default=8, help='The pruning rate / death rate.')
    parser.add_argument('--final_prune_epoch', type=int, default=125, help='The density of the overall sparse network.')
    parser.add_argument('--snip', action='store_true', help='Enable snip initialization. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--pop', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='Multi_Output', help='sparse initialization')
    parser.add_argument('--mix', type=float, default=0.0)
    parser.add_argument('--method', type=str, default='DST', help='method name: DST, dynamic_pruning, GMP, NTK_path')

    # Evaluation task selection
    parser.add_argument('--eval_tasks', type=str, default='mmlu', choices=['mmlu', 'other'], 
                        help='Which evaluation tasks to run: only mmlu or all other tasks')

    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args



def load_model(model_path, args='sd',device="cuda:0", norm_type="pre"):
    # Set environment variables
    os.environ['NORM_TYPE'] = norm_type
    if 'POST_NUM' not in os.environ:
        os.environ['POST_NUM'] = '3'  # Default value, adjust as needed
    
    logger.info(f"Loading model from {model_path} for evaluation")
    
    # Load model configuration
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        model_config = AutoConfig.from_pretrained(model_path)
    else:
        raise ValueError(f"Config not found at {config_path}")
    
    # Check if there's any metadata to determine if it's an ensemble model
    metadata_path = os.path.join(model_path, "training_state.json")
    num_ensemble = args.num_ensemble
    blocks_in_head = args.blocks_in_head
    full_ensemble = args.full_ensemble
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                # Try to extract ensemble info if available
                if "run_config" in metadata and "num_ensemble" in metadata["run_config"]:
                    num_ensemble = metadata["run_config"]["num_ensemble"]
                    if "blocks_in_head" in metadata["run_config"]:
                        blocks_in_head = metadata["run_config"]["blocks_in_head"]
                    if "full_ensemble" in metadata["run_config"]:
                        full_ensemble = metadata["run_config"]["full_ensemble"]
        except:
            logger.warning("Could not parse metadata file for ensemble information")
    
    # Create the appropriate model type
    if num_ensemble == 1:
        model = LlamaForCausalLM(model_config)
    else:
        if full_ensemble:
            print('full')
            model = FullEnsembleLlama(model_config, num_ensemble)
        else:
            model = TreeNetLlama(model_config, num_ensemble, blocks_in_head,device)
    
    # Load the model weights
    if os.path.exists(os.path.join(model_path, "model.safetensors")):
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        model.load_state_dict(state_dict)
    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu"))
    else:
        # Try to load sharded model if available
        try:
            model = modeling_utils.load_sharded_checkpoint(model, model_path)
        except:
            raise ValueError(f"Could not find model weights in {model_path}")
    
    # Move model to device and set to eval mode
    model = model.to(device=device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)
    model.eval()
    if num_ensemble != 1:
        model.device = device if isinstance(device, torch.device) else torch.device(device)
    return model

    
def main(args):

    model = load_model(
        # model_path='checkpoints/single/single/model/',
        # model_path='checkpoints/treenet/model/',
        # model_path='checkpoints/ensemble/model/',
        model_path='checkpoints/neurotrails/model/',
        args=args,
        device='cuda',
        norm_type='pre'
    )

    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
    lm_obj = huggingface.HFLM(pretrained=model, backend='causal', tokenizer=tokenizer, batch_size=args.batch_size)

    # evaluate
    if args.eval_tasks == 'other':
        results = simple_evaluate(
            model=lm_obj,
            tasks=["boolq", "arc_easy", "piqa", "hellaswag", "openbookqa", "winogrande"],
            num_fewshot=0
        )
        print(f"BoolQ Accuracy: {results['results']['boolq']['acc,none']:.4f}")
        print(f"ARC-Easy Accuracy: {results['results']['arc_easy']['acc,none']:.4f}")
        print(f"PIQA Accuracy: {results['results']['piqa']['acc,none']:.4f}")
        print(f"Hellaswag Accuracy: {results['results']['hellaswag']['acc,none']:.4f}")
        print(f"OpenBookQA Accuracy: {results['results']['openbookqa']['acc,none']:.4f}")
        print(f"Winogrande Accuracy: {results['results']['winogrande']['acc,none']:.4f}")
    else:  # MMLU only
        results = simple_evaluate(
            model=lm_obj,
            tasks=["mmlu"],
            num_fewshot=0
        )
        print(f"MMLU Accuracy: {results['results']['mmlu']['acc,none']:.4f}")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
