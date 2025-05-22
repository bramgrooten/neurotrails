import torch
import torch.nn as nn
from peft_pretraining.modeling_llama import LlamaForCausalLM, LlamaConfig


class FullEnsembleLlama(nn.Module):
    """
    Just a simple ensemble of LLaMA models.
    """
    def __init__(self, config: LlamaConfig, num_ensemble: int = 3):
        super().__init__()
        self.num_ensemble = num_ensemble
        for i in range(self.num_ensemble):
            model_i = LlamaForCausalLM(config)
            setattr(self, f'model_{i}', model_i)

    def forward(self, input_ids, attention_mask=None):
        """
        Returns: list[logits_network_0, logits_network_1, ...]
        """
        logits_list = []
        for i in range(self.num_ensemble):
            output = getattr(self, f'model_{i}')(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
            logits = output[0]
            logits_list.append(logits)
        return logits_list
     
    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        state_dict=None,
        save_function=torch.save,
        push_to_hub: bool = False,
        max_shard_size: str = "10GB",
        safe_serialization: bool = False,
        variant=None,
        **kwargs
    ):
        """
        Save the model's state_dict and configuration to a directory.
        This method mimics the PreTrainedModel.save_pretrained method to be compatible
        with the training loop.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the config
        self.config.architectures = [self.__class__.__name__]
        self.config.num_ensemble = self.num_ensemble
        self.config.save_pretrained(save_directory)
        
        # Get state dict
        if state_dict is None:
            state_dict = self.state_dict()
        
        # Save the model
        if max_shard_size is None or max_shard_size == "":
            # No sharding, save as a single file
            model_path = os.path.join(save_directory, WEIGHTS_NAME)
            save_function(state_dict, model_path)
        else:
            # Sharded save - simplified version
            model_path = os.path.join(save_directory, WEIGHTS_NAME)
            save_function(state_dict, model_path)
        
        # Save model structure info
        model_structure = {
            "type": "FullEnsembleLlama",
            "num_ensemble": self.num_ensemble,
            "total_layers": self.config.num_hidden_layers,
        }
        
        with open(os.path.join(save_directory, "model_structure.json"), "w") as f:
            json.dump(model_structure, f, indent=2)
        
        return save_directory

if __name__ == "__main__":
    # Suppose we have a tiny LLaMA config
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=1024,
    )
    multi_model = FullEnsembleLlama(config, heads=3)

    # Quick test
    dummy_input = torch.randint(0, 32000, (2, 16))  # (batch=2, seq=16)
    out = multi_model(dummy_input)
    print(f"Output:")
    print(out)
    print(f"Returned {len(out)} outputs. Each is shape={out[0].shape} (batch=2, seq=16, vocab=32000)")

