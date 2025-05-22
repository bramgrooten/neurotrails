import os
import json
import torch
import torch.nn as nn
from transformers.utils import WEIGHTS_NAME
from peft_pretraining.modeling_llama import LlamaForCausalLM, LlamaConfig


class TreeNetLlama(nn.Module):
    """
    Example "TreeNet (i.e. multi-head) LLaMA":
      - Make (heads + 1) distinct LlamaForCausalLM models, each with random init.
      - One copy is designated as the "shared" backbone (layers[:shared_count]).
      - Each other copy is a "head" (layers[shared_count:]).
      - Then, combine them into a single forward pass.
    The main differences between this architecture and the one used for training is averaging outputs right inside here 
    instead of doing it in a training loop, wrapping model output and adding tie_weights function, which are all needed
    for being suitable for lm_eval.
    """
    def __init__(self, config: LlamaConfig, heads: int = 3, blocks_in_head: int = 5, device: str = "cuda:0"):
        super().__init__()
        self.heads = heads
        self.blocks_in_head = blocks_in_head
        self.config = config
        self.device= device
        total_layers = config.num_hidden_layers
        shared_count = total_layers - blocks_in_head
        if shared_count <= 0 or shared_count >= total_layers:
            raise ValueError(f"Invalid blocks_in_head ({blocks_in_head}) for {total_layers} total layers.")

        # Construct (heads+1) fresh LlamaForCausalLM from scratch so each is uniquely initialized
        self.backbone_model = LlamaForCausalLM(config)   # for the shared backbone
        self.head_models = []
        for _ in range(heads):
            new_head_model = LlamaForCausalLM(config)   # each a separate init, so that initial weights are different
            self.head_models.append(new_head_model)

        # For the backbone copy, remove everything after the shared_count
        self.backbone_model.model.layers = nn.ModuleList(
            self.backbone_model.model.layers[:shared_count]
        )
        del self.backbone_model.lm_head
        # set norm to identity, so it doesn't change the hidden state
        self.backbone_model.model.norm = nn.Identity()

        # For each head, remove the first shared_count layers, so it only has layers[shared_count:].
        for i, hm in enumerate(self.head_models):
            hm.model.layers = nn.ModuleList(
                hm.model.layers[shared_count:]
            )
            del hm.model.embed_tokens
            setattr(self, f'head_model_{i}', hm)

    def forward(self, input_ids, attention_mask=None):
        """
        Example forward pass:
         - pass input through backbone's embed_tokens + layers
         - then for each head, feed that hidden_state into the head's layers + norm + lm_head
         - return list[logits_head_0, logits_head_1, ...]
        """
        # Forward pass through the "backbone" portion
        backbone_outputs = self.backbone_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        shared_hidden_states = backbone_outputs[0]  # shape (batch, seq, hidden)

        # For each head, we call *its* layers, norm, and then lm_head
        logits_list = []
        for i, hm in enumerate(self.head_models):
            head_outputs = hm.model(
                inputs_embeds=shared_hidden_states,
                attention_mask=attention_mask,
            )
            head_hidden = head_outputs[0]
            head_hidden = hm.model.norm(head_hidden)
            logits = hm.lm_head(head_hidden)
            logits_list.append(logits)

        averaged_logits = torch.mean(torch.stack(logits_list), dim=0)
        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits
                
        return ModelOutput(averaged_logits)

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
        self.config.num_ensemble = self.heads
        self.config.blocks_in_head = self.blocks_in_head
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
            # Sharded save
            model_path = os.path.join(save_directory, WEIGHTS_NAME)
            save_function(state_dict, model_path)
        
        # Save model structure info
        model_structure = {
            "type": "TreeNetLlama",
            "heads": self.heads,
            "blocks_in_head": self.blocks_in_head,
            "total_layers": self.config.num_hidden_layers,
        }
        
        with open(os.path.join(save_directory, "model_structure.json"), "w") as f:
            json.dump(model_structure, f, indent=2)
        
        return save_directory

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the `tie_word_embeddings` flag is set in the config, ensure the input 
        and output weights are tied.
        """
        if hasattr(self.config, "tie_word_embeddings") and self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.get_output_embeddings(), self.get_input_embeddings())
        
        # Ensure other necessary tie_weights operations from parent classes are performed
        if hasattr(super(), "tie_weights") and callable(super().tie_weights):
            super().tie_weights()


class FullEnsembleLlama(nn.Module):
    """
    Just a simple ensemble of LLaMA models.
    """
    def __init__(self, config: LlamaConfig, num_ensemble: int = 3):
        super().__init__()
        self.num_ensemble = num_ensemble
        self.config = config
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

        averaged_logits = torch.mean(torch.stack(logits_list), dim=0)
        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits
                
        return ModelOutput(averaged_logits)

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        If the `tie_word_embeddings` flag is set in the config, ensure the input 
        and output weights are tied.
        """
        if hasattr(self.config, "tie_word_embeddings") and self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.get_output_embeddings(), self.get_input_embeddings())
        
        # Ensure other necessary tie_weights operations from parent classes are performed
        if hasattr(super(), "tie_weights") and callable(super().tie_weights):
            super().tie_weights()


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
