import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from tqdm import tqdm

from transformers import OPTForCausalLM, OPTModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from .detox_model_opt import FineGrainedDetoxOPTModel
import logging

class DetoxOPTForCausalLM(OPTForCausalLM):
    def __init__(self, config, forward_mode="reward_guide"):
        super().__init__(config)
        self.model_name = "DetoxOPTForCausalLM"
        self.forward_mode =forward_mode
        if forward_mode == "reward_guide":

            self.model = FineGrainedDetoxOPTModel(config)
            self.model.decoder.is_detox_mode = True
        else:
            self.model = OPTModel(config)
        self.set_forward_mode(forward_mode)

    def set_forward_mode(self, mode="reward_guide"):
        if self.forward_mode == mode:
            return
        
        old_model = self.model
        device = old_model.device
        dtype = old_model.dtype
        
        logging.info(f"Switching model from {type(old_model).__name__} to mode: {mode}")

                     
        if mode == "reward_guide":
            new_model = FineGrainedDetoxOPTModel(self.config)
            new_model.decoder.is_detox_mode = True
        else:
            new_model = OPTModel(self.config)
        
                                             
        new_model.load_state_dict(old_model.state_dict(), strict=False)
        new_model.to(device=device, dtype=dtype)
        
                 
        self.model = new_model
        self.forward_mode = mode
        
               
        del old_model
        torch.cuda.empty_cache()
    
    
    def init_detox_resources(self, reward_stats_path, atlas_path, reward_model_path):
        """
        
        """
        if not isinstance(self.model, FineGrainedDetoxOPTModel):
            print("Current model is standard OPTModel, skipping detox resource initialization.")
            return
                                
        atlas = torch.load(atlas_path)
        self.model.decoder.register_buffer("layer_multi_direction", atlas["layer_multi_direction"])                             
        self.model.decoder.register_buffer("layer_angle_ratios", atlas["layer_angle_ratios"])
        self.model.decoder.register_buffer("layer_multi_stabilities", atlas["layer_multi_stabilities"])

        stats_data = torch.load(reward_stats_path)
                                 
                                   

        if "pos_q" in stats_data:
            self.model.decoder.register_buffer("pos_q", stats_data["pos_q"])
        if "neg_q" in stats_data:
            self.model.decoder.register_buffer("neg_q", stats_data["neg_q"])

        
                                
        reward_head = nn.Sequential(
            nn.Linear(self.config.word_embed_proj_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        reward_head.load_state_dict(torch.load(reward_model_path))
        reward_head = reward_head.to(device=self.model.device, dtype=self.model.dtype)
        reward_head.eval()
        for param in reward_head.parameters():
            param.requires_grad = False
        
                                
        self.model.decoder.reward_head = reward_head

                  
        
        
        
        logging.info("Detox resources initialized and injected into base model.")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        start_index: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if self.forward_mode == "base":
            return self.forward_base(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position= cache_position,
            )

        elif self.forward_mode == "reward_guide":
            return self.forward_guide(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position= cache_position,
                start_index=start_index,
            )
    def forward_base(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                                                                                       
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
                                                                       
            labels = labels.to(logits.device)
                                                
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
                                
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

    def forward_guide(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        start_index: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                                                                                       
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            start_index=start_index,
        )
        hidden_states = outputs[0]

        
                                    
                                                                                                                                          
              
                                                                                                                      
       
        logits = self.lm_head(hidden_states).contiguous()

        loss = None
        if labels is not None:
                                                                       
            labels = labels.to(logits.device)
                                                
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
                                
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    

    def forward_reward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                                                                                       
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        
        lm_logits = self.score(hidden_states)


        loss = None
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
