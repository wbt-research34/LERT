import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from transformers import OPTModel
from transformers.models.opt.modeling_opt import OPTDecoder
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from functools import partial
from transformers.utils import logging as tf_logging
logger = tf_logging.get_logger(__name__)
from transformers.cache_utils import Cache, DynamicCache
                                                                                            
                                    
class FineGrainedDetoxOPTModel(OPTModel):
    def __init__(self, config):
        super(OPTModel, self).__init__(config)
        self.decoder = FineOPTDecoder(config)
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        start_index: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                                                                                          
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

class FineOPTDecoder(OPTDecoder):
    
    def __init__(self, config):
        super().__init__(config)
        self.is_detox_mode = False        
                                
                                
        self.reward_head = None

    def set_model_forward_mode(self, mode: str):
        self.is_detox_mode = (mode == "reward_guide")

        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        start_index: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if input_ids is not None:
            input_ids = input_ids.view(-1, input_ids.shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            if past_key_values is None:
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.53.0. "
                    "You should pass an instance of `DynamicCache` instead, e.g. "
                    "`past_key_values=DynamicCache.from_legacy_cache(past_key_values)`."
                )

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if attention_mask is None:
            seq_length = past_seen_tokens + inputs_embeds.shape[1]
            attention_mask = torch.ones(inputs_embeds.shape[0], seq_length, device=inputs_embeds.device)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

                         
        if position_ids is None:
                                                        
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
                                                        
            position_ids = position_ids[:, past_seen_tokens:]

        pos_embeds = self.embed_positions(attention_mask, past_seen_tokens, position_ids=position_ids)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds.to(inputs_embeds.device)

                        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

                                                                                
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        batch_size, current_seq_len = hidden_states.shape[:2]

                                    
        if start_index is not None:
            target_mask = torch.zeros((batch_size, current_seq_len), dtype=torch.bool, device=hidden_states.device)

                              
            safe_idx = min(max(int(start_index), 0), current_seq_len - 1)

            target_mask[:, safe_idx:] = True

                                                            
            if attention_mask is not None:
                relevant_attn = attention_mask[:, -current_seq_len:].to(hidden_states.device).bool()
                target_mask = target_mask & relevant_attn
        else:
                                          
            target_mask = torch.zeros((batch_size, current_seq_len), dtype=torch.bool, device=hidden_states.device)
            
            if current_seq_len > 1:
                                                    
                if attention_mask is not None:
                    attn_gpu = attention_mask[:, -current_seq_len:].to(hidden_states.device)
                    last_idx = attn_gpu.sum(dim=1) - 1
                                                
                    last_idx = torch.clamp(last_idx, 0, current_seq_len - 1)
                    batch_indices = torch.arange(batch_size, device=hidden_states.device)
                    target_mask[batch_indices, last_idx] = True
                else:
                    target_mask[:, -1] = True
            else:
                                                   
                                                       
                target_mask.fill_(True)


        for i, decoder_layer in enumerate(self.layers):
                                                                                  
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    head_mask[i] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                    position_ids,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    layer_head_mask=(head_mask[i] if head_mask is not None else None),
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]
            if self.is_detox_mode and not self.training:
                target_mask = target_mask.to(hidden_states.device)
                target_h = hidden_states[target_mask]
                if target_h.size(0) > 0:
                    with torch.no_grad():
                        self.reward_head.to(target_h.device) 
                        reward_score = self.reward_head(target_h).squeeze(-1)
                        z=reward_score
                        if i == 0:
                            self._z_early, self._z_mid, self._locked = [], [], None

                                         
                        if 0 <= i <= 6:
                            self._z_early.append(z.detach())        

                                      
                        if 7 <= i <= 13:
                            self._z_mid.append(z.detach())          

                                           
                        if i == 13 and len(self._z_early) == 7 and len(self._z_mid) == 7:
                                                                                           
                                                                                         
                            E = torch.stack([z1.to(hidden_states.device) for z1 in self._z_early], dim=0)
                            M = torch.stack([z2.to(hidden_states.device) for z2 in self._z_mid], dim=0)
                            
                            slopeE = (E[-1] - E[0]) / 6.0
                            slopeM = (M[-1] - M[0]) / 6.0
                            safe = (slopeE > 0) | (slopeM > 0)
                            self._locked = ~safe
                                                                             
                                                                          
                                                              
                                                                                                              
                                    
                        if 14 <= i :

                        
                            step_size = 1
                         
                            toxic_token_mask = self._locked
                            if toxic_token_mask.any():
                                                                             
                                toxic_token_mask = toxic_token_mask.to(target_h.device)
                                toxic_h = target_h[toxic_token_mask]
                                curr_device = toxic_h.device                    
                                angle_ratio = self.layer_angle_ratios[i+1].to(curr_device)
                                stabilities = self.layer_multi_stabilities[i+1].to(curr_device)

                                                               
                                phase_beta = (angle_ratio ** 2) *10
                                                             
                                pos = self.pos_q[1, i+1].to(curr_device)                                          
                                neg = self.neg_q[5, i+1].to(curr_device)
                                                                          
                                                                          
                                z_now = (reward_score - neg) / (pos - neg + 1e-6)

                                                                                             
                                raw_gap = torch.clamp(- z_now[toxic_token_mask], min=0)
                                

                                intensity = raw_gap * 3.5
                                intensity = torch.clamp(intensity, max=10.0)
                                                             
                                v_clusters = self.layer_multi_direction[i+1].to(curr_device)

                                sims = F.cosine_similarity(toxic_h.unsqueeze(1), v_clusters.unsqueeze(0), dim=-1)
                                sim_weights = F.softmax(sims / 0.3, dim=-1)

                                                                        
                                path_weights = torch.where(stabilities > 0.85, stabilities, torch.zeros_like(stabilities))
                                path_weights = path_weights / (path_weights.sum() + 1e-8)

                                combined_weights = (sim_weights * path_weights.unsqueeze(0))
                                combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
                                fine_delta = torch.matmul(combined_weights, v_clusters)    


                                steering_v = ( phase_beta * fine_delta) *\
                                             intensity.unsqueeze(-1) *\
                                             step_size 
                                               
                                target_h[toxic_token_mask] += steering_v
                                hidden_states[target_mask] = target_h

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

                                                       
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

