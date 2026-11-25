import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from functools import partial

                                                                                                                     
                                                                                                                     
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

class FineGrainedDetoxLlamaModel(LlamaModel):
    """
     LlamaModel  Reward 
    """
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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        start_index: Optional[int] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

                                                                                                            
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

                                                                           
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

                        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

                                       

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
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
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
                            E = torch.stack(self._z_early, dim=0)          
                            M = torch.stack(self._z_mid, dim=0)            

                            slopeE = (E[-1] - E[0]) / 6.0
                            slopeM = (M[-1] - M[0]) / 6.0
                            safe = (slopeE > 0) | (slopeM > 0)
                            self._locked = ~safe
                                                                                                                                                       
                                    
                        if 14 <= i :

                            step_size=1
                            toxic_token_mask = self._locked
                            if toxic_token_mask.any():
                                                                             
                                toxic_token_mask = toxic_token_mask.to(target_h.device)
                                toxic_h = target_h[toxic_token_mask]
                                curr_device = toxic_h.device                    
                                angle_ratio = self.layer_angle_ratios[i+1].to(curr_device)
                                stabilities = self.layer_multi_stabilities[i+1].to(curr_device)
                                                               
                                phase_beta = (angle_ratio ** 2) *9
                                                             
                                                                                                                   
                                                                         
                                pos = self.pos_layer[i+1].to(curr_device)
                                neg = self.neg_layer[i+1].to(curr_device)
                                z_now = (reward_score - neg) / (pos - neg + 1e-6)
                                raw_gap = torch.clamp(1.00 - z_now[toxic_token_mask], min=0)
                                                                                        
                                

                                intensity = raw_gap * 3.0
                                intensity = torch.clamp(intensity, max=5.0)
                                                             
                                v_clusters = self.layer_multi_direction[i+1].to(curr_device)

                                sims = F.cosine_similarity(toxic_h.unsqueeze(1), v_clusters.unsqueeze(0), dim=-1)
                                sim_weights = F.softmax(sims / 0.3, dim=-1)

                                                                        
                                path_weights = torch.where(stabilities > 0.85, stabilities, torch.zeros_like(stabilities))
                                path_weights = path_weights / (path_weights.sum() + 1e-8)

                                combined_weights = (sim_weights * path_weights.unsqueeze(0))
                                combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
                                fine_delta = torch.matmul(combined_weights, v_clusters)    
                                                 
                                                                        
                                                

                                steering_v = (phase_beta * fine_delta) *\
                                             intensity.unsqueeze(-1) *\
                                             step_size  
                                               
                                target_h[toxic_token_mask] += steering_v
                                hidden_states[target_mask] = target_h
                            
                    
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                  
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
