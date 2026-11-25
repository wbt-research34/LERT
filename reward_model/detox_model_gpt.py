import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast,BaseModelOutputWithPastAndCrossAttentions
from functools import partial
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from transformers import GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

class FineGrainedDetoxGptModel(GPT2Model):

    def __init__(self, config):
        super().__init__(config)
        self.is_detox_mode = False
        self.reward_head = None
  



    def set_model_forward_mode(self, mode: str):
        self.is_detox_mode = (mode == "reward_guide")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_index: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(dtype=self.dtype)
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None


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



        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
           
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
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

                        if 0 <= i <= 4:
                            self._z_early.append(z.detach())

                        if 5 <= i <= 9:
                            self._z_mid.append(z.detach())

                        if i == 9 and len(self._z_early) == 5 and len(self._z_mid) == 5:
                            E = torch.stack([z1.to(hidden_states.device) for z1 in self._z_early], dim=0)
                            M = torch.stack([z2.to(hidden_states.device) for z2 in self._z_mid], dim=0)
                            
                            slopeE = (E[-1] - E[0]) / 4.0
                            slopeM = (M[-1] - M[0]) / 4.0
                            safe = (slopeE > 0) | (slopeM > 0)
                            self._locked = ~safe
                        if i >= 10:
                          
                            step_size = 1
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



            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
