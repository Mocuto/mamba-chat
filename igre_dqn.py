import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataclasses import dataclass
import random

from igre_v0 import Igre


REUSE_Z = False
USE_TLA_MASKING = False
OPTION_PRUNING_PENALITY = True


@dataclass
class DQNTransition:
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    sequence_ids: torch.Tensor
    sequence_mask: torch.Tensor
    responses_mask: torch.Tensor
    reward: torch.Tensor
    reward_sequence: torch.Tensor
    actor_log_probs: torch.Tensor
    critic_value: torch.Tensor
    is_terminal: bool


class IgreDQNTrainer:
    """Heavily copied from: https://github.com/raghavc/LLM-RLHF-Tuning-with-PPO-and-DPO/blob/main/script/utils/ppo_trainer_with_peft.py#L437"""

    pad_token_id: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: Igre
    tokenizer: AutoTokenizer

    def __init__(self, model: Igre):
        self.model = model
        self.memory = []
        self.eps_clip = 100000
        self.value_clip = 1
        self.gamma = 1.0
        self.critic_loss_weight = 1.0
        self.actor_loss_weight = 1.0
        self.entropy_loss_weight = 0.001
        allowed_tokens = [
            "finish",
            "A",
            "B",
            "C",
            "D",
        ]
        tokenizer = self.model.tokenizer
        # convert tokens to ids
        allowed_token_ids = set([tokenizer.eos_token_id])
        for word in allowed_tokens:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            for token_id in token_ids:
                allowed_token_ids.add(token_id)
        # self.allowed_token_ids = list(allowed_token_ids)
        self.allowed_token_ids = []

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=5e-7)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids,
        return_prompt: bool = True,
    ):
        seqs, _, z = self.model.gen1(  # type: ignore
            prompt_ids, c=0, return_z=True, allowed_tokens=self.allowed_token_ids
        )
        if return_prompt:
            return seqs, z
        return seqs[:, prompt_ids.shape[1] :], z

    def get_response(self, prompt_ids):
        return self.generate(prompt_ids, False)

    def process_sequences(self, prompt_ids, response_ids):
        # seq: [0 0 0 0, prompt, response, 0 0 0 0] change to [prompt, response, 0 0 0 0]
        prompts_without_padding = []
        responses_without_padding = []
        batch_size = 1
        for i in range(batch_size):
            response = response_ids[i]
            prompt = prompt_ids[i]
            prompt_left_padding_length = (prompt == self.pad_token_id).sum().item()
            response_length = (response != self.pad_token_id).sum().item()
            prompt_without_padding = prompt[prompt_left_padding_length:]
            response_without_padding = response[:response_length]

            prompts_without_padding.append(prompt_without_padding)
            responses_without_padding.append(response_without_padding)

        new_sequences = [
            torch.cat([q, r])
            for q, r in zip(prompts_without_padding, responses_without_padding)
        ]
        sequences = torch.nn.utils.rnn.pad_sequence(
            new_sequences, batch_first=True, padding_value=self.pad_token_id
        )

        sequences = {
            "input_ids": sequences,
            "attention_mask": sequences.ne(self.pad_token_id).long().to(self.device),
        }

        return prompts_without_padding, responses_without_padding, sequences

    # def get_last_reward_score(self, values, responses_mask):
    #     batch_size = values.shape[0]
    #     reward_score = []
    #     for i in range(batch_size):
    #         value = values[i]
    #         end_index = responses_mask[i].nonzero()[-1].detach().item()
    #         reward_score.append(value[end_index])

    #     rewards_score = torch.stack(reward_score)

    #     return rewards_score

    @torch.no_grad()
    def get_actor_logits_and_critic_values(self, sequences, z):
        input_ids = sequences["input_ids"].long().to(z.device)
        actor_logits, critic_value = None, None
        if REUSE_Z:
            actor_logits, critic_value = (
                self.model.sys1_logits_and_critic_value_with_option(
                    input_ids, z, allowed_tokens=self.allowed_token_ids
                )
            )
        else:
            actor_logits, critic_value = self.model.sys1_logits_and_critic_value_with_c(
                input_ids, c=0, allowed_tokens=self.allowed_token_ids
            )
        return actor_logits, critic_value.squeeze(1)

    def get_response_mask(self, sequences_mask, prompts_without_padding):
        batch_size = sequences_mask.shape[0]
        responses_mask = []
        for i in range(batch_size):
            prompt = prompts_without_padding[i]
            response_mask = torch.zeros_like(sequences_mask[i])
            response_mask[len(prompt) :] = sequences_mask[i][len(prompt) :]
            responses_mask.append(response_mask)
        return torch.stack(responses_mask)

    def get_entropy(self, logits, mask):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = self.masked_mean(-torch.sum(probs * log_probs, dim=-1), mask)
        return entropy

    def compute_rewards(self, reward, actor_log_probs, responses_mask):
        masks = responses_mask[:, 1:]
        # rewards_score = self.get_last_reward_score(reward, responses_mask)

        batch_size = reward.shape[0]
        rewards_with_kl_penalty = []
        for i in range(batch_size):
            kl = torch.zeros_like(actor_log_probs[i])
            mask = masks[i]
            end_index = mask.nonzero()[-1].detach().item()
            kl_penalty = kl
            kl_penalty[end_index] += 1
            rewards_with_kl_penalty.append(kl_penalty)

        return torch.stack(rewards_with_kl_penalty)

    def get_turn_level_reward_and_advantages(
        self, rewards, is_terminals, critic_values
    ):
        # Compute the expected future reward, i.e., the return
        returns = torch.zeros_like(rewards)
        batch_size = rewards.shape[0]

        discounted_reward = 0
        for i in range(batch_size):
            reverse_i = batch_size - i - 1
            reward = rewards[reverse_i]
            is_terminal = is_terminals[reverse_i]
            if is_terminal:
                discounted_reward = 0
            expected_reward = reward + self.gamma * discounted_reward
            returns[reverse_i] = expected_reward
            discounted_reward = expected_reward

        # Compute the advantages
        advantages = returns.detach() - critic_values.detach()
        return returns, advantages

    def get_token_level_rewards(self, reward_sequences, turn_level_rewards):
        print("reward_sequences")
        print(reward_sequences)
        print(reward_sequences.shape)
        print("turn_level_rewards")
        print(turn_level_rewards)
        print(turn_level_rewards.shape)
        tla = reward_sequences * turn_level_rewards.unsqueeze(-1)
        length = tla.shape[-1]

        for t in reversed(range(length)):
            next_values = tla[:, t + 1] if t < length - 1 else 0.0
            tla[:, t] += 0.99 * next_values

        return tla.detach()

    def add_transition(self, prompt_ids, response_ids, z, reward, is_terminal):
        """
        prompt_ids: Tensor of shape L
        response_ids: Tensor of shape L
        z: Tensor of shape (1, z_dim)
        reward:
        """
        prompts_without_padding, _, sequences = self.process_sequences(
            prompt_ids.unsqueeze(0), response_ids.unsqueeze(0)
        )
        actor_logits, critic_values = self.get_actor_logits_and_critic_values(
            sequences, z
        )
        actor_log_probs = self.get_log_probs(
            actor_logits[:, :-1, :], sequences["input_ids"][:, 1:]
        )

        response_mask = self.get_response_mask(
            sequences["attention_mask"], prompts_without_padding
        ).to(self.device)

        rewards_with_kl_penalty = self.compute_rewards(
            reward, actor_log_probs, response_mask
        )

        rewards_with_kl_penalty = rewards_with_kl_penalty * response_mask[:, 1:]

        transition = DQNTransition(
            prompt_ids=prompt_ids.detach().cpu(),
            response_ids=response_ids.detach().cpu(),
            sequence_ids=sequences["input_ids"].detach().cpu(),
            sequence_mask=sequences["attention_mask"].detach().cpu(),
            responses_mask=response_mask.detach().cpu(),
            reward=reward.detach().cpu(),
            reward_sequence=rewards_with_kl_penalty.detach().cpu(),
            actor_log_probs=actor_log_probs.detach().cpu(),
            critic_value=critic_values.detach().cpu(),
            is_terminal=is_terminal,
        )
        self.memory.append(transition)

    def get_log_probs(self, logits, labels):
        # log_probs = F.log_softmax(logits, dim=-1)
        log_probs = logits
        log_probs_labels = log_probs.gather(
            dim=-1, index=labels.unsqueeze(-1).to(logits.device)
        )
        return log_probs_labels.squeeze(-1)

    def masked_mean(self, data, mask, dim=None, eps=1e-8):
        data = data * mask
        if dim is not None:
            return data.sum(dim=dim) / (mask.sum(dim=dim) + eps)
        else:
            return data.sum() / (mask.sum() + eps)

    def masked_var(self, data, mask, dim=None):
        mean = self.masked_mean(data, mask, dim=dim)
        centered_values = data - mean
        var = self.masked_mean(centered_values**2, mask, dim=dim)
        return var

    def actor_loss(
        self, cur_actor_log_probs, reward_seq, mask, cur_actor_logits, labels
    ):
        q_s_a = cur_actor_log_probs * mask
        y_i = reward_seq * mask
        if USE_TLA_MASKING:
            _, cur_actor_greedy_indices = cur_actor_logits.max(dim=-1)
            tla_mask = torch.ones_like(mask)
            tla_mask_inner = (
                (cur_actor_greedy_indices == labels)
                + (cur_actor_greedy_indices == self.pad_token_id)
                + (cur_actor_greedy_indices == self.model.tokenizer.eos_token_id)
                + (labels == self.pad_token_id)
                + (labels == self.model.tokenizer.eos_token_id)
            )
            print("labels shape")
            print(labels.shape)
            print("cur_actor_greedy_indices shape")
            print(cur_actor_greedy_indices.shape)
            print("tla_mask shape")
            print(tla_mask.shape)
            tla_mask_inner = tla_mask_inner.flip(dims=[-1])
            # Take the cumulative product of the inner to mask tokens that are not the greedy action
            tla_mask_inner = tla_mask_inner.cumprod(dim=-1)
            # Flip the mask back to the original order
            tla_mask_inner = tla_mask_inner.flip(dims=[-1])
            tla_mask[:, :-1] = tla_mask_inner[:, 1:]
            print("DEBUG cur_actor_greedy_indices")
            print(cur_actor_greedy_indices)
            print("DEBUG labels")
            print(labels)
            print("DEBUG tla mask")
            print(tla_mask)
            print("DEBUG mask")
            print(mask)
            mask = mask * tla_mask
            print("DEBUG final mask")
            print(mask)
        # TODO: We need to update this with a value from a target network
        # Take the masked mean square error of the q(s, a) and the y_i
        loss = 0.5 * self.masked_mean((q_s_a - y_i) ** 2, mask)

        if OPTION_PRUNING_PENALITY:
            # For every logit above 0, add a penalty
            # This is to encourage the model to disregard most of the options
            prune_penalty = (torch.relu(cur_actor_logits) * mask.unsqueeze(dim=2)).sum(
                dim=-1
            ).sum(-1).mean() / (cur_actor_logits.shape[-1] * cur_actor_logits.shape[-2])
            loss += (1000 * prune_penalty)
            print("prune_penalty")
            print(prune_penalty)

        return loss

    def critic_loss(self, og_critic_values, cur_critic_values, returns):
        critic_values_clip = torch.clamp(
            cur_critic_values,
            og_critic_values - self.value_clip,
            og_critic_values + self.value_clip,
        )
        values_error = (cur_critic_values - returns) ** 2
        values_clip_error = (critic_values_clip - returns) ** 2
        # loss = 0.5 * self.masked_mean(torch.max(values_error, values_clip_error), mask)
        loss = (
            0.5
            * torch.max(values_error, values_clip_error).sum()
            / (values_error.shape[0] * values_error.shape[1] + 1e-8)
        )

        return loss, values_error

    def get_rewards(self):
        rewards = [transition.reward for transition in self.memory]
        return torch.stack(rewards)

    def get_transitions_as_tensors(self):
        prompt_ids = [transition.prompt_ids for transition in self.memory]
        response_ids = [transition.response_ids for transition in self.memory]
        sequence_ids = [
            transition.sequence_ids.squeeze(0) for transition in self.memory
        ]
        sequence_masks = [
            transition.sequence_mask.squeeze(0) for transition in self.memory
        ]
        responses_mask = [
            transition.responses_mask.squeeze(0) for transition in self.memory
        ]
        rewards = [transition.reward for transition in self.memory]
        reward_sequences = [
            transition.reward_sequence.squeeze(0) for transition in self.memory
        ]
        actor_log_probs = [
            transition.actor_log_probs.squeeze(0) for transition in self.memory
        ]
        critic_values = [transition.critic_value for transition in self.memory]
        is_terminals = [transition.is_terminal for transition in self.memory]

        return (
            torch.stack(prompt_ids).detach(),
            # torch.stack(response_ids).detach(),
            torch.nn.utils.rnn.pad_sequence(
                response_ids, batch_first=True, padding_value=self.pad_token_id
            ).detach(),
            # torch.cat(sequence_ids, dim=0).detach(),
            torch.nn.utils.rnn.pad_sequence(
                sequence_ids, batch_first=True, padding_value=self.pad_token_id
            ).detach(),
            # torch.cat(sequence_masks, dim=0).detach(),
            torch.nn.utils.rnn.pad_sequence(
                sequence_masks, batch_first=True, padding_value=False
            ).detach(),
            # torch.cat(responses_mask, dim=0).detach(),
            torch.nn.utils.rnn.pad_sequence(
                responses_mask, batch_first=True, padding_value=False
            ).detach(),
            torch.cat(rewards, dim=0).detach(),
            # torch.cat(reward_sequences, dim=0).detach(),
            torch.nn.utils.rnn.pad_sequence(
                reward_sequences, batch_first=True, padding_value=0
            ).detach(),
            # torch.cat(actor_log_probs, dim=0).detach(),
            torch.nn.utils.rnn.pad_sequence(
                actor_log_probs, batch_first=True, padding_value=0
            ).detach(),
            torch.cat(critic_values, dim=0).detach(),
            is_terminals,
        )

    def clamp_param_grads(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            # Check for NaNs
            if torch.isnan(param.grad).any():
                # print the name of the parameter
                print("NaN in gradients for " + name)
                print(param.grad)

            param.grad.data.clamp_(-1., 1.)

    def update(self, lr_mult):
        with torch.autograd.set_detect_anomaly(True):
            (
                prompt_ids,
                response_ids,
                sequence_ids,
                sequence_masks,
                responses_mask,
                rewards,
                reward_sequences,
                actor_log_probs,
                critic_values,
                is_terminals,
            ) = self.get_transitions_as_tensors()

            batch_size = 8
            indices = list(range(rewards.shape[0]))
            if len(indices) < batch_size:
                return

            # print the shape of all of these
            # print("prompt_ids")
            # print(prompt_ids.shape)
            # print("response_ids")
            # print(response_ids.shape)
            # print("sequence_ids")
            # print(sequence_ids.shape)
            # print("sequence_masks")
            # print(sequence_masks.shape)
            # print("responses_mask")
            # print(responses_mask.shape)
            # print("rewards")
            # print(rewards.shape)
            # print("reward_sequences")
            # print(reward_sequences.shape)
            # print("actor_log_probs")
            # print(actor_log_probs.shape)
            # print("critic_values")
            # print(critic_values.shape)

            device = self.device
            returns, _ = self.get_turn_level_reward_and_advantages(
                rewards.cpu(), is_terminals, critic_values.cpu()
            )
            token_rewards = self.get_token_level_rewards(
                reward_sequences.cpu(),
                returns,
            )

            og_actor_log_probs = actor_log_probs
            og_critic_values = critic_values

            epochs = 100

            for _ in range(epochs):
                random.shuffle(indices)
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i : i + batch_size]

                    batch_sequence_ids = sequence_ids[batch_indices].detach().to(device)
                    batch_token_level_rewards = (
                        token_rewards[batch_indices].detach().to(device)
                    )
                    batch_mask = responses_mask[batch_indices].detach().to(device)
                    # batch_mask = sequence_masks[batch_indices].detach().to(device)

                    # print("batch_sequence_ids")
                    # print(batch_sequence_ids.shape)
                    # print(sequence_ids.shape)
                    actor_logits, critic_values = (
                        self.model.sys1_logits_and_critic_value_with_c(
                            batch_sequence_ids, c=0
                        )
                    )

                    actor_log_probs = self.get_log_probs(
                        actor_logits[:, :-1, :], batch_sequence_ids[:, 1:]
                    )

                    actor_loss = self.actor_loss(
                        actor_log_probs,
                        batch_token_level_rewards,
                        batch_mask[:, 1:],
                        actor_logits[:, :-1, :],
                        batch_sequence_ids[:, 1:],
                    )

                    # critic_loss, values_error = self.critic_loss(
                    #     batch_og_critic_values,
                    #     critic_values,
                    #     batch_returns,
                    # )

                    loss = self.actor_loss_weight * actor_loss

                    for g in self.optimizer.param_groups:
                        g["lr"] = 1e-4 * lr_mult

                    self.optimizer.zero_grad()
                    loss.backward()
                    # Clamp the gradients
                    self.clamp_param_grads()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    # print loss
                    print("actor_loss")
                    print(actor_loss)
                    print("batch_token_level_rewards")
                    print(batch_token_level_rewards)
                    print(batch_token_level_rewards.shape)
                    print("batch_mask")
                    print(batch_mask)
                    print("actor_log_probs")
                    print(actor_log_probs[:, 10:])
                    # check if actor_log_probs has nan
                    if torch.isnan(actor_log_probs).any():
                        raise Exception("actor_log_probs has nan")

                    # print("sequence_ids")
                    # print(batch_sequence_ids[:, 11:])
                    self.optimizer.step()

                    # print the gradient norm
                    # print("gradient norm")
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None:
                    #         print(name, param.grad.norm())

    def reset_memory(self):
        self.memory = []
