import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataclasses import dataclass
import random

from igre_v0 import Igre


REUSE_Z = False


@dataclass
class PPOTransition:
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


class IgrePPOTrainer:
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
        self.allowed_token_ids = list(allowed_token_ids)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=5e-7)

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
            #! We do not use kl divergence between a reference model, breaking with typical
            #! PPO implementations
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

    def get_token_level_advantages(self, reward_sequences, advantages, response_mask):
        print("reward_sequences")
        print(reward_sequences)
        print(reward_sequences.shape)
        print("advantages")
        print(advantages)
        print(advantages.shape)
        tla = reward_sequences * advantages.unsqueeze(-1)
        length = tla.shape[-1]

        for t in reversed(range(length)):
            next_values = tla[:, t + 1] if t < length - 1 else 0.0
            tla[:, t] += self.gamma * next_values

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

        transition = PPOTransition(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            sequence_ids=sequences["input_ids"],
            sequence_mask=sequences["attention_mask"],
            responses_mask=response_mask,
            reward=reward,
            reward_sequence=rewards_with_kl_penalty,
            actor_log_probs=actor_log_probs,
            critic_value=critic_values,
            is_terminal=is_terminal,
        )
        self.memory.append(transition)

    def get_log_probs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
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

    def actor_loss(self, og_actor_log_probs, cur_actor_log_probs, advantages, mask):
        ratio = torch.exp((cur_actor_log_probs - og_actor_log_probs) * mask)
        loss1 = -advantages * ratio
        loss2 = -advantages * torch.clamp(
            ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip
        )

        loss = self.masked_mean(torch.max(loss1, loss2), mask)
        return loss, ratio

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

    def update(self):
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
        returns, advantages = self.get_turn_level_reward_and_advantages(
            rewards.cpu(), is_terminals, critic_values.cpu()
        )
        token_advantages = self.get_token_level_advantages(
            reward_sequences.cpu(),
            advantages,
            responses_mask,
        )

        og_actor_log_probs = actor_log_probs
        og_critic_values = critic_values

        epochs = 20

        for _ in range(epochs):
            random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]

                batch_sequence_ids = sequence_ids[batch_indices].detach().to(device)
                batch_og_actor_log_probs = (
                    og_actor_log_probs[batch_indices].detach().to(device)
                )
                batch_og_critic_values = (
                    og_critic_values[batch_indices].detach().to(device)
                )
                batch_returns = returns[batch_indices].detach().to(device)
                batch_advantages = token_advantages[batch_indices].detach().to(device)
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

                actor_loss, ratio = self.actor_loss(
                    batch_og_actor_log_probs,
                    actor_log_probs,
                    batch_advantages.unsqueeze(1),
                    batch_mask[:, 1:],
                )

                critic_loss, values_error = self.critic_loss(
                    batch_og_critic_values,
                    critic_values,
                    batch_returns,
                )

                entropy = self.get_entropy(actor_logits[:, :-1, :], batch_mask[:, 1:])

                loss = (
                    (self.actor_loss_weight * actor_loss)
                    + (self.critic_loss_weight * critic_loss)
                    + (self.entropy_loss_weight * entropy)
                )

                self.optimizer.zero_grad()
                loss.backward()
                # Clamp the gradients
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # print loss
                print("actor_loss")
                print(actor_loss)
                print("ratio")
                print(ratio)
                print("critic_loss")
                print(critic_loss)
                print("entropy")
                print(entropy)
                print("batch_advantages")
                print(batch_advantages)
                print("batch_mask")
                print(batch_mask)
                print("actor_log_probs")
                print(actor_log_probs[:, 10:])
                print("og_actor_log_probs")
                print(batch_og_actor_log_probs[:, 10:])
                print("sequence_ids")
                print(batch_sequence_ids[:, 11:])
                print("critic_values")
                print(critic_values)
                self.optimizer.step()

                # print the gradient norm
                # print("gradient norm")
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(name, param.grad.norm())

    def reset_memory(self):
        self.memory = []
