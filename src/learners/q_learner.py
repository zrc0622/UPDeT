import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.pqmix import PQMixer
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.zero_shot = args.zero_shot
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "pqmix":
                self.mixer = PQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        # 从buffer中提取奖励、动作、终止标志、掩码和可用动作
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"] # state 27  action 26  t 26  mask 26

        # Calculate estimated Q-Values
        # 使用mac（主网络）生成每个智能体当前时间步动作的Q值，动作选取方式是直接选取batch中记录的动作
        if self.args.phase_kl:
            mac_out = []
            phase_states = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs, phase_outs = self.mac.forward(batch, t=t, phase_kl=True)
                mac_out.append(agent_outs)
                phase_states.append(phase_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time
            phase_states = th.stack(phase_states, dim=1)
        else:
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        # 使用target mac（目标网络）生成每个智能体下一时间步动作的Q值，动作选取方式是max或双Q
        # max就是直接取每个智能体最大Q值的动作——target网络既选择动作，又评估Q值
        # 双Q就是选取mac输出中最大Q值的动作——target网络只评估Q值，选择动作由主网络负责
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None and self.args.mixer == "pqmix":
            if self.args.agent in ['phase_updet2'] and self.args.pqmix_v2:
                zeros_tensor = th.zeros((batch["phase_representation"].size(0), 1, batch["phase_representation"].size(2))).cuda()
                batch_phase_representation = th.cat((zeros_tensor, batch["phase_representation"]), dim=1)
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch_phase_representation[:, :-2]) # kl_4
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch_phase_representation[:, 1:-1])
            else:
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["phase_representation"][:, :-1]) # kl_4
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], batch["phase_representation"][:, 1:])
        else:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # phase_kl
        if self.args.phase_kl:
            kl_mask = batch["filled"].clone()
            done = batch["terminated"].clone()
            kl_mask[:, 1:] = kl_mask[:, 1:] * (1 - done[:, :-1]) * (1 - done[:, 1:])
            kl_loss = self.uniform_kl_loss(phase_states, kl_mask)
            total_loss = kl_loss * self.args.phase_kl_loss_weight * self.mac.action_selector.epsilon + loss
        else:
            total_loss = loss
            
        # Optimise
        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("total_loss", total_loss.item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            if self.args.phase_kl: self.logger.log_stat("kl_loss", kl_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def uniform_kl_loss(self, action_probs, mask):
        B, T, N, A = action_probs.shape # batchsize, steps, nums, action
    
        # 扩展mask到与action_probs对齐 (B, T, N, 1)
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, -1, N, -1)  # 自动广播到 (B, T, N, 1)
        
        # 计算每个episode每个智能体的有效步数 (B, N, 1)
        valid_steps = mask.sum(dim=1)  # 沿时间轴求和(dim=1, (1, 2)) 改
        
        # 计算每个智能体在各episode的动作概率总和 (B, N, A)
        sum_probs = (action_probs * mask).sum(dim=1)  # 时间轴压缩(dim=1, (1, 2)) 改

        # 计算平均概率分布 (B, N, A)
        avg_probs = sum_probs / valid_steps.clamp(min=1e-8)  # 防止除零

        # 目标均匀分布 (B, N, A) 每个位置为1/A
        target = th.ones_like(avg_probs) / A
        
        # 计算KL散度（注意输入顺序）
        kl_per_agent = F.kl_div(
            input=avg_probs.log(),  # KL(target||input)需要input是log概率
            target=target, 
            reduction='none'
        ).sum(dim=-1)  # (B, N)
        
        # 平均处理：每个batch样本、每个智能体的损失权重相同
        loss = kl_per_agent.mean()  # 先对智能体求平均，再对batch求平均
        
        return loss

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None and not self.zero_shot:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
