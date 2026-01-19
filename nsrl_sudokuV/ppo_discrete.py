import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
from torch.distributions import kl_divergence


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2),  # out: (32, H1, W1)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # out: (64, H2, W2)
            nn.ReLU(),
            nn.Flatten()
        )
        dummy_input = torch.zeros(1, 1, args.state_img_height, args.state_img_width)
        conv_out_dim = self.conv(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, args.action_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return F.softmax(self.fc(x), dim=1)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy_input = torch.zeros(1, 1, args.state_img_height, args.state_img_width)
        conv_out_dim = self.conv(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class PPO_discrete:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        
        self.args = args
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.action_dim = args.action_dim
        self.device = torch.device("cuda")
        
        

        self.actor = Actor(args)
        self.critic = Critic(args)
        
        self.actor.to(self.device)
        self.critic.to(self.device)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = self._flat_to_img_tensor(s)
        a_prob = self.actor(s).detach().cpu().numpy().flatten()
        a = np.argmax(a_prob)
        return a
        
        
    def _flat_to_img_tensor(self, flat_state):
        img = flat_state.reshape(self.args.nrows * self.args.ncols, 28, 28)
        img = img.reshape(self.args.nrows, self.args.ncols, 28, 28).transpose(0, 2, 1, 3).reshape(1, 1, self.args.state_img_height, self.args.state_img_width)
        return torch.tensor(img, dtype=torch.float32).to(self.device)

    def _flat_batch_to_img_tensor(self, batch_flat):
        batch_size = batch_flat.shape[0]
        imgs = batch_flat.reshape(batch_size, self.args.nrows * self.args.ncols, 28, 28)
        imgs = imgs.reshape(batch_size, self.args.nrows, self.args.ncols, 28, 28)
        imgs = imgs.permute(0, 1, 3, 2, 4).reshape(batch_size, 1, self.args.state_img_height, self.args.state_img_width)
        return imgs.to(self.device)




    def choose_action(self, s, action_shield, epsilon=0.0):
        s = self._flat_to_img_tensor(s)
        with torch.no_grad():
            actor_output = self.actor(s)
            dist = Categorical(probs=actor_output)
            dist_masked = Categorical(probs=actor_output * action_shield / (actor_output * action_shield).sum(dim=1, keepdim=True))
            if np.random.uniform() > epsilon:
                a = dist.sample()
            else:
                a = dist_masked.sample()
            
            a_logprob = dist.log_prob(a)
        return a.cpu().numpy()[0], a_logprob.cpu().numpy()[0]

    def update(self, replay_buffer, total_steps, gate, cmpe, is_gate):
        s, a, a_logprob, r, s_, dw, done, performa = replay_buffer.numpy_to_tensor()  # Get training data
        
        # print(s.dtype)
        # print(a.dtype)
        # print(a_logprob.dtype)
        # print(r.dtype)
        # print(s_.dtype)
        # print(dw.dtype)
        # print(done.dtype)
        # exit(0)

        # 不管 numpy_to_tensor 返回啥，统统规范一下
        s   = torch.as_tensor(s,   dtype=torch.float32, device=self.device)
        s_  = torch.as_tensor(s_,  dtype=torch.float32, device=self.device)
        a   = torch.as_tensor(a,   dtype=torch.int64,    device=self.device)      # 动作用 long
        r   = torch.as_tensor(r,   dtype=torch.float32, device=self.device)
        dw  = torch.as_tensor(dw,  dtype=torch.float32, device=self.device)      # 0/1
        done= torch.as_tensor(done,dtype=torch.float32, device=self.device)      # 0/1
        a_logprob = torch.as_tensor(a_logprob, dtype=torch.float32, device=self.device)
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        
        # s = self._flat_batch_to_img_tensor(s)
        # s_ = self._flat_batch_to_img_tensor(s_)
        
        # 2) 你的形状转换函数，最后也要保证在同一 device
        s  = self._flat_batch_to_img_tensor(s).to(self.device)
        s_ = self._flat_batch_to_img_tensor(s_).to(self.device)
        
        with torch.no_grad():  # adv and v_target have no gradient

            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
                
            # for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
            for delta, d in zip(reversed(deltas.flatten()), reversed(done.flatten())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float, device=self.device).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = Categorical(probs=self.actor(s[index]))
                
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
