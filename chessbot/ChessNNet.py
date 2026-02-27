"""
AlphaZero-style ResNet for chess.

Architecture (per the paper):
  - Input conv:  3×3, num_channels filters, BN, ReLU
  - 20 residual blocks: [Conv→BN→ReLU→Conv→BN] + skip, ReLU
  - Policy head: Conv(2 filters,1×1) → BN → ReLU → flatten → FC(action_size) → log-softmax
  - Value head:  Conv(1 filter,1×1) → BN → ReLU → flatten → FC(256) → ReLU → FC(1) → tanh
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from NeuralNet import NeuralNet
from chessbot.ChessBoard import ACTION_SIZE


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ChessNet(nn.Module):
    """
    Input:  (batch, 119, 8, 8)   — channels-first for PyTorch
    Output: (log_policy, value)
      log_policy: (batch, action_size)  log-probabilities
      value:      (batch, 1)            in [-1, 1]
    """

    def __init__(self, action_size: int, num_channels: int = 256, num_res_blocks: int = 20):
        super().__init__()
        self.action_size = action_size

        # Input block
        self.input_conv = nn.Conv2d(119, num_channels, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 8 * 8, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(8 * 8, 256)
        self.value_fc2  = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, 119, 8, 8)
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


class ChessNNet(NeuralNet):
    """
    NeuralNet wrapper around ChessNet for use with MCTS and Coach.
    Accepts board tensors as (8,8,119) numpy arrays.
    """

    def __init__(self, game, args):
        self.action_size = game.getActionSize()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet = ChessNet(
            action_size=self.action_size,
            num_channels=args.num_channels,
            num_res_blocks=args.num_res_blocks,
        ).to(self.device)

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            try:
                self.nnet = torch.compile(self.nnet)
            except Exception as e:
                print(f"torch.compile unavailable ({e}); using eager mode")

    def train(self, examples):
        """
        examples: list of (board_np, pi, v)
          board_np: (8, 8, 119) float32
          pi:       (action_size,) float32
          v:        float
        """
        optimizer = optim.Adam(
            self.nnet.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.l2_reg,
        )
        self.nnet.train()
        use_amp = self.device.type == 'cuda'

        for epoch in range(self.args.epochs):
            # Shuffle
            indices = np.random.permutation(len(examples))
            batch_count = len(examples) // self.args.batch_size

            pi_losses = []
            v_losses  = []

            for b in range(batch_count):
                idx = indices[b * self.args.batch_size:(b + 1) * self.args.batch_size]
                boards, pis, vs = zip(*[examples[i] for i in idx])

                boards = torch.tensor(
                    np.array(boards), dtype=torch.float32, device=self.device
                )   # already (B, 119, 8, 8) — CHW from to_tensor()
                pis = torch.tensor(np.array(pis), dtype=torch.float32, device=self.device)
                vs  = torch.tensor(np.array(vs),  dtype=torch.float32, device=self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                    log_ps, pred_vs = self.nnet(boards)
                    pred_vs = pred_vs.squeeze(1)
                    # Loss: cross-entropy for policy + MSE for value
                    pi_loss = -torch.mean(torch.sum(pis * log_ps, dim=1))
                    v_loss  = torch.mean((vs - pred_vs) ** 2)
                    loss    = pi_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pi_losses.append(pi_loss.item())
                v_losses.append(v_loss.item())

            if batch_count > 0:
                print(f"  Epoch {epoch+1}/{self.args.epochs}  "
                      f"pi_loss={np.mean(pi_losses):.4f}  "
                      f"v_loss={np.mean(v_losses):.4f}")

    def predict(self, board: np.ndarray):
        """
        board: (119, 8, 8) float32 numpy array  (CHW from to_tensor())
        Returns (pi, v):
          pi: (action_size,) numpy float32
          v:  float
        """
        self.nnet.eval()
        use_amp = self.device.type == 'cuda'
        with torch.no_grad():
            x = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 119, 8, 8)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                log_p, v = self.nnet(x)
            pi = torch.exp(log_p).squeeze(0).float().cpu().numpy()
            return pi, float(v.item())

    def predict_batch(self, boards: np.ndarray):
        """
        boards: (N, 119, 8, 8) float32 — CHW stacked batch from to_tensor().
        Returns:
          pis: (N, action_size) float32
          vs:  (N,)             float32
        One GPU forward pass for the entire batch.
        """
        self.nnet.eval()
        use_amp = self.device.type == 'cuda'
        with torch.no_grad():
            x = torch.tensor(boards, dtype=torch.float32, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                log_ps, vs = self.nnet(x)
            pis = torch.exp(log_ps).float().cpu().numpy()
            vs  = vs.squeeze(1).float().cpu().numpy()
            return pis, vs

    def save_checkpoint(self, folder: str = 'checkpoints', filename: str = 'checkpoint.pth.tar'):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        torch.save({'state_dict': self.nnet.state_dict()}, path)

    def load_checkpoint(self, folder: str = 'checkpoints', filename: str = 'checkpoint.pth.tar'):
        path = os.path.join(folder, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.nnet.load_state_dict(checkpoint['state_dict'])
