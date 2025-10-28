"""MLP Trigonometric module for message passing refinement.

This module implements the Mlp_Trigonometric layer from thermo_GNN,
which refines bond-level messages using trigonometric operations.
"""

import torch
import torch.nn as nn
from torch import Tensor


class MlpTrigonometric(nn.Module):
    """An MLP-mixer module with trigonometric operations for message aggregation.

    This module refines bond-level messages in each message passing step by:
    1. Transforming the current message with layer normalization and linear projection
    2. Transforming the original message with linear projection
    3. Applying element-wise multiplication with sine function: m * sin(x)
    4. Applying GELU activation and output projection
    5. Adding residual connection from input

    This is based on the thermo_GNN implementation (mpn.py:42-70).

    Attributes:
        fc_x: Linear layer for transforming current message
        fc_m: Linear layer for transforming original message
        out: Linear layer for output projection
        gelu: GELU activation function
        layer_norm: Layer normalization
    """

    def __init__(self, dim: int):
        """Initialize the MlpTrigonometric module.

        Args:
            dim: Hidden dimension size (typically message_hidden_dim)
        """
        super().__init__()
        self.fc_x = nn.Linear(dim, dim)
        self.fc_m = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        """Refine message using trigonometric MLP-mixer.

        Args:
            x: Current message tensor of shape (num_bonds, hidden_dim)
               This is the message computed in the current depth iteration
            m: Original message tensor of shape (num_bonds, hidden_dim)
               This is the initial message saved at the beginning of message passing

        Returns:
            Refined message tensor of shape (num_bonds, hidden_dim)

        Example:
            >>> mlp_trig = MlpTrigonometric(dim=300)
            >>> current_msg = torch.randn(100, 300)  # 100 bonds, 300 hidden
            >>> original_msg = torch.randn(100, 300)
            >>> refined_msg = mlp_trig(current_msg, original_msg)
            >>> refined_msg.shape
            torch.Size([100, 300])
        """
        # Normalize and transform current message
        x_x = self.layer_norm(x)
        x_x = self.fc_x(x_x)

        # Transform original message
        x_m = self.fc_m(m)

        # Trigonometric mixing: element-wise multiplication with sine
        x_x = x_m * torch.sin(x_x)

        # Activation and output projection
        x_x = self.gelu(x_x)
        h = self.out(x_x) + x  # Residual connection

        return h


def test_mlp_trigonometric():
    """Unit test for MlpTrigonometric module."""
    print("Testing MlpTrigonometric module...")

    # Test 1: Basic functionality
    dim = 300
    num_bonds = 100
    mlp_trig = MlpTrigonometric(dim)

    x = torch.randn(num_bonds, dim)
    m = torch.randn(num_bonds, dim)

    output = mlp_trig(x, m)

    assert output.shape == (num_bonds, dim), f"Expected shape {(num_bonds, dim)}, got {output.shape}"
    print(f"✓ Test 1 passed: Output shape correct {output.shape}")

    # Test 2: Different batch size
    num_bonds_2 = 50
    x2 = torch.randn(num_bonds_2, dim)
    m2 = torch.randn(num_bonds_2, dim)
    output2 = mlp_trig(x2, m2)
    assert output2.shape == (num_bonds_2, dim)
    print(f"✓ Test 2 passed: Variable batch size works {output2.shape}")

    # Test 3: Gradient flow
    x3 = torch.randn(10, dim, requires_grad=True)
    m3 = torch.randn(10, dim, requires_grad=True)
    output3 = mlp_trig(x3, m3)
    loss = output3.sum()
    loss.backward()
    assert x3.grad is not None, "Gradient not computed for x"
    assert m3.grad is not None, "Gradient not computed for m"
    print("✓ Test 3 passed: Gradient flow works")

    # Test 4: Output statistics
    x4 = torch.randn(1000, dim)
    m4 = torch.randn(1000, dim)
    output4 = mlp_trig(x4, m4)
    mean = output4.mean().item()
    std = output4.std().item()
    print(f"✓ Test 4 passed: Output statistics - mean: {mean:.4f}, std: {std:.4f}")

    # Test 5: Module parameters
    param_count = sum(p.numel() for p in mlp_trig.parameters())
    expected_params = dim * dim * 3 + dim * 3 + dim * 2  # 3 Linear layers + biases + LayerNorm
    print(f"✓ Test 5 passed: Parameter count: {param_count} (expected ~{expected_params})")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_mlp_trigonometric()
