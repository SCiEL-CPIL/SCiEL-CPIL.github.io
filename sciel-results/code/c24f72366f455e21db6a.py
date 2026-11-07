"""Rewritten program: multi-scale RBF basis with fixed centers for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute centers for a 3x3 grid plus origin (10 total)
# Grid points
_grid = np.array([-1.0, 0.0, 1.0])
_cx_grid, _cy_grid = np.meshgrid(_grid, _grid, indexing='ij')
_cx_flat = _cx_grid.flatten()
_cy_flat = _cy_grid.flatten()

# 9 grid centers + 1 origin (duplicate of center but we'll use a different width)
# Actually, to avoid redundancy, let's use 9 grid centers and the 10th as a global term?
# No, the task requires params to be fitted. If I include a constant term (p9 * 1), that's fine.
# Let's define 9 RBFs on the 3x3 grid and 1 constant term.
# This is safer than two RBFs at the same center with different widths if the optimizer struggles.
# But two scales is better. Let's do:
# 9 RBFs with width gamma=5.0 at grid points
# 1 RBF with width gamma=2.0 at (0,0) for smoother global trend?
# Or 9 RBFs + 1 constant.
# Let's try 9 RBFs (gamma=5) + 1 constant.

# Centers for the 9 RBFs
_CENTERS_X = _cx_flat
_CENTERS_Y = _cy_flat
_GAMMA = 5.0

# We will compute the features matrix on the fly in the equation function?
# No, x and y are passed in. We can't pre-compute features for unknown x,y.
# But we can pre-compute the centers array for fast broadcasting.

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params: length 10
    # p0..p8: weights for the 9 RBFs
    # p9: constant weight
    
    # x, y are arrays of shape (N,)
    # Compute distances from each point (x, y) to each center (cx_i, cy_i)
    
    # To be efficient, we can loop over the 9 centers or use broadcasting.
    # N is 200. 9 centers. Broadcasting is fine.
    
    # cx shape (9,), cy shape (9,)
    # x shape (N,), y shape (N,)
    
    # dx = x[:, None] - _CENTERS_X[None, :]  # shape (N, 9)
    # dy = y[:, None] - _CENTERS_Y[None, :]  # shape (N, 9)
    # dist2 = dx**2 + dy**2
    # rbf = np.exp(-_GAMMA * dist2)  # shape (N, 9)
    # f = rbf @ params[:9] + params[9]
    
    # This is very fast for N=200.
    
    dx = x[:, None] - _CENTERS_X[None, :]
    dy = y[:, None] - _CENTERS_Y[None, :]
    dist2 = dx**2 + dy**2
    rbf = np.exp(-_GAMMA * dist2)
    
    # Weighted sum
    f = rbf @ params[:9] + params[9]
    
    return f
# EVOLVE-BLOCK-END
