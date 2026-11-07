"""
Strategy: Radial Basis Function (RBF) Network with Learnable Centers and Widths.
Instead of a fixed linear basis, we use a parametric model where the 10 parameters define
the centers and widths of 5 Gaussian RBFs. This allows the model to adapt its local
structure to the data. To ensure stability and prevent BFGS from getting stuck in poor
local minima or producing invalid outputs, we use a robust initialization and careful
parameter scaling. The weights of the RBFs are fixed to 1, and a constant term is added
implicitly by the RBFs' interaction. However, since we only have 10 parameters, we use
5 RBFs with 2 parameters each (cx, cy) and fix the width to a reasonable value, OR we
use 2 RBFs with 5 parameters each (cx, cy, width, weight, bias). 
Actually, a better approach with 10 params:
Use 5 RBFs. Each has a center (cx, cy). Widths are fixed. Weights are fixed to 1.
But this is similar to the parent. 
Let's try: 5 RBFs, each with center (cx, cy) and width (sigma). That's 15 params. Too many.
Let's try: 2 RBFs, each with center (cx, cy), width (sigma), and weight (w). That's 4*2=8 params. 
Plus 2 more: global offset and global scale? 
Or: 5 RBFs, centers (cx, cy) for 5 RBFs = 10 params. Fixed widths. 
This is what the parent tried but with fixed centers. Here, the centers are learnable.
The risk is that BFGS might struggle with non-convexity. 
To mitigate, we can use a hybrid: a fixed polynomial part (using some params) and a small RBF part.
But we only have 10 params. 
Let's stick to the learnable centers for 5 RBFs. 
To help BFGS, we can use a transformation: cx = tanh(p1), cy = tanh(p2) to keep centers in [-1,1].
This makes the landscape smoother.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of 5 Gaussian RBFs with learnable centers.
    The centers are constrained to [-1, 1] via tanh.
    The widths are fixed to 0.5 to provide a balance between local and global coverage.
    The weights of the RBFs are fixed to 1.
    
    params: [cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4, cx5, cy5]
    where cx_i and cy_i are transformed via tanh to ensure they are in [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float).flatten()
    
    if params.size != 10:
        return np.zeros_like(x, dtype=float)
    
    # Transform parameters to valid centers in [-1, 1]
    # Using tanh to constrain the centers
    cx1 = np.tanh(params[0])
    cy1 = np.tanh(params[1])
    cx2 = np.tanh(params[2])
    cy2 = np.tanh(params[3])
    cx3 = np.tanh(params[4])
    cy3 = np.tanh(params[5])
    cx4 = np.tanh(params[6])
    cy4 = np.tanh(params[7])
    cx5 = np.tanh(params[8])
    cy5 = np.tanh(params[9])
    
    # Fixed width for all RBFs
    sigma = 0.5
    
    # Compute the RBFs
    dist1_sq = (x - cx1)**2 + (y - cy1)**2
    b1 = np.exp(-dist1_sq / (sigma**2))
    
    dist2_sq = (x - cx2)**2 + (y - cy2)**2
    b2 = np.exp(-dist2_sq / (sigma**2))
    
    dist3_sq = (x - cx3)**2 + (y - cy3)**2
    b3 = np.exp(-dist3_sq / (sigma**2))
    
    dist4_sq = (x - cx4)**2 + (y - cy4)**2
    b4 = np.exp(-dist4_sq / (sigma**2))
    
    dist5_sq = (x - cx5)**2 + (y - cy5)**2
    b5 = np.exp(-dist5_sq / (sigma**2))
    
    # Linear combination with fixed weights of 1
    # Add a small constant to avoid zero output if all RBFs are far away
    pred = b1 + b2 + b3 + b4 + b5 + 0.1
    
    # Ensure finite output
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
