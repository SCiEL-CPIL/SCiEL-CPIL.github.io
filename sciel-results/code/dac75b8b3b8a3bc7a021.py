"""Symbolic regression: multi-scale RBF basis with 10 parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute centers for a 3x3 grid (9 centers)
_grid = np.array([-1.0, 0.0, 1.0])
_cx_grid, _cy_grid = np.meshgrid(_grid, _grid, indexing='ij')
_CENTERS_X = _cx_grid.flatten()
_CENTERS_Y = _cy_grid.flatten()

# Two RBF scales: narrow for local detail, wide for global trend
_GAMMA_NARROW = 5.0
_GAMMA_WIDE = 1.0

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-scale RBF basis with 10 parameters.
    
    Basis:
      0-4: 5 narrow RBFs (gamma=5.0) at grid points:
            (0,0), (1,0), (-1,0), (0,1), (0,-1)
      5-8: 4 wide RBFs (gamma=1.0) at corner points:
            (1,1), (-1,1), (1,-1), (-1,-1)
      9:   constant term (1.0)
    
    The narrow RBFs capture local features, the wide RBFs capture
    global trends, and the constant provides a baseline. This
    non-parametric-style basis can approximate smooth functions
    well with only 10 parameters, and the exponential form is
    well-behaved for BFGS optimization.
    """
    # x, y: shape (N,)
    # params: shape (10,)
    
    # Narrow RBFs at 5 centers: (0,0), (1,0), (-1,0), (0,1), (0,-1)
    # Centers indexed in _CENTERS_X, _CENTERS_Y (flattened 3x3 grid)
    # Grid layout (i,j) -> index i*3+j:
    # (0,0)=idx4, (1,0)=idx7, (-1,0)=idx1, (0,1)=idx5, (0,-1)=idx3
    
    # Compute distances to all 9 centers at once for efficiency
    dx = x[:, None] - _CENTERS_X[None, :]  # (N, 9)
    dy = y[:, None] - _CENTERS_Y[None, :]  # (N, 9)
    dist2 = dx**2 + dy**2  # (N, 9)
    
    # Narrow RBFs at centers 1, 3, 4, 5, 7 (i.e., (-1,0), (0,-1), (0,0), (0,1), (1,0))
    narrow_centers = np.array([1, 3, 4, 5, 7])
    rbf_narrow = np.exp(-_GAMMA_NARROW * dist2[:, narrow_centers])  # (N, 5)
    
    # Wide RBFs at corner centers 0, 2, 6, 8 (i.e., (-1,-1), (-1,1), (1,-1), (1,1))
    wide_centers = np.array([0, 2, 6, 8])
    rbf_wide = np.exp(-_GAMMA_WIDE * dist2[:, wide_centers])  # (N, 4)
    
    # Constant term
    constant = np.ones_like(x)  # (N,)
    
    # Combine: params[0:5] for narrow, params[5:9] for wide, params[9] for constant
    pred = (rbf_narrow @ params[0:5] + 
            rbf_wide @ params[5:9] + 
            params[9] * constant)
    
    return pred
# EVOLVE-BLOCK-END
