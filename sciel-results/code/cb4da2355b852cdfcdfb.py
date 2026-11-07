"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    # 10-term Fourier series: sin/cos for frequencies 1..5
    # Orthogonal on symmetric intervals -> well-conditioned linear system
    # BFGS on a strictly linear-in-params model converges to least squares
    basis = np.column_stack([
        np.sin(1.0 * x),
        np.cos(1.0 * x),
        np.sin(2.0 * x),
        np.cos(2.0 * x),
        np.sin(3.0 * x),
        np.cos(3.0 * x),
        np.sin(4.0 * x),
        np.cos(4.0 * x),
        np.sin(5.0 * x),
        np.cos(5.0 * x),
    ])
    
    return basis @ params
# EVOLVE-BLOCK-END
