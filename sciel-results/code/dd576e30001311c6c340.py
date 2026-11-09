# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a linear interpolation basis (piecewise linear "hats").
    The basis functions are defined by N evenly spaced nodes in [-2, 2].
    The model is linear in params, ensuring BFGS converges to the global least-squares solution.
    """
    N = 10  # Number of parameters
    # Nodes evenly spaced in [-2, 2]
    nodes = np.linspace(-2.0, 2.0, N)
    
    # Precompute node differences for efficiency
    # dx[i] = nodes[i+1] - nodes[i]
    dx = np.diff(nodes)
    
    # Initialize result
    result = np.zeros_like(x, dtype=np.float64)
    
    # We can vectorize the calculation of basis functions
    # For each x, find which interval it falls into.
    # Since N is small (10), a loop over nodes is fast enough for 200-400 points.
    # However, we can do it more efficiently by checking each basis function's contribution.
    
    for i in range(N):
        if i == 0:
            # Left boundary: active on [nodes[0], nodes[1]]
            # phi_0(x) = (nodes[1] - x) / (nodes[1] - nodes[0])
            active = (x >= nodes[0]) & (x <= nodes[1])
            if np.any(active):
                phi = (nodes[1] - x[active]) / dx[0]
                result[active] += params[i] * phi
        elif i == N - 1:
            # Right boundary: active on [nodes[N-2], nodes[N-1]]
            # phi_{N-1}(x) = (x - nodes[N-2]) / (nodes[N-1] - nodes[N-2])
            active = (x >= nodes[N-2]) & (x <= nodes[N-1])
            if np.any(active):
                phi = (x[active] - nodes[N-2]) / dx[N-2]
                result[active] += params[i] * phi
        else:
            # Interior: active on [nodes[i-1], nodes[i+1]]
            # Left part: [nodes[i-1], nodes[i]]
            active_left = (x >= nodes[i-1]) & (x < nodes[i])
            if np.any(active_left):
                phi_left = (x[active_left] - nodes[i-1]) / dx[i-1]
                result[active_left] += params[i] * phi_left
            
            # Right part: [nodes[i], nodes[i+1]]
            active_right = (x >= nodes[i]) & (x <= nodes[i+1])
            if np.any(active_right):
                phi_right = (nodes[i+1] - x[active_right]) / dx[i]
                result[active_right] += params[i] * phi_right
                
    return result
# EVOLVE-BLOCK-END
