# EVOLVE-BLOCK-START
import time
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Two-layer tanh network with fixed internal weights and 10 learnable
    output-layer coefficients (linear in params, BFGS-friendly).

    Architecture:
      - Input: (x, y)
      - Hidden layer 1: 8 tanh units with fixed weights/biases
      - Hidden layer 2: 8 tanh units with fixed weights/biases
      - Output: linear combination of 8 hidden-2 units + 2 raw inputs
        (total 10 parameters)

    The fixed internal weights are chosen to create diverse non-linear features.
    """
    t0 = time.time()

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    p = np.asarray(params, dtype=np.float64).ravel()

    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size))

    # Fixed weights for hidden layer 1 (8 units, inputs: x, y, 1)
    W1 = np.array([
        [ 1.0,  1.0,  0.0],
        [-1.0,  1.0,  0.0],
        [ 1.0, -1.0,  0.0],
        [-1.0, -1.0,  0.0],
        [ 2.0,  0.5,  0.0],
        [ 0.5,  2.0,  0.0],
        [ 1.5, -1.5,  0.0],
        [-1.5,  1.5,  0.0],
    ])  # shape (8, 3)
    b1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Fixed weights for hidden layer 2 (8 units, inputs: 8 hidden-1 units, 1)
    W2 = np.array([
        [ 1.0,  0.5, -0.5,  0.0,  0.5, -0.5,  0.0,  0.0,  0.0],
        [ 0.5,  1.0,  0.0, -0.5,  0.0,  0.5, -0.5,  0.0,  0.0],
        [-0.5,  0.0,  1.0,  0.5, -0.5,  0.0,  0.5, -0.5,  0.0],
        [ 0.0, -0.5,  0.5,  1.0,  0.0, -0.5,  0.0,  0.5,  0.0],
        [ 0.5,  0.0, -0.5,  0.0,  1.0,  0.5, -0.5,  0.0,  0.0],
        [ 0.0,  0.5,  0.0, -0.5,  0.5,  1.0,  0.0, -0.5,  0.0],
        [-0.5,  0.0,  0.5,  0.0, -0.5,  0.0,  1.0,  0.5,  0.0],
        [ 0.0, -0.5,  0.0,  0.5,  0.0, -0.5,  0.5,  1.0,  0.0],
    ])  # shape (8, 9)
    b2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Build input matrix for hidden layer 1: (n, 3)
    X1 = np.column_stack([x, y, np.ones(n)])

    # Hidden layer 1
    h1 = np.tanh(X1 @ W1.T + b1)  # (n, 8)

    # Build input matrix for hidden layer 2: (n, 9)
    X2 = np.column_stack([h1, np.ones(n)])

    # Hidden layer 2
    h2 = np.tanh(X2 @ W2.T + b2)  # (n, 8)

    # Output: linear combination of h2 (8 units) + x + y
    # params[0:8] -> h2 weights, params[8] -> x weight, params[9] -> y weight
    out = h2 @ p[:8] + p[8] * x + p[9] * y

    # Safety: replace any non-finite with 0
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # Deadline guard
    if time.time() - t0 > 140.0:
        return out

    return out
# EVOLVE-BLOCK-END
