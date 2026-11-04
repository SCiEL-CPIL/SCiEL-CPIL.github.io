import numpy as np
import time

DIMENSIONS = [3, 4, 8]
DEADLINE = 150.0  # wall-clock kill; keep margin

def _verify(U, V, W, n, m, p, tol=1e-6):
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    U_reshaped = U_np.reshape(n, m, -1)
    V_reshaped = V_np.reshape(m, p, -1)
    W_reshaped = W_np.reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
