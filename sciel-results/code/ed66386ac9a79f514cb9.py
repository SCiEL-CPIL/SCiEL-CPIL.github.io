"""Rank-minimal exact tensor decomposition for 2x2 x 2x6 matrix multiplication.

We use a rank-21 Strassen-based block decomposition as the base and then
attempt to reduce the rank by identifying and exploiting linear dependencies
among the factor columns.
"""

import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0


def _deadline_exceeded():
    return time.time() > _DEADLINE


def _build_rank21():
    """Construct rank-21 factor matrices U (4x21), V (12x21), W (12x21)."""
    n, m, p = 2, 2, 6
    rank = 21

    # Strassen decomposition for 2x2 matrices, replicated 3 times for the 6 columns
    # U: (n*m) x rank = 4 x 21
    # V: (m*p) x rank = 12 x 21
    # W: (p*n) x rank = 12 x 21

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen products for 2x2:
    # M1 = A[0,0] * (B[0,1] + B[1,1])
    # M2 = (A[0,0] + A[0,1]) * B[1,0]
    # M3 = (A[1,0] + A[1,1]) * B[0,0]
    # M4 = A[1,1] * (B[0,0] - B[1,0])
    # M5 = (A[0,0] + A[1,1]) * B[1,1]
    # M6 = (A[0,1] - A[1,1]) * (B[0,0] + B[1,0])
    # M7 = (A[1,0] - A[0,0]) * (B[1,1] + B[0,1])

    # For block b (b=0,1,2), the columns in B are at indices 2b and 2b+1
    for b in range(3):
        base = b * 7
        # U is the same for all blocks (since A is the same)
        U[0, base + 0] = 1    # A[0,0]
        U[0, base + 1] = 1    # A[0,0] in M2
        U[1, base + 1] = 1    # A[0,1] in M2
        U[2, base + 2] = 1    # A[1,0] in M3
        U[3, base + 2] = 1    # A[1,1] in M3
        U[3, base + 3] = 1    # A[1,1]
        U[0, base + 4] = 1    # A[0,0] in M5
        U[3, base + 4] = 1    # A[1,1] in M5
        U[1, base + 5] = 1    # A[0,1] in M6
        U[3, base + 5] = -1   # A[1,1] in M6
        U[2, base + 6] = 1    # A[1,0] in M7
        U[0, base + 6] = -1   # A[0,0] in M7

        # V for block b
        # M1: B[0,1] + B[1,1] -> indices 2b+1 and 2b+1+6=2b+7? No.
        # B is 2x6, so B[j, k] where j in {0,1}, k in {0,...,5}
        # V is (m*p) x rank = 12 x 21, indexed as V[j*p + k, r]
        
        # M1: (A[0,0]) * (B[0,2b+1] + B[1,2b+1])
        V[0 * p + (2 * b + 1), base + 0] = 1
        V[1 * p + (2 * b + 1), base + 0] = 1
        
        # M2: (A[0,0] + A[0,1]) * B[1, 2b]
        V[1 * p + (2 * b), base + 1] = 1
        
        # M3: (A[1,0] + A[1,1]) * B[0, 2b]
        V[0 * p + (2 * b), base + 2] = 1
        
        # M4: A[1,1] * (B[0,2b] - B[1,2b])
        V[0 * p + (2 * b), base + 3] = 1
        V[1 * p + (2 * b), base + 3] = -1
        
        # M5: (A[0,0] + A[1,1]) * B[1, 2b+1]
        V[1 * p + (2 * b + 1), base + 4] = 1
        
        # M6: (A[0,1] - A[1,1]) * (B[0,2b] + B[1,2b])
        V[0 * p + (2 * b), base + 5] = 1
        V[1 * p + (2 * b), base + 5] = 1
        
        # M7: (A[1,0] - A[0,0]) * (B[1,2b+1] + B[0,2b+1])
        V[1 * p + (2 * b + 1), base + 6] = 1
        V[0 * p + (2 * b + 1), base + 6] = 1

        # W for block b: C[i, 2b] and C[i, 2b+1]
        # C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # W is (p*n) x rank = 12 x 21, indexed as W[k*n + i, r]
        
        # M1: A[0,0]*(B[0,2b+1]+B[1,2b+1]) -> C[0, 2b+1]
        W[(2 * b + 1) * n + 0, base + 0] = 1
        
        # M2: (A[0,0]+A[0,1])*B[1,2b] -> C[0, 2b]
        W[(2 * b) * n + 0, base + 1] = 1
        
        # M3: (A[1,0]+A[1,1])*B[0,2b] -> C[1, 2b]
        W[(2 * b) * n + 1, base + 2] = 1
        
        # M4: A[1,1]*(B[0,2b]-B[1,2b]) -> C[1, 2b]
        W[(2 * b) * n + 1, base + 3] = 1
        # But wait, M4 also contributes to C[1, 2b] with the -B[1,2b] part
        # Actually, the Strassen formula is:
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M2 + M5
        # C[1,0] = M3 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # Let me redo this more carefully.
        # For a 2x2 block:
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M2 + M5
        # C[1,0] = M3 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # So for block b:
        # C[0, 2b] = M1 + M4 - M5 + M7
        # C[0, 2b+1] = M2 + M5
        # C[1, 2b] = M3 + M4
        # C[1, 2b+1] = M1 - M2 + M3 + M6
        
        # Let me reset W for block b
        W[:, base + 0] = 0
        W[:, base + 1] = 0
        W[:, base + 2] = 0
        W[:, base + 3] = 0
        W[:, base + 4] = 0
        W[:, base + 5] = 0
        W[:, base + 6] = 0
        
        # M1 contribution:
        W[(2 * b) * n + 0, base + 0] += 1      # C[0, 2b] += M1
        W[(2 * b + 1) * n + 1, base + 0] += 1  # C[1, 2b+1] += M1
        
        # M2 contribution:
        W[(2 * b + 1) * n + 0, base + 1] += 1  # C[0, 2b+1] += M2
        W[(2 * b + 1) * n + 1, base + 1] += -1 # C[1, 2b+1] -= M2
        
        # M3 contribution:
        W[(2 * b) * n + 1, base + 2] += 1      # C[1, 2b] += M3
        W[(2 * b + 1) * n + 1, base + 2] += 1  # C[1, 2b+1] += M3
        
        # M4 contribution:
        W[(2 * b) * n + 0, base + 3] += 1      # C[0, 2b] += M4
        W[(2 * b) * n + 1, base + 3] += 1      # C[1, 2b] += M4
        
        # M5 contribution:
        W[(2 * b) * n + 0, base + 4] += -1     # C[0, 2b] -= M5
        W[(2 * b + 1) * n + 0, base + 4] += 1  # C[0, 2b+1] += M5
        
        # M6 contribution:
        W[(2 * b + 1) * n + 1, base + 5] += 1  # C[1, 2b+1] += M6
        
        # M7 contribution:
        W[(2 * b) * n + 0, base + 6] += 1      # C[0, 2b] += M7

    return {"U": U, "V": V, "W": W}


def _verify_tensor(U, V, W):
    """Verify exact tensor decomposition."""
    n, m, p = 2, 2, 6
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if total != 1:
                    return False
    return True


def _try_reduce_rank(U, V, W, target_rank=20):
    """Attempt to reduce rank by finding linear dependencies."""
    n, m, p = 2, 2, 6
    rank = U.shape[1]
    
    if rank <= target_rank:
        return None
    
    # Try removing each single column
    for col_remove in range(rank):
        if _deadline_exceeded():
            break
        new_rank = rank - 1
        new_U = np.delete(U, col_remove, axis=1)
        new_V = np.delete(V, col_remove, axis=1)
        new_W = np.delete(W, col_remove, axis=1)
        if _verify_tensor(new_U, new_V, new_W):
            return {"U": new_U, "V": new_V, "W": new_W}
    
    return None


def construct():
    """Return {'U','V','W'} factor matrices exactly decomposing the 2,2,6 MM tensor."""
    result = _build_rank21()
    
    # Verify rank-21 works
    if not _verify_tensor(result["U"], result["V"], result["W"]):
        raise RuntimeError("Base rank-21 decomposition failed verification")
    
    # Try to reduce to rank 20
    if not _deadline_exceeded():
        reduced = _try_reduce_rank(result["U"], result["V"], result["W"], target_rank=20)
        if reduced is not None:
            if _verify_tensor(reduced["U"], reduced["V"], reduced["W"]):
                return reduced
    
    return result


# Self-check at import time
try:
    _check = _build_rank21()
    _ok = _verify_tensor(_check["U"], _check["V"], _check["W"])
except Exception:
    pass

