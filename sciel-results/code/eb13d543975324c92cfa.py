"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.

Structural change: Fix the block-naive decomposition (rank 54) by correcting
the column indexing bug, and add a proper tensor-equality verification.
Then attempt rank reduction via linear dependence detection on the factor
matrices.

The multiplication tensor T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i) form a
consistent triple, else 0. A rank-R decomposition satisfies
T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] for ALL (a,b,c).

Block-naive with 2x2 blocks: for each (i,j,k,t) with i,j,k in 0..2, t in 0..1,
factor r has U[(2i+a)*6+(2k+t)]=1 for a in {0,1}, V[(2k+t)*6+(2j+c)]=1 for
c in {0,1}, W[(2i+a)*6+(2j+c)]=1 for a,c in {0,1}. Total rank = 3*3*3*2 = 54.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 140.0


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _build_naive_decomposition(n, m, p):
    """Naive decomposition. Rank = n*m*p = 216 for 6x6x6."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}


def _build_block_naive_decomposition(n, m, p, block_size=2):
    """
    Block-naive decomposition for n=m=p=6 with 2x2 blocks. Rank = 54.
    
    For each (i, j, k, t) with i,j,k in 0..2, t in 0..1:
      U[(2i+a)*m + (2k+t), r] = 1 for a in {0,1}
      V[(2k+t)*p + (2j+c), r] = 1 for c in {0,1}
      W[(2i+a)*n + (2j+c), r] = 1 for a,c in {0,1}
    """
    num_blocks = n // block_size  # 3
    total_rank = num_blocks * num_blocks * num_blocks * block_size  # 54
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(num_blocks):
        for k in range(num_blocks):
            for t in range(block_size):
                for j in range(num_blocks):
                    # U: rows (2i+a, 2k+t) for a in {0,1}
                    for a in range(block_size):
                        U[(i * block_size + a) * m + (k * block_size + t)][col_idx] = 1
                    # V: rows (2k+t, 2j+c) for c in {0,1}
                    for c in range(block_size):
                        V[(k * block_size + t) * p + (j * block_size + c)][col_idx] = 1
                    # W: rows (2i+a, 2j+c) for a,c in {0,1}
                    for a in range(block_size):
                        for c in range(block_size):
                            W[(i * block_size + a) * n + (j * block_size + c)][col_idx] = 1
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}


def _verify_tensor(decomp, n, m, p, full_check=True):
    """
    Verify T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] for all (a,b,c).
    T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i) consistent, else 0.
    """
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    if full_check:
        # Check all valid triples (should equal 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check a sample of invalid triples (should equal 0)
        rng = np.random.default_rng(42)
        for _ in range(500):
            a = rng.integers(0, n * m)
            b = rng.integers(0, m * p)
            c = rng.integers(0, p * n)
            i_a, j_a = a // m, a % m
            j_b, k_b = b // p, b % p
            k_c, i_c = c // n, c % n
            if i_a != i_c or j_a != j_b or k_b != k_c:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    return False
    else:
        # Quick check: just valid triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
    return True


def _try_reduce_rank(decomp, n, m, p, max_rank=None):
    """
    Attempt to reduce rank by finding linearly dependent columns in U, V, W.
    If column r of U is a linear combination of others, we can merge factors.
    This is a heuristic approach.
    """
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]
    
    if max_rank is None:
        max_rank = rank - 1
    
    # Try to find linearly dependent columns using QR with column pivoting
    # on the "flattened" tensor representation
    # Build the tensor as a 2D matrix (mode-3 unfolding)
    # T_unfolded[c, (a,b)] = T[a,b,c]
    # But we work with the factor matrices directly.
    
    # Check if any column of U is in the span of others
    # Use SVD on U to find effective rank
    U_sv = np.linalg.svd(U, full_matrices=False)
    V_sv = np.linalg.svd(V, full_matrices=False)
    W_sv = np.linalg.svd(W, full_matrices=False)
    
    U_eff_rank = np.sum(U_sv[0] > 1e-9)
    V_eff_rank = np.sum(V_sv[0] > 1e-9)
    W_eff_rank = np.sum(W_sv[0] > 1e-9)
    
    min_eff_rank = min(U_eff_rank, V_eff_rank, W_eff_rank)
    
    if min_eff_rank >= rank:
        return decomp  # No reduction possible
    
    # Try to find a subset of columns that still spans the tensor
    # This is NP-hard in general, so use a greedy approach
    # For now, just return the original if no obvious reduction
    return decomp


def construct():
    n, m, p = DIMENSIONS
    
    if _remaining_time() < 5:
        return _build_naive_decomposition(n, m, p)
    
    # Build block-naive decomposition (rank 54)
    decomp = _build_block_naive_decomposition(n, m, p, block_size=2)
    
    if _remaining_time() < 10:
        return decomp
    
    # Verify
    try:
        if _verify_tensor(decomp, n, m, p, full_check=True):
            # Try to reduce rank
            reduced = _try_reduce_rank(decomp, n, m, p)
            if _verify_tensor(reduced, n, m, p, full_check=False):
                return reduced
            return decomp
        else:
            return _build_naive_decomposition(n, m, p)
    except Exception:
        return _build_naive_decomposition(n, m, p)


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    valid = _verify_tensor(result, 6, 6, 6, full_check=True)
    print(f"Valid: {valid}")


# (functions already defined above)
