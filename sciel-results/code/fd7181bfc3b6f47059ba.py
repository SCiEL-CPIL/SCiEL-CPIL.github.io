import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 110

    # Build the schoolbook decomposition as the baseline
    U = np.zeros((n * m, rank_schoolbook), dtype=int)
    V = np.zeros((m * p, rank_schoolbook), dtype=int)
    W = np.zeros((p * n, rank_schoolbook), dtype=int)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                col = i * m * p + j * p + k
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1

    def verify(U_mat, V_mat, W_mat):
        """Verify that U,V,W exactly decompose the multiplication tensor."""
        U_mat = np.asarray(U_mat, dtype=float)
        V_mat = np.asarray(V_mat, dtype=float)
        W_mat = np.asarray(W_mat, dtype=float)
        r = U_mat.shape[1]
        # Target tensor: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_mat[a, :], np.multiply(V_mat[b, :], W_mat[c, :]))
                    if abs(val - 1) > 1e-9:
                        return False
        # Check that all other entries are 0
        for a in range(n * m):
            for b in range(m * p):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                if j != j2:
                    continue
                for c in range(p * n):
                    kk = c // n
                    ii = c % n
                    if i == ii and j == j2 and k == kk:
                        continue
                    val = np.dot(U_mat[a, :], np.multiply(V_mat[b, :], W_mat[c, :]))
                    if abs(val) > 1e-9:
                        return False
        return True

    # Verify schoolbook
    assert verify(U, V, W), "Schoolbook decomposition failed verification"

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook

    # Peeling method: try to remove components one at a time
    # For each column index, try to see if we can remove it
    # by checking if the remaining columns still span the required tensor
    
    # We'll work with the current decomposition and try to reduce rank
    U_cur = U.copy().astype(float)
    V_cur = V.copy().astype(float)
    W_cur = W.copy().astype(float)
    cur_rank = rank_schoolbook

    # Try removing columns in a greedy fashion
    # For each column, check if removing it still gives a valid decomposition
    # This is expensive, so we'll be selective
    
    # Actually, a better approach: try to find if any column is redundant
    # A column is redundant if the tensor it represents is already in the span
    # of the other columns
    
    # For small problems, we can try all single-column removals
    # But this is O(n*m*p) checks, each O((n*m*p)^3) for rank computation
    
    # Let's try a smarter approach: use the fact that we have structure
    # Group columns by their (i,j,k) pattern
    
    # For now, let's try a few random column removals and verify
    np.random.seed(42)
    
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        
        # Try to remove a random column
        col_to_try = np.random.randint(0, cur_rank)
        
        # Create candidate without this column
        mask = np.ones(cur_rank, dtype=bool)
        mask[col_to_try] = False
        U_cand = U_cur[:, mask]
        V_cand = V_cur[:, mask]
        W_cand = W_cur[:, mask]
        
        if verify(U_cand, V_cand, W_cand):
            # Successfully removed a column!
            U_cur = U_cand
            V_cand = V_cand
            W_cur = W_cand
            cur_rank -= 1
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
            best_rank = cur_rank
            # Continue trying to remove more
        else:
            # Try a different strategy: perturb and re-solve
            # This is more complex, so for now just continue
            pass

    # Convert back to integer/half-integer if possible
    def to_half_integer(mat):
        """Try to convert to half-integers (multiply by 2 and check if integer)"""
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-9):
            return (rounded / 2).tolist()
        return mat.tolist()

    return {"U": to_half_integer(best_U), "V": to_half_integer(best_V), "W": to_half_integer(best_W)}
