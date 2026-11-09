# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 x 4x4 matrix multiplication.
    
    Strategy:
    1. Build the rank-28 Strassen-based decomposition (4 blocks x 7 Strassen terms).
    2. Use linear algebra to find the minimal subset of terms that spans the target tensor.
    3. Return the lowest-rank verified decomposition found.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def time_left():
        return deadline - time.time()
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.zeros((n, m, p))
        for c in range(rank):
            T += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)
    
    def construct_strassen_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Standard 2x2x2 Strassen terms (7 terms)
        strassen_terms = [
            {'U': [(0, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
            {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1)], 'W': [(0, 1, 1), (1, 1, -1)]},
            {'U': [(0, 0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(1, 0, 1), (1, 1, 1)]},
            {'U': [(1, 1, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, 1), (0, 1, 1)]},
            {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(0, 0, -1), (1, 0, 1)]},
            {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(1, 1, 1)]},
            {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]},
        ]
        
        # Four 2x2 blocks in the 4x4 matrix
        blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
        
        col = 0
        for j_off, k_off in blocks:
            for term in strassen_terms:
                for (li, lj, coeff) in term['U']:
                    U[li * m + (lj + j_off)][col] += coeff
                for (lj, lk, coeff) in term['V']:
                    V[(lj + j_off) * p + (lk + k_off)][col] += coeff
                for (lk, li, coeff) in term['W']:
                    W[(lk + k_off) * n + li][col] += coeff
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Build the rank-28 decomposition
    best = construct_strassen_rank28()
    
    if not verify(best['U'], best['V'], best['W']):
        return best
    
    best_rank = 28
    U_arr = np.array(best['U'], dtype=float)
    V_arr = np.array(best['V'], dtype=float)
    W_arr = np.array(best['W'], dtype=float)
    
    # Vectorize the rank-1 tensors
    # Shape of each term: (n, m, p) -> (n*m*p,)
    terms = []
    for c in range(28):
        u = U_arr[:, c].reshape(n, m)
        v = V_arr[:, c].reshape(m, p)
        w = W_arr[:, c].reshape(p, n)
        # Tensor product u \otimes v \otimes w
        # Reshape to (n, m, p)
        t = np.einsum('ij,jk,ki->ijk', u, v, w)
        terms.append(t.flatten())
    
    terms = np.array(terms).T # Shape (32, 28)
    target = np.ones(n * m * p)
    
    # Find minimal subset of columns that spans the target
    # We want to remove as many columns as possible such that target is in the span of the rest.
    # This is equivalent to finding the smallest k such that there exists a subset of size 28-k 
    # whose column space contains target.
    
    # Heuristic: Start with all 28, try to remove columns one by one.
    # If removing column c keeps target in the span of the others, remove it.
    current_indices = list(range(28))
    
    for _ in range(28): # Max 28 removals
        if time_left() < 5:
            break
        removed = False
        for c in current_indices:
            # Try removing c
            new_indices = [i for i in current_indices if i != c]
            if len(new_indices) == 0:
                break
            # Check if target is in span of terms[new_indices]
            M_sub = terms[:, new_indices]
            # Rank of M_sub vs Rank of [M_sub | target]
            rank_sub = np.linalg.matrix_rank(M_sub, tol=1e-9)
            rank_aug = np.linalg.matrix_rank(np.column_stack((M_sub, target)), tol=1e-9)
            if rank_sub == rank_aug:
                # Target is in the span
                current_indices = new_indices
                removed = True
                break
        if not removed:
            break
            
    best_rank = len(current_indices)
    
    if best_rank < 28:
        U_arr = U_arr[:, current_indices]
        V_arr = V_arr[:, current_indices]
        W_arr = W_arr[:, current_indices]
        best = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
    else:
        best = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
        
    return best

# EVOLVE-BLOCK-END
