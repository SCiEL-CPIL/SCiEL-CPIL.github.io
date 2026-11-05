import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 140  # Safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i,j,k] = 1 if (i,m_idx) and (j,p_idx) and (k,n_idx) align
    # Using trace convention: W[k*n+i, r]
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Sum over r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,k) for matching j
    
    # Actually, for matrix multiplication C = A * B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j; j',k; i',k'] = delta(i,i') * delta(j,j') * delta(k,k')
    # But in our indexing: U[i*m+j, r], V[j'*p+k, r], W[k'*n+i', r]
    # We need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta(i,i')*delta(j,j')*delta(k,k')
    
    # So T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if i==i' and j==j' and k==k', else 0
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook solution
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    def to_half_integer(x):
        """Round to nearest half-integer"""
        return np.round(x * 2) / 2.0
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact"""
        # Compute sum_r U[:,r] * V[:,r] * W[:,r] and compare to T
        # Use einsum
        result = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(result, T, atol=tol)
    
    def search_random(rank, max_iters=500):
        """Try random initialization with half-integer constraints"""
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            # Snap to half-integers
            U = to_half_integer(U)
            V = to_half_integer(V)
            W = to_half_integer(W)
            if verify(U, V, W):
                return U, V, W, rank
        return None
    
    def local_refine(U, V, W, rank, max_steps=200):
        """Try to improve by local perturbations"""
        U, V, W = U.copy(), V.copy(), W.copy()
        current_ok = verify(U, V, W)
        for step in range(max_steps):
            if time.time() > DEADLINE:
                break
            # Try flipping a random entry
            mat = random.choice([U, V, W])
            r, c = random.randint(0, mat.shape[0]-1), random.randint(0, mat.shape[1]-1)
            old_val = mat[r, c]
            # Try adding or subtracting 0.5
            delta = random.choice([0.5, -0.5])
            mat[r, c] = old_val + delta
            if verify(mat if mat is U else U, mat if mat is V else V, mat if mat is W else W):
                # Keep the change
                continue
            else:
                mat[r, c] = old_val
        return U, V, W, rank
    
    # Try to find lower rank decompositions
    # Start from schoolbook and try to merge terms
    for target_rank in range(rank_sb - 1, max(30, rank_sb - 40), -1):
        if time.time() > DEADLINE:
            break
        # Try random search for this rank
        result = search_random(target_rank, max_iters=100)
        if result:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    # Convert to list format
    U_out = [[U_out_val for U_out_val in row] for row in best_U.tolist()]
    V_out = [[V_out_val for V_out_val in row] for row in best_V.tolist()]
    W_out = [[W_out_val for W_out_val in row] for row in best_W.tolist()]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
