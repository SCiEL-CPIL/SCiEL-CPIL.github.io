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
        result = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(result, T, atol=tol)

    def best_rank1_approximation(R, max_iters=100):
        """
        Find the best rank-1 approximation u, v, w for tensor R using ALS.
        """
        # Initialize with random vectors
        u = np.random.randn(R.shape[0])
        v = np.random.randn(R.shape[1])
        w = np.random.randn(R.shape[2])
        
        best_norm = np.inf
        best_u, best_v, best_w = u, v, w
        
        for _ in range(max_iters):
            # Update u
            vw = np.outer(v, w)
            R1 = R.reshape(R.shape[0], -1)
            u_new = R1 @ vw
            if np.linalg.norm(u_new) > 1e-12:
                u = u_new / np.linalg.norm(u_new)
            else:
                u = np.random.randn(R.shape[0])
                u = u / np.linalg.norm(u)
            
            # Update v
            uw = np.outer(u, w)
            R2 = R.reshape(R.shape[1], -1)
            v_new = R2 @ uw
            if np.linalg.norm(v_new) > 1e-12:
                v = v_new / np.linalg.norm(v_new)
            else:
                v = np.random.randn(R.shape[1])
                v = v / np.linalg.norm(v)

            # Update w
            uv = np.outer(u, v)
            R3 = R.reshape(R.shape[2], -1)
            w_new = R3 @ uv
            if np.linalg.norm(w_new) > 1e-12:
                w = w_new / np.linalg.norm(w_new)
            else:
                w = np.random.randn(R.shape[2])
                w = w / np.linalg.norm(w)
            
            # Calculate residual norm
            approx = np.einsum('i,j,k->ijk', u, v, w)
            norm = np.linalg.norm(R - approx)
            
            if norm < best_norm:
                best_norm = norm
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                
            if norm < 1e-12:
                break
                
        return best_u, best_v, best_w, best_norm

    def greedy_decomposition(T, max_rank, time_limit):
        """
        Greedy algorithm to find a low-rank decomposition.
        """
        U_list = []
        V_list = []
        W_list = []
        R = T.copy()
        initial_norm = np.linalg.norm(R)
        
        for rank in range(1, max_rank + 1):
            if time.time() > time_limit:
                break
            
            # Find best rank-1 approx
            u, v, w, norm = best_rank1_approximation(R, max_iters=20)
            
            # Snap to half-integers
            u_h = to_half_integer(u)
            v_h = to_half_integer(v)
            w_h = to_half_integer(w)
            
            # Check if this term is "good"
            approx_h = np.einsum('i,j,k->ijk', u_h, v_h, w_h)
            R_new = R - approx_h
            norm_new = np.linalg.norm(R_new)
            
            # If the snapped term reduces the norm, accept it
            if norm_new < norm:
                U_list.append(u_h)
                V_list.append(v_h)
                W_list.append(w_h)
                R = R_new
                
                if norm_new < 1e-9:
                    U = np.column_stack(U_list)
                    V = np.column_stack(V_list)
                    W = np.column_stack(W_list)
                    return U, V, W, rank
            else:
                # If snapping didn't help, try to force a solution?
                # Or just stop?
                # If we can't reduce the norm with a half-integer term, we might be stuck.
                # We can try to find a term that matches the *largest* element of R?
                # For now, break if we can't make progress.
                if rank > 10: # Give it a few chances
                    break

        return None

    # Try greedy decomposition
    result = greedy_decomposition(T, 95, DEADLINE - 10)
    if result:
        U, V, W, rank = result
        if verify(U, V, W):
            best_U, best_V, best_W, best_rank = U, V, W, rank

    # Convert to list format
    U_out = [[U_out_val for U_out_val in row] for row in best_U.tolist()]
    V_out = [[V_out_val for V_out_val in row] for row in best_V.tolist()]
    W_out = [[W_out_val for W_out_val in row] for row in best_W.tolist()]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
