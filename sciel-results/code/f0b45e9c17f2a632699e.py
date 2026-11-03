import numpy as np
import time
from collections import defaultdict

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Build the list of (a,b,c) indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m + j, j*p + k, k*n + i))
    
    # For each a, list of (b,c) pairs
    a_to_bc = defaultdict(list)
    b_to_ac = defaultdict(list)
    c_to_ab = defaultdict(list)
    for a, b, c in indices:
        a_to_bc[a].append((b, c))
        b_to_ac[b].append((a, c))
        c_to_ab[c].append((a, b))
    
    # Try to find a lower-rank decomposition
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try a few target ranks
    for target_rank in [100, 96, 92, 88, 84, 80, 76, 72, 68, 64]:
        if time.time() > deadline - 15:
            break
        
        np.random.seed(42)
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        converged = False
        for iteration in range(20):
            if time.time() > deadline - 15:
                break
            
            # Update U
            for a in range(n*m):
                if a not in a_to_bc:
                    continue
                bcs = a_to_bc[a]
                A_mat = np.array([V[b, :] * W[c, :] for b, c in bcs])
                b_vec = np.ones(len(bcs))
                U[a, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Update V
            for b in range(m*p):
                if b not in b_to_ac:
                    continue
                acs = b_to_ac[b]
                A_mat = np.array([U[a, :] * W[c, :] for a, c in acs])
                b_vec = np.ones(len(acs))
                V[b, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Update W
            for c in range(p*n):
                if c not in c_to_ab:
                    continue
                abs_ = c_to_ab[c]
                A_mat = np.array([U[a, :] * V[b, :] for a, b in abs_])
                b_vec = np.ones(len(abs_))
                W[c, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            
            # Check convergence
            max_err = 0
            for a, b, c in indices:
                val = np.dot(U[a], V[b] * W[c])
                max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                converged = True
                break
        
        if converged and target_rank < best_rank:
            best_rank = target_rank
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Return schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        for idx, (a, b, c) in enumerate(indices):
            U_sb[a, idx] = 1
            V_sb[b, idx] = 1
            W_sb[c, idx] = 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
