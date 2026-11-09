"""Matrix multiplication 4,4,5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill

    n, m, p = 4, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20

    # Schoolbook rank = n*m*p = 80
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    # Try to find a decomposition of rank r by solving linear equations
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k.
    # This is a bilinear (in U,V,W) system. We can fix W and solve for U,V
    # or use a structured approach.
    #
    # Strategy: Use the fact that for small matrices, known optimal ranks exist.
    # For 4x4 * 4x5, the known upper bound is 80 (schoolbook).
    # We'll attempt to reduce rank using structured patterns.
    #
    # Approach: Parameterize U, V, W with some structure and solve.
    # Since the problem is small, we can try rank 70, 65, 60, ... down to a lower bound.
    # Lower bound: max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # Also, the Strassen lower bound considerations...
    #
    # Let's try a direct numerical approach:
    # For a given rank r, we have 16*r + 20*r + 20*r = 56*r variables.
    # We have 4*4*5 = 80 equations.
    # For r >= 2, we have more variables than equations, so solutions likely exist.
    # But we need a VALID decomposition where the tensor matches exactly.
    #
    # Actually, the decomposition must satisfy:
    # For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a system of 80 cubic equations in 56*r unknowns.
    #
    # A simpler approach: use the "block diagonal" or "Kronecker" structure.
    # Or, use the fact that we can embed a known decomposition.
    #
    # Let's try: for rank r, set up U, V, W as random matrices and check.
    # Actually, a better approach: use the identity
    # A*B = sum_c (U_c * A) * (B * V_c^T) * W_c  (conceptually)
    # But this doesn't directly give the tensor decomposition.
    #
    # The correct formulation: The matrix multiplication tensor T ∈ R^{n×m×p} has
    # T[i,j,k] = 1. A rank-r decomposition is:
    # T[i,j,k] = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Let's try a constructive approach with small rank.
    # For 4x4x4, the best known is 49 (Strassen + others).
    # For 4x4x5, it's at most 80 (schoolbook).
    #
    # Let's try to find a decomposition with rank 70 using random search + refinement.
    # Actually, for the purposes of this task, let's try a few structured decompositions.
    
    # Try rank 80 (schoolbook) as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Try to find lower rank decompositions
    # For each candidate rank, try random initialization and check
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    for r in range(79, 59, -1):  # Try ranks 79 down to 60
        if time.time() > DEADLINE:
            break
        # Try a few random initializations
        found = False
        for attempt in range(50):
            if time.time() > DEADLINE:
                break
            rng = np.random.default_rng(hash((r, attempt)) & 0xFFFFFFFF)
            U = rng.standard_normal((nm, r))
            V = rng.standard_normal((mp, r))
            W = rng.standard_normal((pn, r))
            
            # Compute the tensor
            tensor = np.zeros((n, m, p))
            for c in range(r):
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                # T[i,j,k] += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            
            if np.allclose(tensor, 1.0, atol=1e-8):
                best_U, best_V, best_W, best_rank = U, V, W, r
                found = True
                break
        if found:
            # Try even lower ranks
            break
    
    # If we found a lower rank, verify it more carefully
    if best_rank < rank_sb:
        tensor_check = np.zeros((n, m, p))
        for c in range(best_rank):
            u = best_U[:, c].reshape(n, m)
            v = best_V[:, c].reshape(m, p)
            w = best_W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_check[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        if not np.allclose(tensor_check, 1.0, atol=1e-8):
            # Fallback to schoolbook
            best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
