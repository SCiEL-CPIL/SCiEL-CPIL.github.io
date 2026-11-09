"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a rank-minimal tensor decomposition for 2x3 * 3x6 matrix multiplication.
    """
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (18, rank)  
    # W shape: (p*n, rank) = (12, rank)
    
    # The tensor entries that must be 1:
    # For each (i, j, k) in range(n) x range(m) x range(p):
    #   U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1
    # For all other (u_idx, v_idx, w_idx) combinations, the sum = 0
    
    # Let's enumerate all the "1" positions in the tensor
    # u_idx = i*m + j, v_idx = j*p + k, w_idx = k*n + i
    tensor_ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_ones.append((u_idx, v_idx, w_idx))
    
    # Total number of constraints:
    # "1" constraints: n*m*p = 36
    # "0" constraints: (n*m)*(m*p)*(p*n) - 36 = 6*18*12 - 36 = 1296 - 36 = 1260
    # Total: 1296
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    
    best_result = None
    best_rank = n * m * p  # schoolbook as fallback
    
    # Schoolbook decomposition as fallback
    U_fb = [[0] * best_rank for _ in range(n * m)]
    V_fb = [[0] * best_rank for _ in range(m * p)]
    W_fb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to find lower rank decompositions
    # Use a randomized approach: try to find U, V, W with small entries
    # such that the tensor is exactly decomposed
    
    # Strategy: For a given rank r, we need to find matrices U (6xr), V (18xr), W (12xr)
    # such that for each (u,v,w) in tensor_ones: sum(U[u,r]*V[v,r]*W[w,r]) = 1
    # and for all other (u,v,w): sum = 0
    
    # This is a hard non-linear problem. Let's try a structured approach.
    
    # For 2x3 * 3x6, let's try rank 18, 19, 20, ...
    # Use iterative refinement or algebraic methods
    
    # Actually, let's try a simpler approach: use the fact that we can factor
    # through intermediate matrices.
    
    # A (2x3) * B (3x6) = C (2x6)
    # We can write: C = A * B
    # If we can write A = A1 * A2 where A1 is 2xr and A2 is rx3, and B = B1 * B2 where B1 is 3xs and B2 is sx6,
    # then C = A1 * A2 * B1 * B2, which doesn't directly help.
    
    # Alternative: use block structure. Since n=2, m=3, p=6:
    # Let's try to exploit the small dimensions.
    
    # Try rank from 18 upward
    for r in range(18, min(37, 30)):  # try up to rank 29
        if time.time() > deadline - 10:
            break
            
        # Try to find a decomposition of rank r
        # Use a randomized local search approach
        
        # Initialize with random small values
        np.random.seed(42 + r)
        
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Try to build up the decomposition iteratively
        # Start with a subset of the tensor_ones and assign them to rank-1 components
        
        # Greedy approach: for each rank-1 component, try to cover as many tensor_ones as possible
        # while maintaining the zero constraints
        
        # This is complex. Let's try a different approach:
        # Assume a specific structure for U, V, W.
        
        # For rank r, suppose U has entries in {-1, 0, 1}
        # Try all combinations is infeasible for r >= 18.
        
        # Let's try a simpler structured decomposition:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # = sum_j (sum_t U[i,j,t] * something) ... 
        
        # Actually, let me try a direct algebraic approach.
        # For the tensor to have rank r, there must exist U, V, W such that:
        # T[u,v,w] = sum_t U[u,t] * V[v,t] * W[w,t]
        
        # This is equivalent to saying that the tensor T can be written as a sum of r rank-1 tensors.
        
        # For small ranks, we can try to solve this system.
        # Let's parameterize: fix U and V, then W is determined by the "1" constraints,
        # and we check the "0" constraints.
        
        # For a given U and V, the constraint for (u,v,w) in tensor_ones is:
        # sum_t U[u,t] * V[v,t] * W[w,t] = 1
        # This is a linear system in W.
        
        # Number of "1" constraints: 36
        # Number of unknowns in W: p*n*r = 12*r
        # For r >= 3, 12*r >= 36, so the system is determined or overdetermined.
        
        # For each (u,v) pair, we get a linear equation in W:
        # sum_t U[u,t]*V[v,t]*W[w,t] = 1 for the w such that (u,v,w) is in tensor_ones
        
        # Actually, for each (u,v) pair, there may be multiple w values.
        # Let's check: u = i*m+j, v = j*p+k, w = k*n+i
        # Given u and v, can we determine w? 
        # u determines (i,j), v determines (j,k), so w = k*n+i is determined.
        # So for each (u,v) pair, there's at most one w.
        
        # How many (u,v) pairs? u has 6 values, v has 18 values.
        # But not all (u,v) pairs are valid. A pair (u,v) is valid if:
        # u = i*m+j for some i,j and v = j'*p+k for some j',k with j=j'
        
        # Let's enumerate valid (u,v,w) triples:
        valid_triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    valid_triples.append((u, v, w))
        
        # For each (u,v) pair, how many w values?
        from collections import defaultdict
        uv_to_w = defaultdict(list)
        for u, v, w in valid_triples:
            uv_to_w[(u,v)].append(w)
        
        # Check if any (u,v) has multiple w values
        multi_w = {k: v for k, v in uv_to_w.items() if len(v) > 1}
        # For our case, each (u,v) should have exactly one w since (i,j,k) uniquely determines (u,v,w)
        
        # So we have 36 linear equations in W (12*r unknowns).
        # For r >= 3, this is an underdetermined system (36 eqs, 36+ unknowns for r>=3).
        
        # Strategy: 
        # 1. Randomly initialize U and V
        # 2. Solve for W using the 36 "1" constraints
        # 3. Check if the "0" constraints are satisfied
        # 4. If not, adjust U and V and repeat
        
        # This is a local search approach.
        
        # Let's try a few iterations
        max_iters = 5000
        found = False
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            
            # Build the linear system for W
            # For each (u,v,w) in valid_triples:
            #   sum_t U[u,t]*V[v,t]*W[w,t] = 1
            # This is: (U[u,:] * V[v,:]) dot W[w,:] = 1
            # where U[u,:]*V[v,:] is element-wise product
            
            # Set up matrix A and vector b for the linear system
            # Unknowns: W flattened (12*r variables)
            # Equations: 36
            
            A_mat = np.zeros((len(valid_triples), p*n*r))
            b_vec = np.ones(len(valid_triples))
            
            for idx, (u, v, w) in enumerate(valid_triples):
                # Coefficients for W[w, t] are U[u,t]*V[v,t]
                for t in range(r):
                    A_mat[idx, w*r + t] = U[u, t] * V[v, t]
            
            # Solve for W
            try:
                W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W = W_flat.reshape(p*n, r)
            except:
                continue
            
            # Check all constraints
            valid = True
            # Check "1" constraints
            for u, v, w in valid_triples:
                val = np.dot(U[u, :] * V[v, :], W[w, :])
                if abs(val - 1) > 1e-6:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check "0" constraints (sample some for speed)
            # Total (u,v,w) combos: 6*18*12 = 1296
            # Check all
            for u in range(n*m):
                for v in range(m*p):
                    for w in range(p*n):
                        if (u, v, w) in set(valid_triples):
                            continue
                        val = np.dot(U[u, :] * V[v, :], W[w, :])
                        if abs(val) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                found = True
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Try to simplify to integers or half-integers
                # Round to nearest half-integer
                def round_to_half(x):
                    return 2 * round(x / 2) / 2
                
                U_list = [[round_to_half(x) for x in row] for row in U_list]
                V_list = [[round_to_half(x) for x in row] for row in V_list]
                W_list = [[round_to_half(x) for x in row] for row in W_list]
                
                # Verify after rounding
                valid_rounded = True
                for u, v, w in valid_triples:
                    val = sum(U_list[u][t] * V_list[v][t] * W_list[w][t] for t in range(r))
                    if abs(val - 1) > 1e-6:
                        valid_rounded = False
                        break
                
                if not valid_rounded:
                    # Keep original
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If not found for this rank, try next
        if time.time() > deadline - 10:
            break
    
    return best_result
# EVOLVE-BLOCK-END
