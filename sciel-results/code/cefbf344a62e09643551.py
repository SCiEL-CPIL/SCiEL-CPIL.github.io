"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=6
    
    # For 2x2 * 2x6, the standard block approach gives rank 7*3=21.
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(4, 12, 12) = 12.
    # But for bilinear complexity, we need at least the rank of the tensor.
    # For 2x2*2x2, rank is 7 (Strassen). For 2x2*2x6, we can view it as
    # 3 independent 2x2*2x2 multiplications, giving rank 21.
    # 
    # However, there might be a better decomposition. Let's try to find
    # rank 20 or lower.
    #
    # Actually, for multiplying a 2x2 by a 2x6, we can think of it as
    # multiplying by three 2x2 blocks. The bilinear complexity is at most 21.
    # Can we do better?
    #
    # Let's try a direct search for rank 20.
    # The tensor has 4*12*12 = 576 entries (with structure).
    # 
    # Actually, let me think about this differently. The matrix multiplication
    # tensor T for (2,2,6) has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
    # 
    # Let me try to see if rank 20 is achievable by a small perturbation of Strassen.
    # 
    # For now, let's return the rank-21 Strassen solution, which is valid.
    # The score will be seed/21. If seed=24, score=24/21≈1.1429.
    # 
    # To improve, we need rank 20 or less. Let me attempt a search.
    
    import numpy as np
    import time
    
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the target tensor
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
    # Actually, the tensor for matrix mult is: sum over j of A[i,j] * B[j,k] = C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    # 
    # The decomposition: C[i,k] = sum_t U[i,t] * V[j,t] * W[k,t] ... no.
    # 
    # Actually the standard formulation: 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_t (sum_j A[i,j] V[j,t]) (sum_k' ... ) 
    # 
    # No, the tensor decomposition is:
    # T[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
    # where T[i,j,k] = 1 for the matrix multiplication tensor.
    # 
    # Wait, the interface says:
    # U: (n*m, rank) - so U indexes (i,j) pairs
    # V: (m*p, rank) - so V indexes (j,k) pairs  
    # W: (p*n, rank) - so W indexes (k,i) pairs
    # 
    # And the condition is: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i,j,k]
    # where T[i,j,k] = 1 for matrix multiplication.
    # 
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    # 
    # And for all other (i', j', k') combinations that don't correspond to valid matrix mult,
    # the sum should be 0. But actually, since U, V, W have fixed dimensions, the
    # decomposition only needs to satisfy the equation for valid (i,j,k).
    # 
    # Wait, I need to be more careful. The tensor T is defined on the full space.
    # T[i,j,k] = 1 if (i,j,k) is a valid triple for matrix mult, else 0.
    # But for matrix mult, ALL triples (i,j,k) with i<n, j<m, k<p are "valid" in the sense
    # that A[i,j]*B[j,k] contributes to C[i,k]. So T[i,j,k] = 1 for all i,j,k.
    # 
    # Hmm, but that's not right either. Let me re-read the problem.
    # 
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for (n,m,p) has T[i,j,k] = delta_{i,i}delta_{j,j}delta_{k,k}... 
    # No. The standard matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where the product A*B has entry (i,k) = sum_j A[i,j]B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    # 
    # So the condition is: for all i in [0,2), j in [0,2), k in [0,6):
    #   sum_t U[i*2+j, t] * V[j*6+k, t] * W[k*2+i, t] = 1
    
    # Let's verify the Strassen solution satisfies this.
    
    def build_strassen_21():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l, val in v_coeffs.items():
                    if l == 0:
                        v_idx = 0 * p + (2 * b)
                    elif l == 1:
                        v_idx = 0 * p + (2 * b + 1)
                    elif l == 2:
                        v_idx = 1 * p + (2 * b)
                    elif l == 3:
                        v_idx = 1 * p + (2 * b + 1)
                    else:
                        continue
                    V[v_idx][r] = val
                for (i, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # First, get the baseline
    sol21 = build_strassen_21()
    if not verify(sol21["U"], sol21["V"], sol21["W"]):
        # Should not happen
        return sol21
    
    # Try to find rank 20 by removing one Strassen product and adjusting
    # This is a hard problem. Let's try a random search approach.
    # 
    # Actually, let's think about whether rank 20 is even possible.
    # The bilinear complexity of 2x2*2x2 is 7. For 2x2*2x6, it's at most 21.
    # Is it exactly 21? 
    # 
    # By the submultiplicativity of bilinear complexity, C(2,2,6) <= C(2,2,2)*3 = 21.
    # Lower bound: C(2,2,6) >= C(2,2,2) = 7, but that's weak.
    # A better lower bound: C(n,m,p) >= max(n*m, m*p, p*n) = 12.
    # Even better: C(n,m,p) >= n*m*p / (n+m+p-2) by some bounds... not sure.
    # 
    # For 2x2*2x2, the rank is exactly 7. For 2x2*2x6, I believe the rank is 21
    # because the three 2x2 blocks are "independent" in a certain sense.
    # But this is not necessarily true.
    # 
    # Let me try a random search for rank 20.
    # 
    # Actually, let's try a smarter approach. Start with the rank-21 solution
    # and try to zero out one column by adjusting the others.
    
    import random
    random.seed(42)
    
    best_sol = sol21
    best_rank = 21
    
    # Try to find rank 20
    # For each of the 21 Strassen products, try to eliminate it
    for elim in range(21):
        if time.time() > deadline:
            break
        
        # Copy the solution
        U = [row[:] for row in sol21["U"]]
        V = [row[:] for row in sol21["V"]]
        W = [row[:] for row in sol21["W"]]
        
        # Zero out column elim
        for i in range(n*m):
            U[i][elim] = 0
        for j in range(m*p):
            V[j][elim] = 0
        for k in range(p*n):
            W[k][elim] = 0
        
        # Now we need to adjust the remaining 20 columns to compensate
        # This is a nonlinear problem. Let's try a simple approach:
        # For each (i,j,k), we need sum_{t != elim} U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
        # 
        # This is hard to solve directly. Let's try a different approach:
        # Randomly perturb the solution and see if we can get a valid rank-20.
        
        # Actually, let's just try random rank-20 solutions
        # and check if they work. This is very unlikely to succeed by chance.
    
    # Given the difficulty, let's just return the rank-21 solution.
    # The score is seed/21. If we can't find rank 20, this is the best we can do.
    
    return sol21
# EVOLVE-BLOCK-END
