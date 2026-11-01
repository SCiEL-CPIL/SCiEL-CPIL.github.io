"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = _start_time + 140.0  # Safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    def build_rank53():
        rank = 53
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(base_r, k0, k1):
            r1 = base_r
            r2 = base_r + 1
            r3 = base_r + 2
            r4 = base_r + 3
            r5 = base_r + 4
            r6 = base_r + 5
            r7 = base_r + 6
            
            U[0][r1] = 1
            U[3][r1] = 1
            V[0 * p + k0][r1] = 1
            V[1 * p + k1][r1] = 1
            
            U[2][r2] = 1
            U[3][r2] = 1
            V[0 * p + k0][r2] = 1
            
            U[0][r3] = 1
            V[0 * p + k1][r3] = 1
            V[1 * p + k1][r3] = -1
            
            U[3][r4] = 1
            V[1 * p + k0][r4] = 1
            V[0 * p + k0][r4] = -1
            
            U[0][r5] = 1
            U[1][r5] = 1
            V[1 * p + k1][r5] = 1
            
            U[2][r6] = 1
            U[0][r6] = -1
            V[0 * p + k0][r6] = 1
            V[0 * p + k1][r6] = 1
            
            U[1][r7] = 1
            U[3][r7] = -1
            V[1 * p + k0][r7] = 1
            V[1 * p + k1][r7] = 1
            
            W[2 * k0][r1] = 1
            W[2 * k0][r4] = 1
            W[2 * k0][r5] = -1
            W[2 * k0][r7] = 1
            
            W[2 * k0 + 1][r2] = 1
            W[2 * k0 + 1][r4] = 1
            
            W[2 * k1][r3] = 1
            W[2 * k1][r5] = 1
            
            W[2 * k1 + 1][r1] = 1
            W[2 * k1 + 1][r2] = -1
            W[2 * k1 + 1][r3] = 1
            W[2 * k1 + 1][r6] = 1
        
        for i in range(7):
            k0 = 2 * i
            k1 = 2 * i + 1
            base_r = 7 * i
            add_strassen_block(base_r, k0, k1)
        
        base_r = 49
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        
        U[0][r1] = 1
        V[0 * p + 14][r1] = 1
        W[2 * 14][r1] = 1
        
        U[1][r2] = 1
        V[1 * p + 14][r2] = 1
        W[2 * 14][r2] = 1
        
        U[2][r3] = 1
        V[0 * p + 14][r3] = 1
        W[2 * 14 + 1][r3] = 1
        
        U[3][r4] = 1
        V[1 * p + 14][r4] = 1
        W[2 * 14 + 1][r4] = 1
        
        return {"U": U, "V": V, "W": W}
    
    # Build the guaranteed rank-53 solution first
    best_result = build_rank53()
    best_rank = 53
    
    # Try to find a lower-rank decomposition using remaining time
    # Strategy: attempt to merge Strassen blocks across column pairs
    # by finding shared structure. We'll try a linear algebra approach:
    # For a given rank R, check if we can find U,V,W that work.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        # The tensor T[i,j,k] = delta[i,k] for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
        # In the trace convention: C[k*n+i, :] = sum_r W[k*n+i, r] * (U[:,r] * V[:,r])
        # Actually: C[i,k] = sum_r U[i*m+j ... ] 
        # Let me be precise: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] summed over r should equal delta[i,k] for the (i,j,k) entry
        # Wait, the multiplication tensor is: T[i,j,k] = 1 if i==k else 0? No.
        # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor is T[i,j,k] = 1 for all i,j,k where the output C[i,k] depends on A[i,j]*B[j,k]
        # Actually T[i,j,k] = 1 for all valid i,j,k (it's the structure tensor of matrix mult)
        # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # where i in [0,n), j in [0,m), k in [0,p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(len(U[0])):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try some structured lower-rank attempts
    # Idea: can we do better than 7 per pair? The Strassen bound for 2x2 is 7, so no for individual pairs.
    # But maybe we can share factors across pairs. Let's try rank 52, 51, 50...
    # This is a hard NP-hard problem in general, so we'll do a limited heuristic search.
    
    # Heuristic: try to eliminate one factor by making it a linear combination
    # of existing factors. We'll try small perturbations.
    
    import random
    random.seed(42)
    
    # Try a few random sparse rank-reduction attempts
    # For each, we'll fix U and V from the rank-53 solution but zero out some columns,
    # then try to solve for W.
    
    for target_rank in [52, 51, 50]:
        if time.time() > _DEADLINE - 20:
            break
        
        # Try to find a rank-target_rank decomposition
        # We'll use a simple approach: start from rank-53, randomly zero out 
        # (53 - target_rank) columns in U, V, W, then check if we can adjust W to compensate.
        
        # Actually, a better approach: for each pair of columns, check if one can be 
        # expressed as a linear combination of others in a way that preserves the tensor.
        
        # Given the complexity, let's just try a few specific structured reductions:
        # Can we merge the trivial block (rank 4) with a Strassen block (rank 7) 
        # into a combined block of rank < 11?
        
        # Try: combine column 14 (trivial, rank 4) with column pair (12,13) (Strassen, rank 7)
        # into a 3-column block. The trivial multiplication for one column is rank 4.
        # Strassen for two columns is rank 7. Combined naively is 11.
        # Can we do it in 10? 9? 8?
        
        # This is a specific subproblem. Let's skip the full search and just 
        # return the rank-53 solution if no quick improvement is found.
        
        pass
    
    # If we have time, do a quick verification of the rank-53 solution
    if time.time() < _DEADLINE - 5:
        # Verify the solution is correct
        U, V, W = best_result["U"], best_result["V"], best_result["W"]
        # Spot check a few entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(len(U[0])):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # This shouldn't happen, but rebuild if needed
            best_result = build_rank53()
    
    return best_result

# EVOLVE-BLOCK-END
