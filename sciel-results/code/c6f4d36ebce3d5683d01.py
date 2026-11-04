import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 145s to be safe

    # Use a Kronecker-product structured decomposition
    # The key insight: M_{n,m,p} can be decomposed using M_{n,p} structure
    # For n=3, p=7, the optimal rank is 19 (known from literature)
    # We combine this with the m=4 dimension
    
    # First, let's build a rank-19 decomposition for 3x7 matrix multiplication
    # This is a known result from tensor decomposition theory
    
    # For 3x7, we use a specific structured decomposition
    # The decomposition has rank 19
    
    # Let me construct this carefully
    # The 3x7 multiplication tensor T[i,k] = sum_j A[i,j]*B[j,k]
    # We need U[i,r], V[j,r], W[k,r] such that sum_r U[i,r]*V[j,r]*W[k,r] = delta(i,k)
    
    # Actually, for the full 3x4x7 problem:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # This is equivalent to: for each (i,j,k), we have a 1
    
    # Let me try a different approach: use the fact that we can decompose
    # M_{3,4,7} using a block structure
    
    # Approach: Use Strassen-like decomposition on the 4x4 middle part
    # But first, let me try a simpler structured approach
    
    # Actually, let me try to use a known rank-19 decomposition for 3x7
    # and combine it with the m=4 dimension
    
    # The 3x7 multiplication can be done with rank 19
    # Reference: This is a known optimal decomposition
    
    # Let me construct the rank-19 decomposition for 3x7
    # Then combine with m=4 to get rank 19*4 = 76
    
    # For 3x7 multiplication:
    # A is 3x7, B is 7x7 (wait, that's not right)
    # Actually for M_{3,7}: A is 3x7, B is 7x7? No...
    
    # Let me reconsider. For matrix multiplication M_{n,p}:
    # A is n x p, B is p x p? No, that's not standard.
    
    # Standard: M_{n,m,p} multiplies n x m by m x p to get n x p
    # So M_{3,4,7}: A is 3x4, B is 4x7, result is 3x7
    
    # The tensor is T[i,j,k] where i in [0,3), j in [0,4), k in [0,7)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let me try a Kronecker product approach
    # If we have a decomposition of M_{a,b,c} with rank r,
    # and M_{d,e,f} with rank s,
    # then M_{a*d, b*e, c*f} has rank r*s
    
    # But we need M_{3,4,7}. We can write this as M_{3,4,7}
    # We could try to decompose 4 = 2*2, so M_{3,4,7} ~ M_{3,2,7} x M_{1,2,1}
    # But that doesn't quite work.
    
    # Let me try a different structured approach
    # Use the fact that we can do 4x4 multiplication with rank 7 (Strassen)
    # But we have 3x4 and 4x7, not 4x4.
    
    # Actually, let me try to use a block decomposition
    # Split the 4x7 matrix into blocks
    
    # For now, let me try a random search for rank 83, 82, ...
    # If we find one, return it. Otherwise return schoolbook.
    
    # But first, let me try a simple structured approach that might work
    
    # Try rank 76 using a Kronecker-like structure
    # We can write M_{3,4,7} as a combination of simpler tensors
    
    # Actually, let me just try random search for now
    # and see if we can find rank < 84
    
    import random
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from best_rank - 1 down to 76
    for target_rank in range(best_rank - 1, 75, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random attempts for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Generate random U, V, W with entries in {-1, 0, 1}
            U = np.zeros((n * m, target_rank), dtype=int)
            V = np.zeros((m * p, target_rank), dtype=int)
            W = np.zeros((p * n, target_rank), dtype=int)
            
            # Fill with random -1, 0, 1
            for r in range(target_rank):
                # U column
                U[np.random.choice(n * m, size=min(3, n * m), replace=False), r] = np.random.choice([-1, 1], size=min(3, n * m))
                # V column
                V[np.random.choice(m * p, size=min(4, m * p), replace=False), r] = np.random.choice([-1, 1], size=min(4, m * p))
                # W column
                W[np.random.choice(p * n, size=min(3, p * n), replace=False), r] = np.random.choice([-1, 1], size=min(3, p * n))
            
            # Check if this decomposition works
            # T[i*m+j, j*p+k, k*n+i] should equal 1
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
            
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if val != 1:
                            valid = False
                            break
            
            if valid:
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Return schoolbook decomposition
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
