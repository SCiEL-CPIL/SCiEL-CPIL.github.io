import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Target tensor T[i, j, k] = 1 if (i,j) in U, (j,k) in V, (k,i) in W
    # Actually, the tensor T has T[i, j, k] = 1 for all i,j,k? No.
    # Matrix mult C = A B. C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor T[i, j, k] is 1 if this term is part of the sum.
    # Yes, T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # We want U, V, W such that sum_r U[i, r] V[j, r] W[k, r] = 1 for all i,j,k.
    # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r)
    # Indices:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    # Check ranks from 1 to schoolbook_rank - 1
    for r in range(1, schoolbook_rank):
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of rank r
        # Use random restarts with integer entries in {-1, 0, 1}
        # If we find one, return it.
        
        # Optimization: Use a small number of restarts per rank.
        # For small r, search space is smaller? Not necessarily, but constraints are tighter.
        max_restarts = 1000 if r < 10 else 100
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random integers -1, 0, 1
            U = np.random.randint(-1, 2, size=(n * m, r))
            V = np.random.randint(-1, 2, size=(m * p, r))
            W = np.random.randint(-1, 2, size=(p * n, r))
            
            # Local search: minimize error
            # Error = || sum_r U[:,r] V[:,r] W[:,r] - T ||^2
            # This is a 3-way tensor contraction.
            
            # We can check if the current decomposition is exact.
            # Reconstruct C:
            # C[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Efficient check:
            # Let's compute the reconstructed tensor and compare to 1s.
            
            # Vectorized check:
            # This is tricky to vectorize fully without loops over i,j,k,r.
            # But n=2, m=3, p=4, r<=23.
            # Total ops: 2*3*4*23 = 552 multiplications/additions per check. Very fast.
            
            is_exact = True
            for i in range(n):
                for k in range(p):
                    val = 0
                    for j in range(m):
                        for r_idx in range(r):
                            val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    if val != 1:
                        is_exact = False
                        break
                if not is_exact:
                    break
            if not is_exact:
                continue
                
            if is_exact:
                # Found a decomposition!
                # Convert to lists for return
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
            # If not exact, do a few steps of local search?
            # Given the time budget and low probability, simple random restarts might be enough.
            # Or, we can try to refine the last random guess.
            # Let's try a few random perturbations of the current U, V, W.
            
            for _ in range(50):
                if time.time() > deadline:
                    break
                    
                # Perturb a random element in U, V, or W
                mat = np.random.choice([U, V, W])
                idx = np.random.randint(mat.size)
                old_val = mat.flat[idx]
                mat.flat[idx] = np.random.randint(-1, 2)
                
                # Check again
                is_exact = True
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for j in range(m):
                            for r_idx in range(r):
                                val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        if val != 1:
                            is_exact = False
                            break
                    if not is_exact:
                        break
                if is_exact:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                else:
                    # Revert if it didn't work? Or keep?
                    # Simplest: revert.
                    mat.flat[idx] = old_val

    # If no lower rank found, return schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
