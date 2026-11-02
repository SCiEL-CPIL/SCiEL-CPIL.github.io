"""Matrix multiplication 3x5 x 5x6 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify_full(U, V, W, n=3, m=5, p=6, tol=1e-9):
    """Verify the full tensor decomposition."""
    rank = len(U[0])
    # Check all (i,j,k) triples: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(s - 1) > tol:
                    return False
    # Check all other (a,b,c) entries should be 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i1, j1 = a // m, a % m
                j2, k2 = b // p, b % p
                k1, i2 = c // n, c % n
                if i1 == i2 and j1 == j2:
                    continue  # Already checked
                s = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                if abs(s) > tol:
                    return False
    return True

def _schoolbook(n=3, m=5, p=6):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    return U, V, W

def _try_merge_columns(U, V, W, n=3, m=5, p=6, max_iter=5000, tol=1e-9):
    """Try to reduce rank by finding linear dependencies among columns."""
    rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    # Build the tensor T from the decomposition
    # T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
    # We want to find if any column is a linear combination of others
    
    # For each pair of columns, check if one can be eliminated
    # This is complex, so use a simpler approach: try random linear combinations
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    for _ in range(max_iter):
        if time.time() > 140:
            break
        # Pick two random columns to try to merge
        c1 = random.randint(0, rank - 1)
        c2 = random.randint(0, rank - 1)
        if c1 == c2:
            continue
        
        # Try to express column c1 as alpha * column c2 + beta * (other columns)
        # Simplified: try to see if removing c1 and adjusting c2 works
        # This is a heuristic; in general, this is NP-hard
        
        # Alternative: try to find a 2x2 sub-decomposition that can be reduced
        # For now, just try random perturbations
        
        # Create a copy
        U_new = [row[:] for row in U]
        V_new = [row[:] for row in V]
        W_new = [row[:] for row in W]
        
        # Try to eliminate column c1 by expressing it in terms of c2
        # We need: for all a,b,c, U[a,c1]*V[b,c1]*W[c,c1] = sum_d U[a,d]*V[b,d]*W[c,d] - U[a,c2]*V[b,c2]*W[c,c2]
        # This is generally not possible with just two columns
        
        # Instead, try a different approach: use the fact that the tensor has structure
        # and try to find a lower rank decomposition from scratch using optimization
        
        break  # Skip this for now
    
    return best_U, best_V, best_W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify_full(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook verification failed")
    
    # Try to find a better decomposition using structured methods
    # For 3x5x6, we can try to use the structure of the multiplication
    
    # Approach: Use the fact that we can group terms
    # A (3x5) * B (5x6) = sum_j A[:,j] * B[j,:]
    # Each term is an outer product of a 3x1 and 1x6 vector
    
    # Try to find a decomposition with rank < 90 by exploiting linear dependencies
    # in the space of 3x6 matrices
    
    # The space of 3x6 matrices has dimension 18
    # The multiplication map is: (A, B) -> A*B
    # This is a bilinear map from (R^15 x R^30) -> R^18
    
    # The rank of this bilinear map is at most 15*30/18 = 25 if it were generic,
    # but the actual structure may allow lower rank
    
    # Try a random search for a valid decomposition with rank < 90
    for target_rank in range(89, 50, -1):
        if time.time() > deadline - 10:
            break
        
        # Try to find a random decomposition with this rank
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random small integers
            U_rand = [[0.0] * target_rank for _ in range(n*m)]
            V_rand = [[0.0] * target_rank for _ in range(m*p)]
            W_rand = [[0.0] * target_rank for _ in range(p*n)]
            
            # Use a structured initialization based on the schoolbook
            # Try to use a subset of the schoolbook terms
            indices = random.sample(range(n*m*p), min(target_rank, n*m*p))
            for idx, col_idx in enumerate(indices):
                i = col_idx // (m*p)
                j = (col_idx // p) % m
                k = col_idx % p
                U_rand[i*m + j][idx] = 1.0
                V_rand[j*p + k][idx] = 1.0
                W_rand[k*n + i][idx] = 1.0
            
            # Check if this works
            if _verify_full(U_rand, V_rand, W_rand, n, m, p):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = target_rank
                break
        
        if best_rank < n * m * p:
            break
    
    # If we found a better rank, try to reduce further
    if best_rank < n * m * p:
        for target_rank in range(best_rank - 1, max(20, best_rank - 20), -1):
            if time.time() > deadline - 10:
                break
            
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                
                U_rand = [[0.0] * target_rank for _ in range(n*m)]
                V_rand = [[0.0] * target_rank for _ in range(m*p)]
                W_rand = [[0.0] * target_rank for _ in range(p*n)]
                
                # Use a structured approach: try to find a valid decomposition
                # by solving a system of equations
                
                # For now, just try random
                for c in range(target_rank):
                    for a in range(n*m):
                        U_rand[a][c] = random.choice([-1, 0, 1])
                    for b in range(m*p):
                        V_rand[b][c] = random.choice([-1, 0, 1])
                    for c_idx in range(p*n):
                        W_rand[c_idx][c] = random.choice([-1, 0, 1])
                
                if _verify_full(U_rand, V_rand, W_rand, n, m, p):
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    best_rank = target_rank
                    break
            
            if best_rank <= target_rank + 1:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
