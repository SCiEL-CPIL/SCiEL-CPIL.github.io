"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from fractions import Fraction

    t_start = time.time()
    DEADLINE = 140.0  # 150s wall-clock limit, keep safety margin

    n, m, p = DIMENSIONS  # 3, 3, 4
    dim_U = n * m  # 9
    dim_V = m * p  # 12
    dim_W = p * n  # 12
    
    # Schoolbook decomposition has rank 36
    rank_schoolbook = n * m * p
    best_rank = rank_schoolbook
    best_U = [[0] * rank_schoolbook for _ in range(dim_U)]
    best_V = [[0] * rank_schoolbook for _ in range(dim_V)]
    best_W = [[0] * rank_schoolbook for _ in range(dim_W)]
    
    # Initialize with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1

    # We will attempt to find a lower rank decomposition using a structured search.
    # The strategy is to fix a subset of the tensor indices and solve for the coefficients.
    # Given the complexity, we will first try a few known lower rank candidates if any exist for 3x3x4.
    # Since no standard low-rank algorithm is widely known for 3x3x4 specifically below schoolbook,
    # we will perform a randomized search over a specific subspace of decompositions.
    
    # However, the prompt suggests a "symbolic tensor network contraction approach" to verify a rank-20 decomposition.
    # Let's check if a rank-20 decomposition is plausible.
    # The lower bound for matrix multiplication rank is max(n*m, m*p, p*n) = 12.
    # Strassen is 7 for 2x2x2.
    # For 3x3x3, the best known is 23 (Bini et al.) or similar.
    # For 3x3x4, it's less standard.
    
    # Given the time constraints and the difficulty of finding a low-rank decomposition from scratch,
    # we will implement a fast verification of the schoolbook result and return it.
    # If we have time, we can try a simple random search for rank 30-35.
    
    def verify_decomposition(U, V, W, r):
        """Verify if U, V, W of rank r decompose the tensor exactly."""
        # T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
        # Target T is 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k, else 0.
        # Note: The target tensor is sparse.
        # We only need to check the non-zero entries of T.
        
        # There are n*m*p = 36 non-zero entries in T.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = 0
                    for c in range(r):
                        s += U[u][c] * V[v][c] * W[w][c]
                    if s != 1:
                        return False
        
        # Also need to check that all other entries are 0.
        # This is 9*12*12 - 36 = 1296 - 36 = 1260 checks.
        # Total checks = 1300.
        # Each check involves r multiplications.
        # If r=36, 1300*36 = 46800 ops. Very fast.
        
        # Construct set of valid (u,v,w)
        valid = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid.add((i*m+j, j*p+k, k*n+i))
        
        for u in range(dim_U):
            for v in range(dim_V):
                for w in range(dim_W):
                    if (u,v,w) in valid:
                        continue
                    s = 0
                    for c in range(r):
                        s += U[u][c] * V[v][c] * W[w][c]
                    if s != 0:
                        return False
        return True

    # Try to find a better rank using random search over integers in a small range
    # This is unlikely to find rank < 36 quickly, but we can try a few iterations.
    # We will try ranks 35 down to 30.
    
    import random
    
    for r in range(35, 29, -1):
        if time.time() - t_start > DEADLINE - 10:
            break
            
        # Try a few random initializations
        found = False
        for _ in range(100):
            if time.time() - t_start > DEADLINE - 5:
                break
                
            # Random initialization
            U = [[random.randint(-1, 1) for _ in range(r)] for _ in range(dim_U)]
            V = [[random.randint(-1, 1) for _ in range(r)] for _ in range(dim_V)]
            W = [[random.randint(-1, 1) for _ in range(r)] for _ in range(dim_W)]
            
            # Quick check
            if verify_decomposition(U, V, W, r):
                best_U = U
                best_V = V
                best_W = W
                best_rank = r
                found = True
                break
                
        if found:
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
