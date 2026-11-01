import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (limit is 150s)
    n, m, p = DIMENSIONS
    
    # Trivial decomposition (Rank 45)
    rank_trivial = n * m * p
    U_triv = [[0] * rank_trivial for _ in range(n * m)]
    V_triv = [[0] * rank_trivial for _ in range(m * p)]
    W_triv = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j][col] = 1
                V_triv[j * p + k][col] = 1
                W_triv[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_U = U_triv
    best_V = V_triv
    best_W = W_triv
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Check all specific points
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if val != 1:
                        return False
        # Check off-diagonal points
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    is_consistent = (i == i2 and j == j2 and k == k2)
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if is_consistent:
                        if val != 1:
                            return False
                    else:
                        if val != 0:
                            return False
        return True
    
    # Verify trivial decomposition
    if not verify(U_triv, V_triv, W_triv, rank_trivial):
        return {"U": U_triv, "V": V_triv, "W": W_triv}
    
    # Try to find a better decomposition using random search with integer entries
    # The tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for consistent (i,j,k), else 0
    
    # For matrix multiplication, the rank is at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Also, for M_{n,m,p}, rank >= (nm + mp + pn)/2 for some bounds
    # Let's try ranks from 15 up to 44
    
    # Use a smarter approach: try to find low-rank decompositions
    # The entries should be integers or half-integers
    
    # Strategy: Use random initialization and iterative improvement
    # Or use known results: for 3x3x3, rank is 23 (Strassen's algorithm)
    # For 3x3x5, we can potentially do better than 45
    
    # Let's try a systematic search for small ranks
    # First, let's check if rank 15 is possible (lower bound)
    
    # Actually, let's use a different approach: 
    # The tensor for M_{3,3,5} has specific structure. 
    # We can try to find a decomposition by solving a system of equations.
    
    # For a given rank r, we need to find U (9xr), V (15xr), W (15xr) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # This is a bilinear problem. Let's try random search with integer/half-integer entries.
    
    def random_search(rank, max_iters=10000):
        """Try to find a decomposition of given rank using random search."""
        for attempt in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with random integer/half-integer values
            # Values in {-1, -0.5, 0, 0.5, 1}
            values = [-1, -0.5, 0, 0.5, 1]
            
            U = np.array([[np.random.choice(values) for _ in range(rank)] for _ in range(n * m)], dtype=float)
            V = np.array([[np.random.choice(values) for _ in range(rank)] for _ in range(m * p)], dtype=float)
            W = np.array([[np.random.choice(values) for _ in range(rank)] for _ in range(p * n)], dtype=float)
            
            # Check if this decomposition works
            # Compute the tensor
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.dot(U[a], V[b] * W[c])
                        if abs(val - 1) > 1e-9:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            if not success:
                continue
            
            # Check off-diagonal
            for a in range(n * m):
                if not success:
                    break
                for b in range(m * p):
                    if not success:
                        break
                    for c in range(p * n):
                        i, j = divmod(a, m)
                        j2, k = divmod(b, p)
                        k2, i2 = divmod(c, n)
                        is_consistent = (i == i2 and j == j2 and k == k2)
                        val = np.dot(U[a], V[b] * W[c])
                        if is_consistent:
                            if abs(val - 1) > 1e-9:
                                success = False
                                break
                        else:
                            if abs(val) > 1e-9:
                                success = False
                                break
                    if not success:
                        break
                if not success:
                    break
            if success:
                # Convert to list of lists
                U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n * m)]
                V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m * p)]
                W_list = [[float(W[i][j]) for j in range(rank)] for i in range(p * n)]
                return U_list, V_list, W_list
        
        return None
    
    # Try ranks from lower bound up
    # Lower bound for M_{3,3,5}: at least max(9, 15, 15) = 15
    # Also, rank >= (3*3 + 3*5 + 5*3)/2 = (9+15+15)/2 = 39/2 = 19.5, so at least 20
    # Let's try from 20 up
    
    for rank in range(20, best_rank):
        if time.time() > deadline - 15:
            break
        
        result = random_search(rank, max_iters=5000)
        if result is not None:
            U_list, V_list, W_list = result
            # Verify
            if verify(U_list, V_list, W_list, rank):
                best_rank = rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
