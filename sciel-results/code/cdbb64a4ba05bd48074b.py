import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin before 120s limit
    
    n, m, p = 2, 3, 3
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)
    # W: (p*n, rank) = (6, rank)
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    # Tensor condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,k) for all i,j,k
    # where i in 0..n-1, j in 0..m-1, k in 0..p-1
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        n, m, p = 2, 3, 3
        # For each (i, k, j), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta(i,k)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    target = 1.0 if i == k else 0.0
                    if abs(s - target) > 1e-9:
                        return False
        return True
    
    def verify_numpy(U, V, W):
        """Faster verification using numpy."""
        n, m, p = 2, 3, 3
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each j, we need: sum_r U[:, j, r] * V[:, j, r] outer with W gives identity
        # More precisely: for each j, the matrix M_j[i,k] = sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r]
        # Wait, that's not right either. Let me re-derive.
        # 
        # The condition is: for all i, k, j:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,k)
        #
        # This is a 3D constraint. Let's just check all 18 combinations.
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w])
                    target = 1.0 if i == k else 0.0
                    if abs(s - target) > 1e-9:
                        return False
        return True
    
    # Schoolbook rank 18 fallback
    def schoolbook():
        rank = n * m * p  # 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    fallback = schoolbook()
    
    # Try to find lower rank decompositions
    # Known result: rank of 2x3x3 MM is 15
    # Let's try to find rank 15, 14, 13, 12
    
    def random_decomp(rank, seed=None):
        if seed is not None:
            random.seed(seed)
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
        return U, V, W
    
    def local_search(U, V, W, max_iter=5000, deadline=None):
        """Try to fix a near-solution by flipping individual entries."""
        rank = len(U[0])
        best_err = float('inf')
        
        def error(U, V, W):
            total = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        target = 1.0 if i == k else 0.0
                        total += (s - target) ** 2
            return total
        
        err = error(U, V, W)
        if err < 1e-9:
            return U, V, W, True
        
        for it in range(max_iter):
            if deadline and time.time() > deadline:
                break
            # Pick a random entry to flip
            which = random.randint(0, 2)  # 0=U, 1=V, 2=W
            if which == 0:
                idx = random.randint(0, n * m - 1)
                r = random.randint(0, rank - 1)
                old = U[idx][r]
                U[idx][r] = -old if old != 0 else 1
                new_err = error(U, V, W)
                if new_err < err:
                    err = new_err
                    if err < 1e-9:
                        return U, V, W, True
                else:
                    U[idx][r] = old
            elif which == 1:
                idx = random.randint(0, m * p - 1)
                r = random.randint(0, rank - 1)
                old = V[idx][r]
                V[idx][r] = -old if old != 0 else 1
                new_err = error(U, V, W)
                if new_err < err:
                    err = new_err
                    if err < 1e-9:
                        return U, V, W, True
                else:
                    V[idx][r] = old
            else:
                idx = random.randint(0, p * n - 1)
                r = random.randint(0, rank - 1)
                old = W[idx][r]
                W[idx][r] = -old if old != 0 else 1
                new_err = error(U, V, W)
                if new_err < err:
                    err = new_err
                    if err < 1e-9:
                        return U, V, W, True
                else:
                    W[idx][r] = old
        return U, V, W, False
    
    # Try structured approach: for each j, the slice must give identity
    # Use Strassen-like structure for the 2x2 identity
    
    # Try rank 15 with structured initialization
    def try_structured(rank):
        """Try to build a structured decomposition."""
        if time.time() > DEADLINE:
            return None
        
        # For each j, we need 2x2 identity from the (i,k) indices
        # Use a pattern where for each j, we have some shared terms
        
        # Strategy: use rank 15 = 3*5, try 5 terms per j with some sharing
        # Or use known results
        
        # Let's try a different approach: use the fact that 
        # 2x3x3 can be decomposed using 2x2x3 (rank 7*3=21? No)
        
        # Actually, let's try exhaustive search for small ranks with random restarts
        return None
    
    # Random search for rank 15, 14, 13, 12
    best_result = None
    
    for target_rank in [15, 14, 13, 12]:
        if time.time() > DEADLINE - 5:
            break
        
        found = False
        max_restarts = 200 if target_rank >= 14 else 100
        restarts = 0
        
        while restarts < max_restarts and not found:
            if time.time() > DEADLINE - 5:
                break
            
            U, V, W = random_decomp(target_rank, seed=restarts)
            U, V, W, success = local_search(U, V, W, max_iter=2000, deadline=DEADLINE - 5)
            
            if success:
                # Verify with exact check
                if verify_numpy(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    found = True
                    break
            restarts += 1
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # If no lower rank found, try one more approach: systematic for rank 15
    if time.time() < DEADLINE - 10:
        # Try a more structured search for rank 15
        # Fix W to be sparse (each row has at most 2 non-zeros)
        # and search U, V
        
        for attempt in range(50):
            if time.time() > DEADLINE - 5:
                break
            
            random.seed(attempt * 1000)
            rank = 15
            
            # Initialize W sparsely
            W = [[0] * rank for _ in range(p * n)]
            for row in range(p * n):
                # 1-2 non-zeros per row
                num_nz = random.randint(1, 2)
                cols = random.sample(range(rank), num_nz)
                for c in cols:
                    W[row][c] = random.choice([-1, 1])
            
            # Initialize U, V randomly
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            
            U, V, W, success = local_search(U, V, W, max_iter=3000, deadline=DEADLINE - 5)
            
            if success and verify_numpy(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                break
    
    if best_result is not None:
        return best_result
    
    return fallback
