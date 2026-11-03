import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    deadline = start_time + 130
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # We want a decomposition T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # Shapes: U is (n*m, rank) = (9, rank), V is (m*p, rank) = (12, rank), W is (p*n, rank) = (12, rank)
    
    # Trivial schoolbook decomposition: rank = 36
    def schoolbook():
        rank = n * m * p  # 36
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    best_result = (U_fb, V_fb, W_fb, rank_fb)
    best_rank = rank_fb
    
    # Try to find lower rank decompositions
    # For 3x3x3, rank 23 is known (or 26 for simpler constructions)
    # For 3x3x4, we can try to use block structure
    
    # Strategy: 
    # 1. Try ranks from 20 to 35
    # 2. For each rank, try random search with {-1, 0, 1} and {-0.5, 0, 0.5, 1} entries
    # 3. Use structured approaches based on known 3x3x3 decompositions
    
    # First, let's try to construct a rank-35 decomposition using block structure:
    # The 3x3x4 multiplication can be seen as:
    # - 3x3x3 part (k=0,1,2): rank 23 or 26
    # - 3x3x1 part (k=3): rank 9
    # Total: 32 or 35
    
    # Let's try rank 32 first: 23 (for 3x3x3) + 9 (for 3x3x1)
    # Actually, let me just do random search efficiently
    
    # Efficient random search
    # For each rank, generate random matrices and check if they decompose T
    
    # To speed up, we can use the fact that the tensor is sparse (only 36 non-zero entries)
    # and use structured random matrices
    
    # Let's try a different approach: use the known rank-23 decomposition for 3x3x3
    # and extend it to 3x3x4
    
    # Actually, let me just do efficient random search with early termination
    
    # Precompute for faster checking
    # T_decomp[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    # We need T_decomp == T exactly
    
    # For small ranks, we can use exact arithmetic (integers)
    
    # Try ranks 20-35 with random {-1,0,1} matrices
    for rank in range(20, 36):
        if time.time() > deadline - 15:
            break
        if rank >= best_rank:
            continue
        
        # More attempts for lower ranks
        if rank <= 24:
            attempts = 20000
        elif rank <= 28:
            attempts = 50000
        elif rank <= 32:
            attempts = 100000
        else:
            attempts = 50000
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random matrices with {-1, 0, 1} entries
            # Use a smarter distribution: sparse matrices
            U = np.zeros((n*m, rank), dtype=int)
            V = np.zeros((m*p, rank), dtype=int)
            W = np.zeros((p*n, rank), dtype=int)
            
            # Each column should have some non-zero entries
            # For rank r, we need sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
            
            # Simple approach: random {-1,0,1} with probability 0.3 for non-zero
            for t in range(rank):
                # U column t
                idx = np.random.choice(n*m, size=np.random.randint(1, 4), replace=False)
                U[idx, t] = np.random.choice([-1, 1], size=len(idx))
                
                # V column t
                idx = np.random.choice(m*p, size=np.random.randint(1, 4), replace=False)
                V[idx, t] = np.random.choice([-1, 1], size=len(idx))
                
                # W column t
                idx = np.random.choice(p*n, size=np.random.randint(1, 4), replace=False)
                W[idx, t] = np.random.choice([-1, 1], size=len(idx))
            
            # Check decomposition
            # T_decomp[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
            # Only check non-zero entries of T
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        # Compute sum_t U[a,t]*V[b,t]*W[c,t]
                        s = np.dot(U[a, :], V[b, :] * W[c, :])
                        if s != 1.0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Also need to check that zero entries stay zero
            # This is expensive, so let's sample some zero entries
            valid = True
            for _ in range(20):
                a = np.random.randint(0, n*m)
                b = np.random.randint(0, m*p)
                c = np.random.randint(0, p*n)
                if T[a, b, c] == 0:
                    s = np.dot(U[a, :], V[b, :] * W[c, :])
                    if s != 0:
                        valid = False
                        break
            if valid:
                # Full verification
                T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
                if np.allclose(T_decomp, T, atol=1e-10):
                    best_result = (U, V, W, rank)
                    best_rank = rank
                    break
    
    # Try half-integer entries for some ranks
    for rank in range(22, 30):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
        
        attempts = 20000
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
            
            if np.allclose(T_decomp, T, atol=1e-10):
                best_result = (U, V, W, rank)
                best_rank = rank
                break
    
    U_best, V_best, W_best, _ = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
