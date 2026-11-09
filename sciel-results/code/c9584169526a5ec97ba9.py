import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s kill

    n, m, p = 2, 3, 8

    def schoolbook(n, m, p):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, n, m, p, tol=1e-10):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    U0, V0, W0 = schoolbook(n, m, p)
    assert verify(U0, V0, W0, n, m, p), "Schoolbook failed"

    # Try Winograd-style reduction on k-dimension using 4-point Hadamard
    # H4 = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]], det=16
    # If we transform B's k-dimension by H4 (on groups of 4), the tensor changes
    # but rank is preserved. However, the transformed tensor might have lower rank.
    
    # Actually, let's try a direct construction for lower rank.
    # For [2,3,8], use the structure: T[i,j,k] = 1 for all i,j,k.
    
    # Construction idea: use R = 32 terms.
    # Split k into two groups of 4. For each group, use a 4-point transform.
    # This doesn't directly reduce rank, but let's try a different approach.
    
    # Direct low-rank attempt:
    # Since T is all-ones, try U[i,j,r] = a_r, V[j,k,r] = b_r, W[k*2+i,r] = c_r
    # Then sum_r a_r * b_r * c_r = 1 for all i,j,k, which means a_r*b_r*c_r must be 0 for r>0 and 1 for r=0.
    # This gives rank 1, but that's wrong because W depends on i.
    
    # Let me try: U[i,j,r] depends on i but not j, V[j,k,r] depends on j,k, W[k*2+i,r] depends on k,i.
    
    # Alternative: use the fact that 1 = 1*1*1 and try to find a decomposition
    # where the "1" is shared across multiple (i,j,k) triples.
    
    # For now, return schoolbook. The search for lower rank is complex.
    # Let's try a simple local search: perturb the schoolbook and see if we can reduce rank.
    
    best_rank = n * m * p
    best_decomp = (U0.copy(), V0.copy(), W0.copy())
    
    # Try to find rank 47 by removing one term and adjusting
    import random
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        idx = random.randint(0, best_rank - 1)
        U_mod = np.delete(U0, idx, axis=1)
        V_mod = np.delete(V0, idx, axis=1)
        W_mod = np.delete(W0, idx, axis=1)
        
        # Check if this is valid (it won't be, but check)
        if verify(U_mod, V_mod, W_mod, n, m, p):
            best_rank = best_rank - 1
            best_decomp = (U_mod, V_mod, W_mod)
            break
    
    U_best, V_best, W_best = best_decomp
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
