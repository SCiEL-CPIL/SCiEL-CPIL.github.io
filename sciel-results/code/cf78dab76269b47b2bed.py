import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Schoolbook decomposition
    # Rank R = n * m * p
    R = n * m * p
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i, j] corresponds to row i*m + j
                # V[j, k] corresponds to row j*p + k
                # W[k, i] corresponds to row k*n + i (Trace convention W[k*n+i, r])
                
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = R
    
    # Helper to check if a decomposition is valid
    def check_decomposition(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Greedy removal of terms
    # We try to remove terms one by one if the decomposition remains valid.
    # This is a heuristic that might find a slightly lower rank, though for
    # standard matrix multiplication, the rank is often rigid or close to schoolbook
    # unless specific algebraic structures are exploited (like Strassen).
    # Given the time limit and complexity, we attempt a few passes of greedy removal.
    
    current_rank = R
    removed_any = True
    
    while removed_any and time.time() < deadline:
        removed_any = False
        for r in range(current_rank):
            if time.time() > deadline:
                break
            
            # Try removing term r
            U_new = np.delete(U, r, axis=1)
            V_new = np.delete(V, r, axis=1)
            W_new = np.delete(W, r, axis=1)
            
            if check_decomposition(U_new, V_new, W_new, current_rank - 1):
                U = U_new
                V = V_new
                W = W_new
                current_rank -= 1
                removed_any = True
                break
            # If we don't break, we continue checking other terms in this pass
            # But usually, if one removal fails, we might need to re-evaluate.
            # To be safe and simple, we restart the loop if a removal succeeds.
            # If it fails, we just try the next term.
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_U = U
        best_V = V
        best_W = W
        
    # Convert to lists for the return interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
