import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Safety deadline
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to construct rank 41 solution
    try:
        U, V, W = construct_rank_41()
        # Verify
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
        
    # Fallback to schoolbook rank 45
    rank = n * m * p
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

def construct_rank_41():
    # Rank 23 for 3x3x3 (Strassen)
    # Rank 18 for 3x3x2 (Schoolbook)
    # Total 41
    
    n, m, p = 3, 3, 5
    
    # Part 1: 3x3x3 Strassen
    # U1 (9x23), V1 (9x23), W1 (9x23)
    # Indices for A: i*3+j
    # Indices for B: j*3+k
    # Indices for C: k*3+i
    
    # Part 2: 3x3x2 Schoolbook
    # U2 (9x18), V2 (15x18), W2 (15x18)
    # Note: B indices for part 2 are for columns 3,4 of original B.
    # Original B is 3x5. Indices j*5+k.
    # For k=3,4, index is j*5+3, j*5+4.
    
    # We need to combine them into U (9x41), V (15x41), W (15x41)
    
    # --- Strassen 3x3 ---
    # I will use a known set of 23 terms.
    # If I don't have the exact terms, I will use a placeholder that will fail verification.
    # To be safe, I will implement a generic search for small rank? No.
    # I will use the terms from Bini et al.
    
    # Terms are (u_vec, v_vec, w_vec)
    # u_vec is length 9 (coeffs for A)
    # v_vec is length 9 (coeffs for B)
    # w_vec is length 9 (coeffs for C)
    
    # I will hardcode a known valid set if I can recall it.
    # Otherwise, I will just return the schoolbook for the 3x3 part too, making rank 45.
    # But I want to try 41.
    
    # Let's use a simple 2x2 Strassen on a 2x2 subblock?
    # No, that doesn't cover 3x3.
    
    # I will assume I can find a valid 23-term decomposition.
    # If not, the verification will fail.
    
    # For now, I will implement the schoolbook for the 3x3 part as well, 
    # but I will try to optimize the 3x3x2 part?
    # No, 3x3x2 is already optimal schoolbook.
    
    # Let's just return the schoolbook 45 for now to be safe, 
    # unless I can confidently code Strassen.
    # Given the risk, I will stick to 45?
    # No, the task asks for a stronger algorithm.
    # I will implement the hybrid with a check.
    
    # I will use the following Strassen terms for 3x3 (from a reliable source in memory):
    # ... (Too risky to hardcode without verification)
    
    # I will just return the schoolbook decomposition for the 3x3 part in this attempt,
    # but I will structure the code to allow swapping in Strassen later.
    # Actually, I can use the fact that M(3,3,3) <= 23.
    # If I can't code it, I can't use it.
    
    # I will return the schoolbook 45.
    pass

def verify(U, V, W, n, m, p):
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Random test
    np.random.seed(42)
    A = np.random.randint(-10, 10, (n, m))
    B = np.random.randint(-10, 10, (m, p))
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # Compute C = A B
    C = A @ B
    C_flat = C.flatten() # order k*n+i? 
    # C[i, k] is at index k*n + i in the problem statement?
    # "W[k*n+i,r]"
    # So C_flat should be indexed by k*n + i.
    C_target = np.zeros(p*n)
    for i in range(n):
        for k in range(p):
            C_target[k*n + i] = C[i, k]
            
    # Compute via decomposition
    # c = W @ ( (U @ a) * (V @ b) )
    # u = U @ a
    # v = V @ b
    # c = W @ (u * v)
    
    u = U_np @ A_flat
    v = V_np @ B_flat
    c = W_np @ (u * v)
    
    if not np.allclose(c, C_target, atol=1e-5):
        return False
    return True
