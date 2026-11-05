import numpy as np
import random
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Target ranks to try
    # Schoolbook is 112.
    # We try to find a rank < 112.
    # Known upper bound is 97 (Strassen 49 + Schoolbook 48).
    # We try ranks 96 down to 80.
    
    target_ranks = list(range(96, 79, -1))
    
    # Pre-compute the tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[i, j, k] = 1 if i, j, k form a valid multiplication term
    # i = a*m + b (A index)
    # j = b*p + c (B index)
    # k = c*n + a (W index, trace convention W[k*n+i, r] -> k is row of W? No.)
    
    # Let's clarify the indices for the tensor equation:
    # C = A * B
    # C[a, c] = sum_b A[a, b] B[b, c]
    # U shape (n*m, r). Index i = a*m + b.
    # V shape (m*p, r). Index j = b*p + c.
    # W shape (p*n, r). Index k = c*n + a.
    # Equation: C[a, c] = sum_r U[a*m+b, r] V[b*p+c, r] W[c*n+a, r]
    # The tensor T[i, j, k] is 1 if there exists a, b, c such that:
    # i = a*m + b
    # j = b*p + c
    # k = c*n + a
    # And the term is A[a,b]B[b,c] contributing to C[a,c].
    # So T[i, j, k] = 1 if the indices are consistent.
    
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1
                
    # Flatten T for solving?
    # For each k (0 to p*n-1), we have a system for W[:, k].
    # System: sum_r U[i, r] V[j, r] W[r, k] = T[i, j, k]
    # Let M_k be the matrix where M_k[i, j, r] = U[i, r] V[j, r].
    # This is a linear system for W_k (vector of length r).
    # The "matrix" of the system is not 2D, it's a mapping from (i, j) to r.
    # We can reshape the left side.
    # Let A_sys be a matrix of size (n*m * m*p) x r.
    # Row idx = i * (m*p) + j.
    # Col idx = r.
    # A_sys[row, r] = U[i, r] V[j, r].
    # b_vec[row] = T[i, j, k].
    # Solve A_sys @ W_k = b_vec.
    
    A_sys = np.zeros((n*m * m*p, 100), dtype=float) # Max rank buffer
    
    # Pre-compute A_sys structure? No, U and V change.
    # But we can pre-compute the indices.
    
    def try_rank(r):
        # Randomize U and V
        # Use small integers to keep numbers manageable
        # U: (n*m, r)
        # V: (m*p, r)
        
        for attempt in range(50): # 50 random tries per rank
            U = np.random.randint(-1, 2, size=(n*m, r))
            V = np.random.randint(-1, 2, size=(m*p, r))
            
            # Check if U and V have enough rank?
            # If U or V is singular, we might not be able to represent the tensor.
            # But we are solving for W, so U and V just need to span the space appropriately.
            # Actually, if U and V are random, they are full rank with high probability.
            
            # Build A_sys
            # A_sys is (n*m * m*p) x r
            # We fill it column by column or row by row?
            # A_sys[i * m*p + j, col] = U[i, col] * V[j, col]
            
            A_sys_r = np.zeros((n*m * m*p, r), dtype=float)
            for col in range(r):
                u_col = U[:, col]
                v_col = V[:, col]
                # Outer product of u_col and v_col, flattened
                outer = np.outer(u_col, v_col)
                A_sys_r[:, col] = outer.flatten()
                
            # Solve for W columns
            W = np.zeros((p*n, r), dtype=float)
            valid = True
            
            for k in range(p*n):
                # b_vec for this k
                b_vec = T[:, :, k].flatten()
                
                # Solve A_sys_r @ W_k = b_vec
                # Using least squares
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A_sys_r, b_vec, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check residual
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-9:
                        valid = False
                        break
                else:
                    # If rank < r, check residual manually
                    residual = b_vec - A_sys_r @ sol
                    if np.linalg.norm(residual) > 1e-9:
                        valid = False
                        break
                        
                W[k, :] = sol
            
            if not valid:
                continue
                
            # Check if W entries are integer or half-integer
            # Multiply by 2 to check for integers
            W_2 = W * 2
            W_int = np.round(W_2)
            
            if not np.allclose(W_2, W_int, atol=1e-6):
                continue
                
            # Convert to int
            W_final = W_int.astype(int)
            
            # Verify exactness with integer arithmetic
            # C_calc = U @ W_final @ V.T
            # This is expensive. Let's just trust the float check for now?
            # Better to verify a few entries.
            # Actually, the task requires exact decomposition.
            # Let's verify one entry.
            # C[a, c] = sum_b A[a, b] B[b, c]
            # Our formula: sum_r U[a*m+b, r] V[b*p+c, r] W[c*n+a, r]
            
            # Let's do a full verification using integer math on a small sample or full if fast.
            # Full verification:
            # Compute T_reconstructed
            # T_recon[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            
            # This is a tensor contraction.
            # We can do:
            # Temp[i, k, r] = U[i, r] * W[k, r] -> sum_r? No.
            # We need sum_r U[i, r] V[j, r] W[k, r].
            
            # Let's use np.einsum
            # U: (i, r), V: (j, r), W: (k, r)
            # Result[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            
            # This might be slow for r~90, size 16x28x28.
            # 16*28*28*90 ~ 1.1M operations. Fast enough.
            
            recon = np.einsum('ir,jr,kr->ijk', U, V, W_final, optimize=True)
            
            if not np.array_equal(recon, T):
                continue
                
            # Success!
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W_final.tolist()
            }
            
        return None

    # Try ranks
    for r in target_ranks:
        res = try_rank(r)
        if res is not None:
            return res
            
    # Fallback: Schoolbook Rank 112
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
