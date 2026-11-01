import time
import random
import numpy as np

DIMENSIONS = [2, 2, 14]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i, j, k] = sum_l U[i,l] * V[j,l] * W[k,l]
    # A is n x m, B is m x p, C = A * B is n x p
    # A indices: i = a*n + b where a is row, b is col of A
    # B indices: j = c*m + d where c is col of A (row of B), d is col of B
    # C indices: k = e*n + f where e is row of C, f is col of C
    # C[e,f] = sum_c A[a,c] * B[c,f]
    
    # The tensor T should satisfy: for all a,c,f: T[a*m+c, c*p+f, a*n+f] = 1
    # Wait, let me think about the index convention more carefully.
    
    # U shape: (n*m, rank) - indexed by (i, l) where i corresponds to A entries
    # V shape: (m*p, rank) - indexed by (j, l) where j corresponds to B entries  
    # W shape: (p*n, rank) - indexed by (k, l) where k corresponds to C entries
    
    # A[i] where i = row_A * m + col_A
    # B[j] where j = row_B * p + col_B, but row_B = col_A
    # C[k] where k = row_C * n + col_C, but row_C = row_A, col_C = col_B
    
    # So the tensor element T[i, j, k] should be 1 when:
    # i = a*m + c, j = c*p + f, k = a*n + f
    # And 0 otherwise (for the non-zero pattern of the multiplication tensor)
    
    # Actually, the standard tensor for matrix multiplication is:
    # T[a*m+c, c*p+f, a*n+f] = 1 for all valid a, c, f
    # All other entries are 0.
    
    max_err = 0.0
    for a in range(n):
        for c in range(m):
            for f in range(p):
                i = a * m + c
                j = c * p + f
                k = a * n + f
                # Compute T[i, j, k]
                val = 0.0
                for l in range(rank):
                    val += U[i, l] * V[j, l] * W[k, l]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    # Also check that off-diagonal entries are 0
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                # Check if this is a "valid" multiplication entry
                a = i // m
                c = i % m
                c2 = j // p
                f = j % p
                a2 = k // n
                f2 = k % n
                
                if c != c2 or a != a2 or f != f2:
                    val = 0.0
                    for l in range(rank):
                        val += U[i, l] * V[j, l] * W[k, l]
                    if abs(val) > max_err:
                        max_err = abs(val)
    
    return max_err < tol

def construct():
    """
    Try to find a rank-minimal decomposition for 2x2 * 2x14 matrix multiplication.
    First tries the 4x4 embedding approach, falls back to 7 Strassen blocks (rank 49).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: 7 independent Strassen blocks, rank 49
    def make_strassen_fallback():
        num_blocks = p // 2  # 7 blocks
        rank = num_blocks * 7
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        U_forms = [
            [1, 0, 0, 1],   # M1 = A00 + A11
            [0, 0, 1, 1],   # M2 = A10 + A11
            [1, 0, 0, 0],   # M3 = A00
            [0, 0, 0, 1],   # M4 = A11
            [1, 1, 0, 0],   # M5 = A00 + A01
            [-1, 0, 1, 0],  # M6 = A10 - A00
            [0, 1, 0, -1]   # M7 = A01 - A11
        ]
        
        V_forms = [
            [1, 0, 0, 1],   # M1: e + h
            [1, 0, 0, 0],   # M2: e
            [0, 1, 0, -1],  # M3: f - h
            [-1, 0, 1, 0],  # M4: g - e
            [0, 0, 0, 1],   # M5: h
            [1, 1, 0, 0],   # M6: e + f
            [0, 0, 1, 1]    # M7: g + h
        ]
        
        W_forms = [
            [1, 0, 0, 1],   # M1: C00 + 1, C11 + 1
            [0, 0, 1, -1],  # M2: C10 + 1, C11 - 1
            [0, 1, 0, 1],   # M3: C01 + 1, C11 + 1
            [1, 0, 1, 0],   # M4: C00 + 1, C10 + 1
            [-1, 1, 0, 0],  # M5: C00 - 1, C01 + 1
            [0, 0, 0, 1],   # M6: C11 + 1
            [1, 0, 0, 0]    # M7: C00 + 1
        ]
        
        for r in range(num_blocks):
            v_idx_e = 2 * r
            v_idx_f = 2 * r + 1
            v_idx_g = 14 + 2 * r
            v_idx_h = 14 + 2 * r + 1
            
            w_idx_c00 = 4 * r
            w_idx_c01 = 4 * r + 2
            w_idx_c10 = 4 * r + 1
            w_idx_c11 = 4 * r + 3
            
            for s in range(7):
                c = 7 * r + s
                
                for i in range(4):
                    U[i][c] = U_forms[s][i]
                    
                V[v_idx_e][c] = V_forms[s][0]
                V[v_idx_f][c] = V_forms[s][1]
                V[v_idx_g][c] = V_forms[s][2]
                V[v_idx_h][c] = V_forms[s][3]
                
                W[w_idx_c00][c] = W_forms[s][0]
                W[w_idx_c01][c] = W_forms[s][1]
                W[w_idx_c10][c] = W_forms[s][2]
                W[w_idx_c11][c] = W_forms[s][3]
        
        return {"U": U, "V": V, "W": W}
    
    # Check if we have time to try the 4x4 embedding approach
    if time.time() < deadline - 10.0:
        try:
            result = _try_4x4_embedding(n, m, p, deadline)
            if result is not None:
                # Verify the decomposition
                if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
                    return result
        except Exception:
            pass
    
    # Fallback to Strassen
    return make_strassen_fallback()

def _try_4x4_embedding(n, m, p, deadline):
    """
    Try to use a 4x4 matrix multiplication decomposition to get a lower rank.
    
    The idea: embed the 2x2 * 2x14 problem into a 4x4 * 4x4 problem.
    A (2x2) is embedded into a 4x4 matrix by placing it in the top-left.
    B (2x14) is embedded into a 4x4 matrix.
    
    Actually, this approach is tricky because B is 2x14, not 2x2.
    Let's think differently: use the fact that we can combine multiple 2x2 blocks.
    
    For now, this is a placeholder that returns None to indicate failure.
    """
    # The 4x4 embedding doesn't directly apply because B is 2x14, not 2x2.
    # A proper embedding would require B to be square or we'd need a different approach.
    # Return None to indicate this path is not viable.
    return None
