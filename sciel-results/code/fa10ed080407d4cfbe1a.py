import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _construct_rank_71():
    """
    Returns a rank-71 decomposition for 3x5x5 matrix multiplication.
    Uses 4 disjoint 2x2x2 Strassen blocks and schoolbook for the rest.
    """
    n, m, p = 3, 5, 5
    
    # Define 4 non-overlapping 2x2x2 blocks
    # i in {0,1}, j in {0,1} and {2,3}, k in {0,1} and {2,3}
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    total_strassen_rank = 4 * 7
    total_terms = n * m * p
    strassen_terms_count = 4 * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += -1.0
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[j0*p + k0][c] += -1.0
        W[k0*n + i0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1.0
        U[i0*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1.0
        U[i0*m + j0][c] += -1.0
        V[j0*p + k0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += -1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Identify covered terms
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    # Add schoolbook terms for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1.0
                    V[j*p+k][col] += 1.0
                    W[k*n+i][col] += 1.0
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

def _try_find_rank_70(time_budget=10.0):
    """
    Attempts to find a rank-70 decomposition using random initialization and 
    gradient descent on the tensor reconstruction error.
    """
    n, m, p = 3, 5, 5
    target_rank = 70
    start_time = time.time()
    
    # Target tensor T_ijk = 1 for all i,j,k
    # We want sum_c U_ijk_c V_jkc_c W_kic_c = 1
    
    # Initialize U, V, W randomly
    # To help convergence, we can initialize with the rank-71 solution truncated
    # or just random. Random is better for escaping local minima, but slower.
    # Let's try initializing with the base solution's first 70 columns, scaled slightly.
    
    base = _construct_rank_71()
    U_base = np.array(base["U"])
    V_base = np.array(base["V"])
    W_base = np.array(base["W"])
    
    # We will run a few iterations of a simple multiplicative update / gradient step
    # Error E = sum_ijk (sum_c U_ijk_c V_jkc_c W_kic_c - 1)^2
    
    # This is a non-convex problem. Success is unlikely but possible.
    # We'll try a few random restarts within the time budget.
    
    for restart in range(10):
        if time.time() - start_time > time_budget:
            break
            
        # Initialize
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Precompute indices for speed
        # We can vectorize the tensor contraction
        # T_ijk = sum_c U_ijk_c * V_jkc_c * W_kic_c
        # This is a 3-tensor. We can reshape and use einsum.
        
        # U: (n*m, R) -> (n, m, R)
        # V: (m*p, R) -> (m, p, R)
        # W: (p*n, R) -> (p, n, R)
        
        U_3d = U.reshape(n, m, target_rank)
        V_3d = V.reshape(m, p, target_rank)
        W_3d = W.reshape(p, n, target_rank)
        
        # We want to minimize || U_3d * V_3d * W_3d - 1 ||^2
        # Let's do a few gradient steps.
        # Gradient wrt U_ijk_c: 2 * (T_ijk - 1) * V_jkc_c * W_kic_c
        # Gradient wrt V_jkc_c: 2 * (T_ijk - 1) * U_ijk_c * W_kic_c
        # Gradient wrt W_kic_c: 2 * (T_ijk - 1) * U_ijk_c * V_jkc_c
        
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > time_budget:
                break
                
            # Compute T_ijk
            # T_ijk = sum_c U_ijk_c V_jkc_c W_kic_c
            # Using einsum: 'irc,jsc,ksr->ij... no, indices are tricky'
            # U[i,j,c], V[j,k,c], W[k,i,c]
            # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            
            # This is a bit slow to compute from scratch every step in Python loops.
            # Let's use numpy einsum.
            # U: (n, m, R), V: (m, p, R), W: (p, n, R)
            # We need to sum over c.
            # T[i,j,k] = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            
            T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
            
            # Error
            err = np.sum((T - 1.0)**2)
            
            if err < 1e-6:
                # Success!
                return {"U": U, "V": V, "W": W}
            
            # Gradients
            # dE/dU_ijk_c = 2 * (T_ijk - 1) * V_jkc_c * W_kic_c
            # Note: V_jkc_c is indexed by j,k,c. U is i,j,c.
            # We need to align indices.
            # grad_U[i,j,c] = 2 * sum_k (T[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
            
            # This is also an einsum.
            # Let E = T - 1. Shape (n, m, p).
            E = T - 1.0
            
            # grad_U: (n, m, R)
            # grad_U[i,j,c] = 2 * sum_k E[i,j,k] * V[j,k,c] * W[k,i,c]
            # Indices: i,j,k for E; j,k,c for V; k,i,c for W.
            # Result i,j,c.
            grad_U = 2.0 * np.einsum('ijk,jkc,kic->ijc', E, V_3d, W_3d, optimize=True)
            
            # grad_V: (m, p, R)
            # grad_V[j,k,c] = 2 * sum_i E[i,j,k] * U[i,j,c] * W[k,i,c]
            # Indices: i,j,k for E; i,j,c for U; k,i,c for W.
            # Result j,k,c.
            grad_V = 2.0 * np.einsum('ijk,ijc,kic->jkc', E, U_3d, W_3d, optimize=True)
            
            # grad_W: (p, n, R)
            # grad_W[k,i,c] = 2 * sum_j E[i,j,k] * U[i,j,c] * V[j,k,c]
            # Indices: i,j,k for E; i,j,c for U; j,k,c for V.
            # Result k,i,c.
            grad_W = 2.0 * np.einsum('ijk,ijc,jkc->kic', E, U_3d, V_3d, optimize=True)
            
            # Update
            U_3d -= lr * grad_U
            V_3d -= lr * grad_V
            W_3d -= lr * grad_W
            
            # Reshape back
            U = U_3d.reshape(n*m, target_rank)
            V = V_3d.reshape(m*p, target_rank)
            W = W_3d.reshape(p*n, target_rank)
            
            # Optional: normalize to prevent divergence
            # This is a heuristic.
            
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = 115.0 # Leave 5s margin
    
    # Get the base rank-71 solution
    base_result = _construct_rank_71()
    
    # Try to find a rank-70 solution
    if time.time() < deadline:
        try:
            improved = _try_find_rank_70(time_budget=30.0)
            if improved is not None:
                # Verify the improved solution
                U = np.array(improved["U"])
                V = np.array(improved["V"])
                W = np.array(improved["W"])
                n, m, p = 3, 5, 5
                U_3d = U.reshape(n, m, 70)
                V_3d = V.reshape(m, p, 70)
                W_3d = W.reshape(p, n, 70)
                T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
                if np.max(np.abs(T - 1.0)) < 1e-5:
                    return improved
        except Exception:
            pass
    
    # Return the base rank-71 solution
    return base_result

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    # Verification
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    error += 1
    print(f"Verification errors: {error}")
