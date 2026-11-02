"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find U (nm x r), V (mp x r), W (pn x r) such that
    # for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Start with schoolbook rank 40
    rank_schoolbook = n * m * p  # 40
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :])
                    # W should be such that the product gives 1
                    # Actually the tensor is T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                    # where a=i*m+j, b=j*p+k, c=k*n+i
                    # But W is indexed by c, and we need T[a,b,c]=1 only when c corresponds to (i,k)
                    pass
        # Better verification: reconstruct all entries
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For matrix mult: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # So T[i*m+j, j*p+k, k*n+i] should be 1, and 0 otherwise for valid (i,j,k) combos
        # Actually the tensor is defined as T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # But this is only defined for consistent (i,j,k). The full tensor has entries for all (a,b,c) in [0,nm) x [0,mp) x [0,pn)
        # The matrix multiplication tensor has T[a,b,c] = 1 iff there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
        # For our dimensions, not all (a,b,c) are valid.
        
        # Let's verify by checking that the decomposition reproduces matrix multiplication
        # Create test matrices
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        AB = A @ B
        
        # Using the decomposition: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A_{i,j} * B_{j,k} / (A_{i,j}*B_{j,k})
        # This is getting complicated. Let me use a direct verification.
        
        # The bilinear map: given A and B, compute C_{i,k} = sum_j A_{i,j} B_{j,k}
        # Using decomposition: C_{i,k} = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * A_{i,j}) * (sum_j V[j*p+k,r] * B_{j,k})
        # Wait, that's not right either.
        
        # Correct formulation: The tensor T is such that for inputs x (size nm) and y (size mp),
        # the output z (size pn) has z_c = sum_{a,b} T[a,b,c] * x_a * y_b
        # where x_a = A_{i,j} for a=i*m+j, y_b = B_{j,k} for b=j*p+k, z_c = C_{i,k} for c=k*n+i
        # And T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for the same i,j,k
        # So z_c = sum_{a,b} T[a,b,c] x_a y_b = sum_{j} x_{i*m+j} y_{j*p+k} = sum_j A_{i,j} B_{j,k} = C_{i,k} ✓
        
        # Now with rank-r decomposition: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # So z_c = sum_r W[c,r] * (sum_a U[a,r] x_a) * (sum_b V[b,r] y_b)
        
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            # Compute sum_a U[a,r] x_a for each i
            # x_a = A_{i,j} for a=i*m+j
            left = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    left[i] += U[i*m+j, r] * A[i,j]
            # Compute sum_b V[b,r] y_b for each k
            # y_b = B_{j,k} for b=j*p+k
            right = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    right[k] += V[j*p+k, r] * B[j,k]
            # z_c for c=k*n+i: W[c,r] * left[i] * right[k]
            for k in range(p):
                for i in range(n):
                    C_decomp[i,k] += W[k*n+i, r] * left[i] * right[k]
        
        return np.allclose(C_decomp, AB, atol=tol)
    
    # Try to find lower rank by random search with integer/half-integer entries
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try ranks from 35 down to 25 (optimistic)
    for target_rank in range(35, 24, -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers/half-integers
            scale = 2
            U_try = np.random.randint(-scale, scale+1, (nm, target_rank)) * 0.5
            V_try = np.random.randint(-scale, scale+1, (mp, target_rank)) * 0.5
            W_try = np.random.randint(-scale, scale+1, (pn, target_rank)) * 0.5
            
            # Ensure some non-zero entries
            if np.all(U_try == 0) or np.all(V_try == 0) or np.all(W_try == 0):
                continue
            
            if verify(U_try, V_try, W_try):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
