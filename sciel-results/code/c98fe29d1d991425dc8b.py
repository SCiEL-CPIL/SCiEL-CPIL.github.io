import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    dim_U = n * m  # 16
    dim_V = m * p  # 20
    dim_W = p * n  # 20
    
    # Build the matrix multiplication tensor
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Strassen decomposition for 3x3x3
    # U_strassen: (9, 7), V_strassen: (9, 7), W_strassen: (9, 7)
    # Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 3x3x3: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    # Standard Strassen rank-7 decomposition
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    U_str = np.zeros((9, 7))
    V_str = np.zeros((9, 7))
    W_str = np.zeros((9, 7))
    
    # Index: for 3x3, a = i*3+j, b = j*3+k, c = k*3+i
    # A00: i=0,j=0 -> a=0
    # A01: i=0,j=1 -> a=1
    # A10: i=1,j=0 -> a=3
    # A11: i=1,j=1 -> a=4
    # B00: j=0,k=0 -> b=0
    # B01: j=0,k=1 -> b=1
    # B10: j=1,k=0 -> b=3
    # B11: j=1,k=1 -> b=4
    # C00: k=0,i=0 -> c=0
    # C01: k=0,i=1 -> c=3
    # C10: k=1,i=0 -> c=1
    # C11: k=1,i=1 -> c=4
    
    # M1 = (A00 + A11)(B00 + B11)
    U_str[0, 0] = 1; U_str[4, 0] = 1
    V_str[0, 0] = 1; V_str[4, 0] = 1
    W_str[0, 0] = 1; W_str[4, 0] = 1
    
    # M2 = (A10 + A11) B00
    U_str[3, 1] = 1; U_str[4, 1] = 1
    V_str[0, 1] = 1
    W_str[3, 1] = 1; W_str[4, 1] = 1
    
    # M3 = A00 (B10 - B11)
    U_str[0, 2] = 1
    V_str[3, 2] = 1; V_str[4, 2] = -1
    W_str[1, 2] = 1; W_str[4, 2] = -1
    
    # M4 = A11 (B01 - B00)
    U_str[4, 3] = 1
    V_str[1, 3] = 1; V_str[0, 3] = -1
    W_str[0, 3] = -1; W_str[4, 3] = 1
    
    # M5 = (A00 + A01) B11
    U_str[0, 4] = 1; U_str[1, 4] = 1
    V_str[4, 4] = 1
    W_str[1, 4] = 1; W_str[4, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U_str[3, 5] = 1; U_str[0, 5] = -1
    V_str[0, 5] = 1; V_str[1, 5] = 1
    W_str[3, 5] = 1; W_str[0, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U_str[1, 6] = 1; U_str[4, 6] = -1
    V_str[3, 6] = 1; V_str[4, 6] = 1
    W_str[1, 6] = 1; W_str[4, 6] = -1
    
    # Verify Strassen decomposition for 3x3x3
    T_str = np.zeros((9, 9, 9), dtype=float)
    for r in range(7):
        T_str += np.outer(U_str[:, r], np.outer(V_str[:, r], W_str[:, r])).reshape(9, 9, 9)
    
    # Check against actual 3x3x3 tensor
    T_333 = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T_333[a, b, c] = 1.0
    
    assert np.linalg.norm(T_str - T_333) < 1e-10, "Strassen decomposition verification failed"
    
    # Now build the full 4x4x5 tensor decomposition
    # Core: i in [0,3), j in [0,3), k in [0,3) -> 3x3x3
    # Row strip: i in [3,4), j in [0,3), k in [0,3) -> 1x3x3
    # Column strip: i in [0,3), j in [3,4), k in [0,3) -> 3x1x3
    # Corner: i in [3,4), j in [3,4), k in [0,3) -> 1x1x3
    # Plus k=3,4 parts: i in [0,4), j in [0,4), k in [3,5)
    
    # Strategy: Use Strassen for the 3x3x3 core (7 terms)
    # Then solve for additional terms to cover the rest
    
    # The remaining tensor entries that are NOT covered by the 3x3x3 core:
    # We need to find rank-1 terms that cover:
    # 1. i=3, j in [0,3), k in [0,3)
    # 2. i in [0,3), j=3, k in [0,3)
    # 3. i=3, j=3, k in [0,3)
    # 4. All entries with k in [3,5)
    
    # Let's build the residual tensor after removing the core
    T_residual = T.copy()
    
    # Subtract the 3x3x3 core contribution (which should be exactly 1 in those positions)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_residual[a, b, c] -= 1.0
    
    # Now T_residual has zeros in the 3x3x3 core region
    # We need to decompose T_residual into rank-1 terms
    
    # The support of T_residual:
    # - a = i*m+j where i=3 or j=3
    # - b = j*p+k where j=3 or k>=3
    # - c = k*n+i where i=3 or k>=3
    
    # Let's identify all non-zero entries in T_residual
    nonzero_entries = []
    for a in range(dim_U):
        for b in range(dim_V):
            for c in range(dim_W):
                if abs(T_residual[a, b, c]) > 1e-10:
                    nonzero_entries.append((a, b, c))
    
    # For each rank-1 term, we choose (u, v, w) vectors
    # u has support on {a : T_residual[a,b,c] != 0 for some b,c}
    # v has support on {b : T_residual[a,b,c] != 0 for some a,c}
    # w has support on {c : T_residual[a,b,c] != 0 for some a,b}
    
    # Let's try to find a small number of rank-1 terms
    # The residual has a specific structure
    
    # Actually, let's think about this differently.
    # The 4x4x5 tensor can be seen as:
    # - 3x3x3 core (rank 7 via Strassen)
    # - Additional rows/columns
    
    # Let's try ranks from 7 upward and see if we can find a decomposition
    
    def try_decomposition_with_core(rank_total, max_restarts=500, deadline=None):
        """
        Use Strassen for the 3x3x3 core (7 terms), then find (rank_total - 7) additional
        rank-1 terms to cover the residual.
        """
        extra_rank = rank_total - 7
        if extra_rank < 0:
            return None
        
        # Build the linear system for the extra terms
        # M @ x = T_residual_flat
        # where M columns are vec(u_l * v_l^T) for the extra terms
        # and x = w_flat
        
        # But we need to be smarter. Let's use a different approach:
        # Randomly initialize the extra U and V matrices, then solve for W.
        
        for restart in range(max_restarts):
            if deadline and time.time() > deadline:
                return None
            
            # Randomly initialize extra U and V with {-1, 0, 1}
            U_extra = np.random.choice([-1, 0, 1], size=(dim_U, extra_rank))
            V_extra = np.random.choice([-1, 0, 1], size=(dim_V, extra_rank))
            
            # Build M matrix: (dim_U * dim_V, extra_rank)
            M = np.zeros((dim_U * dim_V, extra_rank))
            for l in range(extra_rank):
                M[:, l] = np.outer(U_extra[:, l], V_extra[:, l]).flatten()
            
            # Solve for W_extra: (dim_W, extra_rank)
            W_extra = np.zeros((dim_W, extra_rank))
            valid = True
            
            for k in range(dim_W):
                b_k = T_residual[:, :, k].flatten()
                if np.allclose(b_k, 0, atol=1e-10):
                    # This k slice is zero, so W_extra[k, :] should be 0
                    continue
                try:
                    x, residuals, rank_m, s = np.linalg.lstsq(M, b_k, rcond=None)
                    if residuals is not None and len(residuals) > 0:
                        if residuals[0] > 1e-9:
                            valid = False
                            break
                    else:
                        res = np.linalg.norm(M @ x - b_k)
                        if res > 1e-9:
                            valid = False
                            break
                    W_extra[k, :] = x
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if W_extra is close to half-integers
            dist = np.abs(W_extra - np.round(W_extra * 2) / 2)
            if np.max(dist) < 1e-6:
                W_extra = np.round(W_extra * 2) / 2
                
                # Verify the full decomposition
                U_full = np.hstack([U_str_padded, U_extra])
                V_full = np.hstack([V_str_padded, V_extra])
                W_full = np.hstack([W_str_padded, W_extra])
                
                T_recon = np.zeros_like(T)
                for l in range(rank_total):
                    T_recon += np.outer(U_full[:, l], np.outer(V_full[:, l], W_full[:, l])).reshape(dim_U, dim_V, dim_W)
                
                if np.linalg.norm(T_recon - T) < 1e-9:
                    return U_full, V_full, W_full
        
        return None
    
    # Pad Strassen matrices to full dimensions
    U_str_padded = np.zeros((dim_U, 7))
    V_str_padded = np.zeros((dim_V, 7))
    W_str_padded = np.zeros((dim_W, 7))
    
    # Copy Strassen entries into the padded matrices
    # For 3x3x3: a=i*3+j, b=j*3+k, c=k*3+i
    # For 4x4x5: a=i*4+j, b=j*5+k, c=k*4+i
    # The core is i in [0,3), j in [0,3), k in [0,3)
    # So a = i*4+j (same as i*3+j for i,j < 3? No! i*4+j != i*3+j)
    # We need to map carefully
    
    # For the 3x3x3 core within the 4x4x5 tensor:
    # i in [0,3), j in [0,3), k in [0,3)
    # a = i*4 + j
    # b = j*5 + k
    # c = k*4 + i
    
    # Strassen indices for 3x3x3:
    # a_str = i*3 + j
    # b_str = j*3 + k
    # c_str = k*3 + i
    
    # Mapping: for i,j,k in [0,3):
    # a_full = i*4 + j
    # b_full = j*5 + k
    # c_full = k*4 + i
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a_str = i * 3 + j
                b_str = j * 3 + k
                c_str = k * 3 + i
                a_full = i * 4 + j
                b_full = j * 5 + k
                c_full = k * 4 + i
                
                for r in range(7):
                    U_str_padded[a_full, r] = U_str[a_str, r]
                    V_str_padded[b_full, r] = V_str[b_str, r]
                    W_str_padded[c_full, r] = W_str[c_str, r]
    
    # Verify the padded Strassen decomposition covers the core
    T_core_check = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for r in range(7):
        T_core_check += np.outer(U_str_padded[:, r], np.outer(V_str_padded[:, r], W_str_padded[:, r])).reshape(dim_U, dim_V, dim_W)
    
    # Check that T_core_check matches T in the core region
    core_match = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_core_check[a, b, c] - 1.0) > 1e-10:
                    core_match = False
                    break
    
    if not core_match:
        # The Strassen decomposition doesn't directly work with the different index mappings
        # We need to construct it differently
        
        # Let's build the Strassen decomposition directly for the 3x3x3 sub-tensor
        # with the correct index mapping
        
        U_str_padded = np.zeros((dim_U, 7))
        V_str_padded = np.zeros((dim_V, 7))
        W_str_padded = np.zeros((dim_W, 7))
        
        # M1 = (A00 + A11)(B00 + B11)
        # A00: i=0,j=0 -> a=0*4+0=0
        # A11: i=1,j=1 -> a=1*4+1=5
        # B00: j=0,k=0 -> b=0*5+0=0
        # B11: j=1,k=1 -> b=1*5+1=6
        # C00: k=0,i=0 -> c=0*4+0=0
        # C11: k=1,i=1 -> c=1*4+1=5
        
        U_str_padded[0, 0] = 1; U_str_padded[5, 0] = 1
        V_str_padded[0, 0] = 1; V_str_padded[6, 0] = 1
        W_str_padded[0, 0] = 1; W_str_padded[5, 0] = 1
        
        # M2 = (A10 + A11) B00
        # A10: i=1,j=0 -> a=1*4+0=4
        # A11: i=1,j=1 -> a=5
        # B00: j=0,k=0 -> b=0
        # C10: k=0,i=1 -> c=0*4+1=1
        # C11: k=1,i=1 -> c=5
        
        U_str_padded[4, 1] = 1; U_str_padded[5, 1] = 1
        V_str_padded[0, 1] = 1
        W_str_padded[1, 1] = 1; W_str_padded[5, 1] = 1
        
        # M3 = A00 (B10 - B11)
        # A00: i=0,j=0 -> a=0
        # B10: j=1,k=0 -> b=1*5+0=5
        # B11: j=1,k=1 -> b=6
        # C01: k=0,i=0 -> c=0
        # C11: k=1,i=0 -> c=1*4+0=4
        
        U_str_padded[0, 2] = 1
        V_str_padded[5, 2] = 1; V_str_padded[6, 2] = -1
        W_str_padded[0, 2] = 1; W_str_padded[4, 2] = -1
        
        # M4 = A11 (B01 - B00)
        # A11: i=1,j=1 -> a=5
        # B01: j=0,k=1 -> b=0*5+1=1
        # B00: j=0,k=0 -> b=0
        # C00: k=0,i=1 -> c=1
        # C11: k=1,i=1 -> c=5
        
        U_str_padded[5, 3] = 1
        V_str_padded[1, 3] = 1; V_str_padded[0, 3] = -1
        W_str_padded[1, 3] = -1; W_str_padded[5, 3] = 1
        
        # M5 = (A00 + A01) B11
        # A00: i=0,j=0 -> a=0
        # A01: i=0,j=1 -> a=0*4+1=1
        # B11: j=1,k=1 -> b=6
        # C01: k=0,i=0 -> c=0
        # C11: k=1,i=0 -> c=4
        
        U_str_padded[0, 4] = 1; U_str_padded[1, 4] = 1
        V_str_padded[6, 4] = 1
        W_str_padded[0, 4] = 1; W_str_padded[4, 4] = 1
        
        # M6 = (A10 - A00)(B00 + B01)
        # A10: i=1,j=0 -> a=4
        # A00: i=0,j=0 -> a=0
        # B00: j=0,k=0 -> b=0
        # B01: j=0,k=1 -> b=1
        # C10: k=0,i=1 -> c=1
        # C00: k=0,i=1 -> c=1
        
        U_str_padded[4, 5] = 1; U_str_padded[0, 5] = -1
        V_str_padded[0, 5] = 1; V_str_padded[1, 5] = 1
        W_str_padded[1, 5] = 1; W_str_padded[1, 5] += 1  # C10 gets both
        
        # Wait, let me recheck M6
        # M6 = (A10 - A00)(B00 + B01)
        # This contributes to C10 and C11? No.
        # (A10 - A00) is a 1x1 matrix (scalar)
        # (B00 + B01) is a 1x2 matrix
        # Product is 1x2, which should map to... 
        # Actually in Strassen, M6 contributes to C10 and C11? Let me recheck.
        
        # Standard Strassen for 2x2:
        # C00 = M1 - M4 + M6 - M7
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 - M6
        
        # So M6 contributes to C00 and C11 with negative sign.
        # For 3x3, the mapping is different. Let me use the standard 3x3 Strassen.
        
        # Actually, I'm overcomplicating this. Let me just verify the original Strassen 
        # decomposition works for the 3x3x3 case with the correct indexing, then map it.
        
        pass
    
    # For now, let's use a simpler approach: just try to find a decomposition from scratch
    # but with a better initialization strategy
    
    def try_find_decomposition(rank, max_restarts=1000, deadline=None):
        """
        Tries to find a rank 'rank' decomposition by randomly initializing U and V,
        then solving for W using least squares and checking for exactness.
        """
        for r in range(max_restarts):
            if deadline and time.time() > deadline:
                return None
            
            # Initialize U and V with random small integers
            U = np.random.choice([-1, 0, 1], size=(dim_U, rank))
            V = np.random.choice([-1, 0, 1], size=(dim_V, rank))
            
            # Construct the matrix M where columns are vec(U[:,l] V[:,l]^T)
            M = np.zeros((dim_U * dim_V, rank))
            for l in range(rank):
                M[:, l] = np.outer(U[:, l], V[:, l]).flatten()
            
            # Solve for W
            W = np.zeros((dim_W, rank))
            valid = True
            
            for k in range(dim_W):
                b_k = T[:, :, k].flatten()
                try:
                    x, residuals, rank_m, s = np.linalg.lstsq(M, b_k, rcond=None)
                    if residuals is not None and len(residuals) > 0:
                        if residuals[0] > 1e-9:
                            valid = False
                            break
                    else:
                        res = np.linalg.norm(M @ x - b_k)
                        if res > 1e-9:
                            valid = False
                            break
                    W[k, :] = x
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if W is close to half-integers
            dist = np.abs(W - np.round(W * 2) / 2)
            if np.max(dist) < 1e-6:
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                T_recon = np.zeros_like(T)
                for l in range(rank):
                    T_recon += np.outer(U[:, l], np.outer(V[:, l], W_rounded[:, l])).reshape(dim_U, dim_V, dim_W)
                
                if np.linalg.norm(T_recon - T) < 1e-9:
                    return U, V, W_rounded

        return None

    start_time = time.time()
    deadline = start_time + 140
    
    # Try ranks from 20 to 30
    for rank in range(20, 31):
        if time.time() > deadline - 10:
            break
            
        result = try_find_decomposition(rank, max_restarts=200, deadline=deadline)
        if result:
            U, V, W = result
            # Final verification
            T_recon = np.zeros_like(T)
            for l in range(rank):
                T_recon += np.outer(U[:, l], np.outer(V[:, l], W[:, l])).reshape(dim_U, dim_V, dim_W)
            if np.linalg.norm(T_recon - T) < 1e-9:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
