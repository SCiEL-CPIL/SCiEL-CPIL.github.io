import time
import numpy as np

DIMENSIONS = [4, 4, 5]
START_TIME = time.time()
DEADLINE = 135.0  # Safety margin for 150s limit

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Generate the base rank-65 decomposition (Strassen 4x4x4 + Schoolbook 4x4x1)
    U_base, V_base, W_base = _build_base_decomposition(n, m, p)
    R_base = U_base.shape[1]
    
    # If time is tight, return the base decomposition immediately
    if time.time() - START_TIME > DEADLINE - 5.0:
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    # 2. Search for linear dependencies to reduce rank
    # We check if any column r can be expressed as a linear combination of other columns.
    # We try to express column r as a sum of a subset of other columns.
    # This is a heuristic search.
    
    U = U_base.copy()
    V = V_base.copy()
    W = W_base.copy()
    
    # We will try to eliminate columns from the end backwards, as they are often "newer" or less integrated.
    # Or we can just iterate through all columns.
    # Given R=65, checking all pairs is 65*64/2 ~ 2000 checks.
    # Each check involves solving a system or checking rank.
    
    # To be safe and fast, we will only try to eliminate columns using a small number of other columns (e.g., 1 or 2).
    # If col r = c * col s, then we can eliminate r.
    # If col r = c1 * col s1 + c2 * col s2, we can eliminate r.
    
    # Let's build the tensor T from the current U, V, W.
    # T[i, j, k, r] = U[i, r] * V[j, r] * W[k, r] (with W indexed as W[k*n+i, r])
    # Actually, the tensor is T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the decomposition is:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  -- NO, j is summed over in matmul.
    # The tensor T for matmul is T[i, j, k] = delta_{i,k}? No.
    # The tensor is T_{ijk} such that C_{ik} = sum_j A_{ij} B_{jk} = sum_j T_{ijk} A_{ij} B_{jk}.
    # So T_{ijk} = 1 if k is the output index? No.
    # T_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # So T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # In standard matmul, A_{ij} B_{jk} contributes to C_{ik}. So T_{ijk} = 1 for all i,j,k?
    # No, C_{ik} = sum_j A_{ij} B_{jk}.
    # So the term A_{ij} B_{jk} contributes 1 to C_{ik}.
    # Thus T_{ijk} = 1 for all i,j,k.
    
    # The decomposition is T_{ijk} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # (Using W indexed by k,i).
    
    # We want to find if column r is linearly dependent on others.
    # This is equivalent to checking if the vector of tensors for column r is in the span of others.
    # Let T_r be the tensor defined by U[:, r], V[:, r], W[:, r].
    # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can check if T_r is in the span of {T_s | s != r}.
    # This is a high-dimensional problem.
    
    # Heuristic: Check if T_r is a scalar multiple of T_s.
    # This is fast.
    
    eliminated = [False] * R_base
    U_new = []
    V_new = []
    W_new = []
    
    # Pre-compute tensors for each column? 65 * 16 * 20 * 20 = 416,000 floats.
    # 65 * 16 * 20 * 20 * 8 bytes = 4.16 MB.
    # This is fine.
    
    T = np.zeros((R_base, n, m, p), dtype=float)
    for r in range(R_base):
        for i in range(n):
            for j in range(m):
                u_val = U[i*m+j, r]
                if u_val == 0: continue
                for k in range(p):
                    v_val = V[j*p+k, r]
                    w_val = W[k*n+i, r]
                    T[r, i, j, k] = u_val * v_val * w_val
    
    # Now check for dependencies.
    # We will iterate r from R_base-1 down to 0.
    # If T_r is dependent on {T_s | s < r and not eliminated[s]}, we eliminate r.
    
    # To speed up, we only check linear dependence on a single other column first.
    for r in range(R_base - 1, -1, -1):
        if eliminated[r]: continue
        
        # Check if T_r is a multiple of T_s for some s < r
        found = False
        for s in range(r):
            if eliminated[s]: continue
            
            # Check if T_r = c * T_s
            # Find a non-zero entry in T_r
            idx = np.nonzero(T[r])
            if len(idx[0]) == 0:
                # T_r is zero tensor. Eliminate it.
                found = True
                break
            
            i0, j0, k0 = idx[0][0], idx[1][0], idx[2][0]
            val_r = T[r, i0, j0, k0]
            val_s = T[s, i0, j0, k0]
            
            if val_s == 0:
                continue
                
            c = val_r / val_s
            
            # Check if T_r == c * T_s
            if np.allclose(T[r], c * T[s], atol=1e-9):
                found = True
                break
        
        if found:
            eliminated[r] = True
    
    # Build new arrays
    keep_indices = [r for r in range(R_base) if not eliminated[r]]
    R_new = len(keep_indices)
    
    if R_new < R_base:
        U_final = U[:, keep_indices]
        V_final = V[:, keep_indices]
        W_final = W[:, keep_indices]
    else:
        U_final = U
        V_final = V
        W_final = W
    
    # Verify
    if not _verify(U_final, V_final, W_final, n, m, p):
        # Fallback to base
        U_final = U_base
        V_final = V_base
        W_final = W_base
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

def _build_base_decomposition(n, m, p):
    # 4x4x4 Strassen (Rank 49)
    # 4x4x1 Schoolbook (Rank 16)
    # Total Rank 65
    
    U4, V4, W4 = _make_4x4x4_strassen()
    R4 = U4.shape[1]
    
    U = np.zeros((n * m, R4 + n * m), dtype=float)
    V = np.zeros((m * p, R4 + n * m), dtype=float)
    W = np.zeros((p * n, R4 + n * m), dtype=float)
    
    U[:, :R4] = U4
    V[:, :R4] = V4
    W[:, :R4] = W4
    
    # Add Schoolbook for k=4
    k_extra = 4
    c = R4
    for i in range(n):
        for j in range(m):
            U[i * m + j, c] = 1.0
            V[j * p + k_extra, c] = 1.0
            W[k_extra * n + i, c] = 1.0
            c += 1
            
    return U, V, W

def _make_4x4x4_strassen():
    # Recursive Strassen for 4x4x4
    # Top level: 7 products of 2x2 blocks.
    # Each 2x2 block product is computed via Strassen (Rank 7).
    # Total Rank 49.
    
    n = m = p = 4
    U = np.zeros((n * m, 49), dtype=float)
    V = np.zeros((m * p, 49), dtype=float)
    W = np.zeros((p * n, 49), dtype=float)
    
    # Strassen 2x2 definitions
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the 7 top-level products.
    # For each product t (0..6), we have:
    # A_comb: 2x2 matrix of coefficients for A blocks
    # B_comb: 2x2 matrix of coefficients for B blocks
    # C_contrib: dict mapping C block to coefficient
    
    # Indices for blocks:
    # A block (br, bc) -> rows br*2..br*2+1, cols bc*2..bc*2+1
    # B block (bc, bk) -> rows bc*2..bc*2+1, cols bk*2..bk*2+1
    # C block (bi, bk) -> rows bi*2..bi*2+1, cols bk*2..bk*2+1
    
    products = [
        # M1
        {
            'A': [[1, 0], [0, 1]],  # A00 + A11
            'B': [[1, 0], [0, 1]],  # B00 + B11
            'C': {0: 1, 3: 1}       # C00 +1, C11 +1
        },
        # M2
        {
            'A': [[0, 0], [1, 1]],  # A10 + A11
            'B': [[1, 0], [0, 0]],  # B00
            'C': {2: 1, 3: -1}      # C10 +1, C11 -1
        },
        # M3
        {
            'A': [[1, 0], [0, 0]],  # A00
            'B': [[0, 1], [0, -1]], # B01 - B11
            'C': {1: 1, 3: 1}       # C01 +1, C11 +1
        },
        # M4
        {
            'A': [[0, 0], [0, 1]],  # A11
            'B': [[0, 0], [1, -1]], # B10 - B00
            'C': {0: -1, 2: 1}      # C00 -1, C10 +1
        },
        # M5
        {
            'A': [[1, 1], [0, 0]],  # A00 + A01
            'B': [[0, 0], [0, 1]],  # B11
            'C': {0: 1, 1: 1}       # C00 +1, C01 +1
        },
        # M6
        {
            'A': [[-1, 0], [1, 0]], # A10 - A00
            'B': [[1, 1], [0, 0]],  # B00 + B01
            'C': {3: 1}             # C11 +1
        },
        # M7
        {
            'A': [[0, 1], [0, -1]], # A01 - A11
            'B': [[0, 0], [1, 1]],  # B10 + B11
            'C': {0: -1}            # C00 -1
        }
    ]
    
    # For each top-level product, we decompose the 2x2 matrix multiplication into 7 scalar products (Strassen inner).
    # The inner Strassen operates on 2x2 matrices.
    # We need to map the inner indices to the global indices.
    
    # Inner Strassen 2x2:
    # A_in: 2x2, B_in: 2x2, C_in: 2x2
    # M1_in = (A00+A11)(B00+B11)
    # ...
    
    # We will hardcode the inner Strassen terms for the 2x2 case.
    # There are 7 terms.
    # For each term, we have U_in (4x1), V_in (4x1), W_in (4x1).
    
    # Let's generate the inner Strassen terms for a generic 2x2.
    # U_in[i, t], V_in[j, t], W_in[k, t]
    
    # We can just call a helper to get the 2x2 Strassen matrices.
    U2, V2, W2 = _make_2x2x2_strassen()
    
    for t_top in range(7):
        prod = products[t_top]
        A_comb = prod['A']
        B_comb = prod['B']
        C_comb = prod['C']
        
        # We need to form the full 4x4 U, V, W for this product.
        # The product is (A_comb * A_blocks) * (B_comb * B_blocks).
        # This is a 4x4 matrix product.
        # We decompose it into 7 inner Strassen products.
        
        # The inner Strassen acts on the 2x2 blocks of the "combined" A and B.
        # Wait, the "combined" A is a 4x4 matrix formed by linear combinations of A blocks.
        # A_combined[i, j] = sum_{br, bc} A_comb[br, bc] * A[br*2+i%2, bc*2+j%2]
        
        # We can view A_combined as a 2x2 block matrix where each block is 2x2.
        # Block (br, bc) of A_combined is A_comb[br, bc] * A_block(br, bc).
        
        # So we apply Strassen to this 2x2 block matrix.
        # The "A00" block for the inner Strassen is A_combined[0:2, 0:2].
        # This block is A_comb[0,0]*A_block(0,0) + A_comb[0,1]*A_block(0,1) ... No.
        # A_comb is 2x2. A_comb[0,0] corresponds to A_block(0,0).
        # So A_combined_block(0,0) = A_comb[0,0] * A_block(0,0).
        # A_combined_block(0,1) = A_comb[0,1] * A_block(0,1).
        
        # So the inner Strassen "A00" is A_comb[0,0] * A_block(0,0).
        # The inner Strassen "A01" is A_comb[0,1] * A_block(0,1).
        
        # We can just scale the U vectors of the inner Strassen by the appropriate coefficients.
        
        # Inner Strassen U vectors:
        # U2[i, t] is the coefficient for A_in[i, j] in term t.
        # A_in is the 2x2 matrix of blocks.
        # A_in[0,0] = A_combined_block(0,0) = A_comb[0,0] * A_block(0,0).
        
        # So for global index (i, j) in A_block(br, bc):
        # i_global = br*2 + i_local
        # j_global = bc*2 + j_local
        # Coefficient in U for term (t_top, t_inner) is:
        # A_comb[br, bc] * U2[i_local, t_inner]
        
        # Similarly for V:
        # B_in[bc, bk] = B_comb[bc, bk] * B_block(bc, bk).
        # Coefficient in V is B_comb[bc, bk] * V2[j_local, t_inner]
        
        # For W:
        # C_in[bi, bk] = C_combined_block(bi, bk).
        # C_combined is the result of the product.
        # The product contributes to C blocks according to C_comb.
        # C_block(bi, bk) += C_comb[bi*2+bk] * C_in[bi, bk]?
        # No, C_comb maps inner C blocks to global C blocks.
        # Inner C block (bi, bk) corresponds to global C block (bi, bk) scaled by C_comb[bi*2+bk].
        
        for t_inner in range(7):
            col = t_top * 7 + t_inner
            
            # Fill U
            for br in range(2):
                for bc in range(2):
                    a_coef = A_comb[br][bc]
                    if a_coef == 0: continue
                    for i_local in range(2):
                        for j_local in range(2):
                            i_global = br * 2 + i_local
                            j_global = bc * 2 + j_global
                            u_val = a_coef * U2[i_local * 2 + j_local, t_inner]
                            if u_val != 0:
                                U[i_global * m + j_global, col] = u_val
                            
            # Fill V
            for bc in range(2):
                for bk in range(2):
                    b_coef = B_comb[bc][bk]
                    if b_coef == 0: continue
                    for j_local in range(2):
                        for k_local in range(2):
                            j_global = bc * 2 + j_local
                            k_global = bk * 2 + k_local
                            v_val = b_coef * V2[j_local * 2 + k_local, t_inner]
                            if v_val != 0:
                                V[j_global * p + k_global, col] = v_val
                                
            # Fill W
            for bi in range(2):
                for bk in range(2):
                    c_coef = C_comb.get(bi * 2 + bk, 0)
                    if c_coef == 0: continue
                    for i_local in range(2):
                        for k_local in range(2):
                            i_global = bi * 2 + i_local
                            k_global = bk * 2 + k_local
                            w_val = c_coef * W2[k_local * 2 + i_local, t_inner]
                            if w_val != 0:
                                W[k_global * n + i_global, col] = w_val
                                
    return U, V, W

def _make_2x2x2_strassen():
    # Returns U (4x7), V (4x7), W (4x7) for 2x2x2 Strassen.
    # Indices:
    # U: i*2+j
    # V: j*2+k
    # W: k*2+i
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00, A11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C00 += M1, C11 += M1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 += M2, C11 -= M2
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C01 += M3, C11 += M3
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C00 -= M4, C10 += M4
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00 += M5, C01 += M5
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C11 += M6
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C00 -= M7
    W[0, 6] = -1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, U.shape[1]):
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False
    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B
    # build C from tensor
    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C, Ctrue, atol=tol)
