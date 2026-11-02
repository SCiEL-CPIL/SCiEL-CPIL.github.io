import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term
    # a, b, c, d are indices in U (local 2x2)
    # e, f, g, h are indices in V (local 2x2)
    # coeffs_u: dict mapping local index to coefficient
    # coeffs_v: dict mapping local index to coefficient
    # coeffs_w: dict mapping local output (0..3) to coefficient
    def add_strassen_term(coeffs_u, coeffs_v, coeffs_w, local_u_base, local_v_base, local_w_base):
        nonlocal col
        # local_u_base: (i, j) offset in U
        # local_v_base: (j, k) offset in V
        # local_w_base: (k, i) offset in W
        
        # coeffs_u keys: 0..3 for 2x2 block
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        for idx, coeff in coeffs_u.items():
            if coeff != 0:
                r, c = divmod(idx, 2)
                u_idx = (local_u_base[0] + r) * m + (local_u_base[1] + c)
                U[u_idx][col] += coeff
                
        for idx, coeff in coeffs_v.items():
            if coeff != 0:
                r, c = divmod(idx, 2)
                v_idx = (local_v_base[0] + r) * p + (local_v_base[1] + c)
                V[v_idx][col] += coeff
                
        for idx, coeff in coeffs_w.items():
            if coeff != 0:
                # idx 0..3 corresponds to C11, C12, C21, C22
                # W index: k*n + i
                # C11: k=local_w_base[0], i=local_w_base[1]
                # C12: k=local_w_base[0]+1, i=local_w_base[1]
                # C21: k=local_w_base[0], i=local_w_base[1]+1
                # C22: k=local_w_base[0]+1, i=local_w_base[1]+1
                if idx == 0:
                    w_idx = local_w_base[0] * n + local_w_base[1]
                elif idx == 1:
                    w_idx = (local_w_base[0] + 1) * n + local_w_base[1]
                elif idx == 2:
                    w_idx = local_w_base[0] * n + (local_w_base[1] + 1)
                elif idx == 3:
                    w_idx = (local_w_base[0] + 1) * n + (local_w_base[1] + 1)
                W[w_idx][col] += coeff
        col += 1

    # Strassen coefficients
    # M1 = (a+d)(e+h)
    add_strassen_term({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:-1}, (0,0), (0,0), (0,0))
    # M2 = (c+d)e
    add_strassen_term({2:1, 3:1}, {0:1}, {2:1}, (0,0), (0,0), (0,0))
    # M3 = a(f-h)
    add_strassen_term({0:1}, {1:1, 3:-1}, {1:1}, (0,0), (0,0), (0,0))
    # M4 = d(g-e)
    add_strassen_term({3:1}, {2:1, 0:-1}, {0:1, 2:1}, (0,0), (0,0), (0,0))
    # M5 = (a+b)h
    add_strassen_term({0:1, 1:1}, {3:1}, {0:-1, 1:1}, (0,0), (0,0), (0,0))
    # M6 = (c-a)(e+f)
    add_strassen_term({2:1, 0:-1}, {0:1, 1:1}, {3:1}, (0,0), (0,0), (0,0))
    # M7 = (b-d)(g+h)
    add_strassen_term({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}, (0,0), (0,0), (0,0))
    
    # This adds 7 terms for the first 2x2x2 block.
    # We need to repeat this for:
    # 3 blocks of B (k=0,1,2)
    # 2 blocks of A (L, R) for each B block
    # Total 3 * 2 * 7 = 42 terms.
    
    # The code above hardcoded local bases (0,0).
    # We need to generalize.
    
    # Let's restart the construction logic properly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen_term_general(coeffs_u, coeffs_v, coeffs_w, u_base_i, u_base_j, v_base_j, v_base_k, w_base_k, w_base_i):
        nonlocal col
        for idx, coeff in coeffs_u.items():
            if coeff != 0:
                r, c = divmod(idx, 2)
                u_idx = (u_base_i + r) * m + (u_base_j + c)
                U[u_idx][col] += coeff
        for idx, coeff in coeffs_v.items():
            if coeff != 0:
                r, c = divmod(idx, 2)
                v_idx = (v_base_j + r) * p + (v_base_k + c)
                V[v_idx][col] += coeff
        for idx, coeff in coeffs_w.items():
            if coeff != 0:
                if idx == 0:
                    w_idx = w_base_k * n + w_base_i
                elif idx == 1:
                    w_idx = (w_base_k + 1) * n + w_base_i
                elif idx == 2:
                    w_idx = w_base_k * n + (w_base_i + 1)
                elif idx == 3:
                    w_idx = (w_base_k + 1) * n + (w_base_i + 1)
                W[w_idx][col] += coeff
        col += 1

    # Strassen Coeffs
    # 0: a, 1: b, 2: c, 3: d
    # 0: e, 1: f, 2: g, 3: h
    # 0: C11, 1: C12, 2: C21, 3: C22
    
    S1_u = {0:1, 3:1}; S1_v = {0:1, 3:1}; S1_w = {0:1, 3:-1}
    S2_u = {2:1, 3:1}; S2_v = {0:1}; S2_w = {2:1}
    S3_u = {0:1}; S3_v = {1:1, 3:-1}; S3_w = {1:1}
    S4_u = {3:1}; S4_v = {2:1, 0:-1}; S4_w = {0:1, 2:1}
    S5_u = {0:1, 1:1}; S5_v = {3:1}; S5_w = {0:-1, 1:1}
    S6_u = {2:1, 0:-1}; S6_v = {0:1, 1:1}; S6_w = {3:1}
    S7_u = {1:1, 3:-1}; S7_v = {2:1, 3:1}; S7_w = {0:1, 3:1}
    
    strassen_terms = [
        (S1_u, S1_v, S1_w),
        (S2_u, S2_v, S2_w),
        (S3_u, S3_v, S3_w),
        (S4_u, S4_v, S4_w),
        (S5_u, S5_v, S5_w),
        (S6_u, S6_v, S6_w),
        (S7_u, S7_v, S7_w)
    ]
    
    # A is 3x4.
    # A1 is rows 0,1. A2 is row 2.
    # B is 4x6.
    # B blocks: cols 0-1, 2-3, 4-5.
    
    for k_block in range(3):
        # B block k_block: cols 2*k_block, 2*k_block+1
        b_col_start = 2 * k_block
        for a_block in range(2):
            # A block a_block: cols 2*a_block, 2*a_block+1
            a_col_start = 2 * a_block
            # U base: row 0 or 1 (for A1), col a_col_start
            u_row_base = 0
            u_col_base = a_col_start
            # V base: row 0 (for B top half) or 2 (for B bottom half)?
            # Wait, B is 4x2.
            # B_top is rows 0,1. B_bottom is rows 2,3.
            # We split B into top 2x2 and bottom 2x2.
            # So for A_L (cols 0,1), we multiply with B_top (rows 0,1) and B_bottom (rows 2,3).
            # This corresponds to:
            # Term 1: A_L * B_top
            # Term 2: A_R * B_bottom
            # Wait, my previous logic was:
            # C = A_L B_T + A_R B_B.
            # This assumes A = [A_L A_R] and B = [B_T; B_B].
            # Yes.
            
            # So for each k_block, we have two Strassen blocks:
            # 1. A_L (cols 0,1) * B_T (rows 0,1)
            # 2. A_R (cols 2,3) * B_B (rows 2,3)
            
            # Block 1:
            u_base_i = 0
            u_base_j = 0
            v_base_j = 0
            v_base_k = b_col_start
            w_base_k = b_col_start
            w_base_i = 0
            
            for tu, tv, tw in strassen_terms:
                add_strassen_term_general(tu, tv, tw, u_base_i, u_base_j, v_base_j, v_base_k, w_base_k, w_base_i)
                
            # Block 2:
            u_base_i = 0
            u_base_j = 2
            v_base_j = 2
            v_base_k = b_col_start
            w_base_k = b_col_start
            w_base_i = 0
            
            for tu, tv, tw in strassen_terms:
                add_strassen_term_general(tu, tv, tw, u_base_i, u_base_j, v_base_j, v_base_k, w_base_k, w_base_i)
                
    # Now C2: Row 2 of A (index 2) * B (4x6)
    # Rank 4.
    for j in range(4):
        # U: row 2, col j
        U[2 * m + j][col] = 1
        # V: row j, all cols? No.
        # V[j, k] = 1 for all k?
        # W[k, 2] = 1 for all k?
        # Let's check:
        # C[2, k] += U[2, j] V[j, k] W[k, 2]
        # If V[j, k] = 1 and W[k, 2] = 1, then C[2, k] += 1 * 1 * 1 = 1.
        # But we need A[2, j] B[j, k].
        # U extracts A[2, j].
        # V extracts B[j, k].
        # So V[j, k] should be 1.
        # W[k, 2] should be 1.
        for k in range(6):
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    import numpy as np
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Verify
    n, m, p = 3, 4, 6
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Compute C_decomp
    # C[i, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] ?
    # No, the decomposition is C = sum_r U_r V_r W_r.
    # U_r is n x m. V_r is m x p. W_r is p x n?
    # Wait, the trace convention W[k*n+i, r] implies W is p x n.
    # The product is not standard matrix multiplication of U_r V_r W_r.
    # It is a tensor contraction.
    # C[i, k] = sum_{j, r} A[i, j] B[j, k] ... no.
    # The decomposition is:
    # C[i, k] = sum_{r} ( sum_{j} U[i*m+j, r] A[i, j] ? No.
    # U, V, W are the factors.
    # The tensor T[i, j, k] = delta_{jk} is decomposed as sum_r U_r[i, j] V_r[j, k] W_r[k, i]?
    # No.
    # Standard tensor decomposition for matrix mult:
    # C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Decomposition:
    # C[i, k] = sum_r ( sum_{j1} U_r[i, j1] A[i, j1] ) * ( sum_{j2} V_r[j2, k] B[j2, k] ) * W_r[k, i]?
    # No.
    # The factors U, V, W are such that:
    # U is n x m x rank.
    # V is m x p x rank.
    # W is p x n x rank.
    # And C[i, k] = sum_{r, j} U[i, j, r] V[j, k, r] W[k, i, r]?
    # Let's check the seed program.
    # U[i*m+j, c] = 1. V[j*p+k, c] = 1. W[k*n+i, c] = 1.
    # This implies:
    # For a specific c (corresponding to i, j, k):
    # U has 1 at (i, j).
    # V has 1 at (j, k).
    # W has 1 at (k, i).
    # The evaluation is:
    # C[i, k] = sum_c ( (U_c . A_i) * (V_c . B_k) * W_c[k, i] )?
    # No, that's not right.
    # Usually, it's:
    # C[i, k] = sum_c U_c[i, j] V_c[j, k] W_c[k, i] A[i, j] B[j, k]?
    
    # Let's look at the standard definition.
    # The tensor is $T_{ijk} = \delta_{jk}$.
    # Decomposition $T = \sum_r u_r \otimes v_r \otimes w_r$.
    # $T_{ijk} = \sum_r u_{r,i} v_{r,j} w_{r,k}$? No.
    # The matrix multiplication tensor is $M_{ijk} = \delta_{jk}$? No.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This is a bilinear map.
    # The tensor is $T_{i, j, k} = \delta_{jk}$? No, indices are $i$ (out row), $j$ (inner), $k$ (out col).
    # $T_{i, j, k} = \delta_{jk}$ is not correct because $j$ and $k$ are different dimensions in $A$ and $B$.
    # $A$ is $n \times m$. $B$ is $m \times p$.
    # $i \in 1..n, j \in 1..m, k \in 1..p$.
    # The tensor is $T_{i, j, k} = \delta_{jk}$? No, $j$ is from $A$, $k$ is from $B$.
    # The term is $A_{ij} B_{jk}$.
    # So the tensor is $T_{i, j, k} = 1$ if we sum over $j$?
    # No, the tensor defines the multiplication.
    # $C_{ik} = \sum_j T_{i, j, k} A_{ij} B_{jk}$.
    # Here $T_{i, j, k} = 1$ for all $i, j, k$? No.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # So $T_{i, j, k} = \delta_{jk}$? No, $j$ and $k$ are not the same index.
    # The tensor is simply 1 for all valid $i, j, k$?
    # No, the sum is over $j$.
    # $C_{ik} = \sum_j 1 \cdot A_{ij} B_{jk}$.
    # So the tensor $T_{i, j, k} = 1$?
    # If $T_{i, j, k} = 1$, then $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # Yes.
    # So we are decomposing the tensor $T_{i, j, k} = 1$ (for all $i,j,k$) into rank 1 tensors.
    # $T_{i, j, k} = \sum_r U_r[i, j] V_r[j, k] W_r[k, i]$?
    # No, $W$ has indices $k, i$.
    # $U$ has indices $i, j$.
    # $V$ has indices $j, k$.
    # $W$ has indices $k, i$.
    # The product is $U_r[i, j] V_r[j, k] W_r[k, i]$.
    # Sum over $r$.
    # This must equal 1 for all $i, j, k$.
    
    # Let's check the seed program with this formula.
    # For c corresponding to (i, j, k):
    # U[i, j] = 1. V[j, k] = 1. W[k, i] = 1.
    # Sum over r:
    # Term r=(i, j, k) gives 1*1*1 = 1.
    # Other terms are 0.
    # So sum is 1. Correct.
    
    # So my construction must satisfy:
    # $\sum_r U_r[i, j] V_r[j, k] W_r[k, i] = 1$ for all $i, j, k$.
    
    # Let's verify my Strassen construction.
    # For a Strassen term, we have $U_r, V_r, W_r$.
    # $U_r[i, j]$ is non-zero only for specific $i, j$.
    # $V_r[j, k]$ is non-zero only for specific $j, k$.
    # $W_r[k, i]$ is non-zero only for specific $k, i$.
    # The product $U_r[i, j] V_r[j, k] W_r[k, i]$ must sum to 1.
    
    # In Strassen, we compute $C = AB$.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # My Strassen terms compute linear combinations.
    # $M = L_A \cdot L_B$.
    # $L_A = \sum u_{ij} A_{ij}$.
    # $L_B = \sum v_{jk} B_{jk}$.
    # $M = (\sum u_{ij} A_{ij}) (\sum v_{jk} B_{jk})$.
    # This is a bilinear form.
    # In the tensor decomposition, this corresponds to a rank 1 tensor?
    # No, $M$ is a scalar.
    # The contribution to $C_{ik}$ is $w_{ik} M$.
    # $w_{ik}$ is the coefficient from $W_r$.
    # So the term is $w_{ik} (\sum u_{ij} A_{ij}) (\sum v_{jk} B_{jk})$.
    # $= \sum_{ij, jk} w_{ik} u_{ij} v_{jk} A_{ij} B_{jk}$.
    # The coefficient of $A_{ij} B_{jk}$ in $C_{ik}$ is $w_{ik} u_{ij} v_{jk}$.
    # We need the sum of these coefficients over $r$ to be 1 for all $i, j, k$.
    # i.e., $\sum_r W_r[k, i] U_r[i, j] V_r[j, k] = 1$.
    
    # My `add_strassen_term` sets:
    # $U_r[i, j] = u_{coeff}$.
    # $V_r[j, k] = v_{coeff}$.
    # $W_r[k, i] = w_{coeff}$.
    # This matches.
    
    # So the logic holds.
    
    # One detail: W index is $k \cdot n + i$.
    # In `add_strassen_term_general`, I used `w_idx = w_base_k * n + w_base_i`.
    # This is correct.
    
    # The C2 part:
    # $U[2, j] = 1$.
    # $V[j, k] = 1$.
    # $W[k, 2] = 1$.
    # Product: $1 \cdot 1 \cdot 1 = 1$.
    # Sum over $j$ (via rank):
    # For fixed $i=2, k$, sum over $r$ (which corresponds to $j$):
    # $\sum_j 1 \cdot 1 \cdot 1 = 4$?
    # No, for a specific $r$ (which is $j$), the term is non-zero only for that $j$.
    # So for a specific $A_{2j} B_{jk}$, the coefficient is 1.
    # Sum over $r$ gives $\sum_j 1 \cdot A_{2j} B_{jk}$.
    # This is correct.
    
    print("Constructed rank", rank)
    # Quick check with random matrices
    # C_decomp[i, k] = sum_r ( sum_j U[i*m+j, r] A[i, j] ) * ( sum_j V[j*p+k, r] B[j, k] ) * W[k*n+i, r] ?
    # No, the formula is:
    # C[i, k] = sum_{r, j} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i, j] B[j, k]
    # This is not efficient to compute directly.
    # But we can check if the tensor is 1.
    
    # Check tensor identity
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Should be 1 for all i, j, k.
    
    T_check = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_check[i, j, k] = s
                
    if np.allclose(T_check, 1.0):
        print("Tensor decomposition is EXACT.")
    else:
        print("Tensor decomposition FAILED.")
        print("Max error:", np.max(np.abs(T_check - 1.0)))

