import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # U: 16 x 49
    # V: 16 x 49
    # W: 16 x 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to flatten A index (i, j) to row index in U
    # U row corresponds to A[i, j] where i, j in 0..3
    # Index = i * m + j = i * 4 + j
    
    # Helper to flatten B index (i, j) to row index in V
    # V row corresponds to B[i, j]
    # Index = i * p + j = i * 4 + j
    
    # Helper to flatten C index (i, j) to row index in W
    # W row corresponds to C[i, j]
    # Note: W[k*n+i, r] convention in task description?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i, j] = sum A[i, k] B[k, j].
    # The tensor decomposition is C[i, j] = sum_r U[i, m, r] V[m, j, r] W[r, i, j]?
    # No, the interface is U, V, W matrices.
    # U shape (n*m, rank). Row index i*m+j corresponds to A[i, j]?
    # V shape (m*p, rank). Row index j*p+k corresponds to B[j, k]?
    # W shape (p*n, rank). Row index k*n+i?
    # The product is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
    # This would sum over j, k? No, the sum is only over r.
    # The result is a 3-tensor?
    # The task says "Matrix multiplication 4,4,4".
    # The output is C[i, k] = sum_j A[i, j] B[j, k].
    # The decomposition is C[i, k] = sum_r U[i, j, r] V[j, k, r] ...?
    # No, the standard bilinear form decomposition is:
    # C_{ik} = \sum_r U_{i, j, r} V_{j, k, r} is not right.
    # It is C_{ik} = \sum_r A_{i, j(r)} B_{k(r), k} ...?
    
    # Let's check the shapes:
    # U: (n*m, rank). Row index corresponds to (i, j) of A.
    # V: (m*p, rank). Row index corresponds to (j, k) of B.
    # W: (p*n, rank). Row index corresponds to (k, i) of C?
    # The task says: "W[k*n+i,r]".
    # This implies the row index of W is $k \cdot n + i$.
    # If we compute $\sum_r U[i, j, r] V[j, k, r] W[k, i, r]$, this doesn't look like matrix multiplication.
    
    # Standard tensor decomposition for matrix multiplication $C = AB$:
    # $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
    # This can be written as $\sum_{r} \lambda_r A_{i, \alpha_r} B_{\beta_r, k}$.
    # The matrices U, V, W in the prompt are likely:
    # U[i*m + j, r] is the coefficient of A[i, j] in the r-th term.
    # V[j*p + k, r] is the coefficient of B[j, k] in the r-th term.
    # W[k*n + i, r] is the coefficient of the term r in C[i, k]?
    # Wait, the index of W is $k \cdot n + i$.
    # If the term is $A_{i, j} B_{j, k}$, it contributes to $C_{i, k}$.
    # So W should have an entry at index corresponding to $C_{i, k}$.
    # The prompt says W has shape (p*n, rank).
    # And index is $k \cdot n + i$.
    # This corresponds to $C[i, k]$ if we interpret the first index as row and second as col?
    # Usually $C$ is $n \times p$.
    # If $W$ index is $k \cdot n + i$, and $k$ goes 0..p-1, $i$ goes 0..n-1.
    # This is a flattened $C$ matrix with row $k$ and col $i$?
    # But $C$ is $n \times p$. Row $i$, Col $k$.
    # So $W$ index $k \cdot n + i$ corresponds to $C[i, k]$?
    # Yes, if we flatten $C$ by columns (Fortran order) or if the index is simply mapping to the pair $(k, i)$.
    # Given the name "trace convention", it might be related to $Tr(C A^T B)$?
    # Regardless, we just need to fill W such that the decomposition holds.
    
    # Let's assume the standard bilinear decomposition:
    # $C_{ik} = \sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}$ is NOT correct because U, V, W are just coefficients.
    # The decomposition is:
    # $C = \sum_r W_r \otimes V_r \otimes U_r$?
    # No, the score is the rank.
    # The condition is:
    # For all i, k: $C_{ik} = \sum_j A_{ij} B_{jk} = \sum_r U[i, j, r] V[j, k, r] W[k, i, r]$?
    # No, the sum over r must produce the value.
    # Usually, the decomposition is $C_{ik} = \sum_r \lambda_r A_{i, \alpha_r} B_{\beta_r, k}$.
    # Here, U, V, W are matrices.
    # U[row_u, r], V[row_v, r], W[row_w, r].
    # The product is $\sum_r U[row_u, r] V[row_v, r] W[row_w, r]$.
    # This must equal $C_{ik}$ if $(row_u, row_v, row_w)$ correspond to $(A_{i, j}, B_{j, k}, C_{i, k})$?
    # No, the sum is over r.
    # It must be that for each r, the term is a product of a linear form in A, a linear form in B, and a coefficient for C.
    # So $C_{ik} = \sum_r U_{A, r} V_{B, r} W_{C, r}$.
    # Where $U_{A, r}$ is a linear combination of A entries.
    # $V_{B, r}$ is a linear combination of B entries.
    # $W_{C, r}$ is a linear combination of C entries?
    # No, W is a matrix.
    # The condition is:
    # $\sum_r U[i, j, r] V[j, k, r] W[k, i, r]$? No.
    
    # Let's look at the seed code.
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # Sum over c: $\sum_c U[i, j, c] V[j, k, c] W[k, i, c]$.
    # In seed, for a specific c, i, j, k are fixed.
    # So the sum is over c.
    # If we set U, V, W as in seed, the term for c corresponding to (i, j, k) is 1*1*1 = 1.
    # This contributes to $C[i, k]$?
    # The index of W is $k \cdot n + i$.
    # So the term is associated with $C[i, k]$.
    # So the decomposition is:
    # $C[i, k] = \sum_j A[i, j] B[j, k] = \sum_c (\sum_j U[i, j, c]) (\sum_k V[j, k, c]) W[k, i, c]$?
    # No.
    # The standard form is:
    # $C[i, k] = \sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}$ is not right.
    
    # It is:
    # $C[i, k] = \sum_r A_{i, j_r} B_{k_r, k} \lambda_r$?
    # The matrices U, V, W represent the coefficients.
    # $U$ row $i \cdot m + j$ has non-zero at col r if $A_{i, j}$ is used in term r.
    # $V$ row $j \cdot p + k$ has non-zero at col r if $B_{j, k}$ is used in term r.
    # $W$ row $k \cdot n + i$ has non-zero at col r if term r contributes to $C_{i, k}$.
    
    # So, for a term r:
    # $A_{lin} = \sum_{i, j} U[i, j, r] A_{i, j}$
    # $B_{lin} = \sum_{j, k} V[j, k, r] B_{j, k}$
    # $C_{contrib} = W[k, i, r] A_{lin} B_{lin}$?
    # No, the product is $A_{lin} B_{lin}$.
    # This product is a scalar.
    # It is added to $C_{i, k}$ with coefficient $W[k, i, r]$.
    # So $C_{i, k} = \sum_r W[k, i, r] A_{lin, r} B_{lin, r}$.
    
    # This matches the seed code logic.
    
    # So, for our 49-rank decomposition:
    # For each column r (0..48):
    # 1. Compute $A_{lin, r}$: linear combination of A entries.
    #    This is given by $U[:, r]$.
    # 2. Compute $B_{lin, r}$: linear combination of B entries.
    #    This is given by $V[:, r]$.
    # 3. Compute the contribution to C.
    #    $A_{lin, r} B_{lin, r}$ is the scalar product.
    #    It contributes to $C_{i, k}$ with weight $W[k, i, r]$.
    
    # In our derivation:
    # $P_t = L_A(X_s) L_B(Y_s)$.
    # $L_A(X_s)$ is a linear combination of A entries.
    # $L_B(Y_s)$ is a linear combination of B entries.
    # So $U[:, r]$ should be the coefficients of $L_A(X_s)$.
    # $V[:, r]$ should be the coefficients of $L_B(Y_s)$.
    # $P_t$ contributes to $C_{i, k}$ with coefficient $W[k, i, r]$.
    # We calculated $E_{s, t}$ and $W_{block}(s)$.
    # The coefficient of $P_t$ in $C_{i, k}$ is determined by these.
    
    # Let's refine the W calculation.
    # $C_{block\_i, block\_j}$ is a $2 \times 2$ submatrix.
    # Entries are $C[2bi + di, 2bj + dj]$.
    # $M_s$ contributes to $C_{block\_i, block\_j}$ with coeff $w_{bi, bj}$.
    # $P_t$ contributes to $M_s$ entry $(di, dj)$ with coeff $E_{s, t}[di, dj]$.
    # So $C[2bi + di, 2bj + dj]$ gets contribution $w_{bi, bj} \cdot E_{s, t}[di, dj]$.
    # The index in W is $k \cdot n + i = (2bj + dj) \cdot 4 + (2bi + di)$.
    
    # We can now code this.
    
    # Precompute X_s, Y_s linear maps
    # X_s is 2x2. Each entry is a linear combo of A.
    # A is 4x4.
    
    # Let's define the coefficients for X_s and Y_s.
    # coeff_X[s, xi, yi, ai, aj] = coeff of A[ai, aj] in X_s[xi, yi]
    # coeff_Y[s, xi, yi, bi, bj] = coeff of B[bi, bj] in Y_s[xi, yi]
    
    coeff_X = np.zeros((7, 2, 2, 4, 4))
    coeff_Y = np.zeros((7, 2, 2, 4, 4))
    
    # s=0: X0 = A11 + A22
    # A11 is A[0:2, 0:2], A22 is A[2:4, 2:4]
    for i in range(2):
        for j in range(2):
            coeff_X[0, i, j, i, j] = 1
            coeff_X[0, i, j, i+2, j+2] = 1
            
    # s=1: X1 = A21 + A22
    # A21 is A[2:4, 0:2], A22 is A[2:4, 2:4]
    for i in range(2):
        for j in range(2):
            coeff_X[1, i, j, i+2, j] = 1
            coeff_X[1, i, j, i+2, j+2] = 1
            
    # s=2: X2 = A11
    for i in range(2):
        for j in range(2):
            coeff_X[2, i, j, i, j] = 1
            
    # s=3: X3 = A22
    for i in range(2):
        for j in range(2):
            coeff_X[3, i, j, i+2, j+2] = 1
            
    # s=4: X4 = A11 + A12
    # A11 is A[0:2, 0:2], A12 is A[0:2, 2:4]
    for i in range(2):
        for j in range(2):
            coeff_X[4, i, j, i, j] = 1
            coeff_X[4, i, j, i, j+2] = 1
            
    # s=5: X5 = A21 - A11
    for i in range(2):
        for j in range(2):
            coeff_X[5, i, j, i+2, j] = 1
            coeff_X[5, i, j, i, j] = -1
            
    # s=6: X6 = A12 - A22
    for i in range(2):
        for j in range(2):
            coeff_X[6, i, j, i, j+2] = 1
            coeff_X[6, i, j, i+2, j+2] = -1
            
    # Similarly for Y_s (B)
    # s=0: Y0 = B11 + B22
    for i in range(2):
        for j in range(2):
            coeff_Y[0, i, j, i, j] = 1
            coeff_Y[0, i, j, i+2, j+2] = 1
            
    # s=1: Y1 = B11
    for i in range(2):
        for j in range(2):
            coeff_Y[1, i, j, i, j] = 1
            
    # s=2: Y2 = B12 - B22
    for i in range(2):
        for j in range(2):
            coeff_Y[2, i, j, i, j+2] = 1
            coeff_Y[2, i, j, i+2, j+2] = -1
            
    # s=3: Y3 = B21 - B11
    for i in range(2):
        for j in range(2):
            coeff_Y[3, i, j, i+2, j] = 1
            coeff_Y[3, i, j, i, j] = -1
            
    # s=4: Y4 = B22
    for i in range(2):
        for j in range(2):
            coeff_Y[4, i, j, i+2, j+2] = 1
            
    # s=5: Y5 = B11 + B12
    for i in range(2):
        for j in range(2):
            coeff_Y[5, i, j, i, j] = 1
            coeff_Y[5, i, j, i, j+2] = 1
            
    # s=6: Y6 = B21 + B22
    for i in range(2):
        for j in range(2):
            coeff_Y[6, i, j, i+2, j] = 1
            coeff_Y[6, i, j, i+2, j+2] = 1
            
    # Inner Strassen L_A, L_B coefficients for x, y
    # L_A[t] is linear combo of x00, x01, x10, x11
    # x_idx: 0->00, 1->01, 2->10, 3->11
    LA_coeffs = [
        [1, 0, 0, 1], # P1: x00 + x11
        [0, 0, 1, 1], # P2: x10 + x11
        [1, 0, 0, 0], # P3: x00
        [0, 0, 0, 1], # P4: x11
        [1, 1, 0, 0], # P5: x00 + x01
        [0, 0, 1, -1],# P6: x10 - x00
        [0, 1, 0, -1] # P7: x01 - x11
    ]
    
    LB_coeffs = [
        [1, 0, 0, 1], # P1: y00 + y11
        [1, 0, 0, 0], # P2: y00
        [0, 1, 0, -1],# P3: y01 - y11
        [0, 0, 1, -1],# P4: y10 - y00
        [0, 0, 0, 1], # P5: y11
        [1, 1, 0, 0], # P6: y00 + y01
        [0, 0, 1, 1]  # P7: y10 + y11
    ]
    
    # E_st: contribution of P_t to Z_s entries (z00, z01, z10, z11)
    # z_idx: 0->00, 1->01, 2->10, 3->11
    E_st = [
        [1, 0, 0, 1], # P1 -> z00, z11
        [0, 0, 1, -1],# P2 -> z10, z11(-)
        [0, 1, 0, 1], # P3 -> z01, z11
        [1, 0, 1, 0], # P4 -> z00, z10
        [-1, 1, 0, 0],# P5 -> z00(-), z01
        [0, 0, 0, 1], # P6 -> z11
        [1, 0, 0, 0]  # P7 -> z00
    ]
    
    # W_block[s]: contribution of M_s to C blocks (C00, C01, C10, C11)
    # block_idx: 0->00, 1->01, 2->10, 3->11
    W_block = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1],# M2 -> C10, C11(-)
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> C00(-), C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # Calculate U[:, r]
            # U[i, j, r] = sum_{xi, yi} LA_coeffs[t][xi*2+yi] * coeff_X[s, xi, yi, i, j]
            for i in range(4):
                for j in range(4):
                    val = 0
                    for xi in range(2):
                        for yi in range(2):
                            val += LA_coeffs[t][xi * 2 + yi] * coeff_X[s, xi, yi, i, j]
                    U[i * 4 + j, r] = val
                    
            # Calculate V[:, r]
            # V[i, j, r] = sum_{xi, yi} LB_coeffs[t][xi*2+yi] * coeff_Y[s, xi, yi, i, j]
            for i in range(4):
                for j in range(4):
                    val = 0
                    for xi in range(2):
                        for yi in range(2):
                            val += LB_coeffs[t][xi * 2 + yi] * coeff_Y[s, xi, yi, i, j]
                    V[i * 4 + j, r] = val
                    
            # Calculate W[:, r]
            # W[k, i, r] is coeff for C[i, k]
            # C[i, k] is in block (bi, bj) where i = 2*bi + di, k = 2*bj + dj
            # Contribution = W_block[s][block_idx] * E_st[t][z_idx]
            # block_idx = bi * 2 + bj
            # z_idx = di * 2 + dj
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    w_block_coeff = W_block[s][block_idx]
                    if w_block_coeff == 0:
                        continue
                    for di in range(2):
                        for dj in range(2):
                            z_idx = di * 2 + dj
                            e_coeff = E_st[t][z_idx]
                            if e_coeff == 0:
                                continue
                            i = 2 * bi + di
                            k = 2 * bj + dj
                            # W index is k * n + i = k * 4 + i
                            W[k * 4 + i, r] += w_block_coeff * e_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
