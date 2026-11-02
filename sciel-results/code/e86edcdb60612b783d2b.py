import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # U: (n*m, R) = (6, R)
    # V: (m*p, R) = (15, R)
    # W: (p*n, R) = (10, R)
    
    R = 17
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_idx: (i, j) -> i*m + j
    # v_idx: (j, k) -> j*p + k
    # w_idx: (k, i) -> k*n + i
    # coeff_u, coeff_v, coeff_w are the multipliers for this term
    def add_term(i, j, k, r, c_u, c_v, c_w):
        U[i * m + j][r] += c_u
        V[j * p + k][r] += c_v
        W[k * n + i][r] += c_w

    # Block 1: j=2, k in 0..4
    # Term: A_{i,2} * B_{2,k} -> C_{i,k}
    # This is a single rank-1 term covering all i, k.
    r = 0
    for i in range(n):
        add_term(i, 2, 0, r, 1, 1, 1) # Wait, this adds to W[k*n+i].
        # But B_{2,k} depends on k. U depends on i.
        # If I set U[i,2,r]=1 and V[2,k,r]=1, then sum is 1 for all k.
        # But V[2,k,r] must be 1 for ALL k in this term?
        # No, V is a matrix. V[j*p+k, r].
        # If I set V[2*p + k, r] = 1 for all k, and U[i*m + 2, r] = 1 for all i,
        # and W[k*n + i, r] = 1 for all k,i,
        # Then term is 1 for all i,k.
        # Is this correct?
        # C_{i,k} += A_{i,2} B_{2,k}.
        # Yes, this term is exactly that product.
        # So for r=0:
        for k in range(p):
            add_term(i, 2, k, r, 1, 1, 1)
            # This will sum up to p for U and V? No.
            # U[i*m+2, r] will be incremented p times?
            # No, add_term adds to specific indices.
            # U[i*m+2, r] is the same index for all k.
            # So U[i*m+2, r] becomes p.
            # V[2*p+k, r] is different for each k. So V[2*p+k, r] becomes 1.
            # W[k*n+i, r] is different for each k. So W[k*n+i, r] becomes 1.
            # Then product sum: U[i*m+2, r] * V[2*p+k, r] * W[k*n+i, r] = p * 1 * 1 = p.
            # We want 1. So we must scale.
            # Better to just define the matrices directly or scale.
            pass
    
    # Let's restart the filling logic to be cleaner.
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    # Block 1: j=2
    # Rank 0
    r = 0
    for i in range(n):
        U[i * m + 2][r] = 1.0
    for k in range(p):
        V[2 * p + k][r] = 1.0
        for i in range(n):
            W[k * n + i][r] = 1.0
    # Check: sum_r U[i,m+2,r] V[2p+k,r] W[kn+i,r]
    # = 1 * 1 * 1 = 1. Correct for j=2 part.
    # For other j, U is 0. Correct.
    
    # Block 2: j in {0,1}, k in {0,1} (Strassen 2x2x2)
    # Rank 1-7
    base_r = 1
    # Strassen coefficients for A (2x2) and B (2x2)
    # We map local indices (li, lj) to global (i, j)
    # i is 0..1, j is 0..1. k is 0..1.
    
    def add_strassen_block(offset_r, i_off, j_off, k_off):
        # i_off, j_off, k_off are global indices for local 0
        # Local i in 0,1 -> Global i = i_off + li
        # Local j in 0,1 -> Global j = j_off + lj
        # Local k in 0,1 -> Global k = k_off + lk
        
        # P1 = (A00 + A11)(B00 + B11)
        r = offset_r
        U[(i_off+0)*m + (j_off+0)][r] = 1.0
        U[(i_off+1)*m + (j_off+1)][r] = 1.0
        V[(j_off+0)*p + (k_off+0)][r] = 1.0
        V[(j_off+1)*p + (k_off+1)][r] = 1.0
        # W contributions:
        # C00 += P1, C01 -= P1, C10 -= P1, C11 += P1 ? No.
        # C00 = P1 - P4 - P6 - P7
        # C01 = P3 + P7
        # C10 = P2 + P5
        # C11 = P1 - P2 - P3 + P5
        # So W for P1:
        # W[k0*n + i0] += 1 (for C00)
        # W[k1*n + i0] -= 1 (for C01) -> Wait, C01 is row 0, col 1. Index k=1, i=0.
        # W[k0*n + i1] -= 1 (for C10)
        # W[k1*n + i1] += 1 (for C11)
        
        # Let's write W explicitly for each P
        
        W[(k_off+0)*n + (i_off+0)][r] = 1.0
        W[(k_off+1)*n + (i_off+0)][r] = -1.0
        W[(k_off+0)*n + (i_off+1)][r] = -1.0
        W[(k_off+1)*n + (i_off+1)][r] = 1.0
        
        # P2 = (A10 + A11) B00
        r = offset_r + 1
        U[(i_off+1)*m + (j_off+0)][r] = 1.0
        U[(i_off+1)*m + (j_off+1)][r] = 1.0
        V[(j_off+0)*p + (k_off+0)][r] = 1.0
        # C10 += P2, C11 -= P2
        W[(k_off+0)*n + (i_off+1)][r] = 1.0
        W[(k_off+1)*n + (i_off+1)][r] = -1.0
        
        # P3 = A00 (B10 - B00)
        r = offset_r + 2
        U[(i_off+0)*m + (j_off+0)][r] = 1.0
        V[(j_off+1)*p + (k_off+0)][r] = 1.0
        V[(j_off+0)*p + (k_off+0)][r] = -1.0
        # C01 += P3, C11 -= P3
        W[(k_off+1)*n + (i_off+0)][r] = 1.0
        W[(k_off+1)*n + (i_off+1)][r] = -1.0
        
        # P4 = A11 (B01 - B10)
        r = offset_r + 3
        U[(i_off+1)*m + (j_off+1)][r] = 1.0
        V[(j_off+0)*p + (k_off+1)][r] = 1.0
        V[(j_off+1)*p + (k_off+0)][r] = -1.0
        # C00 -= P4, C11 += P4
        W[(k_off+0)*n + (i_off+0)][r] = -1.0
        W[(k_off+1)*n + (i_off+1)][r] = 1.0
        
        # P5 = (A00 + A01) B11
        r = offset_r + 4
        U[(i_off+0)*m + (j_off+0)][r] = 1.0
        U[(i_off+0)*m + (j_off+1)][r] = 1.0
        V[(j_off+1)*p + (k_off+1)][r] = 1.0
        # C10 += P5, C11 += P5
        W[(k_off+0)*n + (i_off+1)][r] = 1.0
        W[(k_off+1)*n + (i_off+1)][r] = 1.0
        
        # P6 = (A10 - A00) (B00 + B01)
        r = offset_r + 5
        U[(i_off+1)*m + (j_off+0)][r] = 1.0
        U[(i_off+0)*m + (j_off+0)][r] = -1.0
        V[(j_off+0)*p + (k_off+0)][r] = 1.0
        V[(j_off+0)*p + (k_off+1)][r] = 1.0
        # C00 -= P6
        W[(k_off+0)*n + (i_off+0)][r] = -1.0
        
        # P7 = (A01 - A11) (B10 + B11)
        r = offset_r + 6
        U[(i_off+0)*m + (j_off+1)][r] = 1.0
        U[(i_off+1)*m + (j_off+1)][r] = -1.0
        V[(j_off+1)*p + (k_off+0)][r] = 1.0
        V[(j_off+1)*p + (k_off+1)][r] = 1.0
        # C00 -= P7, C01 += P7
        W[(k_off+0)*n + (i_off+0)][r] = -1.0
        W[(k_off+1)*n + (i_off+0)][r] = 1.0

    # Block 2: i=0..1, j=0..1, k=0..1
    add_strassen_block(1, 0, 0, 0)
    
    # Block 3: i=0..1, j=0..1, k=2..3
    add_strassen_block(8, 0, 0, 2)
    
    # Block 4: j in {0,1}, k=4
    # 2x2x1
    # C_i4 = A_i0 B_04 + A_i1 B_14
    r = 15
    # Term 1: A_i0 B_04
    for i in range(n):
        U[i*m + 0][r] = 1.0
        W[4*n + i][r] = 1.0
    V[0*p + 4][r] = 1.0
    
    r = 16
    # Term 2: A_i1 B_14
    for i in range(n):
        U[i*m + 1][r] = 1.0
        W[4*n + i][r] = 1.0
    V[1*p + 4][r] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
