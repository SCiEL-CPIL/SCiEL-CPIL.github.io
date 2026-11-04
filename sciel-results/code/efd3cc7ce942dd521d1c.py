"""Matrix multiplication 4x4 * 4x5: rank-minimal exact tensor decomposition.

Strategy: Build a rank-160 decomposition via Strassen-style block factorization
using Winograd 2x2x2 rank-7 sub-decompositions, combined with a direct
rank-20 decomposition for the remaining 4x5 column interactions.

We decompose 4x4 * 4x5 into:
  C = A * B where A is 4x4, B is 4x5, C is 4x5.

We use the block structure:
  A = [[A00, A01], [A10, A11]]  (each 2x2)
  B = [[B00, B01], [B10, B11]]  (each 2x5)

Then C = [[C00, C01], [C10, C11]] where each C block is 2x5.

We can write:
  C00 = A00*B00 + A01*B10
  C01 = A00*B01 + A01*B11
  C10 = A10*B00 + A11*B10
  C11 = A10*B01 + A11*B11

Each 2x2 * 2x5 product can be decomposed.
For a 2x2 * 2x5 product, we can use the rank-7 Winograd 2x2x2 for the first 2 columns
and extend to 5 columns.

Actually, a cleaner approach: treat the 4x4 * 4x5 as 4x4 * 4x4 (ignore column 5)
plus 4x4 * 4x1 (column 5). But that's not how it works.

Let me use a direct approach:
The matrix multiplication tensor for 4x4 * 4x5 has rank at most 4*4*5 = 80 (schoolbook).
We can do better with Strassen.

Using Strassen on 4x4 blocks:
  M1 = (A00+A11)*(B00+B11)
  M2 = (A10+A11)*B00
  M3 = A00*(B10-B00)
  M4 = A11*(B01-B11)
  M5 = (A00+A01)*B11
  M6 = (A10-A00)*(B00+B01)
  M7 = (A01-A11)*(B10+B11)

But B is 4x5, so B blocks are 2x5. This complicates things.

Simpler approach: Just use the schoolbook with rank 80, but try to find a lower rank.

Actually, let me construct a rank-56 decomposition using 2x2 blocks:
For each pair of rows and columns, use Winograd.

Let me use a straightforward block decomposition with rank 4*4*5 = 80 and see if I can reduce it.

Given the complexity, I'll implement the schoolbook (rank 80) as the baseline and add a
verification function.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _verify_decomposition(U, V, W, n=4, m=4, p=5):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u_col = U_np[:, c].reshape(n, m)
        v_col = V_np[:, c].reshape(m, p)
        w_col = W_np[:, c].reshape(p, n)
        # T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
        for i in range(n):
            for j in range(m):
                if u_col[i, j] != 0:
                    for k in range(p):
                        T[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
    
    # The matrix multiplication tensor should be T[i,j,k] = 1 if j==k else 0
    # Wait, that's not right. The tensor for matrix multiplication is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] should be 1 for all valid i,j,k (it's the identity in j,k pairing)
    # Actually, the tensor is T[i,j,k] = delta_{j,k} ... no.
    
    # The multiplication tensor is: C[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form: C = <A, B, W> where W[i,j,k] = 1 for the multiplication.
    # Actually, the canonical tensor for matrix mult n x m * m x p has T[i,j,k] = 1
    # for all i in [0,n), j in [0,m), k in [0,p). That's not right either.
    
    # The tensor T such that C[i,k] = sum_{j,c} ... 
    # Let me re-derive: We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{j,k} * ... 
    # No. We want: for all i,k: C[i,k] = sum_j A[i,j] B[j,k]
    # If we write C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_j V[j,k,c] B[j,k]) W[k,i,c]
    # That doesn't directly work.
    
    # The standard decomposition: C[i,k] = sum_c (U_col_c . A) (V_col_c . B) W[i,k,c]
    # where U_col_c is a vector in R^{n*m}, V_col_c in R^{m*p}, W in R^{p*n}
    # and the dots are with A flattened and B flattened.
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_c U[i,j,c] V[j',k,c] W[k,i,c] A[i,j] B[j',k] = A[i,j] B[j,k] * delta_{j,j'}
    # So: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{j,j'} for j'=j, i.e., = 1
    # And for j' != j, sum_c U[i,j,c] V[j',k,c] W[k,i,c] = 0
    
    # Wait, that's not quite right. Let me re-index.
    # A is n x m, flattened to index (i,j) -> i*m+j
    # B is m x p, flattened to index (j,k) -> j*p+k
    # C is n x p, but W is p x n, index (k,i) -> k*n+i
    
    # C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) * W[k*n+i, c]
    # We want this to equal sum_j A[i,j] B[j,k]
    # So we need: sum_c U[i*m+j1, c] V[j2*p+k, c] W[k*n+i, c] = delta_{j1, j2}
    
    # Let's verify this.
    errors = 0.0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    expected = 1.0 if j1 == j2 else 0.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m + j1][c] * V[j2*p + k][c] * W[k*n + i][c]
                    if abs(actual - expected) > 1e-9:
                        errors += abs(actual - expected)
    return errors < 1e-9

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 120
    
    # Schoolbook rank = n*m*p = 80
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # Use Strassen-like block decomposition
    
    # For 4x4 * 4x5, we can decompose into 2x2 * 2x5 blocks
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01], [B10, B11]] each 2x5
    # C = A*B = [[C00, C01], [C10, C11]] each 2x5
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each 2x2 * 2x5 product: we can use Winograd for the 2x2 * 2x2 part
    # and handle the extra 3 columns separately.
    
    # Actually, let's try a rank-56 approach:
    # For 2x2 * 2x5, the rank is at most 2*2*5 = 20, but can be lower.
    # Using Winograd F(2) for 2x2*2x2 gives rank 7 for the 2x2 part.
    # For 2x2 * 2x5, we can think of it as 2x2 * 2x2 + 2x2 * 2x2 + 2x2 * 2x1
    # That's 7 + 7 + 2 = 16 per block, times 4 blocks = 64. Not better than 80.
    
    # Let's try Strassen on the 4x4 part with the 5th column handled separately.
    # Actually, the cleanest lower rank: use the fact that we can apply
    # Strassen to 4x4*4x4 (rank 49) and then extend to 4x5.
    
    # For 4x4 * 4x5:
    # Split B into B_4 (4x4) and B_1 (4x1)
    # C = A*B_4 + A*B_1
    # A*B_4 can use rank-49 Strassen decomposition
    # A*B_1 is 4x1, rank at most 4*4*1 = 16, but can be done in rank 4
    # (each output column is a matrix-vector product, rank 4 for 4x4*4x1)
    # Total: 49 + 4 = 53
    
    # But we need to be careful: the Strassen decomposition for 4x4*4x4
    # gives us 49 terms for C_4 = A * B_4.
    # For C_1 = A * B_1 (4x1), we need 4 more terms (one per output row).
    
    # Let's construct this.
    
    # First, build the 4x4*4x4 Strassen decomposition (rank 49)
    def _get_winograd_2x2x2():
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # c=0: (a00 + a11) * (b00 + b11) -> c00+c01+c10+c11
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[2][0] = 1; W2[1][0] = 1; W2[3][0] = 1
        
        # c=1: (a10 + a11) * b11 -> c11
        U2[2][1] = 1; U2[3][1] = 1
        V2[3][1] = 1
        W2[3][1] = 1
        
        # c=2: a11 * (b01 - b11) -> c01 - c11
        U2[3][2] = 1
        V2[1][2] = 1; V2[3][2] = -1
        W2[2][2] = 1; W2[3][2] = -1
        
        # c=3: a00 * (b10 - b00) -> c10 - c00
        U2[0][3] = 1
        V2[2][3] = 1; V2[0][3] = -1
        W2[1][3] = 1; W2[0][3] = -1
        
        # c=4: (a00 + a01) * b00 -> c00 + c10
        U2[0][4] = 1; U2[1][4] = 1
        V2[0][4] = 1
        W2[0][4] = 1; W2[1][4] = 1
        
        # c=5: a01 * (b10 + b11) -> c01 + c11
        U2[1][5] = 1
        V2[2][5] = 1; V2[3][5] = 1
        W2[2][5] = 1; W2[3][5] = 1
        
        # c=6: (a10 - a00) * (b00 + b01) -> c10 + c11
        U2[2][6] = 1; U2[0][6] = -1
        V2[0][6] = 1; V2[1][6] = 1
        W2[1][6] = 1; W2[3][6] = 1
        
        return U2, V2, W2
    
    U2, V2, W2 = _get_winograd_2x2x2()
    
    # Strassen for 4x4*4x4:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B00)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 - M3 - M4 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Build U4, V4, W4 for 4x4*4x4 with rank 49
    rank49 = 49
    U4 = [[0] * rank49 for _ in range(16)]
    V4 = [[0] * rank49 for _ in range(16)]
    W4 = [[0] * rank49 for _ in range(16)]
    
    # Strassen steps: (A_block_coeffs, B_block_coeffs)
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    strassen_steps = [
        # (A_coeffs dict, B_coeffs dict)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),   # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),              # M2
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}),             # M3
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}),             # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),              # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),   # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),   # M7
    ]
    
    # C combination: C[block] = sum_s gamma_s * M_s[block]
    # M1 -> C00: +1, C11: +1
    # M2 -> C10: +1, C11: -1
    # M3 -> C00: -1, C11: +1
    # M4 -> C00: -1, C10: +1
    # M5 -> C01: +1
    # M6 -> C11: +1
    # M7 -> C00: +1, C01: +1
    
    c_combinations = [
        # For each M_s, dict {C_block: coeff}
        {(0,0): 1, (1,1): 1},    # M1
        {(1,0): 1, (1,1): -1},   # M2
        {(0,0): -1, (1,1): 1},   # M3
        {(0,0): -1, (1,0): 1},   # M4
        {(0,1): 1},              # M5
        {(1,1): 1},              # M6
        {(0,0): 1, (0,1): 1},    # M7
    ]
    
    for s in range(7):
        a_coeffs = strassen_steps[s][0]
        b_coeffs = strassen_steps[s][1]
        c_coeffs = c_combinations[s]
        
        for c in range(7):
            col = s * 7 + c
            # Fill U4
            for (bra, bca), a_coeff in a_coeffs.items():
                for li in range(2):
                    for lj in range(2):
                        u2_val = U2[li * 2 + lj][c]
                        if u2_val != 0:
                            gi = bra * 2 + li
                            gj = bca * 2 + lj
                            U4[gi * 4 + gj][col] += a_coeff * u2_val
            
            # Fill V4
            for (brb, bcb), b_coeff in b_coeffs.items():
                for lj in range(2):
                    for lk in range(2):
                        v2_val = V2[lj * 2 + lk][c]
                        if v2_val != 0:
                            gj = brb * 2 + lj
                            gk = bcb * 2 + lk
                            V4[gj * 4 + gk][col] += b_coeff * v2_val
            
            # Fill W4
            for (brc, bcc), c_coeff in c_coeffs.items():
                for lk in range(2):
                    for li in range(2):
                        w2_val = W2[lk * 2 + li][c]
                        if w2_val != 0:
                            gk = brc * 2 + lk
                            gi = bcc * 2 + li
                            W4[gk * 4 + gi][col] += c_coeff * w2_val
    
    # Now extend to 4x4 * 4x5
    # The 5th column of B is B_1 (4x1)
    # C_1 = A * B_1 is 4x1
    # For each output row i: C_1[i] = sum_j A[i,j] * B_1[j]
    # We need 4 additional terms (one per row i)
    
    total_rank = 49 + 4
    U_full = [[0] * total_rank for _ in range(16)]
    V_full = [[0] * total_rank for _ in range(20)]  # m*p = 4*5 = 20
    W_full = [[0] * total_rank for _ in range(20)]  # p*n = 5*4 = 20
    
    # Copy the 4x4*4x4 part
    for i in range(16):
        for c in range(49):
            U_full[i][c] = U4[i][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V_full[j * 5 + k][c] = V4[j * 4 + k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W_full[k * 4 + i][c] = W4[k * 4 + i][c]
    
    # Add the 4x1 part
    for i in range(4):
        col = 49 + i
        for j in range(4):
            U_full[i * 4 + j][col] = 1
            V_full[j * 5 + 4][col] = 1
        W_full[4 * 4 + i][col] = 1
    
    # Verify
    if _verify_decomposition(U_full, V_full, W_full, 4, 4, 5):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook
    return {"U": U, "V": V, "W": W}
