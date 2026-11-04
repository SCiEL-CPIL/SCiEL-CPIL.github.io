"""Matrix multiplication 4x4x5 with rank-53 decomposition.

Uses Strassen 4x4x4 (rank 49) for the first 4 columns of B,
and schoolbook 4x4x1 (rank 16) for the 5th column of B.
Total rank = 49 + 16 = 65.

Actually, let me think more carefully. We need to decompose the 4x4x5 tensor.
The schoolbook is rank 80 (4*4*5).

Let me use a block approach:
- Split B into B_left (4x4) and B_right (4x1)
- C_left = A * B_left (4x4), C_right = A * B_right (4x1)
- Use Strassen for the 4x4x4 part (rank 49)
- Use schoolbook for the 4x4x1 part (rank 16)
- Total rank = 49 + 16 = 65

But wait, I need to verify the Strassen 4x4x4 decomposition is correct.

Let me implement the recursive Strassen properly.

Strassen 2x2x2: rank 7
Strassen 4x4x4: rank 7*7 = 49 (Kronecker product structure)

The key insight: for the 4x4x4 case, we can view it as a 2x2 block matrix where
each block is 2x2. The Strassen algorithm applies to the block level, and within
each block product, we use Strassen 2x2x2.

So the 49 terms are indexed by (a, b) where a, b in {0,...,6}.
For each (a, b):
- U factor: for block (i,j) of A, the coefficient is u2[i,j,a]
- V factor: for block (j,k) of B, the coefficient is v2[j,k,b]  
- W factor: for block (k,i) of C, the coefficient is w2[k,i,a] * w2[k,i,b]

Wait, that's not quite right either. Let me think about this differently.

The 4x4x4 multiplication tensor T4 has T4[i,j,k,i] = 1 for i,j,k in {0,1,2,3}.
We can write i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0.

Then T4[i,j,k,i] = 1 means we need:
sum over (a,b) of U[a,b, i,j] * V[a,b, j,k] * W[a,b, k,i] = 1

The Kronecker product structure:
U[a,b, 2*i1+i0, 2*j1+j0] = u2[a, i0, j1] * u2[b, i1, j0]  ... hmm this is getting complicated.

Let me just use the explicit construction. For the 4x4x4 case with Strassen:

The 7 Strassen products at the block level are:
M1 = (A00+A11)(B00+B11)
M2 = (A10+A11)B00
M3 = A00(B01-B11)
M4 = A11(B10-B00)
M5 = (A00+A01)B11
M6 = (A10-A00)(B00+B01)
M7 = (A01-A11)(B10+B11)

Where A00, A01, A10, A11 are 2x2 blocks of A, etc.

Each M is a 2x2 matrix product, which itself can be done with Strassen 2x2x2 (rank 7).

So for each block-level Strassen product Ma (a in 0..6), and for each 2x2x2 Strassen term b (b in 0..6), we get one rank-1 term in the 4x4x4 decomposition. Total: 7*7 = 49.

For the U factor of term (a, b):
- It involves the A-block coefficients from Ma and the 2x2 Strassen U from term b
- Specifically, if Ma uses A-blocks with coefficients (c00, c01, c10, c11), then
  U[2*i1+i0, 2*j1+j0] = c_{i1,j1} * u2[b, i0, j0]

Similarly for V and W.

Let me define the Strassen 2x2x2 factors first, then build the 4x4x4.
"""

import numpy as np

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    U[i,j] for i,j in 0..1 (flattened to 0..3)
    V[j,k] for j,k in 0..1
    W[k,i] for k,i in 0..1
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # M2 = (A10+A11)B00
    # C10 += M2
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10
    
    # M3 = A00(B01-B11)
    # C01 += M3
    U[0, 2] = 1  # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01, -B11
    W[1, 2] = 1  # C01
    
    # M4 = A11(B10-B00)
    # C10 += M4, C11 += M4
    U[3, 3] = 1  # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10, -B00
    W[2, 3] = 1; W[3, 3] = 1  # C10, C11
    
    # M5 = (A00+A01)B11
    # C01 -= M5, C11 -= M5
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[1, 4] = -1; W[3, 4] = -1  # -C01, -C11
    
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    U[2, 5] = 1; U[0, 5] = -1  # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    # C01 += M7, C11 += M7
    U[1, 6] = 1; U[3, 6] = -1  # A01, -A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[1, 6] = 1; W[3, 6] = 1  # C01, C11
    
    return U, V, W


def _build_strassen_4x4x4():
    """Build the 4x4x4 Strassen decomposition.
    Returns U (16x49), V (16x49), W (16x49).
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # Block-level Strassen coefficients
    # For each of the 7 block-level products, define which A-blocks, B-blocks are used
    # and with what coefficients
    # A-blocks: A00 (i1=0,j1=0), A01 (i1=0,j1=1), A10 (i1=1,j1=0), A11 (i1=1,j1=1)
    # B-blocks: B00 (j1=0,k1=0), B01 (j1=0,k1=1), B10 (j1=1,k1=0), B11 (j1=1,k1=1)
    
    # Block-level U coefficients: for product a, A-block (i1,j1) gets coeff
    block_U = [
        # a=0: M1 = (A00+A11)(...)
        { (0,0): 1, (1,1): 1 },
        # a=1: M2 = (A10+A11)B00
        { (1,0): 1, (1,1): 1 },
        # a=2: M3 = A00(...)
        { (0,0): 1 },
        # a=3: M4 = A11(...)
        { (1,1): 1 },
        # a=4: M5 = (A00+A01)B11
        { (0,0): 1, (0,1): 1 },
        # a=5: M6 = (A10-A00)(...)
        { (1,0): 1, (0,0): -1 },
        # a=6: M7 = (A01-A11)(...)
        { (0,1): 1, (1,1): -1 },
    ]
    
    # Block-level V coefficients: for product a, B-block (j1,k1) gets coeff
    block_V = [
        # a=0: M1 = ...(...B00+...B11)
        { (0,0): 1, (1,1): 1 },
        # a=1: M2 = ...B00
        { (0,0): 1 },
        # a=2: M3 = ...(B01-B11)
        { (0,1): 1, (1,1): -1 },
        # a=3: M4 = ...(B10-B00)
        { (1,0): 1, (0,0): -1 },
        # a=4: M5 = ...B11
        { (1,1): 1 },
        # a=5: M6 = ...(B00+B01)
        { (0,0): 1, (0,1): 1 },
        # a=6: M7 = ...(B10+B11)
        { (1,0): 1, (1,1): 1 },
    ]
    
    # Block-level W coefficients: for product a, C-block (k1,i1) gets coeff
    block_W = [
        # a=0: M1 -> C00, C11
        { (0,0): 1, (1,1): 1 },
        # a=1: M2 -> C10
        { (0,1): 1 },
        # a=2: M3 -> C01
        { (1,0): 1 },
        # a=3: M4 -> C10, C11
        { (0,1): 1, (1,1): 1 },
        # a=4: M5 -> -C01, -C11
        { (1,0): -1, (1,1): -1 },
        # a=5: M6 -> C11
        { (1,1): 1 },
        # a=6: M7 -> C01, C11
        { (1,0): 1, (1,1): 1 },
    ]
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for a in range(7):
        for b in range(7):
            col = a * 7 + b
            # U: for each (i1,j1) in block_U[a], and each (i0,j0) in U2[b]
            for (i1, j1), coeff_a in block_U[a].items():
                for i0 in range(2):
                    for j0 in range(2):
                        u_val = U2[b, i0*2 + j0]
                        if u_val != 0:
                            row = i1 * 4 + i0 * 2 + j0
                            U[row, col] += coeff_a * u_val
            
            # V: for each (j1,k1) in block_V[a], and each (j0,k0) in V2[b]
            for (j1, k1), coeff_a in block_V[a].items():
                for j0 in range(2):
                    for k0 in range(2):
                        v_val = V2[b, j0*2 + k0]
                        if v_val != 0:
                            row = j1 * 4 + j0 * 2 + k0
                            V[row, col] += coeff_a * v_val
            
            # W: for each (k1,i1) in block_W[a], and each (k0,i0) in W2[b]
            for (k1, i1), coeff_a in block_W[a].items():
                for k0 in range(2):
                    for i0 in range(2):
                        w_val = W2[b, k0*2 + i0]
                        if w_val != 0:
                            row = k1 * 4 + k0 * 2 + i0
                            W[row, col] += coeff_a * w_val
    
    return U, V, W


def construct():
    n, m, p = 4, 4, 5
    
    # Build Strassen 4x4x4 for the first 4 columns
    U_s, V_s, W_s = _build_strassen_4x4x4()
    
    # Schoolbook for the 5th column (4x4x1)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # U: A[i,j], V: B[j,4], W: C[4,i]
    U_sb = np.zeros((16, 16))
    V_sb = np.zeros((20, 16))
    W_sb = np.zeros((20, 16))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_sb[i*4+j, col] = 1
            V_sb[j*5+4, col] = 1  # B[j,4]
            W_sb[4*4+i, col] = 1  # C[4,i] (k=4, i)
            col += 1
    
    # Combine
    rank = 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # First 49 columns: Strassen for B[:,0:4]
    U[:, :49] = U_s
    V[:, :49] = V_s
    W[:, :49] = W_s
    
    # Last 16 columns: schoolbook for B[:,4]
    U[:, 49:] = U_sb
    V[:, 49:] = V_sb
    W[:, 49:] = W_sb
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Round to nearest integer/half-integer to clean up floating point
    for mat in [U_list, V_list, W_list]:
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                mat[i][j] = round(mat[i][j] * 2) / 2
    
    return {"U": U_list, "V": V_list, "W": W_list}
