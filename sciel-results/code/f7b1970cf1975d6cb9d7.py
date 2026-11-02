"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    The tensor for $C = A B$ where $A$ is $2 \times 2$ and $B$ is $2 \times 10$ can be viewed as
    5 independent $2 \times 2 \times 2$ multiplication problems, one for each pair of columns in $B$.
    Specifically, let $B = [B_0, B_1, B_2, B_3, B_4]$ where each $B_i$ is a $2 \times 2$ matrix.
    Then $C = A B = [A B_0, A B_1, A B_2, A B_3, A B_4]$.
    
    The rank of the $2 \times 2 \times 2$ matrix multiplication tensor is 7 (Strassen's algorithm).
    Since the 5 blocks are independent (the input $A$ is the same, but the outputs and the specific
    $B$ slices are disjoint in the tensor space), the total rank is the sum of the ranks of the blocks.
    Total Rank = 5 * 7 = 35.
    
    The interface requires:
    U: shape (n*m, rank) = (4, 35)
    V: shape (m*p, rank) = (20, 35)
    W: shape (p*n, rank) = (20, 35)
    
    Indices:
    U index: i*m + j  (i in 0..1, j in 0..1) -> 0,1,2,3
    V index: j*p + k  (j in 0..1, k in 0..9) -> 0..19
    W index: k*n + i  (k in 0..9, i in 0..1) -> 0..19 (Official trace convention)
    
    Strassen's 7 products for multiplying 2x2 matrices $M_1$ and $M_2$:
    $M_1 = [[a, b], [c, d]]$, $M_2 = [[e, f], [g, h]]$
    $P_1 = (a+d)(e+h)$
    $P_2 = (c+d)e$
    $P_3 = a(f-h)$
    $P_4 = d(g-e)$
    $P_5 = (a+b)h$
    $P_6 = (c-a)(e+f)$
    $P_7 = (b-d)(g+h)$
    
    Result $M = M_1 M_2$:
    $M_{0,0} = P_1 + P_4 - P_5 + P_7$
    $M_{0,1} = P_3 + P_5$
    $M_{1,0} = P_2 + P_4$
    $M_{1,1} = P_1 - P_2 + P_3 + P_6$
    
    We will construct the 35 terms corresponding to these 7 products for each of the 5 column pairs.
    """
    n, m, p = DIMENSIONS
    # rank = 35
    rank = 35
    
    U = np.zeros((n * m, rank), dtype=float)  # 4 x 35
    V = np.zeros((m * p, rank), dtype=float)  # 20 x 35
    W = np.zeros((p * n, rank), dtype=float)  # 20 x 35
    
    # Strassen coefficients for the 7 products
    # Each product r is defined by coefficients for the 4 inputs of A and 4 inputs of B
    # A inputs: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B inputs: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # U coefficients (for A) and V coefficients (for B) for each of the 7 Strassen products
    # Product 1: (a+d)(e+h) -> A: [1, 0, 0, 1], B: [1, 0, 0, 1] (wait, (a+d) is [1,0,0,1] dot A? No.
    # Let's map A indices: 0:a, 1:b, 2:c, 3:d.
    # P1: (a+d)(e+h). A_vec = [1, 0, 0, 1]. B_vec = [1, 0, 0, 1] (for e,h? No. e=B00, f=B01, g=B10, h=B11).
    # e+h corresponds to B indices 0 and 3. So B_vec = [1, 0, 0, 1].
    
    # P2: (c+d)e. A_vec = [0, 0, 1, 1]. B_vec = [1, 0, 0, 0].
    # P3: a(f-h). A_vec = [1, 0, 0, 0]. B_vec = [0, 1, 0, -1].
    # P4: d(g-e). A_vec = [0, 0, 0, 1]. B_vec = [-1, 0, 1, 0].
    # P5: (a+b)h. A_vec = [1, 1, 0, 0]. B_vec = [0, 0, 0, 1].
    # P6: (c-a)(e+f). A_vec = [-1, 0, 1, 0]. B_vec = [1, 1, 0, 0].
    # P7: (b-d)(g+h). A_vec = [0, 1, 0, -1]. B_vec = [0, 0, 1, 1].
    
    strassen_A = np.array([
        [1, 0, 0, 1],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, 0, -1]  # P7
    ])
    
    strassen_B = np.array([
        [1, 0, 0, 1],  # P1 (e+h)
        [1, 0, 0, 0],  # P2 (e)
        [0, 1, 0, -1], # P3 (f-h)
        [-1, 0, 1, 0], # P4 (g-e)
        [0, 0, 0, 1],  # P5 (h)
        [1, 1, 0, 0],  # P6 (e+f)
        [0, 0, 1, 1]   # P7 (g+h)
    ])
    
    # W coefficients (for C)
    # C indices in tensor W: k*n + i.
    # For a 2x2 block, k is local 0 or 1, i is 0 or 1.
    # Local C indices:
    # C00: i=0, k_local=0
    # C01: i=0, k_local=1
    # C10: i=1, k_local=0
    # C11: i=1, k_local=1
    
    # M00 = P1 + P4 - P5 + P7
    # M01 = P3 + P5
    # M10 = P2 + P4
    # M11 = P1 - P2 + P3 + P6
    
    # We need to map these to the global W indices.
    # Global W index = k_global * n + i = k_global * 2 + i.
    # For block b (0 to 4), the columns are 2*b and 2*b+1.
    # So k_global = 2*b + k_local.
    # W_index = (2*b + k_local) * 2 + i.
    
    # Let's define the W contribution for each Strassen product P_r to the local C matrix.
    # w_contrib[r][local_c_idx] = coefficient
    # local_c_idx: 0->C00, 1->C01, 2->C10, 3->C11
    
    w_contrib = np.zeros((7, 4), dtype=float)
    
    # P1 contributes to C00 (+1) and C11 (+1)
    w_contrib[0, 0] = 1
    w_contrib[0, 3] = 1
    
    # P2 contributes to C10 (+1) and C11 (-1)
    w_contrib[1, 2] = 1
    w_contrib[1, 3] = -1
    
    # P3 contributes to C01 (+1) and C11 (+1)
    w_contrib[2, 1] = 1
    w_contrib[2, 3] = 1
    
    # P4 contributes to C00 (+1) and C10 (+1)
    w_contrib[3, 0] = 1
    w_contrib[3, 2] = 1
    
    # P5 contributes to C00 (-1) and C01 (+1)
    w_contrib[4, 0] = -1
    w_contrib[4, 1] = 1
    
    # P6 contributes to C11 (+1)
    w_contrib[5, 3] = 1
    
    # P7 contributes to C00 (+1)
    w_contrib[6, 0] = 1
    
    col = 0
    for b in range(5):
        for r in range(7):
            # U: same for all blocks
            U[:, col] = strassen_A[r]
            
            # V: specific to block b
            # B inputs for block b:
            # e = B[0, 2b] -> index 0*p + 2b = 2b
            # f = B[0, 2b+1] -> index 0*p + 2b+1 = 2b+1
            # g = B[1, 2b] -> index 1*p + 2b = 10 + 2b
            # h = B[1, 2b+1] -> index 1*p + 2b+1 = 10 + 2b+1
            
            V[2*b, col] = strassen_B[r, 0]
            V[2*b+1, col] = strassen_B[r, 1]
            V[10+2*b, col] = strassen_B[r, 2]
            V[10+2*b+1, col] = strassen_B[r, 3]
            
            # W: specific to block b
            # C outputs for block b:
            # C00: i=0, k_global=2b -> index (2b)*2 + 0 = 4b
            # C01: i=0, k_global=2b+1 -> index (2b+1)*2 + 0 = 4b+2
            # C10: i=1, k_global=2b -> index (2b)*2 + 1 = 4b+1
            # C11: i=1, k_global=2b+1 -> index (2b+1)*2 + 1 = 4b+3
            
            W[4*b, col] = w_contrib[r, 0]
            W[4*b+2, col] = w_contrib[r, 1]
            W[4*b+1, col] = w_contrib[r, 2]
            W[4*b+3, col] = w_contrib[r, 3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
