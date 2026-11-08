import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # The decomposition is based on 3 blocks of 2x3x2 multiplication.
    # Rank of 2x3x2 is 11.
    # Total rank = 3 * 11 = 33.
    
    # Define the 11 terms for a single 2x3x2 block.
    # A is 2x3, B is 3x2, C is 2x2.
    # Indices:
    # A: 0,1,2,3,4,5 (row-major)
    # B: 0,1,2,3,4,5 (row-major)
    # C: 0,1,2,3 (row-major)
    
    # We define the U, V, W vectors for the 11 terms.
    # U is length 6, V is length 6, W is length 4.
    
    U_block = np.zeros((6, 11), dtype=float)
    V_block = np.zeros((6, 11), dtype=float)
    W_block = np.zeros((4, 11), dtype=float)
    
    # Term 0: (A00+A11)(B00+B11)
    U_block[0, 0] = 1; U_block[4, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1; W_block[3, 0] = 1
    
    # Term 1: (A10+A11)(B00)
    U_block[3, 1] = 1; U_block[4, 1] = 1
    V_block[0, 1] = 1
    W_block[2, 1] = 1; W_block[3, 1] = -1
    
    # Term 2: A00(B01-B11)
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[3, 2] = -1
    W_block[1, 2] = 1; W_block[3, 2] = 1
    
    # Term 3: A11(B10-B00)
    U_block[4, 3] = 1
    V_block[2, 3] = 1; V_block[0, 3] = -1
    W_block[0, 3] = 1; W_block[2, 3] = 1
    
    # Term 4: (A00+A01)(B11)
    U_block[0, 4] = 1; U_block[1, 4] = 1
    V_block[3, 4] = 1
    W_block[0, 4] = -1; W_block[1, 4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    U_block[3, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = 1
    
    # Term 6: (A01-A11)(B10+B11)
    U_block[1, 6] = 1; U_block[4, 6] = -1
    V_block[2, 6] = 1; V_block[3, 6] = 1
    W_block[0, 6] = 1
    
    # Term 7: A02 * B20
    U_block[2, 7] = 1
    V_block[4, 7] = 1
    W_block[0, 7] = 1
    
    # Term 8: A02 * B21
    U_block[2, 8] = 1
    V_block[5, 8] = 1
    W_block[1, 8] = 1
    
    # Term 9: A12 * B20
    U_block[5, 9] = 1
    V_block[4, 9] = 1
    W_block[2, 9] = 1
    
    # Term 10: A12 * B21
    U_block[5, 10] = 1
    V_block[5, 10] = 1
    W_block[3, 10] = 1
    
    # Now construct the full matrices using Kronecker products / block repetition.
    # U is 6x33. It is just [U_block, U_block, U_block]
    U = np.hstack([U_block, U_block, U_block])
    
    # V is 18x33.
    # B is 3x6. We split into 3 blocks of 3x2.
    # V_block is 6x11.
    # The global V has 18 rows.
    # Row indices 0-5 correspond to B[:, 0:2]
    # Row indices 6-11 correspond to B[:, 2:4]
    # Row indices 12-17 correspond to B[:, 4:6]
    V = np.zeros((18, 33))
    for i in range(3):
        V[i*6:(i+1)*6, i*11:(i+1)*11] = V_block
        
    # W is 12x33.
    # C is 2x6. We split into 3 blocks of 2x2.
    # W_block is 4x11.
    # The global W has 12 rows.
    # Convention: W[k*n + i, r].
    # C index (i, k). i in 0..1, k in 0..5.
    # Local block i, k_local.
    # Global index: k_global = i*2 + k_local? No.
    # C is 2x6. Row i, Col k.
    # Flat index: i*6 + k.
    # But the contract says W[k*n + i, r].
    # So W index is k*n + i = k*2 + i.
    # For block 0 (k=0,1):
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # This matches W_block order (0,1,2,3) if W_block[0]=C00, W_block[1]=C01, W_block[2]=C10, W_block[3]=C11?
    # Let's check W_block definition.
    # W_block[0] corresponds to C[0,0] (i=0, k=0)
    # W_block[1] corresponds to C[0,1] (i=0, k=1)
    # W_block[2] corresponds to C[1,0] (i=1, k=0)
    # W_block[3] corresponds to C[1,1] (i=1, k=1)
    #
    # Global W indices:
    # Block 0 (k=0,1):
    # k=0, i=0 -> 0*2+0=0
    # k=0, i=1 -> 0*2+1=1
    # k=1, i=0 -> 1*2+0=2
    # k=1, i=1 -> 1*2+1=3
    # So W[0:4, 0:11] = W_block
    
    # Block 1 (k=2,3):
    # k=2, i=0 -> 2*2+0=4
    # k=2, i=1 -> 2*2+1=5
    # k=3, i=0 -> 3*2+0=6
    # k=3, i=1 -> 3*2+1=7
    # So W[4:8, 11:22] = W_block
    
    # Block 2 (k=4,5):
    # k=4, i=0 -> 4*2+0=8
    # k=4, i=1 -> 4*2+1=9
    # k=5, i=0 -> 5*2+0=10
    # k=5, i=1 -> 5*2+1=11
    # So W[8:12, 22:33] = W_block
    
    W = np.zeros((12, 33))
    for i in range(3):
        W[i*4:(i+1)*4, i*11:(i+1)*11] = W_block
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
