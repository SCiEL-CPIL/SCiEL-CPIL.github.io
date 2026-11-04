"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p

    def check_decomposition(U, V, W):
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    # Build Strassen 2x2x2 base case (rank 7)
    # A: 2x2, indices a=i*2+j; B: 2x2, indices b=k*2+l; C: 2x2, indices c=m*2+n
    # Tensor: T[a,b,c] = 1 iff a//2==c//2 and a%2==c%2 (i.e., row of C = row of A, col of C = col of B)
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        # M2 = (A10+A11)B00 -> C10
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1
        # M3 = A00(B01-B11) -> C01
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1
        # M4 = A11(B10-B00) -> C10, C11
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[2,3]=1; W[3,3]=1
        # M5 = (A00+A01)B11 -> C01, C11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[1,4]=1; W[3,4]=1
        # M6 = (A10-A00)(B00+B01) -> C00, C10
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[0,5]=1; W[2,5]=1
        # M7 = (A01-A11)(B10+B11) -> C00, C01
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=-1; W[1,6]=1
        return U, V, W

    U2, V2, W2 = strassen_2x2()

    # Build 4x4x4 via Kronecker/Strassen: 7 products of 2x2 blocks, each using 7 scalar mults = 49
    # A 4x4: blocks A11(0-1,0-1), A12(0-1,2-3), A21(2-3,0-1), A22(2-3,2-3)
    # Same for B and C.
    # For each of 7 Strassen terms, define extraction maps LA(4x16), LB(4x16), LC(16x4)
    def build_444():
        U44 = np.zeros((16, 49))
        V44 = np.zeros((16, 49))
        W44 = np.zeros((16, 49))
        
        # Block index helpers: block(row_block, col_block) gives 4 indices in 16-vector
        def blk(rb, cb):
            # rb, cb in {0,1}; returns list of 4 flat indices for A (row-major 4x4)
            return [rb*4 + cb*2 + di for di in range(4)]
        
        # For each of the 7 Strassen terms, define which blocks to sum/diff
        # Term structure: (sum of A blocks)(sum of B blocks) -> sum of C blocks
        # A_blocks: list of (coeff, rb, cb)
        # B_blocks: list of (coeff, rb, cb)  
        # C_blocks: list of (coeff, rb, cb)
        terms = [
            # M1: (A11+A22)(B11+B22) -> C11 + C22
            ([ (1,0,0),(1,1,1) ], [ (1,0,0),(1,1,1) ], [ (1,0,0),(1,1,1) ]),
            # M2: (A21+A22)B11 -> C21
            ([ (1,1,0),(1,1,1) ], [ (1,0,0) ], [ (1,1,0) ]),
            # M3: A11(B12-B22) -> C12
            ([ (1,0,0) ], [ (1,0,1),(-1,1,1) ], [ (1,0,1) ]),
            # M4: A22(B21-B11) -> C21 + C22
            ([ (1,1,1) ], [ (1,1,0),(-1,0,0) ], [ (1,1,0),(1,1,1) ]),
            # M5: (A11+A12)B22 -> C12 + C22
            ([ (1,0,0),(1,0,1) ], [ (1,1,1) ], [ (1,0,1),(1,1,1) ]),
            # M6: (A21-A11)(B11+B12) -> C11 + C21
            ([ (1,1,0),(-1,0,0) ], [ (1,0,0),(1,0,1) ], [ (1,0,0),(1,1,0) ]),
            # M7: (A12-A22)(B21+B22) -> C11 + C12
            ([ (1,0,1),(-1,1,1) ], [ (1,1,0),(1,1,1) ], [ (1,0,0),(1,0,1) ]),
        ]
        
        for t_idx, (a_blks, b_blks, c_blks) in enumerate(terms):
            # Build LA: 4x16, extracts 2x2 "matrix" from A blocks
            LA = np.zeros((4, 16))
            for coeff, rb, cb in a_blks:
                idxs = blk(rb, cb)
                for di in range(4):
                    LA[di, idxs[di]] += coeff
            
            LB = np.zeros((4, 16))
            for coeff, rb, cb in b_blks:
                idxs = blk(rb, cb)
                for di in range(4):
                    LB[di, idxs[di]] += coeff
            
            LC = np.zeros((16, 4))
            for coeff, rb, cb in c_blks:
                idxs = blk(rb, cb)
                for di in range(4):
                    LC[idxs[di], di] += coeff
            
            for q in range(7):
                col = t_idx * 7 + q
                U44[:, col] = LA.T @ U2[:, q]
                V44[:, col] = LB.T @ V2[:, q]
                W44[:, col] = LC @ W2[:, q]
        
        return U44, V44, W44

    U44, V44, W44 = build_444()

    # Now handle 4x4 x 4x5 = 4x5
    # Split B into B_left (4x4) and B_right (4x1)
    # C into C_left (4x4) and C_right (4x1)
    # Use Strassen 4x4x4 for left part (49 terms), schoolbook 4x4x1 for right part (16 terms)
    # Total rank: 65
    
    U_total = np.zeros((16, 65))
    V_total = np.zeros((20, 65))
    W_total = np.zeros((20, 65))
    
    # First 49 columns: Strassen for B_left -> C_left
    U_total[:, :49] = U44
    V_total[:16, :49] = V44
    W_total[:16, :49] = W44
    
    # Last 16 columns: schoolbook for B_right (4x1) -> C_right (4x1)
    # A is 4x4 (16 entries), B_right is 4x1 (4 entries: j=0..3, k=4), C_right is 4x1 (4 entries: i=0..3)
    # For each (i,j) in A: U[i*4+j, col] = 1
    # B_right: V[j*5+4, col] = 1  (j in 0..3, k=4, so flat index j*5+4)
    # C_right: W[4*i+0, col] = 1  (k=4, i in 0..3, flat index k*4+i = 4*4+i = 16+i)
    # Wait: W index is k*n+i where n=4, so for k=4: 4*4+i = 16+i
    for i in range(4):
        for j in range(4):
            col = 49 + i*4 + j
            U_total[i*4+j, col] = 1
            V_total[j*5+4, col] = 1   # B flat: j*p+k = j*5+4
            W_total[4*4+i, col] = 1    # W flat: k*n+i = 4*4+i = 16+i
    
    # Verify
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    if check_decomposition(U_list, V_list, W_list):
        if 65 < best_rank:
            best_rank = 65
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
# EVOLVE-BLOCK-END
