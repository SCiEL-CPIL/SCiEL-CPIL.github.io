"""Tensor decomposition for 6x6 matrix multiplication using Strassen-based block decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: Partition the 6x6 matrices into 3x3 blocks of 2x2 sub-matrices.
    Apply Strassen's algorithm to each 2x2 block multiplication (rank 7 per 2x2).
    Total rank = 3 * 3 * 7 = 63... wait, that's not right.
    
    Actually, for 6x6 matrices partitioned into 2x2 blocks:
    - A is 6x6, partitioned into 3x3 blocks of 2x2
    - B is 6x6, partitioned into 3x3 blocks of 2x2
    - C is 6x6, partitioned into 3x3 blocks of 2x2
    
    For each block position (i,j) in the 3x3 grid, we compute C[i,j] = sum_k A[i,k]*B[k,j]
    where each A[i,k], B[k,j], C[i,j] is a 2x2 matrix.
    
    To use Strassen on the 3x3 block matrix multiplication:
    We can apply Strassen recursively or use a direct approach.
    
    Simpler approach: For each of the 9 output blocks C[i,j], compute it as sum of 3 products.
    But that's schoolbook at the block level.
    
    Better: Use Strassen on the 3x3 block structure. But Strassen is for 2x2.
    
    Alternative: Use the fact that 6x6 can be decomposed using known algorithms.
    The best known rank for 6x6 is 216 (schoolbook) or potentially lower with Strassen.
    
    Let me think about this differently. The standard Strassen algorithm reduces 2x2 from 8 to 7 multiplications.
    For 6x6, if we partition into 2x2 blocks (3x3 grid), we have 9 block multiplications, each requiring 3 block products (schoolbook at block level) = 27 block products, each of rank 7 = 189 total rank.
    
    So rank 189 is achievable.
    
    Let me construct this:
    - 3x3 grid of 2x2 blocks
    - For each output block C[i,j], C[i,j] = sum_{k=0}^{2} A[i,k] * B[k,j]
    - Each 2x2 matrix product uses Strassen with 7 scalar multiplications
    
    Total rank = 9 * 3 * 7 = 189
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block structure: 3x3 grid of 2x2 blocks
    # Each block is 2x2, so block indices: 0,1,2 for rows and cols
    # Within each block: local indices 0,1
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i_block in range(3):  # row block of A, col block of C
        for j_block in range(3):  # col block of B, row block of C
            for k_block in range(3):  # col block of A, row block of B
                # Strassen decomposition for 2x2 * 2x2 = 2x2
                # A_block = A[i_block, k_block], B_block = B[k_block, j_block]
                # C_block = C[i_block, j_block] += A_block * B_block
                
                # Strassen's 7 products for 2x2:
                # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
                # M1 = (a+d)(e+h)
                # M2 = (c+d)e
                # M3 = a(f-h)
                # M4 = d(g-e)
                # M5 = (a+b)h
                # M6 = (c-a)(e+f)
                # M7 = (b-d)(g+h)
                # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
                
                strassen_factors = [
                    # (u_row, u_col, v_row, v_col, w_row, w_col, coeff)
                    # M1 = (a+d)(e+h)
                    # a is at local (0,0), d is at local (1,1)
                    # e is at local (0,0), h is at local (1,1)
                    # C contribution: M1 to C[0,0] and C[1,1]
                    # C[0,0] += M1, C[1,1] += M1
                    # For M1: U gets a and d, V gets e and h
                    # a: U[0,0]=1, d: U[1,1]=1
                    # e: V[0,0]=1, h: V[1,1]=1
                    # W: C[0,0] gets +1, C[1,1] gets +1
                    
                    # Let me define local indices:
                    # A local: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
                    # B local: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
                    # C local: (0,0), (0,1), (1,0), (1,1)
                    
                    # M1 = (a+d)(e+h)
                    # U: a=(0,0), d=(1,1) -> U_local = [[1,0],[0,1]]
                    # V: e=(0,0), h=(1,1) -> V_local = [[1,0],[0,1]]
                    # W: C[0,0]+=1, C[1,1]+=1 -> W_local = [[1,0],[0,1]]
                    ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
                    
                    # M2 = (c+d)e
                    # U: c=(1,0), d=(1,1) -> U_local = [[0,0],[1,1]]
                    # V: e=(0,0) -> V_local = [[1,0],[0,0]]
                    # W: C[1,0]+=1 -> W_local = [[0,0],[1,0]]
                    ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
                    
                    # M3 = a(f-h)
                    # U: a=(0,0) -> U_local = [[1,0],[0,0]]
                    # V: f=(0,1), h=(1,1) -> V_local = [[0,1],[0,-1]]
                    # W: C[0,1]+=1 -> W_local = [[0,1],[0,0]]
                    ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]),
                    
                    # M4 = d(g-e)
                    # U: d=(1,1) -> U_local = [[0,0],[0,1]]
                    # V: g=(1,0), e=(0,0) -> V_local = [[0,-1],[1,0]]
                    # W: C[0,0]+=1, C[1,0]+=1 -> W_local = [[1,0],[1,0]]
                    ([[0,0],[0,1]], [[0,-1],[1,0]], [[1,0],[1,0]]),
                    
                    # M5 = (a+b)h
                    # U: a=(0,0), b=(0,1) -> U_local = [[1,1],[0,0]]
                    # V: h=(1,1) -> V_local = [[0,0],[0,1]]
                    # W: C[0,0]-=1, C[0,1]+=1 -> W_local = [[-1,1],[0,0]]
                    ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,1],[0,0]]),
                    
                    # M6 = (c-a)(e+f)
                    # U: c=(1,0), a=(0,0) -> U_local = [[-1,0],[1,0]]
                    # V: e=(0,0), f=(0,1) -> V_local = [[1,1],[0,0]]
                    # W: C[1,1]+=1 -> W_local = [[0,0],[0,1]]
                    ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
                    
                    # M7 = (b-d)(g+h)
                    # U: b=(0,1), d=(1,1) -> U_local = [[0,1],[-1,0]]
                    # V: g=(1,0), h=(1,1) -> V_local = [[0,0],[1,1]]
                    # W: C[0,0]+=1 -> W_local = [[1,0],[0,0]]
                    ([[0,1],[-1,0]], [[0,0],[1,1]], [[1,0],[0,0]]),
                ]
                
                for u_local, v_local, w_local in strassen_factors:
                    u_local = np.array(u_local, dtype=float)
                    v_local = np.array(v_local, dtype=float)
                    w_local = np.array(w_local, dtype=float)
                    
                    # Map local 2x2 indices to global indices
                    # A: global row = i_block*2 + local_row, global col = k_block*2 + local_col
                    # B: global row = k_block*2 + local_row, global col = j_block*2 + local_col
                    # C: global row = i_block*2 + local_row, global col = j_block*2 + local_col
                    # W index: k*n + i where k is B's col (j_block*2 + local_col_B), i is A's row (i_block*2 + local_row_A)
                    
                    for lr in range(2):  # local row
                        for lc in range(2):  # local col
                            u_val = u_local[lr, lc]
                            v_val = v_local[lr, lc]
                            w_val = w_local[lr, lc]
                            
                            if u_val != 0:
                                a_row = i_block * 2 + lr
                                a_col = k_block * 2 + lc
                                U[a_row * m + a_col, col] += u_val
                            
                            if v_val != 0:
                                b_row = k_block * 2 + lr
                                b_col = j_block * 2 + lc
                                V[b_row * p + b_col, col] += v_val
                            
                            if w_val != 0:
                                # W index: k*n + i
                                # k is B's column index = j_block*2 + lc
                                # i is A's row index = i_block*2 + lr
                                w_k = j_block * 2 + lc
                                w_i = i_block * 2 + lr
                                W[w_k * n + w_i, col] += w_val
                    
                    col += 1
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
