"""Matrix multiplication 4x5x6 via Strassen-based decomposition."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 if we're looking at the (i,k) entry of A*B given A[i,j]*B[j,k]
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (it's the identity in the middle index)
    # Wait, let me reconsider. The decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # No, that's not right either. Let me think again.
    #
    # The matrix multiplication tensor M has M[i,j,k] = 1 for all valid i,j,k.
    # A decomposition of rank r means: M[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # with indexing U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    #
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7)."""
    # A is 2x2, B is 2x2, C = A*B is 2x2
    # n=2, m=2, p=2
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products for 2x2 matrices:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # M1 = (c+d)(e+f)     -> contributes to p,q,r,s
    # M2 = (a+b)h          -> contributes to q,s
    # M3 = a(e-g)          -> contributes to p,r
    # M4 = d(f-h)          -> contributes to q,s
    # M5 = (a+d)(h-g)      -> contributes to p,s
    # M6 = (c-a)(e+g)      -> contributes to p,r
    # M7 = (b-d)(f+h)      -> contributes to q,s
    #
    # C[0,0] = M1 - M3 + M5 - M7
    # C[0,1] = M2 + M4
    # C[1,0] = M3 - M5 + M6
    # C[1,1] = M1 - M2 + M4 + M7
    #
    # Indices: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    #          B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    #          C[0,0]=p, C[0,1]=q, C[1,0]=r, C[1,1]=s
    #
    # U index: i*2+j, so A[0,0]->0, A[0,1]->1, A[1,0]->2, A[1,1]->3
    # V index: j*2+k, so B[0,0]->0, B[0,1]->1, B[1,0]->2, B[1,1]->3
    # W index: k*2+i, so C[0,0]->0, C[1,0]->1, C[0,1]->2, C[1,1]->3
    #   (k=0,i=0 -> 0; k=1,i=0 -> 1; k=0,i=1 -> 2; k=1,i=1 -> 3)
    
    # M1 = (c+d)(e+f): U has c(A[1,0]->2) and d(A[1,1]->3); V has e(B[0,0]->0) and f(B[0,1]->1)
    # W contributions: C[0,0]->+1 (idx 0), C[0,1]->+1 (idx 2), C[1,0]->+1 (idx 1), C[1,1]->+1 (idx 3)
    U[2][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # M2 = (a+b)h: U has a(0), b(1); V has h(3)
    # W: C[0,1]->+1 (idx 2), C[1,1]->-1 (idx 3)
    U[0][1] = 1; U[1][1] = 1
    V[3][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a(e-g): U has a(0); V has e(0), g(2)
    # W: C[0,0]->-1 (idx 0), C[1,0]->+1 (idx 1)
    U[0][2] = 1
    V[0][2] = 1; V[2][2] = -1
    W[0][2] = -1; W[1][2] = 1
    
    # M4 = d(f-h): U has d(3); V has f(1), h(3)
    # W: C[0,1]->+1 (idx 2), C[1,1]->+1 (idx 3)
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # M5 = (a+d)(h-g): U has a(0), d(3); V has h(3), g(2)
    # W: C[0,0]->+1 (idx 0), C[1,1]->-1 (idx 3)
    U[0][4] = 1; U[3][4] = 1
    V[3][4] = 1; V[2][4] = -1
    W[0][4] = 1; W[3][4] = -1
    
    # M6 = (c-a)(e+g): U has c(2), a(0); V has e(0), g(2)
    # W: C[0,0]->-1 (idx 0), C[1,0]->+1 (idx 1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[0][5] = -1; W[1][5] = 1
    
    # M7 = (b-d)(f+h): U has b(1), d(3); V has f(1), h(3)
    # W: C[0,0]->-1 (idx 0), C[1,1]->+1 (idx 3)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # For 4x5x6, we can use a block approach.
    # Split A (4x5) into blocks, B (5x6) into blocks.
    # 
    # Strategy: Use Strassen for 2x2x2 subproblems where possible.
    # 
    # Let's think about block sizes.
    # A is 4x5, B is 5x6, C is 4x6.
    #
    # One approach: treat the middle dimension (5) as the "m" dimension.
    # We can split the 4 rows of A and 6 columns of B into 2x2 blocks of 2 rows/cols,
    # but 5 doesn't split evenly into 2s. 
    #
    # Alternative: Use a mixed approach.
    # Split n=4 into 2+2, p=6 into 2+2+2, m=5 into 2+2+1 or similar.
    #
    # Actually, let's use a cleaner approach:
    # We can express the 4x5x6 multiplication as a combination of smaller multiplications.
    #
    # Let A be 4x5, B be 5x6.
    # Split A into A1 (2x5) and A2 (2x5) [top and bottom halves]
    # Split B into B1 (5x2), B2 (5x2), B3 (5x2) [three 2-column blocks]
    #
    # C1 = A1*B1 (2x2), C2 = A1*B2 (2x2), C3 = A1*B3 (2x2)
    # C4 = A2*B1 (2x2), C5 = A2*B2 (2x2), C6 = A2*B3 (2x2)
    #
    # Each 2x2x2 multiplication can use Strassen (rank 7) instead of schoolbook (rank 8).
    # Total: 6 * 7 = 42 rank contributions... but wait, this isn't quite right.
    #
    # The issue is that the Strassen decomposition for each 2x2x2 block operates on 
    # specific sub-blocks of A and B. Let me be more careful.
    #
    # For C1 = A1*B1: A1 is rows 0-1 of A (2x5), B1 is cols 0-1 of B (5x2)
    #   But Strassen requires 2x2 inputs. So we need A1 to be 2x2 and B1 to be 2x2.
    #   That means we need to also split the middle dimension.
    #
    # Let me reconsider. The middle dimension is 5. We need to split it.
    # 5 = 2 + 3 or 1 + 4 or other splits.
    #
    # Actually, for the tensor decomposition, we don't need to physically block.
    # We can directly construct U, V, W for the full 4x5x6 problem.
    #
    # Let me try a different strategy: use the fact that we can combine 
    # Strassen decompositions for 2x2x2 sub-tensors with identity mappings for 
    # the remaining dimensions.
    #
    # More concretely:
    # The tensor T[i,j,k] for 4x5x6 can be decomposed as follows:
    # For each pair of row-blocks and column-blocks, we handle the multiplication.
    #
    # Let's split:
    # - Rows of A (and C): i = 0,1,2,3. Split into blocks: {0,1} and {2,3}
    # - Cols of B (and C): k = 0,1,2,3,4,5. Split into blocks: {0,1}, {2,3}, {4,5}
    # - Middle: j = 0,1,2,3,4. Split into blocks: {0,1}, {2,3}, {4}
    #
    # For block (row_block r, col_block c, mid_block s):
    #   A_block is A[rows_r, cols_s] - size (2 x |s|)
    #   B_block is B[rows_s, cols_c] - size (|s| x 2)
    #   C_block is C[rows_r, cols_c] - size (2 x 2)
    #
    # If |s| = 2, we can use Strassen (rank 7).
    # If |s| = 1, we use schoolbook (rank 1*1*... well, it's just 1*2*2 = 4 but actually 
    #   A_block is 2x1, B_block is 1x2, so C = A_block * B_block is 2x2.
    #   This is rank 2*1*2 = 4 schoolbook, or we can think of it differently.
    #
    # Let me just directly build the decomposition.
    
    # I'll construct U, V, W by handling each (row_block, col_block, mid_block) combination.
    
    # Row blocks: [{0,1}, {2,3}]
    row_blocks = [(0, 2), (2, 4)]
    # Col blocks: [{0,1}, {2,3}, {4,5}]
    col_blocks = [(0, 2), (2, 4), (4, 6)]
    # Mid blocks: [{0,1}, {2,3}, {4}]
    mid_blocks = [(0, 2), (2, 4), (4, 5)]
    
    U = [[0] * 0 for _ in range(n * m)]
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    col_offset = 0
    
    for r0, r1 in row_blocks:
        for c0, c1 in col_blocks:
            for s0, s1 in mid_blocks:
                nr = r1 - r0  # 2
                nc = c1 - c0  # 2
                ns = s1 - s0  # 2 or 1
                
                if ns == 2:
                    # Use Strassen for 2x2x2
                    sU, sV, sW = _strassen_2x2x2()
                    rank_block = 7
                    for c in range(7):
                        # Map local indices to global
                        # Local A index (li, lj) -> global (r0+li, s0+lj) -> U[(r0+li)*m + (s0+lj), col_offset+c]
                        # Local B index (mj, mk) -> global (s0+mj, c0+mk) -> V[(s0+mj)*p + (c0+mk), col_offset+c]
                        # Local C index (lk, li) -> global (k=c0+lk, i=r0+li) -> W[(c0+lk)*n + (r0+li), col_offset+c]
                        for li in range(2):
                            for lj in range(2):
                                if sU[li * 2 + lj][c] != 0:
                                    U[(r0 + li) * m + (s0 + lj)][col_offset + c] = sU[li * 2 + lj][c]
                        for mj in range(2):
                            for mk in range(2):
                                if sV[mj * 2 + mk][c] != 0:
                                    V[(s0 + mj) * p + (c0 + mk)][col_offset + c] = sV[mj * 2 + mk][c]
                        for lk in range(2):
                            for li in range(2):
                                if sW[lk * 2 + li][c] != 0:
                                    W[(c0 + lk) * n + (r0 + li)][col_offset + c] = sW[lk * 2 + li][c]
                else:
                    # ns = 1: schoolbook for 2x1x2 (rank 4)
                    rank_block = nr * ns * nc  # 2*1*2 = 4
                    for a_i in range(nr):
                        for a_j in range(ns):
                            for b_k in range(nc):
                                U[(r0 + a_i) * m + (s0 + a_j)][col_offset] = 1
                                V[(s0 + a_j) * p + (c0 + b_k)][col_offset] = 1
                                W[(c0 + b_k) * n + (r0 + a_i)][col_offset] = 1
                                col_offset += 1
                    col_offset -= rank_block  # reset, we'll add below
                    # Actually, let me redo this more carefully
                    for idx in range(rank_block):
                        pass  # already handled in loop
                    # The loop above already incremented col_offset by rank_block
                    continue
                
                col_offset += rank_block
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
