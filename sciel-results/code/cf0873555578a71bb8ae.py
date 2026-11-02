"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build the target tensor T[i,j,k] = 1 if this is a valid matmul index
    # T_{i,j,k} corresponds to C[i,k] += A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k)
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == T[i,j,k]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Target: 1.0
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W):
    """Fast verification using tensor contraction."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float).reshape(n, m, rank)
    V = np.array(V, dtype=float).reshape(m, p, rank)
    W = np.array(W, dtype=float).reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Contract over r
    UV = np.einsum('ijr,jkr->ikr', U, V)  # This isn't quite right
    # Better: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Let's do it step by step
    # First: A[i,j,k] = sum_r U[i,j,r]*V[j,k,r]
    A = np.einsum('ijr,jkr->ijk', U, V)
    # Then: T[i,j,k] = A[i,j,k]*W[k,i,r]... no, W is indexed by (k,i,r)
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # = sum_r (U[i,j,r]*V[j,k,r]) * W[k,i,r]
    # Let B[i,j,k,r] = U[i,j,r]*V[j,k,r]
    # T[i,j,k] = sum_r B[i,j,k,r] * W[k,i,r]
    B = np.einsum('ijr,jkr->ijk r'.replace(' ',''), U, V)  # shape (n,m,p,rank)
    T = np.einsum('ijk r,kir->ijk', B, W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i,j,k] - 1.0) > 1e-9:
                    return False
    return True

def _block_decomposition():
    """Block decomposition using Strassen for 2x2x2 blocks, rank 44."""
    n, m, p = DIMENSIONS
    # We'll use a block approach:
    # Split the 3x4x4 problem into blocks.
    # Idea: treat rows 0,1 together with Strassen (rank 7 for 2x2x2), 
    # and row 2 separately.
    # Actually, let's think more carefully.
    # 
    # For a 3x4x4 matmul: C = A @ B where A is 3x4, B is 4x4, C is 3x4.
    # 
    # One approach: split A into rows. Let A1 = A[0:2,:] (2x4), A2 = A[2,:] (1x4).
    # C1 = A1 @ B (2x4), C2 = A2 @ B (1x4).
    # 
    # For C1 = A1 @ B: A1 is 2x4, B is 4x4.
    # Split B into two 2x4 column blocks: B = [B_left, B_right] where B_left is 4x2, B_right is 4x2.
    # C1_left = A1 @ B_left (2x2), C1_right = A1 @ B_right (2x2).
    # Each 2x2 matmul can be done with Strassen rank 7.
    # So C1 needs rank 7+7 = 14.
    # 
    # For C2 = A2 @ B: A2 is 1x4, B is 4x4.
    # C2 = A2 @ B (1x4). This is just a row vector times a matrix.
    # C2[k] = sum_j A2[j] * B[j,k] for k=0..3.
    # This is rank 4 (each output element is a separate dot product).
    # 
    # Total: 14 + 4 = 18? That seems too low. Let me reconsider.
    #
    # Wait, I need to be more careful. The tensor decomposition is:
    # T[i,j,k] = 1 for all i,j,k (it's the full matmul tensor).
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    #
    # Let me try a different block strategy.
    #
    # Actually, let me just implement the known rank-44 construction from the cell findings.
    # "block decomposition that combines Strassen's rank-7 algorithm for 2x2x2 with 
    #  trivial rank-4 for the remaining row, reducing the rank of 3x2x2 blocks to 11, 
    #  and tiling 4 such blocks to achieve a total rank of 44"
    #
    # So for a 3x2x2 block: rank 11 (instead of trivial 12)
    # We tile 4 such blocks: 4 * 11 = 44
    #
    # The 3x2x2 block: multiply a 3x2 matrix by a 2x2 matrix to get a 3x2 result.
    # T[i,j,k] where i in {0,1,2}, j in {0,1}, k in {0,1}
    #
    # For the 3x2x2 case:
    # Rows 0,1 form a 2x2x2 subproblem (Strassen rank 7)
    # Row 2 is a 1x2x2 subproblem (rank 2, since it's just two dot products)
    # But we can potentially share some terms.
    #
    # Let me construct the 3x2x2 block with rank 11:
    # Strassen for rows 0,1 (2x2x2): 7 terms
    # Row 2: needs 2 more terms for the two outputs
    # But maybe we can combine: row 2's computation can reuse some Strassen terms?
    #
    # Actually, let me just hardcode a rank-44 solution based on the block structure.
    # 
    # For the full 3x4x4:
    # Split columns of B (and rows of A) into 2 blocks of size 2.
    # Block (row_block, col_block): 3x2x2 subproblem, rank 11 each.
    # 4 blocks * 11 = 44.
    #
    # Let me define the 3x2x2 Strassen-like decomposition with rank 11.
    
    # For 3x2x2: i in {0,1,2}, j in {0,1}, k in {0,1}
    # Strassen for 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1}):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B10)
    # M6 = (A01-A11)(B01+B11)
    #
    # C00 = M0 + M3 - M5 - M6
    # C01 = M2 + M4 + M6
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M4
    #
    # For row 2 (i=2):
    # C20 = A20*B00 + A21*B10
    # C21 = A20*B01 + A21*B11
    #
    # We need to express these with the minimum number of additional terms.
    # C20 and C21 are bilinear in A's row 2 and B's columns.
    # They can't be expressed using the Strassen M's directly since those involve rows 0,1.
    # So we need 2 more terms: one for C20, one for C21? 
    # But C20 = A20*B00 + A21*B10 requires 2 multiplications if done separately.
    # However, as a tensor term, each "term" r contributes U[i,j,r]*V[j,k,r]*W[k,i,r].
    # For i=2, j=0, k=0: need U[2,0,r]*V[0,0,r]*W[0,2,r] to sum to 1
    # For i=2, j=1, k=0: need U[2,1,r]*V[1,0,r]*W[0,2,r] to sum to 1
    # For i=2, j=0, k=1: need U[2,0,r]*V[0,1,r]*W[1,2,r] to sum to 1
    # For i=2, j=1, k=1: need U[2,1,r]*V[1,1,r]*W[1,2,r] to sum to 1
    #
    # With 2 additional terms (r=7, r=8):
    # Term 7: U[2,0,7]=1, V[0,0,7]=1, W[0,2,7]=1 -> handles (2,0,0)
    #          U[2,1,7]=1, V[1,0,7]=1, W[0,2,7]=1 -> also contributes to (2,1,0)
    # But we need (2,1,0) to be 1, and (2,0,0) to be 1.
    # If term 7 gives both, that's fine if no other terms contribute.
    # But wait, we also need (2,0,1) and (2,1,1).
    # Term 8: U[2,0,8]=1, V[0,1,8]=1, W[1,2,8]=1 -> handles (2,0,1)
    #          U[2,1,8]=1, V[1,1,8]=1, W[1,2,8]=1 -> also contributes to (2,1,1)
    #
    # So with 2 additional terms, we can handle row 2!
    # Total for 3x2x2: 7 + 2 = 9? But the cell finding says rank 11.
    #
    # Let me re-examine. The issue is that the Strassen terms also have non-zero
    # entries for i=2 in U, V, W. We need to make sure they're zero.
    #
    # In the Strassen decomposition for 2x2x2, U[i,j,r] is non-zero only for i in {0,1}.
    # So for i=2, all Strassen terms contribute 0. Good.
    # Similarly, V[j,k,r] and W[k,i,r] are defined for the 2x2x2 case.
    #
    # So the 3x2x2 block can be done with 7 + 2 = 9 terms?
    # But the cell finding says 11. Maybe I'm missing something.
    #
    # Let me just try rank 9 for the 3x2x2 block and see if it works.
    # If not, I'll increase.
    
    # Actually, let me just implement the full 3x4x4 with rank 44 as stated,
    # and then try to search for improvements.
    
    # Let me construct it explicitly.
    # 
    # For the 3x4x4 problem, split into 4 blocks of 3x2x2:
    # Block (b1, b2) where b1 in {0,1} (row block of A / col block of B), 
    # b2 in {0,1} (col block of output / col block of B)
    #
    # Actually, let me think of it as:
    # A is 3x4: rows i=0,1,2; cols j=0,1,2,3
    # B is 4x4: rows j=0,1,2,3; cols k=0,1,2,3
    # C is 3x4: rows i=0,1,2; cols k=0,1,2,3
    #
    # Split j into {0,1} and {2,3}, split k into {0,1} and {2,3}.
    # 4 blocks, each 3x2x2.
    #
    # For each block, use rank-9 (or 11) decomposition.
    # Total: 4*9 = 36 or 4*11 = 44.
    #
    # Let me try rank 9 first for each block.
    
    rank = 4 * 9  # 36, if 3x2x2 can be done in rank 9
    U = [[0] * rank for _ in range(3 * 4)]  # 12 x rank
    V = [[0] * rank for _ in range(4 * 4)]  # 16 x rank
    W = [[0] * rank for _ in range(4 * 3)]  # 12 x rank
    
    # Strassen coefficients for 2x2x2
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B10)
    # M6 = (A01-A11)(B01+B11)
    #
    # U for 2x2x2 (i in {0,1}, j in {0,1}):
    # U[0,0,0]=1, U[1,1,0]=1  (M0: A00+A11)
    # U[1,0,1]=1, U[1,1,1]=1  (M1: A10+A11)
    # U[0,0,2]=1               (M2: A00)
    # U[1,1,3]=1               (M3: A11)
    # U[0,0,4]=1, U[0,1,4]=1  (M4: A00+A01)
    # U[1,0,5]=1, U[0,0,5]=-1 (M5: A10-A00)
    # U[0,1,6]=1, U[1,1,6]=-1 (M6: A01-A11)
    #
    # V for 2x2x2 (j in {0,1}, k in {0,1}):
    # V[0,0,0]=1, V[1,1,0]=1  (M0: B00+B11)
    # V[0,0,1]=1               (M1: B00)
    # V[1,0,2]=1, V[1,1,2]=-1 (M2: B10-B11)
    # V[0,1,3]=1, V[0,0,3]=-1 (M3: B01-B00)
    # V[1,1,4]=1               (M4: B11)
    # V[0,0,5]=1, V[1,0,5]=1  (M5: B00+B10)
    # V[0,1,6]=1, V[1,1,6]=1  (M6: B01+B11)
    #
    # W for 2x2x2 (k in {0,1}, i in {0,1}):
    # C00 = M0 + M3 - M5 - M6
    # C01 = M2 + M4 + M6
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M4
    # W[0,0,0]=1, W[0,0,3]=1, W[0,0,5]=-1, W[0,0,6]=-1
    # W[1,0,2]=1, W[1,0,4]=1, W[1,0,6]=1
    # W[0,1,1]=1, W[0,1,5]=1
    # W[1,1,0]=1, W[1,1,1]=-1, W[1,1,2]=1, W[1,1,4]=1
    
    # Now for each of the 4 blocks, assign the 9 terms.
    # Block (jb, kb) where jb in {0,1} (j offset: 0 or 2), kb in {0,1} (k offset: 0 or 2)
    
    for jb in range(2):
        for kb in range(2):
            base = (jb * 2 + kb) * 9  # base index for this block's 9 terms
            j_off = jb * 2  # offset in j
            k_off = kb * 2  # offset in k
            
            # Strassen terms (r = 0..6)
            # U: rows i=0,1; cols j=j_off, j_off+1
            U[0 * 4 + j_off][base + 0] = 1
            U[1 * 4 + (j_off+1)][base + 0] = 1
            U[1 * 4 + j_off][base + 1] = 1
            U[1 * 4 + (j_off+1)][base + 1] = 1
            U[0 * 4 + j_off][base + 2] = 1
            U[1 * 4 + (j_off+1)][base + 3] = 1
            U[0 * 4 + j_off][base + 4] = 1
            U[0 * 4 + (j_off+1)][base + 4] = 1
            U[1 * 4 + j_off][base + 5] = 1
            U[0 * 4 + j_off][base + 5] = -1
            U[0 * 4 + (j_off+1)][base + 6] = 1
            U[1 * 4 + (j_off+1)][base + 6] = -1
            
            # V: rows j=j_off, j_off+1; cols k=k_off, k_off+1
            V[j_off * 4 + k_off][base + 0] = 1
            V[(j_off+1) * 4 + (k_off+1)][base + 0] = 1
            V[j_off * 4 + k_off][base + 1] = 1
            V[(j_off+1) * 4 + k_off][base + 2] = 1
            V[(j_off+1) * 4 + (k_off+1)][base + 2] = -1
            V[j_off * 4 + (k_off+1)][base + 3] = 1
            V[j_off * 4 + k_off][base + 3] = -1
            V[(j_off+1) * 4 + (k_off+1)][base + 4] = 1
            V[j_off * 4 + k_off][base + 5] = 1
            V[(j_off+1) * 4 + k_off][base + 5] = 1
            V[j_off * 4 + (k_off+1)][base + 6] = 1
            V[(j_off+1) * 4 + (k_off+1)][base + 6] = 1
            
            # W: rows k=k_off, k_off+1; cols i=0,1
            W[k_off * 3 + 0][base + 0] = 1
            W[k_off * 3 + 0][base + 3] = 1
            W[k_off * 3 + 0][base + 5] = -1
            W[k_off * 3 + 0][base + 6] = -1
            W[(k_off+1) * 3 + 0][base + 2] = 1
            W[(k_off+1) * 3 + 0][base + 4] = 1
            W[(k_off+1) * 3 + 0][base + 6] = 1
            W[k_off * 3 + 1][base + 1] = 1
            W[k_off * 3 + 1][base + 5] = 1
            W[(k_off+1) * 3 + 1][base + 0] = 1
            W[(k_off+1) * 3 + 1][base + 1] = -1
            W[(k_off+1) * 3 + 1][base + 2] = 1
            W[(k_off+1) * 3 + 1][base + 4] = 1
            
            # Row 2 terms (r = 7, 8)
            # Term 7: handles (i=2, j=j_off, k=k_off) and (i=2, j=j_off+1, k=k_off)
            U[2 * 4 + j_off][base + 7] = 1
            U[2 * 4 + (j_off+1)][base + 7] = 1
            V[j_off * 4 + k_off][base + 7] = 1
            V[(j_off+1) * 4 + k_off][base + 7] = 1
            W[k_off * 3 + 2][base + 7] = 1
            
            # Term 8: handles (i=2, j=j_off, k=k_off+1) and (i=2, j=j_off+1, k=k_off+1)
            U[2 * 4 + j_off][base + 8] = 1
            U[2 * 4 + (j_off+1)][base + 8] = 1
            V[j_off * 4 + (k_off+1)][base + 8] = 1
            V[(j_off+1) * 4 + (k_off+1)][base + 8] = 1
            W[(k_off+1) * 3 + 2][base + 8] = 1
    
    # Verify
    if not _verify(U, V, W):
        # Fall back to schoolbook
        rank2 = 3 * 4 * 4
        U2 = [[0] * rank2 for _ in range(12)]
        V2 = [[0] * rank2 for _ in range(16)]
        W2 = [[0] * rank2 for _ in range(12)]
        col = 0
        for i in range(3):
            for j in range(4):
                for k in range(4):
                    U2[i * 4 + j][col] = 1
                    V2[j * 4 + k][col] = 1
                    W2[k * 3 + i][col] = 1
                    col += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with the block decomposition
    result = _block_decomposition()
    best_rank = len(result["U"][0])
    
    # Try to improve via random search
    # This is a placeholder for future improvements
    # For now, return the block decomposition
    
    return result
# EVOLVE-BLOCK-END
