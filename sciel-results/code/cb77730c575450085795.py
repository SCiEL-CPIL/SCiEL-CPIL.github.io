"""Block-structured decomposition for 4x5x5 matrix multiplication."""
import numpy as np
from itertools import product
import time
import sys

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 140.0

def _strassen_2x2():
    """Return U, V, W for 2x2x2 matrix mult with rank 7 (Strassen)."""
    # Strassen's 7 multiplications for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B
    # Indexing: A[i,j] with i,j in 0..1, B[j,k], C[i,k]
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where n=2,m=2,p=2
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each multiplication c, we need U[i,j,c], V[j,k,c], W[k,i,c]
    # such that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = C[i,k]
    
    # Let's define the 7 terms:
    # Term 0: (A00+A11)*(B00+B11)
    #   A part: A00=1, A11=1 -> U[0,0,0]=1, U[1,1,0]=1
    #   B part: B00=1, B11=1 -> V[0,0,0]=1, V[1,1,0]=1
    #   C part: contributes to C00(+1), C01(0), C10(0), C11(+1)
    #   Wait, this is a scalar multiplication, so W should be all 1s? No.
    #   Actually, each term is a scalar product. The contribution to C[i,k] is U[i,j]*V[j,k]*W[k,i]
    #   For a single scalar product M, we have one rank-1 term.
    #   Let me reconsider. Each of the 7 multiplications gives a scalar.
    #   The final C is a linear combination of these 7 scalars.
    #   So for each scalar M_c, we need to express its contribution to each C[i,k].
    
    # Let me use a different approach: directly define the 7 rank-1 terms.
    # Each term c has U_c, V_c, W_c vectors.
    
    # M1 = (A00+A11)*(B00+B11)
    #   U[0,0,0]=1, U[1,1,0]=1
    #   V[0,0,0]=1, V[1,1,0]=1
    #   W[0,0,0]=1, W[0,1,0]=1, W[1,0,0]=1, W[1,1,0]=1
    #   This gives: for all i,k: sum_j U[i,j]*V[j,k] = (i==0 and k==0: 1, i==1 and k==1: 1, else 0)
    #   That's not right. Let me think more carefully.
    
    # Actually, the standard way: for each of the 7 multiplications, we have a product of a linear form in A and a linear form in B.
    # The result is a scalar that contributes to multiple entries of C.
    # So W[k,i,c] indicates the coefficient with which scalar M_c contributes to C[i,k].
    
    # Let me define:
    # c=0: M1 = (A00+A11)*(B00+B11)
    U[0,0] = 1; U[3,0] = 1  # A00, A11 (indices: 0=00, 1=01, 2=10, 3=11)
    V[0,0] = 1; V[3,0] = 1  # B00, B11
    W[0,0] = 1; W[1,0] = 1; W[2,0] = 1; W[3,0] = 1  # contributes to all C entries? No.
    # M1 contributes: C00 += M1, C01 += 0, C10 += 0, C11 += M1? No.
    # From Strassen: C00 = M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    # So M1 contributes +1 to C00 and +1 to C11.
    W[0,0] = 1  # C00 (k=0,i=0)
    W[3,0] = 1  # C11 (k=1,i=1)
    W[1,0] = 0; W[2,0] = 0
    
    # c=1: M2 = (A10+A11)*B00
    U[2,1] = 1; U[3,1] = 1  # A10, A11
    V[0,1] = 1  # B00
    W[2,1] = 1  # C10 (k=0,i=1)
    
    # c=2: M3 = A00*(B01-B11)
    U[0,2] = 1  # A00
    V[1,2] = 1; V[3,2] = -1  # B01, -B11
    W[1,2] = 1  # C01 (k=1,i=0)
    
    # c=3: M4 = A11*(B10-B00)
    U[3,3] = 1  # A11
    V[2,3] = 1; V[0,3] = -1  # B10, -B00
    W[0,3] = 1  # C00
    W[2,3] = 1  # C10
    
    # c=4: M5 = (A00+A01)*B11
    U[0,4] = 1; U[1,4] = 1  # A00, A01
    V[3,4] = 1  # B11
    W[0,4] = -1  # C00
    W[1,4] = 1   # C01
    
    # c=5: M6 = (A10-A00)*(B00+B01)
    U[2,5] = 1; U[0,5] = -1  # A10, -A00
    V[0,5] = 1; V[1,5] = 1  # B00, B01
    W[3,5] = 1  # C11
    
    # c=6: M7 = (A01-A11)*(B10+B11)
    U[1,6] = 1; U[3,6] = -1  # A01, -A11
    V[2,6] = 1; V[3,6] = 1  # B10, B11
    W[0,6] = 1  # C00
    W[3,6] = 1  # C11
    
    return U, V, W, r

def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m x p."""
    r = n * m * p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    for c, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        U[i*m + j, c] = 1.0
        V[j*p + k, c] = 1.0
        W[k*n + i, c] = 1.0
    return U, V, W, r

def _combine_blocks(blocks):
    """
    Combine block decompositions into a global decomposition.
    blocks: list of (n_i, m_i, p_i, U_i, V_i, W_i, r_i)
    The global problem is sum of blocks where:
    - Global A is block-diagonal with blocks A_i of size n_i x m_i
    - Global B is block-diagonal with blocks B_i of size m_i x p_i  
    - Global C is block-diagonal with blocks C_i of size n_i x p_i
    
    But 4x5x5 doesn't decompose cleanly into independent blocks this way.
    Instead, we'll use a different strategy: split the 4x5 into row blocks,
    5x5 into corresponding column blocks.
    """
    pass

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5
    r_school = n * m * p  # 100
    
    # Build schoolbook as fallback
    U_sb = np.zeros((n * m, r_school))
    V_sb = np.zeros((m * p, r_school))
    W_sb = np.zeros((p * n, r_school))
    for c, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        U_sb[i * m + j, c] = 1.0
        V_sb[j * p + k, c] = 1.0
        W_sb[k * n + i, c] = 1.0
    
    def _pack(U, V, W, r):
        return U[:, :r].tolist(), V[:, :r].tolist(), W[:, :r].tolist()
    
    def _is_valid(U, V, W, r):
        if time.time() - start_time > TIME_LIMIT - 10:
            return False
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        A = np.einsum('jkc,kic->jki', V3, W3)
        T = np.einsum('ijc,jki->ijk', U3, A)
        if not np.allclose(T, 1.0, atol=1e-10):
            return False
        return True
    
    # Strategy: Split 4x5x5 into blocks:
    # A is 4x5, B is 5x5, C is 4x5
    # Split rows of A/C into groups, columns of A / rows of B into groups, columns of B / C into groups.
    # 
    # Let's try: split 4 rows into 2+2, 5 cols into 2+3, 5 cols of B into 2+3.
    # Block (0,0): A[0:2,0:2] (2x2) * B[0:2,0:2] (2x2) -> C[0:2,0:2] (2x2)
    # Block (0,1): A[0:2,2:5] (2x3) * B[2:5,0:2] (3x2) -> C[0:2,0:2] (2x2)
    # Block (1,0): A[2:4,0:2] (2x2) * B[0:2,2:5] (2x3) -> C[2:4,2:5] (2x3)
    # Block (1,1): A[2:4,2:5] (2x3) * B[2:5,2:5] (3x3) -> C[2:4,2:5] (2x3)
    #
    # Wait, the block structure needs to be consistent. Let me use a proper block decomposition.
    # 
    # Partition:
    # A: rows 0:4, cols 0:5. Split rows at 2, cols at 2.
    #   A00: 2x2, A01: 2x3, A10: 2x2, A11: 2x3
    # B: rows 0:5, cols 0:5. Split rows at 2, cols at 2.
    #   B00: 2x2, B01: 2x3, B10: 3x2, B11: 3x3
    # C: rows 0:4, cols 0:5. Split rows at 2, cols at 2.
    #   C00: 2x2, C01: 2x3, C10: 2x2, C11: 2x3
    #
    # C = A*B:
    # C00 = A00*B00 + A01*B10  (2x2 = 2x2*2x2 + 2x3*3x2)
    # C01 = A00*B01 + A01*B11  (2x3 = 2x2*2x3 + 2x3*3x3)
    # C10 = A10*B00 + A11*B10  (2x2 = 2x2*2x2 + 2x3*3x2)
    # C11 = A10*B01 + A11*B11  (2x3 = 2x2*2x3 + 2x3*3x3)
    #
    # This is getting complex. Let me try a simpler block structure.
    # 
    # Alternative: Use the fact that 4x5x5 can be seen as two 2x5x5 problems stacked.
    # C_top = A_top * B (2x5 * 5x5 = 2x5)
    # C_bot = A_bot * B (2x5 * 5x5 = 2x5)
    # 
    # For each 2x5x5, we can try to find a good decomposition.
    # Schoolbook for 2x5x5 is 50. Two of them = 100.
    # 
    # Let me try a different approach: use known results.
    # The rank of 2x2x2 is 7 (Strassen).
    # The rank of 2x3x3 is at most 2*3*3=18, but likely lower.
    # 
    # Let me try to construct a block decomposition where I use Strassen for 2x2x2 sub-blocks.
    # 
    # Partition 4x5x5 as:
    # A = [[A00, A01], [A10, A11]] where A00,A10 are 2x2, A01,A11 are 2x3
    # B = [[B00, B01], [B10, B11]] where B00 is 2x2, B01 is 2x3, B10 is 3x2, B11 is 3x3
    # C = [[C00, C01], [C10, C11]] where C00,C10 are 2x2, C01,C11 are 2x3
    #
    # This doesn't lead to independent sub-problems easily.
    #
    # Let me try yet another approach: use the 2x2x2 Strassen decomposition
    # and embed it in the larger problem.
    #
    # Actually, let me just try to find a decomposition with rank < 100 by
    # using a structured approach with smaller blocks.
    
    # Try: split into 2x2x2 blocks where possible
    # 4x5x5: think of it as (2+2)x(2+3)x(2+3)
    
    # Let me build the decomposition block by block.
    # The key insight: for the 2x2x2 sub-problems, use Strassen (rank 7 instead of 8).
    
    # Define the partition:
    # A rows: 0,1,2,3 -> split at 2: group0=[0,1], group1=[2,3]
    # A cols/B rows: 0,1,2,3,4 -> split at 2: group0=[0,1], group1=[2,3,4]
    # B cols/C cols: 0,1,2,3,4 -> split at 2: group0=[0,1], group1=[2,3,4]
    
    # Sub-problems:
    # (gA=0, gB_row=0, gB_col=0): A[0:2,0:2] (2x2) * B[0:2,0:2] (2x2) -> contributes to C[0:2,0:2]
    # (gA=0, gB_row=1, gB_col=0): A[0:2,2:5] (2x3) * B[2:5,0:2] (3x2) -> contributes to C[0:2,0:2]
    # (gA=0, gB_row=0, gB_col=1): A[0:2,0:2] (2x2) * B[0:2,2:5] (2x3) -> contributes to C[0:2,2:5]
    # (gA=0, gB_row=1, gB_col=1): A[0:2,2:5] (2x3) * B[2:5,2:5] (3x3) -> contributes to C[0:2,2:5]
    # (gA=1, gB_row=0, gB_col=0): A[2:4,0:2] (2x2) * B[0:2,0:2] (2x2) -> contributes to C[2:4,0:2]
    # (gA=1, gB_row=1, gB_col=0): A[2:4,2:5] (2x3) * B[2:5,0:2] (3x2) -> contributes to C[2:4,0:2]
    # (gA=1, gB_row=0, gB_col=1): A[2:4,0:2] (2x2) * B[0:2,2:5] (2x3) -> contributes to C[2:4,2:5]
    # (gA=1, gB_row=1, gB_col=1): A[2:4,2:5] (2x3) * B[2:5,2:5] (3x3) -> contributes to C[2:4,2:5]
    
    # This is 8 sub-problems. The 2x2x2 ones can use Strassen.
    # But the 2x3x2, 2x2x3, 2x3x3 ones need their own decompositions.
    
    # This is getting very complex. Let me try a simpler structural approach:
    # Just use schoolbook for everything but replace 2x2x2 blocks with Strassen.
    
    # Actually, let me try a completely different structural approach:
    # Use the fact that we can express 4x5x5 as a sum of tensor products of smaller decompositions.
    
    # For now, let me try to get a rank improvement by using Strassen on the 2x2x2 
    # sub-blocks and schoolbook elsewhere.
    
    # Partition:
    # n_split = [2, 2]
    # m_split = [2, 3]
    # p_split = [2, 3]
    
    n_splits = [2, 2]
    m_splits = [2, 3]
    p_splits = [2, 3]
    
    # Build the full decomposition
    total_rank = 0
    all_U = []
    all_V = []
    all_W = []
    
    n_offsets = [0, 2]
    m_offsets = [0, 2]
    p_offsets = [0, 2]
    
    for ia in range(2):
        for jb in range(2):
            for kb in range(2):
                ni = n_splits[ia]
                mj = m_splits[jb]
                pk = p_splits[kb]
                
                # This sub-problem: A[ia, jb] (ni x mj) * B[jb, kb] (mj x pk) -> C[ia, kb] (ni x pk)
                # Use schoolbook for this sub-block
                r_sub = ni * mj * pk
                U_sub = np.zeros((ni * mj, r_sub))
                V_sub = np.zeros((mj * pk, r_sub))
                W_sub = np.zeros((pk * ni, r_sub))
                
                for c, (i, j, k) in enumerate(product(range(ni), range(mj), range(pk))):
                    U_sub[i * mj + j, c] = 1.0
                    V_sub[j * pk + k, c] = 1.0
                    W_sub[k * ni + i, c] = 1.0
                
                # Now we need to embed this into the global decomposition.
                # The global indices:
                # A row: n_offsets[ia] + i, A col: m_offsets[jb] + j
                # B row: m_offsets[jb] + j, B col: p_offsets[kb] + k
                # C row: n_offsets[ia] + i, C col: p_offsets[kb] + k
                
                # Create global U, V, W for this sub-block
                U_g = np.zeros((n * m, r_sub))
                V_g = np.zeros((m * p, r_sub))
                W_g = np.zeros((p * n, r_sub))
                
                for i in range(ni):
                    for j in range(mj):
                        U_g[(n_offsets[ia] + i) * m + (m_offsets[jb] + j), :] = U_sub[i * mj + j, :]
                
                for j in range(mj):
                    for k in range(pk):
                        V_g[(m_offsets[jb] + j) * p + (p_offsets[kb] + k), :] = V_sub[j * pk + k, :]
                
                for k in range(pk):
                    for i in range(ni):
                        W_g[(p_offsets[kb] + k) * n + (n_offsets[ia] + i), :] = W_sub[k * ni + i, :]
                
                all_U.append(U_g)
                all_V.append(V_g)
                all_W.append(W_g)
                total_rank += r_sub
    
    # This gives total_rank = sum of all sub-block ranks = 100 (same as schoolbook)
    # Because we're just using schoolbook for each block.
    
    # To improve, we need to use lower-rank decompositions for some blocks.
    # The 2x2x2 blocks can use Strassen (rank 7 instead of 8).
    # There are 4 such blocks: (ia=0,jb=0,kb=0), (ia=0,jb=0,kb=1), (ia=1,jb=0,kb=0), (ia=1,jb=0,kb=1)
    # Wait, no. The 2x2x2 blocks are where ni=2, mj=2, pk=2.
    # That's only (ia=0 or 1, jb=0, kb=0): ni=2, mj=2, pk=2. Two blocks.
    # And (ia=0 or 1, jb=0, kb=1): ni=2, mj=2, pk=3. Not 2x2x2.
    # (ia=0 or 1, jb=1, kb=0): ni=2, mj=3, pk=2. Not 2x2x2.
    # (ia=0 or 1, jb=1, kb=1): ni=2, mj=3, pk=3. Not 2x2x2.
    
    # So only 2 blocks are 2x2x2. Using Strassen saves 2*(8-7)=2 rank. Total: 98.
    
    # Let me redo with Strassen for 2x2x2 blocks.
    total_rank = 0
    all_U = []
    all_V = []
    all_W = []
    
    for ia in range(2):
        for jb in range(2):
            for kb in range(2):
                ni = n_splits[ia]
                mj = m_splits[jb]
                pk = p_splits[kb]
                
                if ni == 2 and mj == 2 and pk == 2:
                    U_sub, V_sub, W_sub, r_sub = _strassen_2x2()
                else:
                    U_sub, V_sub, W_sub, r_sub = _schoolbook_block(ni, mj, pk)
                
                U_g = np.zeros((n * m, r_sub))
                V_g = np.zeros((m * p, r_sub))
                W_g = np.zeros((p * n, r_sub))
                
                for i in range(ni):
                    for j in range(mj):
                        U_g[(n_offsets[ia] + i) * m + (m_offsets[jb] + j), :] = U_sub[i * mj + j, :]
                
                for j in range(mj):
                    for k in range(pk):
                        V_g[(m_offsets[jb] + j) * p + (p_offsets[kb] + k), :] = V_sub[j * pk + k, :]
                
                for k in range(pk):
                    for i in range(ni):
                        W_g[(p_offsets[kb] + k) * n + (n_offsets[ia] + i), :] = W_sub[k * ni + i, :]
                
                all_U.append(U_g)
                all_V.append(V_g)
                all_W.append(W_g)
                total_rank += r_sub
    
    # Concatenate
    U_full = np.hstack(all_U)
    V_full = np.hstack(all_V)
    W_full = np.hstack(all_W)
    
    # Verify
    if _is_valid(U_full, V_full, W_full, total_rank):
        if time.time() - start_time < TIME_LIMIT - 5:
            U_list, V_list, W_list = _pack(U_full, V_full, W_full, total_rank)
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_list, V_list, W_list = _pack(U_sb, V_sb, W_sb, r_school)
    return {"U": U_list, "V": V_list, "W": W_list}
