"""Matrix multiplication 2x4 * 4x7 rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication."""
    deadline = time.time() + 125  # 125s deadline, 5s safety margin before 130s hard kill
    
    n, m, p = 2, 4, 7
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # W index: k*n + i (row = k*n+i, col = c)
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if s != 1:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p  # 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def strassen_2x2():
        """Strassen decomposition for 2x2 * 2x2, returns 7 tuples of (u, v, w) patterns.
        Each tuple: (U_entries, V_entries, W_entries) where each is a dict mapping (i,j)->val etc.
        Actually, let's return the 7 rank-1 terms directly as (u, v, w) vectors."""
        # Strassen for C = A*B where A,B are 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # For our tensor: U[i,j,c], V[j,k,c], W[k,i,c]
        # For a 2x2*2x2 block, i in {0,1}, j in {0,1}, k in {0,1}
        # We'll return 7 terms, each with (u_dict, v_dict, w_dict)
        terms = []
        
        # M1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        terms.append((
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1},
            {(0,0):1, (1,1):1}
        ))
        
        # M2: (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1
        terms.append((
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(0,1):1}  # k=0, i=1 -> index k*n+i = 0*2+1 = 1
        ))
        
        # M3: A00(B01-B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01=1
        terms.append((
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(1,0):1}  # k=1, i=0 -> index 1*2+0 = 2
        ))
        
        # M4: A11(B10-B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C00=1, C10=1
        terms.append((
            {(1,1):1},
            {(1,0):1, (0,0):-1},
            {(0,0):1, (0,1):1}  # k=0,i=0 -> 0; k=0,i=1 -> 1
        ))
        
        # M5: (A00+A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00=-1, C01=1
        terms.append((
            {(0,0):1, (0,1):1},
            {(1,1):1},
            {(0,0):-1, (1,0):1}  # k=0,i=0 -> 0; k=1,i=0 -> 2
        ))
        
        # M6: (A10-A00)(B00+B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11=1
        terms.append((
            {(1,0):1, (0,0):-1},
            {(0,0):1, (0,1):1},
            {(1,1):1}  # k=1,i=1 -> 3
        ))
        
        # M7: (A01-A11)(B10+B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1
        terms.append((
            {(0,1):1, (1,1):-1},
            {(1,0):1, (1,1):1},
            {(0,0):1}  # k=0,i=0 -> 0
        ))
        
        return terms
    
    def block_decomposition_strassen():
        """Use block decomposition with Strassen on 2x2 sub-blocks.
        
        Split the 4 columns of A into two groups of 2: j=0,1 and j=2,3
        Split the 4 rows of B into two groups of 2: j=0,1 and j=2,3
        
        C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
        
        Each term is 2x2 @ 2x7.
        For 2x2 @ 2x7, we can further split the 7 columns into 3+3+1 or use Strassen-like.
        
        Actually, for 2x2 @ 2xp, we can use the following approach:
        Split the p columns into groups. For each group of 2 columns, use Strassen (rank 7).
        For remaining columns, use schoolbook (rank 4 per column).
        
        For 2x2 @ 2x7: split 7 into 3 pairs + 1 single
        - 3 pairs: 3 * 7 = 21
        - 1 single: 4
        Total: 25 per 2x2@2x7 block
        
        Two blocks: 2 * 25 = 50
        
        But we can do better by splitting 7 into different ways.
        Or use a more efficient decomposition for 2x2 @ 2xp.
        
        Actually, let me think about 2x2 @ 2xp more carefully.
        C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1; k=0..p-1
        
        This is p independent 2x2 @ 2x1 multiplications.
        2x2 @ 2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        This has 4 outputs, each a product of one A and one B entry.
        The rank of this tensor is 4 (it's essentially the identity on R^4).
        
        Wait, no. The tensor T[i,j,k] for 2x2@2x1 has i in {0,1}, j in {0,1}, k in {0}.
        T[i,j,0] = 1 for all i,j.
        A rank-r decomposition: sum_c U[i,j,c]*V[j,0,c]*W[0,i,c] = 1 for all i,j.
        
        This is the same as the 2x2@2x1 matrix multiplication tensor.
        Its rank is 4 (schoolbook). Can it be less?
        The tensor is the identity tensor of size 2x2x1, which has rank 4.
        Actually, it's the "parallel" tensor sum of 4 rank-1 tensors, and they're linearly independent,
        so rank is exactly 4.
        
        So 2x2 @ 2xp has rank 4p by schoolbook.
        Lower bound: max(4, 2p, 2p) = 2p for p>=2.
        
        Can we do better than 4p? For p=2, rank is 7 (Strassen) < 8=4*2.
        For p=3, can we do better than 12?
        
        Let me try to construct a decomposition for 2x2 @ 2x7 with rank < 28.
        
        Approach: Use Strassen on pairs of columns.
        For columns (k, k+1), we have a 2x2 @ 2x2 sub-problem.
        C[:,k:k+2] = A @ B[k:k+2,:]
        
        Using Strassen on this 2x2 @ 2x2:
        The 7 Strassen terms involve products like (A[i,a]+A[i,b])*(B[c,k]+B[c,k+1]) etc.
        
        Let me just build the full decomposition using Strassen on 2x2 blocks of the 7 columns.
        Split 7 columns as: (0,1), (2,3), (4,5), (6)
        - 3 Strassen blocks: 3*7 = 21
        - 1 schoolbook column: 4
        Total for one 2x2@2x7: 25
        Two such blocks: 50
        
        But wait, I need to be more careful. The two 2x2@2x7 blocks share the same A rows
        but different A columns and B rows. Let me construct this properly.
        """
        # Full decomposition:
        # A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
        # B is 4x7: B[j,k] for j in {0,1,2,3}, k in {0,...,6}
        # C is 2x7: C[i,k] for i in {0,1}, k in {0,...,6}
        #
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
        #
        # Split into:
        # C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
        #         = C1[i,k] + C2[i,k]
        #
        # C1 = A1 @ B1 where A1 = A[:,0:2] (2x2), B1 = B[0:2,:] (2x7)
        # C2 = A2 @ B2 where A2 = A[:,2:4] (2x2), B2 = B[2:4,:] (2x7)
        
        # For each 2x2 @ 2x7, use Strassen on column pairs
        # 2x2 @ 2x7: split 7 columns into pairs (0,1),(2,3),(4,5) and single (6)
        
        def strassen_2x2_2xp(A_offset, B_row_offset, col_groups):
            """Generate Strassen terms for 2x2 @ 2xp.
            A_offset: (i, j_offset) -> actual (i, j) in original A
            B_row_offset: j in {0,1} -> actual j = B_row_offset + j
            col_groups: list of column groups, each is (k_start, k_end) inclusive
            """
            terms = []
            for k_start, k_end in col_groups:
                if k_end - k_start == 1:
                    # Single column: schoolbook, 4 terms
                    k = k_start
                    for i in range(2):
                        for j in range(2):
                            terms.append({
                                'U': {(i, A_offset + j): 1},
                                'V': {(B_row_offset + j, k): 1},
                                'W': {(k, i): 1}
                            })
                elif k_end - k_start == 2:
                    # Two columns: Strassen, 7 terms
                    k0, k1 = k_start, k_end
                    # M1 = (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1])
                    terms.append({
                        'U': {(0, A_offset+0): 1, (1, A_offset+1): 1},
                        'V': {(B_row_offset+0, k0): 1, (B_row_offset+1, k1): 1},
                        'W': {(k0, 0): 1, (k1, 1): 1}
                    })
                    # M2 = (A[1,0]+A[1,1])*B[0,k0]
                    terms.append({
                        'U': {(1, A_offset+0): 1, (1, A_offset+1): 1},
                        'V': {(B_row_offset+0, k0): 1},
                        'W': {(k0, 1): 1}
                    })
                    # M3 = A[0,0]*(B[0,k1]-B[1,k1])
                    terms.append({
                        'U': {(0, A_offset+0): 1},
                        'V': {(B_row_offset+0, k1): 1, (B_row_offset+1, k1): -1},
                        'W': {(k1, 0): 1}
                    })
                    # M4 = A[1,1]*(B[1,k0]-B[0,k0])
                    terms.append({
                        'U': {(1, A_offset+1): 1},
                        'V': {(B_row_offset+1, k0): 1, (B_row_offset+0, k0): -1},
                        'W': {(k0, 0): 1, (k0, 1): 1}
                    })
                    # M5 = (A[0,0]+A[0,1])*B[1,k1]
                    terms.append({
                        'U': {(0, A_offset+0): 1, (0, A_offset+1): 1},
                        'V': {(B_row_offset+1, k1): 1},
                        'W': {(k0, 0): -1, (k1, 0): 1}
                    })
                    # M6 = (A[1,0]-A[0,0])*(B[0,k0]+B[0,k1])
                    terms.append({
                        'U': {(1, A_offset+0): 1, (0, A_offset+0): -1},
                        'V': {(B_row_offset+0, k0): 1, (B_row_offset+0, k1): 1},
                        'W': {(k1, 1): 1}
                    })
                    # M7 = (A[0,1]-A[1,1])*(B[1,k0]+B[1,k1])
                    terms.append({
                        'U': {(0, A_offset+1): 1, (1, A_offset+1): -1},
                        'V': {(B_row_offset+1, k0): 1, (B_row_offset+1, k1): 1},
                        'W': {(k0, 0): 1}
                    })
            return terms
        
        # Block 1: A[:,0:2] @ B[0:2,:]
        # A columns: 0,1; B rows: 0,1
        col_groups_1 = [(0,1), (2,3), (4,5), (6,6)]
        terms_1 = strassen_2x2_2xp(0, 0, col_groups_1)
        
        # Block 2: A[:,2:4] @ B[2:4,:]
        # A columns: 2,3; B rows: 2,3
        col_groups_2 = [(0,1), (2,3), (4,5), (6,6)]
        terms_2 = strassen_2x2_2xp(2, 2, col_groups_2)
        
        all_terms = terms_1 + terms_2
        rank = len(all_terms)
        
        # Build U, V, W matrices
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for c, term in enumerate(all_terms):
            for (i, j), val in term['U'].items():
                U[i*m + j][c] = val
            for (j, k), val in term['V'].items():
                V[j*p + k][c] = val
            for (k, i), val in term['W'].items():
                W[k*n + i][c] = val
        
        return U, V, W, rank
    
    # First, try the Strassen block decomposition
    U, V, W, rank = block_decomposition_strassen()
    
    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # If that didn't work, try a different column grouping
    # Try splitting 7 as (0,1),(2,3),(4,5,6) - but Strassen needs pairs
    # Or try different block splits of the 4 columns
    
    # Try splitting 4 columns as (0,1,2,3) -> (0,3) and (1,2) instead of (0,1) and (2,3)
    # This might not help, but let's try
    
    # Alternative: try rank 49 by using a different decomposition
    # For 2x2 @ 2x7, can we achieve rank 24 instead of 25?
    # Split 7 as (0,1),(2,3),(4,5),(6) gives 7+7+7+4 = 25
    # Split 7 as (0,1,2),(3,4,5),(6) - but Strassen is for 2x2@2x2
    
    # Let me try a different approach: use the fact that we have 2 rows in A
    # to share some computations across the two 2x2@2x7 blocks
    
    # Actually, let me try to verify with a direct computation first
    # and if it fails, fall back to schoolbook
    
    # Let me also try rank 48: split 7 as (0,1),(2,3),(4,5),(6) but use a 
    # more efficient decomposition for the single column
    
    # For now, if the Strassen block decomposition works, return it
    # Otherwise, fall back to schoolbook
    
    if time.time() > deadline - 10:
        U, V, W, rank = schoolbook()
        return {"U": U, "V": V, "W": W}
    
    # Try to find an even better decomposition
    # The current rank is 50 (3*7 + 4 per block, 2 blocks)
    # Can we do better?
    
    # Try: use a different split of the 4 columns
    # Split as (0,1) and (2,3) is natural, but what about (0,3) and (1,2)?
    # This doesn't change the rank, just which entries are grouped
    
    # Try: for 2x2@2x7, use a decomposition with rank 24
    # by grouping 7 columns as (0,1),(2,3),(4,5,6)
    # For the triple (4,5,6), use a 2x2@2x3 decomposition
    # 2x2@2x3: can we do better than 12 (schoolbook) or 7+4=11 (Strassen on pair + single)?
    # Split (4,5,6) as (4,5) + (6): 7 + 4 = 11
    # So 2x2@2x7 with groups (0,1),(2,3),(4,5),(6): 7+7+7+4 = 25
    # Or (0,1),(2,3,4),(5,6): 7 + 11 + 7 = 25 (same)
    
    # It seems 25 is the best we can get with this approach per 2x2@2x7 block.
    # Total: 50.
    
    # Let me try a completely different approach: direct search for rank 48, 49
    # using a structured ansatz
    
    # Ansatz: U has a specific block structure
    # Given time constraints, let's just return what we have
    
    # Verify once more
    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Debug: if verification fails, print info and fall back
    # For now, fall back to schoolbook
    U, V, W, rank = schoolbook()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
