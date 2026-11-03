"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Best known rank so far
    best_rank = n * m * p  # schoolbook: 96
    best_decomp = _schoolbook(n, m, p)
    
    # Try Strassen-based decomposition
    # Strategy: Use Strassen on 2x2x2 blocks where possible
    # For 3x4x8, we can partition as:
    # Rows: 3 = 1 + 2 (or 2 + 1)
    # Cols: 4 = 2 + 2
    # Output cols: 8 = 4 + 4 (or other splits)
    
    # Let's try: A is 3x4, B is 4x8
    # Partition A into [A1; A2] where A1 is 1x4, A2 is 2x4
    # Partition B into [B1 B2] where B1 is 4x4, B2 is 4x4
    # Then C = A1*B1 + A2*B1 + A1*B2 + A2*B2 (this is just schoolbook split)
    
    # Better: Use Strassen on the 2x2x2 subproblem
    # A2 is 2x4, B1 is 4x4 -> result is 2x4
    # We can compute this using Strassen-like approach
    
    # Actually, let's think more carefully.
    # The tensor for (3,4,8) matrix mult has rank at most 96 (schoolbook).
    # We know rank(2,2,2) <= 7 (Strassen)
    # rank(2,2,n) can be bounded using Strassen.
    
    # Let's try a direct approach: use known low-rank decompositions
    # For (2,2,2): Strassen gives rank 7
    # For (2,2,8): We can use Strassen on the 2x2 part, treating the 8 as 4x2
    #   So (2,2,8) = (2,2,4) * (4,2,2) in some sense... 
    #   Actually, rank(2,2,n) <= 7 * ceil(n/2) for Strassen-based? No, that's not right.
    #   
    # Let me think differently. The matrix multiplication tensor T_{m,n,p} 
    # for multiplying m x n by n x p.
    # 
    # We can use the fact that if we have a decomposition of T_{a,b,c} with rank r,
    # then T_{ka, kb, kc} has rank at most r * k^2 (by Kronecker product).
    #
    # For (3,4,8):
    # Option 1: (3,4,8) = (1,1,1) x (3,4,8) -- not helpful
    # Option 2: Use (2,2,2) Strassen (rank 7)
    #   (3,4,8) can be decomposed as:
    #   A = [A_top; A_bottom] where A_top is 1x4, A_bottom is 2x4
    #   B = [B_left B_right] where B_left is 4x4, B_right is 4x4
    #   
    #   C = A_top * B_left + A_top * B_right + A_bottom * B_left + A_bottom * B_right
    #   
    #   A_top * B_left: (1,4,4) tensor, rank <= 16 (schoolbook)
    #   A_top * B_right: (1,4,4) tensor, rank <= 16
    #   A_bottom * B_left: (2,4,4) tensor
    #   A_bottom * B_right: (2,4,4) tensor
    #
    #   For (2,4,4): We can use Strassen on the 2x2 part.
    #   A_bottom is 2x4 = [A_bl A_br] where each is 2x2
    #   B_left is 4x4 = [B_tl B_tr; B_bl B_br] where each is 2x2
    #   
    #   A_bottom * B_left = [A_bl A_br] * [B_tl B_tr; B_bl B_br]
    #                     = A_bl*B_tl + A_br*B_bl, A_bl*B_tr + A_br*B_br
    #   
    #   Each 2x2*2x2 -> 2x2 uses Strassen (rank 7)
    #   So (2,4,4) rank <= 7 + 7 = 14? Wait, no.
    #   
    #   Actually, the (2,4,4) tensor is the multiplication of a 2x4 matrix by a 4x4 matrix.
    #   We can view this as: for each output column, we compute a linear combination.
    #   
    #   Let me use a different approach. Use the known result:
    #   rank(2,2,n) <= 7n/2 for even n (using Strassen recursively)
    #   Actually, rank(2,2,2) = 7, and rank(2,2,4) <= 14 (Kronecker: 7*2)
    #   rank(2,2,8) <= 28 (Kronecker: 7*4)
    #
    #   For (2,4,4): Think of it as computing 2 outputs, each is a 4-dim vector.
    #   C[i,j] = sum_k A[i,k] * B[k,j]
    #   For fixed i, this is a row of A times B.
    #   
    #   Alternative: (2,4,4) can be seen as (2,2,4) x (2,1,1) in some way... 
    #   This is getting complicated. Let me just try to construct a decomposition
    #   with rank < 96 and verify it.
    
    # Let's try: Use Strassen for the 2x2x2 core
    # Decompose 3x4x8 as follows:
    # Split rows of A: i=0 (1 row), i=1,2 (2 rows)
    # Split cols of B: k=0,1,2,3 -> split into k=0,1 and k=2,3
    # Split output cols: j=0..7 -> split into j=0..3 and j=4..7
    
    # C[0,0:4] = A[0,0:2]*B[0:2,0:4] + A[0,2:4]*B[2:4,0:4]
    # C[0,4:8] = A[0,0:2]*B[0:2,4:8] + A[0,2:4]*B[2:4,4:8]
    # C[1:3,0:4] = A[1:3,0:2]*B[0:2,0:4] + A[1:3,2:4]*B[2:4,0:4]
    # C[1:3,4:8] = A[1:3,0:2]*B[0:2,4:8] + A[1:3,2:4]*B[2:4,4:8]
    
    # Each term is a matrix multiplication:
    # (1,2,4), (1,2,4), (2,2,4), (2,2,4), (1,2,4), (1,2,4), (2,2,4), (2,2,4)
    
    # Rank of (1,2,4): schoolbook = 8, but can we do better?
    # (1,2,4): C[j] = sum_k A[k] * B[k,j] for j=0..3
    # This is just a linear map from R^2 x R^8 to R^4
    # Rank <= 8 (schoolbook)
    
    # Rank of (2,2,4): 
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] for i=0,1; j=0..3
    # This is 2x2 times 2x4.
    # Schoolbook rank = 2*2*4 = 16
    # Using Strassen on the 2x2 part: rank(2,2,2)=7, so rank(2,2,4) <= 7*2 = 14
    
    # Total: 4*(1,2,4) + 4*(2,2,4)
    # With schoolbook: 4*8 + 4*16 = 32 + 64 = 96 (same as before)
    # With Strassen for (2,2,4): 4*8 + 4*14 = 32 + 56 = 88
    
    # So we can get rank 88!
    # Let's construct this.
    
    decomp_88 = _strassen_based(n, m, p)
    if decomp_88 is not None:
        best_decomp = decomp_88
        best_rank = 88
    
    # Try to do even better
    # Can we reduce the (1,2,4) terms?
    # (1,2,4): C[j] = A[0]*B[0,j] + A[1]*B[1,j]
    # This is essentially: for each j, C[j] = dot(A, B[:,j])
    # The tensor is in R^2 x R^8 x R^4
    # Rank of this tensor: It's the sum of 4 rank-1 tensors (one per j)
    # So rank <= 4? No wait, each "rank-1" term is A[0]*B[0,j] for fixed j...
    # Actually, the (1,2,4) tensor: T[a,k,j] = delta(a,0) * delta(k, ?) ... 
    # Let me think. T_{a,k,j} where a in {0}, k in {0,1}, j in {0..3}
    # T[0,k,j] = 1 if we're computing C[j] = sum_k A[k] B[k,j]
    # So T[0,k,j] = 1 for all k,j? No, that's not right either.
    # 
    # The matrix multiplication tensor T_{m,n,p} has entries T_{i,k,j} = delta(i,k) * delta(k,j)? No.
    # T_{i,k,j} = 1 if the (i,j) entry of the product uses A[i,k] and B[k,j].
    # So T_{i,k,j} = delta(i,k) * delta(k,j) is wrong.
    # Actually, T_{i,k,j} = 1 for all valid i,k,j in the schoolbook, but that's not a tensor.
    # 
    # The tensor is: T = sum_{i,k,j} e_i ⊗ e_k ⊗ e_j * (coefficient)
    # For matrix mult: C[i,j] = sum_k A[i,k] B[k,j]
    # So T_{i,k,j} = 1 (the coefficient is 1 for each term A[i,k]B[k,j])
    # Wait, that means T = sum_{i,k,j} e_i ⊗ e_k ⊗ e_j, which has rank m*n*p.
    # That's the schoolbook.
    
    # OK so the (1,2,4) tensor has rank 1*2*4 = 8 in schoolbook.
    # Can we do better? The tensor is:
    # T[0,k,j] for k=0,1 and j=0,1,2,3
    # T = sum_{k,j} e_0 ⊗ e_k ⊗ e_j
    # This is e_0 ⊗ (sum_k e_k ⊗ (sum_j e_j))
    # = e_0 ⊗ (sum_k e_k) ⊗ (sum_j e_j)
    # This is a rank-1 tensor! So rank(1,2,4) = 1?
    # Wait, that can't be right. Let me reconsider.
    #
    # Oh I see my confusion. The tensor for (m,n,p) matrix mult is:
    # T_{i,k,j} = 1 if we're using the k-th column of A's i-th row and k-th row of B's j-th column
    # Actually no. The standard definition:
    # The multiplication tensor μ_{m,n,p} has entries:
    # μ_{i,k,j} = 1 for all i in [m], k in [n], j in [p]
    # And the multiplication is: C = sum_{k} (A[:,k] ⊗ B[k,:]) ... no
    #
    # Let me be precise. The tensor is in R^{m*n} ⊗ R^{n*p} ⊗ R^{m*p}
    # Wait, the problem says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # So the tensor is in R^{m*n} ⊗ R^{m*p} ⊗ R^{p*n}
    #
    # The entries: For the (i,j) output (i in [n], j in [p]), we have:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # In the tensor formulation:
    # T[a, b, c] where a indexes A (size m*n), b indexes B (size m*p), c indexes C (size p*n)
    # 
    # A is n x m, so a = i*m + k (row i, col k)
    # B is m x p, so b = k*p + j (row k, col j)  
    # C is n x p, but W uses (p*n, rank) with W[k*n+i, r]
    # So c = j*n + i? Let me re-read: "W[k*n+i,r]" and "output factor follows W[k*n+i,r]"
    # And C[i,j] = sum_k A[i,k] B[k,j]
    # So the output index for C[i,j] in W is... if W has shape (p*n, rank), 
    # and W[k*n+i, r], then the index is k*n+i. But k ranges over m (the inner dimension),
    # not p. This is confusing.
    #
    # Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So:
    # U has rows indexed by i*m+j where i in [n], j in [m] -> n*m rows
    # V has rows indexed by j*p+k where j in [m], k in [p] -> m*p rows
    # W has rows indexed by k*n+i where k in [p], i in [n] -> p*n rows
    #
    # And the multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # where A is n x m, B is m x p, C is n x p
    #
    # In the tensor: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should give the coefficient of A[i,j]*B[j,k] in C[i,k]
    #
    # For schoolbook: each term A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1
    # So for each (i,j,k), we have one rank-1 term:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    #
    # So the tensor is: sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # OK so for (1,2,4) subproblem: n=1, m=2, p=4
    # Tensor: sum_{i=0}^{0} sum_{j=0}^{1} sum_{k=0}^{3} e_{i*2+j} ⊗ e_{j*4+k} ⊗ e_{k*1+i}
    # = sum_{j=0}^{1} sum_{k=0}^{3} e_j ⊗ e_{j*4+k} ⊗ e_k
    # = e_0 ⊗ (sum_{k=0}^{3} e_{4+k}) ⊗ (sum_{k=0}^{3} e_k) + e_1 ⊗ (sum_{k=0}^{3} e_{4+k}) ⊗ (sum_{k=0}^{3} e_k)
    # Wait, that's not right. Let me redo:
    # = sum_{j=0}^{1} sum_{k=0}^{3} e_j ⊗ e_{j*4+k} ⊗ e_k
    # For j=0: sum_{k=0}^{3} e_0 ⊗ e_k ⊗ e_k = e_0 ⊗ (e_0⊗e_0 + e_1⊗e_1 + e_2⊗e_2 + e_3⊗e_3)
    # For j=1: sum_{k=0}^{3} e_1 ⊗ e_{4+k} ⊗ e_k = e_1 ⊗ (e_4⊗e_0 + e_5⊗e_1 + e_6⊗e_2 + e_7⊗e_3)
    #
    # So T = e_0 ⊗ (sum_k e_k⊗e_k) + e_1 ⊗ (sum_k e_{4+k}⊗e_k)
    # The first part is rank 1 (it's e_0 ⊗ S where S = sum_k e_k⊗e_k)
    # The second part is rank 1 (it's e_1 ⊗ S' where S' = sum_k e_{4+k}⊗e_k)
    # So total rank = 2!
    #
    # Wait, that seems too good. Let me verify: 
    # T = e_0 ⊗ S + e_1 ⊗ S'
    # This is a sum of 2 rank-1 tensors, so rank <= 2.
    # And it's clearly not rank 1 since S and S' are different.
    # So rank(1,2,4) = 2.
    
    # Similarly, let's check (2,2,4): n=2, m=2, p=4
    # T = sum_{i=0}^{1} sum_{j=0}^{1} sum_{k=0}^{3} e_{i*2+j} ⊗ e_{j*4+k} ⊗ e_{k*2+i}
    # For i=0, j=0: sum_k e_0 ⊗ e_k ⊗ e_{2k} = e_0 ⊗ (e_0⊗e_0 + e_1⊗e_2 + e_2⊗e_4 + e_3⊗e_6)
    # For i=0, j=1: sum_k e_1 ⊗ e_{4+k} ⊗ e_{2k} = e_1 ⊗ (e_4⊗e_0 + e_5⊗e_2 + e_6⊗e_4 + e_7⊗e_6)
    # For i=1, j=0: sum_k e_2 ⊗ e_k ⊗ e_{2k+1} = e_2 ⊗ (e_0⊗e_1 + e_1⊗e_3 + e_2⊗e_5 + e_3⊗e_7)
    # For i=1, j=1: sum_k e_3 ⊗ e_{4+k} ⊗ e_{2k+1} = e_3 ⊗ (e_4⊗e_1 + e_5⊗e_3 + e_6⊗e_5 + e_7⊗e_7)
    #
    # Each of these 4 terms is rank 1, so rank <= 4.
    # Can we do better? Possibly, but let's use 4 for now.
    
    # So total rank: 4*(1,2,4) + 4*(2,2,4) = 4*2 + 4*4 = 8 + 16 = 24??
    # That seems way too low. Let me double-check.
    
    # Actually, I think I'm making an error in the tensor structure. Let me re-examine.
    # The issue is that when we split the problem into subproblems, the subproblem tensors
    # are not independent - they share the same A and B indices.
    
    # Let me just try to construct a rank-88 decomposition using Strassen for (2,2,2) blocks
    # and verify it numerically.
    
    if time.time() > deadline:
        return best_decomp
    
    # Let's try a more systematic approach:
    # Use the known Strassen decomposition for (2,2,2) and build up from there.
    
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For (2,2,2), the tensor is:
    # T = sum_{i,j,k} e_{i*2+j} ⊗ e_{j*2+k} ⊗ e_{k*2+i}
    
    # Let me just implement the rank-88 construction and verify.
    
    decomp = _construct_strassen_88(n, m, p)
    if decomp is not None:
        # Verify
        if _verify(decomp, n, m, p):
            best_decomp = decomp
            best_rank = 88
    
    if time.time() > deadline:
        return best_decomp
    
    # Try to find an even better decomposition using optimization
    # Start from the rank-88 decomposition and try to reduce rank
    
    # For now, return the best we have
    return best_decomp

def _schoolbook(n, m, p):
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

def _strassen_based(n, m, p):
    """Construct a Strassen-based decomposition for 3x4x8 with rank 88."""
    # We'll use the following strategy:
    # Split A (3x4) into A1 (1x4) and A2 (2x4)
    # Split B (4x8) into B1 (4x4) and B2 (4x4)
    # 
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    # 
    # For A1*B1: (1,4,4) tensor
    # For A1*B2: (1,4,4) tensor
    # For A2*B1: (2,4,4) tensor
    # For A2*B2: (2,4,4) tensor
    #
    # (1,4,4) rank: using the analysis above, it's 4 (one per output col pair)
    # Actually, let me just use schoolbook for (1,4,4): rank = 16
    # And for (2,4,4): use Strassen on the 2x2 part
    #
    # (2,4,4): A2 is 2x4 = [A2a A2b] where A2a, A2b are 2x2
    #           B is 4x4 = [B11 B12; B21 B22] where each is 2x2
    # C = A2a*B11 + A2b*B21, A2a*B12 + A2b*B22
    #
    # Each 2x2*2x2 -> 2x2 uses Strassen (rank 7)
    # So (2,4,4) rank <= 7 + 7 = 14
    
    # Total: 2*(1,4,4) + 2*(2,4,4)
    # With schoolbook for (1,4,4): 2*16 + 2*14 = 32 + 28 = 60
    # Hmm, that's even better than 88!
    
    # Let me construct this carefully.
    
    # Actually, I realize I need to be more careful about the tensor structure.
    # Let me just build it up from Strassen blocks.
    
    # For now, let's return None and use a simpler approach
    return None

def _construct_strassen_88(n, m, p):
    """Construct rank-88 decomposition using Strassen for 2x2x2 blocks."""
    # This is a placeholder - let's implement it properly
    # For now, return None
    return None

def _verify(decomp, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    import numpy as np
    
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    rank = U.shape[1]
    
    # Test with random matrices
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                # C[i,j] = sum_c U[i*m+?, c] * V[?*p+j, c] * W[j*n+i, c]
                # Actually, we need to sum over all (i',j',k') such that
                # the tensor term contributes to C[i,j]
                # 
                # The decomposition gives:
                # C[i,j] = sum_{a,b,c: W index = j*n+i} U[a,c] * V[b,c] * W[j*n+i, c]
                # where a = i'*m + j', b = j'*p + k
                # and the term is nonzero only if i'=i, j'=k, and the W index matches
                #
                # Actually, the correct formula is:
                # C[i,j] = sum_{i',j',k} A[i',j'] * B[j',k] * (tensor coefficient for (i',j',k) -> (i,j))
                #
                # In the decomposition:
                # sum_c U[i'*m+j', c] * V[j'*p+k, c] * W[k*n+i, c] should give
                # 1 if (i',j',k) contributes to C[i,j] with coefficient 1, else 0
                #
                # For matrix mult, A[i',j'] * B[j',k] contributes to C[i',k] (not C[i,j] in general)
                # So the tensor coefficient for (i',j',k) -> (i,j) is delta(i',i) * delta(k,j)
                
                pass
        
        # Let me just directly compute the tensor and check
        # The reconstructed C should satisfy:
        # C[i,j] = sum_{i',j',k} A[i',j'] * B[j',k] * T[i',j',k,i,j]
        # where T is the tensor defined by the decomposition
        
        # Easier: just check that for the standard basis vectors, the decomposition works
        # Or, compute the full tensor and compare
    
    # For a quick check, let's verify with specific values
    # Set A to have a 1 at position (i0, j0) and 0 elsewhere
    # Set B to have a 1 at position (j0, k0) and 0 elsewhere
    # Then C should have a 1 at position (i0, k0)
    
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                A = np.zeros((n, m))
                A[i0, j0] = 1
                B = np.zeros((m, p))
                B[j0, k0] = 1
                C_expected = A @ B  # should be 1 at (i0, k0), 0 elsewhere
                
                # Compute using decomposition
                # C[i,j] = sum_c sum_{i',j',k} A[i',j'] * U[i'*m+j', c] * B[j',k] * V[j'*p+k, c] * W[k*n+i, c]
                # With A having only A[i0,j0]=1 and B having only B[j0,k0]=1:
                # C[i,j] = U[i0*m+j0, c] * V[j0*p+k0, c] * W[k0*n+i, c] summed over c
                # = sum_c U[i0*m+j0, c] * V[j0*p+k0, c] * W[k0*n+i, c]
                
                C_computed = np.zeros((n, p))
                for i in range(n):
                    for j in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i0*m+j0, c] * V[j0*p+k0, c] * W[k0*n+i, c]
                        C_computed[i, j] = val
                
                if not np.allclose(C_computed, C_expected, atol=1e-9):
                    return False
    
    return True

# EVOLVE-BLOCK-END
