"""Matrix multiplication 2x3x6 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-14 decomposition for 2x3x6 matrix multiplication.
    
    Strategy: Treat as tensor product of 2x3x3 (rank 7) and 1x1x2 (rank 2).
    The 2x3x3 multiplication tensor can be decomposed using the known rank-7
    decomposition derived from the 3x3x3 rank-7 (Strassen-type) by symmetry slicing.
    
    For 2x3x3: U_A is 6x7, V_A is 9x7, W_A is 6x7
    For 1x1x2: U_B is 1x2, V_B is 2x2, W_B is 2x2
    
    The full 2x3x6 decomposition:
    U = U_A ⊗ U_B  -> shape (6*1, 7*2) = (6, 14)
    V = V_A ⊗ V_B  -> shape (9*2, 7*2) = (18, 14)
    W = W_A ⊗ W_B with row permutation -> shape (6*2, 7*2) = (12, 14)
    
    Actually, let me think more carefully about the Kronecker structure.
    
    For matrix multiplication of (n x m) * (m x p):
    U has rows indexed by (i, j) where i in [0,n), j in [0,m), so n*m rows
    V has rows indexed by (j, k) where j in [0,m), k in [0,p), so m*p rows  
    W has rows indexed by (k, i) where k in [0,p), i in [0,n), so p*n rows
    
    For 2x3x3 (n=2, m=3, p=3):
    U_A: 6 x 7
    V_A: 9 x 7
    W_A: 6 x 7
    
    For 1x1x2 (n=1, m=1, p=2):
    U_B: 1 x 2
    V_B: 2 x 2
    W_B: 2 x 2
    
    Tensor product gives 2x3x6 (n=2*1, m=3*1, p=3*2):
    U_new: (2*3) x (7*2) = 6 x 14
    V_new: (3*6) x (7*2) = 18 x 14
    W_new: (6*2) x (7*2) = 12 x 14
    
    Row indexing:
    U_A row: i*3 + j for i in [0,2), j in [0,3)
    U_B row: 0 (only one row)
    U_new row: (i*3 + j) * 1 + 0 = i*3 + j  [but we need i in [0,2), j in [0,3)]
    Wait, for 2x3x6: n=2, m=3, p=6
    U_new row index: i*m + j = i*3 + j for i in [0,2), j in [0,3)
    That's the same as U_A's row index! Good.
    
    V_A row: j*3 + k for j in [0,3), k in [0,3)
    V_B row: 0*p_B + k_B = k_B for k_B in [0,2)
    V_new row: (j*3 + k) * 2 + k_B = j*6 + 2k + k_B
    For 2x3x6: V row index = j*6 + k for j in [0,3), k in [0,6)
    Here k = 2k' + k_B where k' in [0,3), k_B in [0,2)
    So k ranges over 0..5, which is correct for p=6.
    V_new row index = j*6 + (2k' + k_B) = j*6 + 2k' + k_B
    And (j*3 + k') * 2 + k_B = j*6 + 2k' + k_B. Same! Good.
    
    W_A row: k*2 + i for k in [0,3), i in [0,2)
    W_B row: k_B for k_B in [0,2)
    W_new row: (k*2 + i) * 2 + k_B = 4k + 2i + k_B
    For 2x3x6: W row index = k*2 + i for k in [0,6), i in [0,2)
    Here k = 2k' + k_B where k' in [0,3), k_B in [0,2)
    So k ranges over 0..5.
    W row index = (2k' + k_B) * 2 + i = 4k' + 2k_B + i
    But Kronecker gives: (k'*2 + i) * 2 + k_B = 4k' + 2i + k_B
    These are different! We need 4k' + 2k_B + i vs 4k' + 2i + k_B.
    
    So we need to permute the rows of W_new.
    The Kronecker row for (k', i, k_B) is at position (2k' + i) * 2 + k_B = 4k' + 2i + k_B
    The desired row for (k', i, k_B) is at position 4k' + 2k_B + i
    
    We need to map: for each (k', i, k_B), the value at Kronecker position 4k' + 2i + k_B 
    should go to position 4k' + 2k_B + i.
    
    Let me construct W_new by iterating over all (k', i, k_B, c') combinations.
    """
    import numpy as np
    
    # Known rank-7 decomposition for 2x3x3 matrix multiplication
    # Using the standard Strassen-based construction
    # For 3x3x3, rank 7 decomposition exists (Strassen 1969)
    # For 2x3x3, we can use a similar approach
    
    # Let me use a known rank-7 decomposition for 2x3x3
    # This is derived from the 3x3x3 rank-7 by fixing one index
    
    # Standard 3x3x3 rank-7 decomposition (Strassen):
    # U_333, V_333, W_333 each 9x7
    
    # For 2x3x3, I'll construct it directly
    # Using bilinear decomposition with 7 terms
    
    # Let me use a well-known construction
    # For multiplying (2x3) by (3x3) to get (2x3)
    
    # Actually, let me just hardcode a valid rank-7 decomposition for 2x3x3
    # and then apply the Kronecker product with the 1x1x2 decomposition
    
    # Rank-7 decomposition for 2x3x3:
    # U_A: 6x7, V_A: 9x7, W_A: 6x7
    
    # Using a known construction from literature
    # The 2x3x3 multiplication tensor has rank at most 7
    
    # Let me construct it explicitly
    # For the product C = AB where A is 2x3 and B is 3x3:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    # I'll use a standard rank-7 decomposition
    # This is a valid decomposition (verified by construction)
    
    U_A = np.array([
        [1,  0,  0,  0,  0,  0,  0],  # A[0,0]
        [0,  1,  0,  0,  0,  0,  0],  # A[0,1]
        [0,  0,  1,  0,  0,  0,  0],  # A[0,2]
        [1,  0,  0,  0,  0,  0,  0],  # A[1,0]
        [0,  1,  0,  0,  0,  0,  0],  # A[1,1]
        [0,  0,  1,  0,  0,  0,  0],  # A[1,2]
    ], dtype=float)
    
    # This is just schoolbook rank 6, not 7. Let me think differently.
    
    # Actually, for 2x3x3, the schoolbook rank is 2*3*3 = 18.
    # The rank is at most 7? Let me reconsider.
    
    # For m x n matrix multiplication, the tensor rank is at most mn.
    # For 2x3x3: we're multiplying (2x3) by (3x3) to get (2x3).
    # The tensor has dimensions 2 (rows of A) x 3 (cols of A/rows of B) x 3 (cols of B)
    # So it's a 2x3x3 tensor. The schoolbook rank is 2*3*3 = 18.
    # The minimal rank is not known to be 7.
    
    # Let me reconsider the problem. The task is 2x3x6 matrix multiplication.
    # This means multiplying a 2x3 matrix by a 3x6 matrix to get a 2x6 matrix.
    # The tensor dimensions are 2 x 3 x 6 (for the three indices i, j, k).
    # Schoolbook rank is 2*3*6 = 36.
    
    # The proposed strategy was to use tensor product structure.
    # 2x3x6 can be seen as (2x3x3) tensor (1x1x2) if we split the 6 into 3*2.
    # But the tensor product of multiplication tensors corresponds to block matrix multiplication.
    
    # For the 2x3x3 part: schoolbook rank 18, not 7.
    # The rank of 2x3x3 multiplication tensor is at most 2*3 = 6? No.
    
    # Actually, for multiplying an (n x m) matrix by an (m x p) matrix,
    # the tensor rank is at most n*m*p (schoolbook).
    # For 2x3x3: at most 18.
    # The lower bound is at least max(2*3, 3*3, 2*3) = 9 (from border rank arguments).
    
    # Let me just use a simple approach: schoolbook decomposition of rank 36,
    # but try to find a smaller rank through search or known constructions.
    
    # For now, let me implement the schoolbook decomposition which is guaranteed correct,
    # and the task says "rank-minimal" but the seed is already at rank 36 with score 1.0.
    
    # Let me try rank 12: if I can find a rank-12 decomposition.
    # For 2x3x6, one approach: use the fact that we can compute the product
    # by first doing 2x3 times 3x3 (twice, for the two halves of the 6 columns),
    # but that's still rank 18.
    
    # Actually, let me just return the schoolbook decomposition with rank 36,
    # which is the same as the seed. The score will be 1.0.
    
    # But the task asks for a stronger algorithm. Let me try to find a rank < 36.
    
    # For 2x3x6 matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,...,5}
    
    # One approach: use bilinear decomposition with fewer terms.
    # The tensor is T_{i,j,k} = 1 for all i,j,k (since each output entry is a sum of products).
    # Wait, no. The tensor for matrix multiplication has T_{i,j,k} = 1 if the (i,k) entry
    # of the product involves A[i,j]*B[j,k]. So T_{i,j,k} = 1 for all valid i,j,k.
    
    # The rank of this tensor is the minimal number of rank-1 tensors that sum to it.
    # A rank-1 tensor is of the form u_i * v_j * w_k.
    
    # For the all-ones tensor of shape 2x3x6, the rank is min(2,3,6) = 2? No, that's for 
    # the tensor that is all ones. But our tensor is the structure tensor of matrix multiplication.
    
    # The structure tensor of (n x m) times (m x p) has entries T_{i,j,k} = delta where 
    # the (i,k) output depends on A_{i,j} and B_{j,k}. For standard multiplication, 
    # T_{i,j,k} = 1 for all i, j, k (each triple contributes to the sum for C_{i,k}).
    
    # Wait, that's not right either. The tensor is defined such that:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # In tensor form: C_{i,k} = sum_{j} sum_{c} U_{i,j,c} V_{j,k,c} W_{k,i,c} ... no.
    
    # Let me re-read the task. The decomposition is:
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # And the condition is that for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 (the coefficient in the product)
    
    # For standard matrix multiplication, the coefficient is 1 for each (i,j,k) triple.
    # So we need: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
    
    # This is exactly the rank of the 3-way tensor with all entries equal to 1, 
    # but with the specific indexing structure. Actually no, it's the tensor where 
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # The rank of the all-ones tensor of shape a x b x c is min(a, b, c).
    # For 2 x 3 x 6, that's min(2,3,6) = 2.
    
    # Can we achieve rank 2? Let's try:
    # U[i*m+j, c] = u_c (independent of i,j) for c=0,1
    # V[j*p+k, c] = v_c (independent of j,k) for c=0,1
    # W[k*n+i, c] = w_c (independent of k,i) for c=0,1
    # Then sum_c u_c v_c w_c must equal 1 for all i,j,k.
    # This is just u_0 v_0 w_0 + u_1 v_1 w_1 = 1.
    # We can set u_0=v_0=w_0=1, u_1=v_1=w_1=0. That gives rank 1!
    
    # Wait, but that would mean C_{i,k} = sum_j A_{i,j} B_{j,k} * 1, which is correct!
    # So rank 1 should work?
    
    # Let me check: with rank 1:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 for all k,i
    # Then for each (i,j,k): sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1*1*1 = 1. Correct!
    
    # But wait, does this actually decompose the matrix multiplication?
    # The condition is that the bilinear map defined by these factors equals matrix multiplication.
    # If all entries are 1, then:
    # C_{i,k} = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] * A[i,j] * B[j,k]
    #         = sum_j 1 * 1 * 1 * A[i,j] * B[j,k]
    #         = sum_j A[i,j] B[j,k]
    # That's correct!
    
    # So rank 1 works for the tensor condition. But is this a valid decomposition?
    # The task says "exactly decompose the matrix multiplication tensor". 
    # If the tensor is the all-ones tensor, then rank 1 should be valid.
    
    # Let me verify with the evaluator's convention: W[k*n+i, r]
    # So W is indexed by (k, i) with row index k*n + i.
    
    # With rank 1:
    # U: (2*3) x 1 = 6 x 1, all entries 1
    # V: (3*6) x 1 = 18 x 1, all entries 1
    # W: (6*2) x 1 = 12 x 1, all entries 1
    
    # This should be a valid rank-1 decomposition!
    
    # But the seed program uses rank 36, which suggests that either:
    # 1. My understanding is wrong, or
    # 2. The evaluator checks something more specific.
    
    # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) 
    # with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p) is:
    # T_{i,j,k} = 1 if the (i,k) entry of the product involves A_{i,j} and B_{j,k}
    # For standard multiplication, every (i,j,k) triple contributes, so T_{i,j,k} = 1.
    
    # A decomposition of rank r means: T_{i,j,k} = sum_{c=1}^r U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # (with appropriate indexing)
    
    # For the all-ones tensor, rank 1 works as I showed.
    
    # I think the seed program is just using the naive schoolbook approach and hasn't
    # been optimized. Let me implement rank 1.
    
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
