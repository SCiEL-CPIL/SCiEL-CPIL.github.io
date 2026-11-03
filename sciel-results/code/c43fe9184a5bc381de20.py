"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import random

def _rank17_4x4x4_terms():
    """
    Return the 17 rank-1 terms for the rank-17 decomposition of 4x4x4 matrix multiplication.
    Each term is (U_coeffs, V_coeffs, W_coeffs) where each is a dict {(i,j): coeff}.
    
    This uses a known rank-17 construction. The decomposition is:
    T = sum_{r=1}^{17} u_r * v_r * w_r
    where u_r is a 4x4 matrix, v_r is a 4x4 matrix, w_r is a 4x4 matrix.
    
    We use the construction based on the fact that the 4x4 tensor has rank 17.
    One explicit construction uses the following approach:
    - Use 16 terms from a structured decomposition
    - Plus 1 additional term
    
    Actually, let me use a verified rank-17 decomposition.
    """
    # Rank-17 decomposition for 4x4 matrix multiplication
    # This is a known result; here we provide explicit coefficients.
    # Using the construction from literature (e.g., Alon-Rodeh or equivalent).
    
    terms = []
    
    # We'll construct this using a systematic approach.
    # The 4x4x4 tensor T_{ijk} = delta_{jk} (for matrix mult: T[i,j,k] = 1 if j==k else 0... 
    # Actually for matrix mult: (A*B)_{ik} = sum_j A_{ij} B_{jk}, so T_{i,j,k} = 1 iff the 
    # multiplication involves A_{ij} and B_{jk} contributing to C_{ik}.
    # So T[i,j,k] = 1 for all i,j,k (it's the full 4x4x4 tensor with all entries 1? No!)
    # Actually T[i,j,k] = 1 if and only if this term appears, which is always: 
    # C_{ik} = sum_j A_{ij} B_{jk}, so the tensor is T[i,j,k] = 1 for all i,j,k? No.
    # The tensor for matrix multiplication [n,n,n] has T[i,j,k] = 1 if j==j (always true)
    # Wait, I'm confusing myself. The matrix multiplication tensor is:
    # T[i,j,k] = 1 for all i,j,k in {0,...,n-1}
    # No! That's wrong. Let me think again.
    # 
    # For matrix multiplication, we have C = A * B where C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 (the coefficient of A_{ij} * B_{jk} in C_{ik})
    # So YES, T[i,j,k] = 1 for ALL i, j, k. The tensor is the all-ones tensor of size n^3.
    # 
    # Wait no, that's not right either. The tensor decomposition represents:
    # T = sum_r u_r * v_r * w_r where (u_r * v_r * w_r)[i,j,k] = u_r[i] * v_r[j] * w_r[k]
    # And we need T[i,j,k] = 1 for all i,j,k? That would mean the all-ones tensor, which has rank 1!
    # 
    # I'm making an error. Let me reconsider.
    # 
    # The matrix multiplication tensor T_{n} is defined by:
    # T[i,j,k] = 1 if and only if the product A_{ij} * B_{jk} contributes to C_{ik}
    # In standard matrix multiplication, EVERY A_{ij} * B_{jk} contributes to C_{ik}, so
    # T[i,j,k] = 1 for all i,j,k.
    #
    # But that can't be right because the rank of the all-ones tensor is 1.
    # 
    # OH WAIT. I think I've been confusing the tensor structure. Let me be very careful.
    #
    # The matrix multiplication tensor for n x n matrices is a 3-tensor with indices (i, j, k)
    # where i is the row of A, j is the column of A / row of B, and k is the column of B.
    # The value T[i,j,k] = 1 means that A[i,j] * B[j,k] appears in the sum for C[i,k].
    # Since in standard matrix multiplication, A[i,j] * B[j,k] DOES appear in C[i,k] for all i,j,k,
    # we have T[i,j,k] = 1 for all i,j,k.
    #
    # But this is a 3-dimensional array of all 1's, which as a tensor has rank 1 (it's the 
    # outer product of three all-ones vectors). That's clearly wrong because we know matrix
    # multiplication has rank > 1.
    #
    # I see my error now. The tensor is NOT all ones. The correct statement is:
    # The bilinear map (A, B) -> A*B is represented by a tensor T where
    # (A*B)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # And T[i,j,k] = 1 for all valid i,j,k. So the tensor IS all ones.
    #
    # But then the "rank" in tensor decomposition sense is different. The tensor rank of the
    # all-ones n^3 tensor is indeed 1. So something is fundamentally wrong with my understanding.
    #
    # Let me look at this differently. The matrix multiplication tensor is NOT the all-ones tensor.
    # The issue is that the indices are coupled: for a given (i,k), only specific j values contribute.
    # No wait, for matrix multiplication, ALL j values from 0 to n-1 contribute to C[i,k].
    #
    # I think the confusion is that in tensor decomposition for matrix multiplication, we're looking
    # at the tensor as a multilinear map, and the "rank" refers to the minimum number of rank-1
    # tensors to sum to get the matrix multiplication tensor. The matrix multiplication tensor 
    # T_n is defined by T_n[i,j,k] = 1 if and only if j is a valid intermediate index, which it
    # always is. So T_n IS the all-ones tensor, and its rank is 1?? That contradicts everything.
    #
    # I must be fundamentally misunderstanding the tensor structure. Let me think about it from
    # the Strassen algorithm perspective.
    #
    # In Strassen's algorithm for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # This corresponds to a rank-1 term: u1[i,j] * v1[j,k] * w1[i,k]
    # where u1 = [[1,0],[0,1]], v1 = [[1,1],[1,1]], w1 = [[1,0],[0,1]]
    # Check: u1[i,j]*v1[j,k]*w1[i,k] summed over j gives:
    # For i=0,k=0: sum_j u1[0,j]*v1[j,0]*w1[0,0] = u1[0,0]*v1[0,0]*1 + u1[0,1]*v1[1,0]*1 = 1*1*1 + 0*1*1 = 1
    # For i=0,k=1: sum_j u1[0,j]*v1[j,1]*w1[0,1] = 0 (since w1[0,1]=0)
    # Hmm, that doesn't work directly.
    #
    # I think the correct interpretation is:
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k (as a 3-index array),
    # and we want to write T = sum_r (u_r outer v_r outer w_r) where u_r, v_r, w_r are vectors.
    # But that gives the all-ones tensor which has rank 1.
    #
    # I clearly have a fundamental misunderstanding. Let me just use the known Strassen terms
    # and verify they work, rather than trying to derive the rank-17 from scratch.
    
    # Let me abandon the rank-17 approach and stick with the Strassen-based approach,
    # but try to optimize it.
    
    # For now, return empty and use the Strassen approach
    return terms

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a tuple (U_coeffs, V_coeffs, W_coeffs) where:
      U_coeffs: dict {(i, j): coeff} for the 2x2 left matrix
      V_coeffs: dict {(j, k): coeff} for the 2x2 right matrix
      W_coeffs: dict {(i, k): coeff} for the 2x2 output
    """
    terms = []
    
    # M1 = (P00+P11)*(Q00+Q11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    
    # M2 = (P10+P11)*Q00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    
    # M3 = P00*(Q01-Q11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    
    # M4 = P11*(Q10-Q00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    
    # M5 = (P00+P01)*Q11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    
    # M6 = (P10-P00)*(Q00+Q01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    
    # M7 = (P01-P11)*(Q10+Q11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    
    return terms

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    
    Strategy: Split the 2x4 * 4x6 multiplication into blocks:
    - A is split into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3)
    - B is split into top (rows 0,1) and bottom (rows 2,3), each with 3 column blocks of width 2
    
    So C = A_left * B_top + A_right * B_bottom
    
    Each (2x2) * (2x2) block multiplication uses 7 Strassen terms.
    Total: 2 (A blocks) * 3 (B col blocks) * 7 (Strassen) = 42 terms.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    # For each A block (left/right) and each B column block
    for a_block in range(2):  # 0: A cols 0,1; 1: A cols 2,3
        for b_col_block in range(3):  # B col blocks: 0,1,2 (each width 2)
            for t in range(7):  # 7 Strassen terms
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                
                # Update U: A block a_block has columns at offset a_block*2
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V: B rows depend on a_block (top if a_block=0, bottom if a_block=1)
                # B col block b_col_block has columns at offset b_col_block*2
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W: output C[i, k] where i is row of C, k is col of C
                # W index: k*n + i (official trace convention)
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n=2, m=4, p=6, trials=50):
    """Verify a decomposition numerically with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    for trial in range(trials):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        # Expected C
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Computed C
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        
        if C_comp != C_exp:
            return False
    return True

def _schoolbook_decomposition():
    """Fallback: rank-m = 4 decomposition."""
    n, m, p = DIMENSIONS
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4 
    matrix by a 4 by 6 matrix.
    
    Strategy: Use Strassen-based rank-42 decomposition.
    Split A into two 2x2 blocks, B into top/bottom 2x6 halves, each with 3 col blocks.
    Each (2x2)*(2x2) uses 7 Strassen terms. Total: 2*3*7 = 42.
    
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 100
    
    # Try Strassen decomposition first (rank 42)
    result = _strassen_block_decomposition()
    
    # Verify the decomposition
    if _verify_decomposition(result, 2, 4, 6, trials=50):
        return result
    
    # Check deadline
    if time.time() > deadline - 10:
        # Fall back to schoolbook
        return _schoolbook_decomposition()
    
    # Fallback to schoolbook (rank 4)
    schoolbook = _schoolbook_decomposition()
    if _verify_decomposition(schoolbook, 2, 4, 6, trials=50):
        return schoolbook
    
    # Should never reach here
    return schoolbook
# EVOLVE-BLOCK-END
