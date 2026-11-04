"""Matrix multiplication 4x4x5 rank-minimal decomposition.

Strategy:
  1. Build the schoolbook rank-80 decomposition (guaranteed correct).
  2. Attempt to find a lower-rank decomposition via structured search:
     - Use the known rank-7 Strassen decomposition for 2x2x2.
     - The 4x4x4 tensor can be decomposed using block-Strassen:
       Partition 4x4 into 2x2 blocks. The block multiplication tensor
       is a 2x2x2 tensor over 2x2 matrices. Each 2x2x2 scalar tensor
       has rank 7, but here each "entry" is a 2x2 matrix, so we need
       to handle the internal structure.
       
     Actually, the correct approach: The 4x4x4 matrix multiplication tensor
     T[i,j,k] = 1 (for the standard MM tensor) can be decomposed as follows.
     
     Let me use a different, proven approach:
     - For 4x4x4, the schoolbook rank is 64.
     - Strassen's algorithm gives rank 7^2 = 49 for 4x4x4 (applying 2x2x2
       Strassen at the block level, where each block multiplication is itself
       a 2x2x2 problem of rank 7, giving 7*7=49 total rank).
     
     Wait, that's not quite right either. Let me think carefully.
     
     The 4x4x4 MM tensor: C_{ik} = sum_j A_{ij} B_{jk}
     Partition into 2x2 blocks: A = [[A00,A01],[A10,A11]], each 2x2.
     B = [[B00,B01],[B10,B11]], each 2x2.
     C = [[C00,C01],[C10,C11]], each 2x2.
     
     C00 = A00*B00 + A01*B10
     C01 = A00*B01 + A01*B11
     C10 = A10*B00 + A11*B10
     C11 = A10*B01 + A11*B11
     
     Each product like A00*B00 is a 2x2x2 MM tensor. Using Strassen's rank-7
     decomposition for each 2x2x2, we get 4 * 7 = 28 rank terms for the 4x4x4.
     
     But we can do better by using Strassen at the block level:
     M1 = (A00+A11)(B00+B11)
     M2 = (A10+A11)B00
     M3 = A00(B01-B11)
     M4 = A11(B10-B00)
     M5 = (A00+A01)B11
     M6 = (A10-A00)(B00+B01)
     M7 = (A01-A11)(B10+B11)
     
     Each M_i is a product of two 2x2 matrices, which is a 2x2x2 MM tensor
     of rank 7. So total rank for 4x4x4 = 7 * 7 = 49.
     
     For 4x4x5:
     The 5th column of B can be handled separately.
     C_{i,4} = sum_{j=0}^{3} A_{ij} * B_{j,4}
     This is 4 schoolbook terms (one for each j), each contributing to all 4 rows of C's 5th column.
     So we need 4 additional rank terms.
     
     Total rank = 49 + 4 = 53.
     
     Let me verify this is correct and implement it.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _strassen_2x2_factors():
    """Return U, V, W factor matrices (each 4x7) for 2x2x2 MM tensor.
    
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    where i,j,k in {0,1} and c in {0,...,6}.
    """
    # Strassen's 7 terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    # i,j,k in {0,1}
    
    # M1 = (A00+A11)(B00+B11)
    # A factors: A00 (i=0,j=0) and A11 (i=1,j=1)
    # B factors: B00 (j=0,k=0) and B11 (j=1,k=1)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    # C00 gets +1, C11 gets +1
    W[0*2+0, 0] = 1  # C00 (k=0,i=0)
    W[1*2+1, 0] = 1  # C11 (k=1,i=1)
    
    # M2 = (A10+A11)B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    # C10 gets +1
    W[0*2+1, 1] = 1  # C10 (k=0,i=1)
    
    # M3 = A00(B01-B11)
    U[0*2+0, 2] = 1  # A00
    V[0*2+1, 2] = 1  # B01
    V[1*2+1, 2] = -1 # B11
    # C01 gets +1
    W[1*2+0, 2] = 1  # C01 (k=1,i=0)
    
    # M4 = A11(B10-B00)
    U[1*2+1, 3] = 1  # A11
    V[1*2+0, 3] = 1  # B10
    V[0*2+0, 3] = -1 # B00
    # C00 gets +1, C10 gets +1
    W[0*2+0, 3] = 1  # C00
    W[0*2+1, 3] = 1  # C10
    
    # M5 = (A00+A01)B11
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01
    V[1*2+1, 4] = 1  # B11
    # C00 gets -1, C01 gets +1
    W[0*2+0, 4] = -1 # C00
    W[1*2+0, 4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1  # B00
    V[0*2+1, 5] = 1  # B01
    # C11 gets +1
    W[1*2+1, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # A11
    V[1*2+0, 6] = 1  # B10
    V[1*2+1, 6] = 1  # B11
    # C00 gets +1
    W[0*2+0, 6] = 1  # C00
    
    return U, V, W

def _build_4x4x4_block_strassen():
    """Build rank-49 decomposition for 4x4x4 MM using block Strassen.
    
    Returns U (16x49), V (16x49), W (16x49).
    """
    sU, sV, sW = _strassen_2x2_factors()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # For each of the 7 Strassen terms at the block level,
    # we have a 2x2x2 sub-problem. The block indices (I,J,K) in {0,1}^3
    # map to actual indices: i = 2*I + i', j = 2*J + j', k = 2*K + k'
    # where i',j',k' in {0,1}.
    
    # The 7 Strassen terms define how blocks combine:
    # Term t uses A-blocks with (I,J) pairs and B-blocks with (J,K) pairs.
    # From the 2x2 Strassen:
    # M1: A blocks (0,0) and (1,1); B blocks (0,0) and (1,1)
    # M2: A blocks (1,0) and (1,1); B block (0,0)
    # M3: A block (0,0); B blocks (0,1) and (1,1)
    # M4: A block (1,1); B blocks (1,0) and (0,0)
    # M5: A blocks (0,0) and (0,1); B block (1,1)
    # M6: A blocks (1,0) and (0,0); B blocks (0,0) and (0,1)
    # M7: A blocks (0,1) and (1,1); B blocks (1,0) and (1,1)
    
    # For each term t, the contribution to C-block (I,K) is given by sW.
    # The internal 2x2x2 structure is given by sU, sV, sW.
    
    # Term definitions: list of (a_I, a_J, coeff_A) and (b_J, b_K, coeff_B) and (c_I, c_K, coeff_C)
    terms = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for t in range(7):
        a_terms, b_terms, c_terms = terms[t]
        
        # For this Strassen term, we need to expand the 2x2x2 internal structure.
        # The A-block pairs, B-block pairs, and C-block pairs are fixed by the term.
        # But within each block, we have the 2x2 internal structure.
        
        # Actually, the way this works: each "block product" like A_block(I,J) * B_block(J,K)
        # is itself a 2x2x2 MM problem. So for each of the 7 Strassen terms at the top level,
        # and for each pair of A-blocks and B-blocks involved, we need the full 2x2x2 decomposition.
        
        # Let me reconsider. The top-level Strassen gives:
        # M_t = (sum_{(I,J)} cA_{I,J} * A_{I,J}) * (sum_{(J,K)} cB_{J,K} * B_{J,K})
        # This is a product of two "sums of blocks". Each block is 2x2.
        # The product is a 2x2 matrix (the result block).
        # 
        # To decompose this into scalar rank terms, we need to expand:
        # M_t = sum_{(I1,J1)} sum_{(J2,K2)} cA_{I1,J1} * cB_{J2,K2} * A_{I1,J1} * B_{J2,K2}
        # But A_{I1,J1} * B_{J2,K2} only makes sense if J1 == J2 (for the contraction).
        # Wait, no. In the block Strassen, the multiplication is:
        # M_t = (sum cA * A_blocks) * (sum cB * B_blocks)
        # where the * is matrix multiplication of 2x2 matrices.
        # So (sum cA * A) * (sum cB * B) = sum_{I1,J1,J2,K2} cA_{I1,J1} cB_{J2,K2} A_{I1,J1} B_{J2,K2}
        # But A_{I1,J1} B_{J2,K2} is only defined when J1 == J2.
        # Hmm, this is getting complicated. Let me use a cleaner formulation.
        
        # Alternative: just enumerate all scalar rank terms directly.
        # For the 4x4x4 tensor, use the Kronecker product structure.
        # T_4x4x4 = T_2x2x2 ⊗ T_2x2x2 (in a certain sense)
        # Actually, T_4x4x4[i,j,k] = 1, and we can write it as:
        # T_4x4x4 = sum over block-Strassen terms...
        
        # Let me just directly construct the rank-49 decomposition.
        # For each of the 7 top-level Strassen terms, and for each of the 7 internal
        # Strassen terms (for the 2x2x2 sub-problem), we get one rank term.
        
        # The mapping:
        # Top-level term t has A-block coefficients a_t[(I,J)] and B-block coefficients b_t[(J,K)].
        # For the actual multiplication, we need J to match between A and B blocks.
        # So for term t, the valid (I,J,K) combinations are those where (I,J) is in a_t
        # and (J,K) is in b_t, with the same J.
        
        # For each such (I,J,K), the contribution is a_t[(I,J)] * b_t[(J,K)] * (2x2x2 MM tensor for blocks I,J,K)
        # The 2x2x2 MM tensor for blocks I,J,K is decomposed using the 7 Strassen terms.
        
        # So for each top-level term t, and for each valid (I,J,K), and for each internal
        # Strassen term s, we get a rank term.
        
        # Total: sum over t of (number of valid (I,J,K) for term t) * 7
        # For M1: (I,J) in {(0,0),(1,1)}, (J,K) in {(0,0),(1,1)}, matching J: (0,0,0) and (1,1,1) -> 2
        # For M2: (I,J) in {(1,0),(1,1)}, (J,K) in {(0,0)}, matching J: (1,0,0) -> 1
        # For M3: (I,J) in {(0,0)}, (J,K) in {(0,1),(1,1)}, matching J: (0,0,1) -> 1
        # For M4: (I,J) in {(1,1)}, (J,K) in {(1,0),(0,0)}, matching J: (1,1,0) -> 1
        # For M5: (I,J) in {(0,0),(0,1)}, (J,K) in {(1,1)}, matching J: (0,1,1) -> 1
        # For M6: (I,J) in {(1,0),(0,0)}, (J,K) in {(0,0),(0,1)}, matching J: (1,0,0),(0,0,0),(0,0,1) -> 3
        # For M7: (I,J) in {(0,1),(1,1)}, (J,K) in {(1,0),(1,1)}, matching J: (0,1,1),(1,1,0),(1,1,1) -> 3
        # Total: 2+1+1+1+1+3+3 = 12 block combinations, each with 7 internal terms = 84?
        
        # That's more than 49. Let me reconsider.
        
        # I think the issue is that the top-level Strassen already accounts for the
        # block structure, and each "term" M_t is a single 2x2x2 MM problem.
        # So M_t = X_t * Y_t where X_t and Y_t are 2x2 matrices (linear combinations of blocks).
        # The 2x2x2 MM tensor for X_t * Y_t has rank 7.
        # So total rank = 7 * 7 = 49.
        
        # The key insight: M_t is a product of two 2x2 matrices, and this product
        # is a 2x2x2 MM tensor. We decompose it using Strassen's rank-7.
        # The 2x2 matrices X_t and Y_t are linear combinations of the original blocks.
        # So the rank-49 decomposition is:
        # For each t in {0,...,6} and s in {0,...,6}:
        #   rank term (t,s) contributes to U, V, W.
        
        # X_t = sum_{(I,J)} a_t[(I,J)] * A_{I,J}
        # Y_t = sum_{(J,K)} b_t[(J,K)] * B_{J,K}
        # M_t = X_t * Y_t = sum_{i',j',k'} (X_t)_{i'j'} (Y_t)_{j'k'} C_{i'k'}
        
        # Using Strassen for the 2x2x2:
        # M_t = sum_s [sum_{i'j'} sU[i'2+j',s] (X_t)_{i'j'}] * [sum_{j'k'} sV[j'2+k',s] (Y_t)_{j'k'}] * [sum_{i'k'} sW[k'2+i',s] C_{i'k'}]
        
        # Substituting:
        # (X_t)_{i'j'} = sum_{(I,J)} a_t[(I,J)] A_{I*2+i', J*2+j'}
        # (Y_t)_{j'k'} = sum_{(J,K)} b_t[(J,K)] B_{J*2+j', K*2+k'}
        
        # So the rank term (t,s) has:
        # U[i*4+j, (t,s)] = sum_{(I,J)} a_t[(I,J)] * sU[(i mod 2)*2 + (j mod 2), s] * delta_{i//2, I} * delta_{j//2, J}
        # Wait, i = I*2 + i', j = J*2 + j'. So:
        # U[i*4+j, (t,s)] = a_t[(i//2, j//2)] * sU[(i%2)*2 + (j%2), s] if (i//2, j//2) in a_t else 0
        
        # Similarly for V and W.
        
        col = t * 7 + s
        for i in range(4):
            for j in range(4):
                I, J = i // 2, j // 2
                ip, jp = i % 2, j % 2
                if (I, J) in a_terms:
                    U[i*4+j, col] = a_terms[(I,J)] * sU[ip*2+jp, s]
        
        for j in range(4):
            for k in range(4):
                J, K = j // 2, k // 2
                jp, kp = j % 2, k % 2
                if (J, K) in b_terms:
                    V[j*4+k, col] = b_terms[(J,K)] * sV[jp*2+kp, s]
        
        for k in range(4):
            for i in range(4):
                K, I = k // 2, i // 2
                kp, ip = k % 2, i % 2
                if (I, K) in c_terms:
                    W[k*4+i, col] = c_terms[(I,K)] * sW[kp*2+ip, s]
    
    return U, V, W

def _build_4x4x5():
    """Build rank-53 decomposition for 4x4x5 MM."""
    U4, V4, W4 = _build_4x4x4_block_strassen()
    
    n, m, p = 4, 4, 5
    rank = 53
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # First 49 columns: 4x4x4 part (first 4 columns of output)
    U[:, :49] = U4
    V[:, :49] = V4
    W[:, :49] = W4
    
    # Last 4 columns: schoolbook for 5th column
    # C_{i,4} = sum_{j=0}^{3} A_{i,j} * B_{j,4}
    for j in range(4):
        col = 49 + j
        U[j, col] = 1  # A_{i,j} for all i -> actually U[i*4+j, col] = 1 for all i
        for i in range(4):
            U[i*4+j, col] = 1
        V[j*4+4, col] = 1  # B_{j,4}
        for i in range(4):
            W[4*4+i, col] = 1  # C_{i,4}
    
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=5, tol=1e-10):
    """Verify that the decomposition correctly represents the MM tensor."""
    # T[i,j,k] should be 1 for all i,j,k (coefficient of A_{ij}B_{jk} in C_{ik})
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Compute all T[i,j,k]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                expected = 1.0
                if abs(val - expected) > tol:
                    return False, (i, j, k, val, expected)
    return True, None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, build the schoolbook rank-80 as a fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try the rank-53 decomposition
    try:
        U53, V53, W53 = _build_4x4x5()
        valid, info = _verify(U53, V53, W53)
        if valid:
            # Convert to lists
            return {
                "U": U53.tolist(),
                "V": V53.tolist(),
                "W": W53.tolist()
            }
        else:
            # Fall back to schoolbook
            pass
    except Exception:
        pass
    
    # Fallback: schoolbook rank-80
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
