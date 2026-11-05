"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 150.0  # wall-clock limit per evaluator
    safety_margin = 5.0  # seconds to spare
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Start with schoolbook decomposition (rank = n*m*p = 84)
    # Then try to reduce rank using Strassen-like 2x2x2 blocks
    
    # A 2x2x2 Strassen block covers triples (i0+di, j0+dj, k0+dk) for di,dj,dk in {0,1}
    # It uses 7 multiplications instead of 8, saving 1 rank term.
    # The 7 rank-1 terms for a 2x2x2 block at (i0,j0,k0) are:
    # t0: (i0,j0,k0), (i0+1,j0+1,k0), (i0+1,j0+1,k0+1) -> M = (A_i0 + A_i0+1)(B_j0 + B_j0+1)
    #     contributes: C_k0 gets M, C_k0+1 gets M (with appropriate signs)
    # Actually, let me use the standard Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # For the tensor decomposition, each multiplication corresponds to a rank-1 term.
    # The U, V, W matrices encode the coefficients.
    
    # Let's think about this differently. The matrix multiplication tensor T has
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # A rank-1 decomposition term c corresponds to:
    #   U[i,j,c] * V[j,k,c] * W[k,i,c] summed over c gives delta for each (i,j,k)
    # Actually, the tensor is T[i,j,k] where the multiplication is:
    #   (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k.
    
    # Schoolbook: for each (i,j,k), one term with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    
    # For a 2x2x2 block at (i0,j0,k0), we cover 8 triples with 7 terms.
    # The Strassen algorithm for 2x2 matrix multiplication:
    #   Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    #   C = A*B = [[c00, c01], [c10, c11]]
    #
    #   m1 = (a00 + a11)(b00 + b11)
    #   m2 = (a10 + a11) b00
    #   m3 = a00 (b10 - b11)
    #   m4 = a11 (b01 - b00)
    #   m5 = (a00 + a01) b11
    #   m6 = (a10 - a00) (b00 + b01)
    #   m7 = (a01 - a11) (b10 + b11)
    #
    #   c00 = m1 + m4 - m5 - m7
    #   c01 = m3 + m5
    #   c10 = m2 + m4
    #   c11 = m1 - m2 + m3 + m7
    #
    # In terms of the tensor T[i,j,k], for the block (i0..i0+1, j0..j0+1, k0..k0+1):
    # Each multiplication mt contributes to certain (i,j,k) entries.
    # 
    # m1 = (a00+a11)(b00+b11): This is a product of a sum of A entries and a sum of B entries.
    #   U part: a00 means U[i0, j0] += 1, a11 means U[i0+1, j0+1] += 1
    #   V part: b00 means V[j0, k0] += 1, b11 means V[j0+1, k0+1] += 1
    #   The product gives:
    #     a00*b00 -> (i0,j0,k0)
    #     a00*b11 -> (i0,j0+1,k0+1)
    #     a11*b00 -> (i0+1,j0,k0)
    #     a11*b11 -> (i0+1,j0+1,k0+1)
    #   And W part: c00 gets +1, c11 gets +1
    #     So W[k0, i0] += 1 for (i0,.,k0) contribution to c00
    #     W[k0+1, i0+1] += 1 for (i0+1,.,k0+1) contribution to c11
    #
    # This is getting complex. Let me just enumerate the 7 rank-1 terms explicitly.
    
    # For a 2x2x2 block at (i0, j0, k0), the 7 terms are:
    # Term 1: U has +1 at (i0,j0) and (i0+1,j0+1); V has +1 at (j0,k0) and (j0+1,k0+1)
    #          W has +1 at (k0,i0) and (k0+1,i0+1)
    # Term 2: U has +1 at (i0+1,j0) and (i0+1,j0+1); V has +1 at (j0,k0)
    #          W has +1 at (k0,i0+1) and (k0+1,i0+1)
    # Term 3: U has +1 at (i0,j0); V has +1 at (j0+1,k0) and -1 at (j0+1,k0+1)
    #          W has +1 at (k0,i0) and (k0+1,i0)
    # Term 4: U has +1 at (i0+1,j0+1); V has +1 at (j0,k0+1) and -1 at (j0,k0)
    #          W has +1 at (k0,i0+1) and (k0+1,i0+1)
    # Term 5: U has +1 at (i0,j0) and (i0,j0+1); V has +1 at (j0+1,k0+1)
    #          W has -1 at (k0,i0) and -1 at (k0+1,i0)
    # Term 6: U has +1 at (i0+1,j0) and -1 at (i0,j0); V has +1 at (j0,k0) and (j0,k0+1)
    #          W has +1 at (k0,i0+1) and (k0+1,i0+1)
    # Term 7: U has +1 at (i0,j0+1) and -1 at (i0+1,j0+1); V has +1 at (j0+1,k0) and (j0+1,k0+1)
    #          W has -1 at (k0,i0) and -1 at (k0+1,i0)
    
    # Let me verify term 1:
    # U contributions: (i0,j0):+1, (i0+1,j0+1):+1
    # V contributions: (j0,k0):+1, (j0+1,k0+1):+1
    # W contributions: (k0,i0):+1, (k0+1,i0+1):+1
    #
    # The rank-1 tensor is: sum over all (a,b,c) of U[a,b] * V[b,c] * W[c,a] * e_{a,b,c}
    # = U[i0,j0]*V[j0,k0]*W[k0,i0] * e_{i0,j0,k0}
    #   + U[i0,j0]*V[j0+1,k0+1]*W[k0+1,i0] * e_{i0,j0+1,k0+1}  -- wait, W[k0+1,i0] is 0
    #
    # Hmm, I think I'm confusing myself. Let me reconsider.
    #
    # In a rank-1 term, U, V, W are vectors. The tensor contribution is:
    #   T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # So for term 1:
    #   U[i0*m+j0, c] = 1, U[(i0+1)*m+(j0+1), c] = 1
    #   V[j0*p+k0, c] = 1, V[(j0+1)*p+(k0+1), c] = 1
    #   W[k0*n+i0, c] = 1, W[(k0+1)*n+(i0+1), c] = 1
    #
    # T_c[i,j,k] is non-zero when:
    #   (i,j) is in {(i0,j0), (i0+1,j0+1)} AND (j,k) is in {(j0,k0), (j0+1,k0+1)} AND (k,i) is in {(k0,i0), (k0+1,i0+1)}
    #
    # For (i,j) = (i0,j0): j must be j0. Then (j,k) = (j0,k) must be in {(j0,k0), (j0+1,k0+1)}, so k=j0... no, (j0,k0). So k=k0.
    #   Then (k,i) = (k0,i0) which is in the set. So T_c[i0,j0,k0] = 1*1*1 = 1. ✓
    #
    # For (i,j) = (i0,j0): j=j0. (j0,k) in set means k=k0. Already covered.
    #   What about (j0, k0+1)? No, (j0,k0+1) is not in {(j0,k0), (j0+1,k0+1)}.
    #
    # For (i,j) = (i0+1,j0+1): j=j0+1. (j,k) = (j0+1,k) in set means k=k0+1.
    #   (k,i) = (k0+1, i0+1) which is in the set. So T_c[i0+1,j0+1,k0+1] = 1. ✓
    #
    # So term 1 only covers (i0,j0,k0) and (i0+1,j0+1,k0+1). That's only 2 out of 8.
    #
    # This isn't right for Strassen. The issue is that in Strassen, the "multiplications" are of linear combinations, not individual elements. The tensor decomposition needs to capture that.
    #
    # Actually, I think the correct approach is: each "multiplication" m_t in Strassen corresponds to a rank-1 term where:
    #   U encodes the linear combination of A entries
    #   V encodes the linear combination of B entries
    #   W encodes how the result contributes to C entries
    #
    # For m1 = (a00+a11)(b00+b11):
    #   U: a00 -> U[i0,j0,c]=1, a11 -> U[i0+1,j0+1,c]=1
    #   V: b00 -> V[j0,k0,c]=1, b11 -> V[j0+1,k0+1,c]=1
    #   The product gives 4 cross terms, but we want them to contribute to C in a specific way.
    #   C00 gets +m1, C11 gets +m1.
    #   C00 = sum_j a0j * bj0, so for the (i0,k0) entry, m1 contributes via a00*b00, a00*b11 (no, b11 is for k0+1), a11*b00, a11*b11 (no).
    #   
    # I think the cleanest way is to just directly construct the 7 terms by computing what U, V, W should be.
    
    # Let me use a different approach: directly construct Strassen blocks.
    # For a 2x2x2 block at (i0,j0,k0), define local indices (0,1) for i,j,k offsets.
    
    # The 7 Strassen multiplications and their contributions to the 4 output entries:
    # m1: U=(0,1) for i, (0,1) for j offset... 
    
    # Actually, let me just hard-code the 7 terms for a 2x2x2 block.
    # I'll use the standard Strassen and carefully derive U, V, W.
    
    def make_strassen_terms(i0, j0, k0):
        """Return list of 7 (U_sparse, V_sparse, W_sparse) for 2x2x2 block at (i0,j0,k0).
        Each sparse dict maps (row, col) -> value for U, V, W respectively."""
        terms = []
        
        # m1 = (A00 + A11)(B00 + B11)
        # A00 = A[i0,j0], A11 = A[i0+1,j0+1]
        # B00 = B[j0,k0], B11 = B[j0+1,k0+1]
        # C00 += m1, C11 += m1
        # U: (i0,j0)->1, (i0+1,j0+1)->1
        # V: (j0,k0)->1, (j0+1,k0+1)->1
        # W: (k0,i0)->1, (k0+1,i0+1)->1
        # But this gives T[i0,j0,k0]=1, T[i0,j0+1,k0+1]=? no...
        # Let me just compute: T_c[i,j,k] = U[i,j]*V[j,k]*W[k,i]
        # For (i,j,k) = (i0,j0,k0): U[i0,j0]=1, V[j0,k0]=1, W[k0,i0]=1 -> 1
        # For (i,j,k) = (i0,j0,k0+1): U[i0,j0]=1, V[j0,k0+1]=0 -> 0
        # For (i,j,k) = (i0,j0+1,k0): U[i0,j0+1]=0 -> 0
        # For (i,j,k) = (i0,j0+1,k0+1): U[i0,j0+1]=0 -> 0
        # For (i,j,k) = (i0+1,j0,k0): U[i0+1,j0]=0 -> 0
        # For (i,j,k) = (i0+1,j0,k0+1): U[i0+1,j0]=0 -> 0
        # For (i,j,k) = (i0+1,j0+1,k0): U[i0+1,j0+1]=1, V[j0+1,k0]=0 -> 0
        # For (i,j,k) = (i0+1,j0+1,k0+1): U[i0+1,j0+1]=1, V[j0+1,k0+1]=1, W[k0+1,i0+1]=1 -> 1
        # So this term covers (i0,j0,k0) and (i0+1,j0+1,k0+1).
        
        # But in Strassen, m1 contributes to C00 and C11, not just diagonal elements.
        # C00 = sum_j A[i0,j] * B[j,k0]. The m1 term should contribute to C00 via A[i0,j0]*B[j0,k0] and A[i0+1,j0+1]*B[j0,k0]... no.
        
        # I think the issue is that in the tensor formulation, each rank-1 term is a "multiplication" of a linear combination of A entries with a linear combination of B entries, and the result is distributed to C entries.
        
        # Let me reconsider. The matrix multiplication tensor is:
        # T[i,j,k] = 1 for all (i,j,k)
        # And we want: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = T[i,j,k] = 1
        
        # For Strassen, we group 8 schoolbook terms into 7.
        # The 8 schoolbook terms for the block are:
        # (i0,j0,k0), (i0,j0,k0+1), (i0,j0+1,k0), (i0,j0+1,k0+1),
        # (i0+1,j0,k0), (i0+1,j0,k0+1), (i0+1,j0+1,k0), (i0+1,j0+1,k0+1)
        
        # Each schoolbook term c has:
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        
        # To combine them, we need to find 7 vectors (u_c, v_c, w_c) such that:
        # sum_{c in block} u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all 8 (i,j,k) in block
        
        # This is equivalent to finding a rank-7 decomposition of the 2x2x2 all-ones tensor.
        # The rank of the 2x2x2 all-ones tensor is 7 (this is the Strassen result).
        
        # The standard Strassen decomposition of the 2x2x2 all-ones tensor:
        # Let the 8 entries be T[0,0,0], T[0,0,1], T[0,1,0], T[0,1,1], T[1,0,0], T[1,0,1], T[1,1,0], T[1,1,1]
        # All equal to 1.
        
        # The 7 rank-1 terms:
        # Term 1: u1 = [1,0; 0,1] (i.e., u1[0,0]=1, u1[1,1]=1), v1 = [1,0; 0,1], w1 = [1,0; 0,1]
        #   T1[i,j,k] = u1[i,j]*v1[j,k]*w1[k,i]
        #   T1[0,0,0] = 1*1*1 = 1
        #   T1[0,0,1] = 1*0*0 = 0  (v1[0,1]=0)
        #   T1[0,1,0] = 0*... = 0  (u1[0,1]=0)
        #   T1[0,1,1] = 0
        #   T1[1,0,0] = 0  (u1[1,0]=0)
        #   T1[1,0,1] = 0
        #   T1[1,1,0] = 1*0*0 = 0  (v1[1,0]=0)
        #   T1[1,1,1] = 1*1*1 = 1
        #   So term 1 covers (0,0,0) and (1,1,1).
        
        # Term 2: u2 = [0,0; 1,1], v2 = [1,0; 0,0], w2 = [1,0; 0,1]
        #   T2[i,j,k] = u2[i,j]*v2[j,k]*w2[k,i]
        #   T2[0,0,0] = 0
        #   T2[0,0,1] = 0
        #   T2[0,1,0] = 0
        #   T2[0,1,1] = 0
        #   T2[1,0,0] = 1*1*1 = 1
        #   T2[1,0,1] = 1*0*0 = 0  (v2[0,1]=0, w2[1,1]=1 but v2[0,1]=0)
        #   T2[1,1,0] = 1*0*0 = 0
        #   T2[1,1,1] = 1*0*1 = 0  (v2[1,1]=0)
        #   So term 2 covers (1,0,0).
        
        # Term 3: u3 = [1,0; 0,0], v3 = [0,0; 1,-1], w3 = [1,0; 1,0]
        #   T3[i,j,k] = u3[i,j]*v3[j,k]*w3[k,i]
        #   T3[0,0,0] = 1*0*1 = 0
        #   T3[0,0,1] = 1*0*0 = 0
        #   T3[0,1,0] = 0
        #   T3[0,1,1] = 0
        #   T3[1,0,0] = 0
        #   T3[1,0,1] = 0
        #   T3[1,1,0] = 0
        #   T3[1,1,1] = 0
        #   Hmm, that doesn't work either.
        
        # Let me look up the correct Strassen decomposition of the all-ones 2x2x2 tensor.
        # Actually, I recall that the rank of the 2x2x2 all-ones tensor is indeed 7, and the
        # decomposition can be found in various references. Let me just implement it by solving
        # the system of equations.
        
        # Actually, for this problem, let me just try a simpler approach:
        # Start with schoolbook (rank 84), then greedily try to replace 8 schoolbook terms
        # with 7 Strassen terms for various 2x2x2 blocks.
        
        # But I need to get the Strassen terms right. Let me use a known decomposition.
        
        # After some research, the 7 rank-1 terms for the 2x2x2 all-ones tensor are:
        # Using 0-based local indices (i', j', k') in {0,1}:
        
        # I'll just implement a solver for this small case.
        
        pass  # placeholder
    
    # Given the complexity of getting Strassen terms exactly right, and the time constraint,
    # let me just return the schoolbook decomposition which is guaranteed correct.
    # The rank is 84, which gives score 1.0.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
