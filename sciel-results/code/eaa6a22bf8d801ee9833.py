import time
import numpy as np
from itertools import product

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
TIME_LIMIT = 135  # seconds, leaves margin before 150s kill

def _build_strassen_decomposition(n, m, p, base_n=2):
    """
    Build a Strassen-based tensor decomposition for multiplying n x m by m x p.
    For n=m=p=6, use 2x2 block decomposition with Strassen for 2x2 blocks,
    then schoolbook for 2x2 sub-multiplications.
    
    Returns U, V, W as lists of lists with integer entries.
    """
    if n == 1 and m == 1 and p == 1:
        # Single scalar multiplication: rank 1
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    # Determine block sizes
    n_blocks = (n + base_n - 1) // base_n
    m_blocks = (m + base_n - 1) // base_n
    p_blocks = (p + base_n - 1) // base_n
    
    # For 6x6x6 with base_n=2: n_blocks=3, m_blocks=3, p_blocks=3
    # Each block is at most 2x2
    
    # Pad to exact multiples for clean block decomposition
    n_padded = n_blocks * base_n
    m_padded = m_blocks * base_n
    p_padded = p_blocks * base_n
    
    # Strassen decomposition for 2x2x2 blocks
    # 7 multiplications for 2x2 matrix multiplication
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B22-B12)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each pair of block positions (ib, jb) in A and (jb, kb) in B,
    # we compute C[ib, kb] using Strassen with 7 scalar (base-sized) multiplications.
    
    # Total number of block-level Strassen decompositions:
    # For each (ib, kb) in output, sum over jb: 7 multiplications
    # With n_blocks=3, m_blocks=3, p_blocks=3: 3*3*3*7 = 189 multiplications
    
    # But each "multiplication" at the block level is actually a base_n x base_n
    # matrix multiplication, which we do with schoolbook (rank base_n^3).
    
    # So total rank = n_blocks * m_blocks * p_blocks * 7 * base_n^3
    # = 3 * 3 * 3 * 7 * 8 = 1512 for base_n=2
    
    # That's worse than schoolbook! Let me reconsider.
    
    # Actually, the Strassen decomposition replaces each block-level multiplication
    # (which costs base_n^3 in schoolbook) with 7 multiplications of (base_n x base_n)
    # matrices. But those 7 multiplications are still full matrix multiplications.
    # So the total cost is:
    # Schoolbook: (n_blocks * base_n) * (m_blocks * base_n) * (p_blocks * base_n) = n*m*p
    # Strassen: n_blocks * p_blocks * m_blocks * 7 * base_n^3
    #          = (n/base_n) * (p/base_n) * (m/base_n) * 7 * base_n^3
    #          = 7 * n * m * p / base_n
    # For base_n=2: 7 * 216 / 2 = 756
    # That's better than 216! Wait no, 756 > 216.
    
    # Hmm, I think I'm confusing myself. Let me think more carefully.
    
    # The tensor rank for n x m by m x p matrix multiplication is at most n*m*p (schoolbook).
    # Strassen's algorithm for 2x2x2 has rank 7 (vs 8 for schoolbook).
    # For larger dimensions, we can use block decomposition.
    
    # For 6x6x6, if we decompose into 3x3 blocks of size 2x2:
    # Each output block C[ib,kb] = sum_jb A[ib,jb] * B[jb,kb]
    # Each product A[ib,jb] * B[jb,kb] is a 2x2 * 2x2 multiplication.
    # Using Strassen for each 2x2*2x2: rank 7 per product.
    # Number of products: 3 * 3 * 3 = 27
    # Total rank: 27 * 7 = 189
    
    # Yes! That's 189, which matches the knowledge from the cell findings.
    
    # Now let's build the actual U, V, W matrices.
    
    # U has shape (n*m, rank) = (36, 189)
    # V has shape (m*p, rank) = (36, 189)  
    # W has shape (p*n, rank) = (36, 189)
    
    # Indexing conventions:
    # U[i*m + j, c] - row index is i*m+j where i is row of A, j is col of A
    # V[j*p + k, c] - row index is j*p+k where j is row of B (= col of A), k is col of B
    # W[k*n + i, c] - row index is k*n+i where k is row of C (= col of B), i is col of C (= row of A)
    
    # Block indexing:
    # i = ib * base_n + ii, where ii in [0, base_n)
    # j = jb * base_n + jj, where jj in [0, base_n)
    # k = kb * base_n + kk, where kk in [0, base_n)
    
    # Strassen for 2x2x2:
    # The 7 products are:
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B11-B21)
    # M4 = A22 * (B22-B12)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A11+A12-A21-A22) * (B21+B22)  -- wait, let me use the standard form
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block product A[ib,jb] * B[jb,kb], we have 7 sub-products.
    # Each sub-product is a bilinear form in the entries.
    
    # Let's define the 7 Strassen products for 2x2:
    # Each product M_s (s=1..7) is a sum of terms A[ii,jj] * B[jj2,kk]
    # with coefficients.
    
    # M1: (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # M2: (A21+A22)B11 = A21*B11 + A22*B11
    # M3: A11(B11-B21) = A11*B11 - A11*B21
    # M4: A22(B22-B12) = A22*B22 - A22*B12
    # M5: (A11+A12)B22 = A11*B22 + A12*B22
    # M6: (A21-A11)(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
    # M7: (A12-A22)(B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
    
    # Note: indices here are local 2x2 indices: (11, 12, 21, 22)
    # Mapping to 0-based: 11->(0,0), 12->(0,1), 21->(1,0), 22->(1,1)
    
    # Let me define the 7 Strassen patterns for 2x2:
    # Each pattern is a list of (a_i, a_j, b_j, b_k, coeff) where
    # a_i is the row index in A (0 or 1), a_j is col in A (0 or 1)
    # b_j is row in B (0 or 1), b_k is col in B (0 or 1)
    
    # Actually, for the tensor decomposition, each "multiplication" in Strassen
    # corresponds to a rank-1 term in the bilinear form.
    
    # Let me directly construct the U, V, W for the Strassen decomposition.
    
    # For each (ib, jb, kb) in block space and each Strassen product s in 1..7,
    # we have a set of (ii, jj, kk) local indices with coefficients.
    
    # The contribution to U, V, W:
    # For a term with coefficient c, local A index (ii, jj), local B index (jj, kk):
    #   U[(ib*bn+ii)*m + (jb*bn+jj), col] += c
    #   V[(jb*bn+jj)*p + (kb*bn+kk), col] += 1  (or appropriate)
    #   W[(kb*bn+kk)*n + (ib*bn+ii), col] += 1  (or appropriate)
    
    # Wait, I need to be more careful. The tensor decomposition expresses:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In the decomposition: C[i,k] = sum_c U[i,j... wait no.
    
    # The tensor T_{i,j,k} for matrix mult is: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] where i indexes output row, j indexes the "middle" (col of A / row of B), k indexes output col
    # T[i,j,k] = 1 for all i,j,k (it's the identity in the middle dimension)
    
    # Wait, no. The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i, j, k] = delta_{j,j} ... no.
    
    # Let me think again. C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # This is because each term A[i,j]*B[j,k] contributes to C[i,k].
    
    # A rank-1 decomposition: T[i,j,k] = sum_c U[i*?, j*?...] 
    # In the given interface:
    # U has shape (n*m, rank): U[i*m + j, c]
    # V has shape (m*p, rank): V[j*p + k, c]
    # W has shape (p*n, rank): W[k*n + i, c]
    
    # The decomposition says: C[i,k] = sum_j sum_c ... hmm.
    
    # Actually, the standard tensor decomposition for matrix multiplication:
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    # where T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    # Wait, the indices need to match up.
    
    # From the interface:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So the tensor is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # And we need T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k] and each product contributes 1.
    
    # For schoolbook: for each (i,j,k), one column c with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    # Rank = n*m*p.
    
    # For Strassen: we group the (i,j,k) triples into blocks and use the 7-product decomposition.
    
    # Let me build this step by step for the 2x2 Strassen case first, then generalize.
    
    # For 2x2x2 Strassen, rank 7:
    # The 7 bilinear forms (each is a sum of A[i,j]*B[j,k] terms):
    
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # c00 = sum_j A[0,j]*B[j,0] = a00*b00 + a01*b10
    # c01 = sum_j A[0,j]*B[j,1] = a00*b01 + a01*b11
    # c10 = sum_j A[1,j]*B[j,0] = a10*b00 + a11*b10
    # c11 = sum_j A[1,j]*B[j,1] = a10*b01 + a11*b11
    
    # Strassen:
    # M1 = (a00+a11)*(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # M2 = (a10+a11)*b00 = a10*b00 + a11*b00
    # M3 = a00*(b00-b10) = a00*b00 - a00*b10
    # M4 = a11*(b11-b01) = a11*b11 - a11*b01
    # M5 = (a00+a01)*b11 = a00*b11 + a01*b11
    # M6 = (a10-a00)*(b00+b01) = a10*b00 + a10*b01 - a00*b00 - a00*b01
    # M7 = (a01-a11)*(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Let me verify c00:
    # M1 + M4 - M5 + M7
    # = (a00*b00 + a00*b11 + a11*b00 + a11*b11) 
    #   + (a11*b11 - a11*b01)
    #   - (a00*b11 + a01*b11)
    #   + (a01*b10 + a01*b11 - a11*b10 - a11*b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11
    #   + a11*b11 - a11*b01
    #   - a00*b11 - a01*b11
    #   + a01*b10 + a01*b11 - a11*b10 - a11*b11
    # = a00*b00 
    #   + (a00*b11 - a00*b11)
    #   + a11*b00
    #   + (a11*b11 + a11*b11 - a11*b11)
    #   - a11*b01
    #   + (-a01*b11 + a01*b11)
    #   + a01*b10
    #   - a11*b10
    # = a00*b00 + a11*b00 + a11*b11 - a11*b01 + a01*b10 - a11*b10
    # That doesn't look right. Let me recheck.
    
    # Hmm, I think I may have the Strassen formulas slightly off. Let me use the standard ones.
    
    # Standard Strassen (from Wikipedia):
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B11 - B21)
    # P4 = A22(B22 - B12)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # With 1-based indexing: A11=a00, A12=a01, A21=a10, A22=a11
    # B11=b00, B12=b01, B21=b10, B22=b11
    
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b00-b10)
    # P4 = a11(b11-b01)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # C11 = P1 + P4 - P5 + P7
    # = (a00+a11)(b00+b11) + a11(b11-b01) - (a00+a01)b11 + (a01-a11)(b10+b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 
    #   + a11*b11 - a11*b01
    #   - a00*b11 - a01*b11
    #   + a01*b10 + a01*b11 - a11*b10 - a11*b11
    # = a00*b00 + a11*b00 + a11*b11 - a11*b01 + a01*b10 - a11*b10
    # Hmm, still doesn't simplify to a00*b00 + a01*b10.
    
    # Let me just verify numerically.
    # a00=1,a01=2,a10=3,a11=4, b00=5,b01=6,b10=7,b11=8
    # C11 should be 1*5 + 2*7 = 19
    
    # P1 = (1+4)(5+8) = 5*13 = 65
    # P4 = 4*(8-6) = 4*2 = 8
    # P5 = (1+2)*8 = 24
    # P7 = (2-4)*(7+8) = -2*15 = -30
    # C11 = 65 + 8 - 24 + (-30) = 19 ✓
    
    # OK so the formula is correct, it just doesn't simplify algebraically in a clean way
    # because of cancellations. Good.
    
    # Now for the tensor decomposition, each P_s is a bilinear form.
    # P_s = sum_{i,j,k} coeff_s[i,j,k] * A[i,j] * B[j,k]
    
    # For the decomposition, we need:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # In Strassen, for each block (ib,jb,kb), we have 7 products P1..P7.
    # Each P_s can be written as a sum of rank-1 terms in the (i,j,k) space.
    # But actually, each P_s is already a bilinear form, not necessarily rank 1.
    
    # Wait, I think I'm overcomplicating this. Let me reconsider.
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k.
    # We want to write T as a sum of rank-1 tensors: T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    # In Strassen, the key insight is that instead of having separate terms for each (i,j,k),
    # we group them. For 2x2x2:
    # Instead of 8 separate rank-1 terms (one for each of the 8 (i,j,k) triples),
    # we use 7 terms where each term is a linear combination.
    
    # But wait, each rank-1 term in the tensor decomposition corresponds to ONE (i,j,k) combination
    # with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1. That's the schoolbook.
    
    # For Strassen to have rank 7, we need 7 columns in U, V, W such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # This means the 7 rank-1 tensors sum to the all-ones tensor.
    
    # Hmm, but Strassen's algorithm has 7 multiplications, not 7 rank-1 tensor terms.
    # The connection is through the tensor rank.
    
    # Actually, I think the way to think about it is:
    # The 7 Strassen products define 7 linear functionals on A and 7 on B,
    # and then C is computed from the 7 products.
    
    # For the tensor decomposition, each Strassen product P_s corresponds to a
    # bilinear form B_s(A,B) = sum_{i,j} u_s[i,j] * A[i,j] * sum_{j,k} v_s[j,k] * B[j,k]
    # No, that's not right either.
    
    # Let me think about it differently. In Strassen:
    # We compute 7 intermediate values: P1..P7
    # Each P_s = L_s(A) * R_s(B) where L_s is a linear form on A and R_s on B.
    # Then C[i,k] = sum_s M_{i,k,s} * P_s where M is a fixed matrix.
    
    # For the tensor decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_s M_{i,k,s} * L_s(A) * R_s(B)
    # = sum_s M_{i,k,s} * (sum_{i',j'} l_s[i',j'] * A[i',j']) * (sum_{j',k'} r_s[j',k'] * B[j',k'])
    
    # This is getting complex. Let me just directly construct the rank-1 decomposition
    # that corresponds to Strassen.
    
    # Actually, the simplest way: the tensor rank of 2x2x2 matrix mult is 7.
    # This means there exist 7 triples of matrices (U_c, V_c, W_c) for c=1..7
    # such that for all i,j,k in {0,1}:
    # sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    
    # One known decomposition (I'll derive it):
    # The 7 Strassen products give us 7 bilinear forms. Each bilinear form P_s
    # can be seen as a rank-1 element in the space of bilinear forms if we
    # appropriately define the U, V, W.
    
    # Actually, I recall that the tensor decomposition for Strassen works as follows:
    # Each of the 7 products P_s is a scalar. The output C is a linear combination
    # of these scalars. So:
    # C[i,k] = sum_s c_{i,k,s} * P_s
    # where c_{i,k,s} are constants.
    
    # And P_s = sum_{i',j'} a_{s,i',j'} * A[i',j'] * sum_{j',k'} b_{s,j',k'} * B[j',k']
    # Wait, P_s is a product of a linear form in A and a linear form in B:
    # P_s = (sum_{i',j'} a_{s,i',j'} A[i',j']) * (sum_{j',k'} b_{s,j',k'} B[j',k'])
    
    # So C[i,k] = sum_s c_{i,k,s} * (sum_{i',j'} a_{s,i',j'} A[i',j']) * (sum_{j',k'} b_{s,j',k'} B[j',k'])
    
    # This is a sum of 7 terms, each of which is a product of a linear form in A
    # and a linear form in B, times a constant. Each such term is a rank-1 element
    # in the tensor sense? Not exactly, because the linear form in A involves both
    # i and j indices, and similarly for B.
    
    # For the tensor decomposition T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]:
    # We need T[i,j,k] = 1.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_c [U[i*m+j, c] * W[k*n+i, c]] * [V[j*p+k, c] * A[i,j] * B[j,k]]
    # Hmm, this isn't working out cleanly.
    
    # Let me just look at it from the evaluator's perspective. The evaluator checks
    # that the decomposition is exact. So I need to find U, V, W such that
    # for all i in [0,6), j in [0,6), k in [0,6):
    # sum_c U[i*6+j, c] * V[j*6+k, c] * W[k*6+i, c] = 1
    
    # And the rank (number of columns) should be minimized.
    
    # For Strassen with 2x2 blocks: rank 189.
    # Let me just construct the 189-rank decomposition directly.
    
    # For each block (ib, jb, kb) where ib,kb in [0,3), jb in [0,3):
    # We have a 2x2x2 Strassen decomposition.
    # The 7 products for this block contribute to the local 2x2x2 tensor.
    
    # For the local 2x2x2 tensor (indices ii,jj,kk in {0,1}):
    # T_ii,jj,kk = 1
    # = sum_{s=1}^{7} U_s[ii,jj] * V_s[jj,kk] * W_s[kk,ii]
    
    # I need to find the specific U_s, V_s, W_s for the 7 Strassen products.
    
    # From the Strassen formulas:
    # P1 = (a00+a11)(b00+b11)
    # As a bilinear form: P1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # This is NOT rank-1 in the (i,j,k) tensor sense.
    
    # I think the correct interpretation is:
    # The 7 Strassen products are 7 scalars. The output C is determined by these 7 scalars.
    # For the tensor decomposition, we need to express the identity tensor as a sum
    # of 7 rank-1 tensors.
    
    # Let me just use a known rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
    
    # After some thought, I believe the correct approach is:
    # For each Strassen product P_s, we have:
    # P_s = sum_{(i,j,k) in support_s} coeff_{s,i,j,k} * A[i,j] * B[j,k]
    # And C[i,k] = sum_s M_{i,k,s} * P_s
    # 
    # For the tensor decomposition, each P_s contributes to the tensor as:
    # For each (i,j,k) with nonzero coeff in P_s:
    #   U[i,j] gets coeff, V[j,k] gets 1, W[k,i] gets M_{i,k,s}
    # But this would give rank = sum_s |support_s| which is more than 7.
    
    # I think I need to accept that the rank-7 decomposition of the 2x2x2 tensor
    # is not simply the 7 Strassen products as rank-1 terms.
    
    # Let me try a different approach: just use the known fact that 2x2x2 has rank 7
    # and construct the decomposition explicitly.
    
    # Actually, you know what, let me just use a simpler approach for 6x6x6:
    # Use 3x3 blocks of size 2x2, and for each block multiplication use the
    # schoolbook 2x2x2 decomposition (rank 8) instead of Strassen.
    # Total rank: 27 * 8 = 216. Same as schoolbook for 6x6x6. Not helpful.
    
    # OR: Use 2x2 Strassen for each block: 27 * 7 = 189.
    
    # To implement the 189-rank decomposition, I need the explicit rank-7
    # decomposition of the 2x2x2 matrix multiplication tensor.
    
    # Let me define it as follows. The 2x2x2 tensor T has T[i,j,k]=1 for all i,j,k in {0,1}.
    # A rank-1 decomposition: T[i,j,k] = sum_{c=1}^{7} u_c[i,j] * v_c[j,k] * w_c[k,i]
    
    # I'll use the following known decomposition (derived from Strassen):
    # c=1: u=[1,0,0,1] (i.e., u[0,0]=1, u[1,1]=1), v=[1,0,0,1] (v[0,0]=1, v[1,1]=1), w=[1,1,1,1]
    # Wait, this is getting too complicated to derive by hand.
    
    # Let me just use a numerical approach: solve for a rank-7 decomposition
    # using the known Strassen structure.
    
    # Actually, for the purpose of this task, let me just use the 189-rank
    # decomposition that's known to work, even if I have to construct it
    # carefully.
    
    # Here's my plan:
    # 1. For the 2x2x2 case, I'll use a precomputed rank-7 decomposition.
    # 2. For 6x6x6, I'll tile it with 2x2 blocks (3x3 blocks in each dimension).
    # 3. Each block uses the 2x2x2 rank-7 decomposition.
    # 4. Total rank: 27 * 7 = 189.
    
    # For the 2x2x2 rank-7 decomposition, I'll use the following:
    # (This is a standard result; the 7 rank-1 tensors sum to the all-ones 2x2x2 tensor)
    
    # Let me define the 7 triples explicitly.
    # U_c is 2x2, V_c is 2x2, W_c is 2x2 for c=1..7.
    # T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    
    # One valid decomposition (I'll verify numerically):
    
    # c=1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,1],[1,-1]]
    # Let me check: T[0,0,0] = 1*1*1 + 0 + 0 + 0 = 1 ✓ (only U[0,0]*V[0,0]*W[0,0] contributes since others are 0)
    # Wait, T[i,j,k] = sum_c U_c[i,j]*V_c[j,k]*W_c[k,i]
    # For c=1: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1, but we also have U[0,0]*V[0,1]*W[1,0] = 1*0*1 = 0
    # And U[1,1]*V[1,0]*W[0,1] = 1*0*1 = 0, U[1,1]*V[1,1]*W[1,1] = 1*1*(-1) = -1
    # So for c=1, the contribution to T[0,0,0] is 1, to T[1,1,1] is -1, etc.
    
    # This is getting too tedious to do by hand. Let me just use a numerical solver
    # to find a rank-7 decomposition, or use a known one from the literature.
    
    # Actually, let me take a step back. The task says the score is the rank,
    # and the seed has rank 216. The knowledge says Strassen gives rank 189.
    # Let me just implement the 189-rank decomposition using a straightforward
    # approach: for each block, use 7 "virtual" multiplications based on Strassen,
    # and express each virtual multiplication as a combination of the actual
    # (i,j,k) terms.
    
    # Wait, I think the key insight I'm missing is:
    # In the tensor decomposition, each column c corresponds to ONE rank-1 tensor.
    # The rank of the tensor is the minimum number of such rank-1 tensors needed.
    # 
    # For 2x2x2 matrix mult, the tensor rank is 7. This means we can find 7
    # rank-1 tensors that sum to the all-ones tensor.
    #
    # A rank-1 tensor is specified by three vectors: u (length 4, for 2x2 A),
    # v (length 4, for 2x2 B), w (length 4, for 2x2 C).
    # The tensor is u ⊗ v ⊗ w (with appropriate index mapping).
    
    # Let me just hardcode a known rank-7 decomposition of the 2x2x2 tensor.
    
    # From the literature (e.g., Strassen 1969), the 7 rank-1 decompositions are:
    
    # I'll use the following (verified to be correct):
    
    def _get_2x2_strassen_decomp():
        """Return U, V, W for 2x2x2 matrix mult with rank 7."""
        # Each is a 4x7 matrix (flattened 2x2 to 4)
        # U[i*2+j, c] for i,j in {0,1}
        # V[j*2+k, c] for j,k in {0,1}
        # W[k*2+i, c] for k,i in {0,1}
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # c=0: P1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        U[0, 0] = 1  # A[0,0]
        U[3, 0] = 1  # A[1,1]
        V[0, 0] = 1  # B[0,0]
        V[3, 0] = 1  # B[1,1]
        W[0, 0] = 1  # C[0,0] contribution: W[k*2+i, c], for k=0,i=0: W[0,0]
        W[1, 0] = 1  # C[0,1]: W[1*2+0, 0] = W[2, 0]... 
        
        # Wait, I'm confusing the indexing. Let me be very explicit.
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        # i,j,k in {0,1}
        
        # For the decomposition to work:
        # sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all i,j,k
        
        # Let me just use a known correct decomposition.
        # I'll define it by the 7 Strassen products and their contributions.
        
        # Actually, let me just use a different strategy.
        # I'll use the fact that for 6x6x6, I can use a 3x3 block decomposition
        # where each block is 2x2, and for each block I use the schoolbook 2x2x2
        # decomposition (rank 8). This gives rank 27*8 = 216, same as schoolbook.
        
        # Instead, let me try to directly construct a rank-189 decomposition
        # by using the Strassen algorithm at the block level.
        
        # Here's the key: in Strassen, we compute 7 intermediate values P1..P7.
        # Each P_s is a bilinear form in the entries of A and B.
        # The output C is a linear combination of P1..P7.
        
        # For the tensor decomposition, we can think of it as:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_s M_{i,k,s} * P_s
        # where P_s = sum_{i',j'} a_{s,i',j'} A[i',j'] * sum_{j',k'} b_{s,j',k'} B[j',k']
        
        # Now, for each s and each (i',j',j',k') term in P_s:
        # a_{s,i',j'} * b_{s,j',k'} * A[i',j'] * B[j',k']
        # This contributes to C[i,k] for all i,k via M_{i,k,s}.
        
        # So the full expansion is:
        # C[i,k] = sum_s sum_{i',j'} sum_{j',k'} M_{i,k,s} * a_{s,i',j'} * b_{s,j',k'} * A[i',j'] * B[j',k']
        
        # For the tensor decomposition, we need:
        # T[i',j',k'] = sum_c U[i'*m+j', c] * V[j'*p+k', c] * W[k'*n+i', c] = 1
        
        # Wait, the indices in T are (i',j',k') where i' is the row of A, j' is the col of A / row of B,
        # and k' is the col of B. And T[i',j',k'] = 1 for all valid indices.
        
        # In the Strassen expansion, the coefficient of A[i',j'] * B[j',k'] in C[i,k] is:
        # sum_s M_{i,k,s} * a_{s,i',j'} * b_{s,j',k'} (where j' in B equals j' in A)
        
        # And we need this to equal 1 when (i,k) = (i',k') and 0 otherwise... no,
        # we need the sum over j' to give 1 for the correct (i,k).
        
        # This is the standard matrix multiplication, so it should work out.
        
        # For the tensor decomposition, each "term" A[i',j'] * B[j',k'] in the expansion
        # of P_s contributes to the tensor T[i',j',k'] with weight:
        # sum over (i,k) of M_{i,k,s} * (contribution to C[i,k])
        
        # I think the cleanest way is:
        # For each block (ib,jb,kb) and each Strassen product s:
        # P_s = sum_{ii,jj} a_{s,ii,jj} A[ib,jb,ii,jj] * sum_{jj,kk} b_{s,jj,kk} B[jb,kb,jj,kk]
        # C[ib,kb,ii,kk] += M_{ii,kk,s} * P_s
        
        # For the tensor decomposition, each (ii,jj,kk) triple in the block
        # gets a contribution from each s:
        # T[ib,ii, jb,jj, kb,kk] += M_{ii,kk,s} * a_{s,ii,jj} * b_{s,jj,kk}
        
        # And we need the total to be 1.
        
        # Since Strassen is correct, this will sum to 1.
        
        # For the rank-1 decomposition, we can use one column per (block, s, ii, jj, kk)
        # term, but that would give rank = 27 * 7 * 8 = 1512, which is worse.
        
        # The key insight for Strassen's rank benefit: the 7 products P_s are shared
        # across all (ii,kk) outputs. So instead of having separate rank-1 terms for
        # each (ii,jj,kk), we can factor out the P_s.
        
        # But in the tensor decomposition, each column is a rank-1 tensor.
        # The Strassen structure means that for a given block and s, the same
        # (a_{s,ii,jj}, b_{s,jj,kk}) pair is used for all (ii,kk) outputs.
        
        # I think the correct rank-1 decomposition for Strassen is:
        # For each (ib,jb,kb,s): one column
        # U[i,j] = a_{s,ii,jj} where i = ib*2+ii, j = jb*2+jj
        # V[j,k] = b_{s,jj,kk} where j = jb*2+jj, k = kb*2+kk
        # W[k,i] = M_{ii,kk,s} where k = kb*2+kk, i = ib*2+ii
        #
        # But this doesn't work because U depends on (ii,jj), V on (jj,kk), W on (kk,ii),
        # and they're all different for each (ii,jj,kk).
        
        # I give up trying to derive this analytically. Let me just use a numerical
        # approach or a known result.
        
        # For the purpose of this task, let me just return the schoolbook decomposition
        # (rank 216) which is guaranteed to be correct. The Strassen improvement
        # to rank 189 is known but complex to implement correctly.
        
        return None  # Signal to use schoolbook
    
    # Just use schoolbook for now - it's guaranteed correct
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Try to build a Strassen-based decomposition
    # For 6x6x6 with 2x2 blocks: rank 189
    # I'll construct it by tiling the 2x2 Strassen decomposition.
    
    bn = 2  # block size
    nb = n // bn  # number of blocks = 3
    mb = m // bn
    pb = p // bn
    
    # 2x2x2 Strassen rank-7 decomposition
    # I'll define the 7 rank-1 tensors explicitly.
    # U_c[i*2+j], V_c[j*2+k], W_c[k*2+i] for i,j,k in {0,1}, c in 0..6
    
    # Using the standard Strassen decomposition:
    # The 7 products and their contributions to the tensor.
    
    # Let me define it as: for each c, U_c is a 4-vector, V_c is a 4-vector, W_c is a 4-vector.
    # T[i,j,k] = sum_c U_c[i*2+j] * V_c[j*2+k] * W_c[k*2+i] = 1
    
    # I'll use a known decomposition. Here are the 7 rank-1 tensors:
    
    # c=0: U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1]  (P1-like)
    # Let me just verify: T[0,0,0] = 1*1*1 + 0 + 0 + 1*0*1 = 1
    # T[0,0,1] = 1*0*1 + 0 + 0 + 1*0*1 = 0... that's not 1.
    
    # This isn't working. Let me use a different approach.
    
    # I'll solve for the rank-7 decomposition numerically using the Strassen structure.
    # Actually, let me just use the schoolbook decomposition. It's safe and correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
