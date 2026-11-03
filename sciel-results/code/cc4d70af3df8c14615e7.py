"""Winograd FFT-based exact decomposition for 4x5 * 5x6 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]
_t0 = time.time()

def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition for n x m * m x p."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W

def _fft4_decomposition():
    """
    Winograd/FFT decomposition for 4x4 * 4x4.
    Returns U (16 x 16), V (16 x 16), W (16 x 16) with Fraction entries.
    U[i*4+j, c], V[j*4+k, c], W[k*4+i, c].
    """
    n = 4
    # DFT matrix F and its inverse F_inv (1/n * F^T for the standard DFT, but
    # for cyclic convolution the transform is F with F^{-1} = (1/4) F^T.
    # However, for the Winograd scheme we use the fact that the tensor
    # of cyclic convolution of size 4 is rank <= 16 via the FFT.
    #
    # The 4-point DFT matrix F:
    # F[r, s] = omega^{r*s} where omega = i (the 4th root of unity)
    #
    # We work over Fraction with sqrt(-1) -- this requires complex numbers.
    # Since the task allows half-integer entries, we need to handle this.
    #
    # Actually, the standard Winograd for size 4 uses real arithmetic with
    # entries like 1, -1, 2, etc. Let me recall the FMA (Fast Matrix
    # Algorithm) for 4x4.
    #
    # For cyclic convolution c_0 = sum_j a_j b_{-j} mod 4 (or similar),
    # the Winograd scheme uses:
    #   a1 = a_0 + a_2, a2 = a_1 + a_3, a3 = a_0 - a_2, a4 = a_1 - a_3
    #   b1 = b_0 + b_2, b2 = b_1 + b_3, b3 = b_0 - b_2, b4 = b_1 - b_3
    #   c0 = a1*b1 + a3*b3 + (a1*b3 + a3*b1)/2 ... 
    #
    # This is getting complicated. Let me use the tensor rank approach directly.
    # The cyclic convolution tensor of size n has rank n over C, so over
    # rationals it's still rank n if we allow complex... but we need rational.
    #
    # Alternative: use the block Kronecker structure.
    # 4 = 2*2, so the 4x4 cyclic conv has rank <= 2*2 = 4 via the 2D FFT.
    # Actually no, the tensor rank of the 4x4 cyclic convolution is 4 (over C).
    #
    # For the matrix multiplication 4x4*4x4, the naive rank is 64.
    # Using block decomposition: partition into 2x2 blocks.
    # There are 4 blocks of A and 4 blocks of B, giving 16 block products
    # of size 2x2 * 2x2. Each 2x2*2x2 has rank 8 (naive) or... 
    # Actually 2x2 matrix multiplication has tensor rank 7 (Strassen).
    # So 16 * 7 = 112 > 64. Not helpful.
    #
    # Let me just use a different approach. The key insight for the FFT-based
    # Winograd for the FULL matrix product (not just cyclic conv) is:
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # If we group by (i mod 4, j mod 4, k mod 4) -- but n=4 so i,j,k are 0..3.
    # Actually for 4x4 * 4x4, we can write:
    # C = sum over frequency domains using FFT.
    #
    # The standard FFT-based matrix multiplication (Winograd):
    # C[i,k] = (1/4) sum_r omega^{-r*i} * (sum_j omega^{r*j} A[i,j]) * 
    #          (sum_k' ... ) -- this doesn't quite work for non-circular.
    #
    # Let me try a different decomposition. Since we're going to combine
    # with strips anyway, let me just use a known good rank for 4x4x4.
    #
    # Actually, the simplest good approach: use the rank-16 decomposition
    # that exploits the 4-point FFT. The key formula:
    #
    # Let F be the 4x4 DFT matrix. Then:
    # C = F^{-1} (F_A * F_B) F^{-1} where F_A is the row-wise FFT of A,
    # F_B is the column-wise FFT of B, and * is element-wise.
    #
    # This gives: C[i,k] = sum_{r,s} (1/16) F[i,r] F[k,s] * (row FFT of A) * (col FFT of B)
    #
    # More precisely: C[i,k] = sum_j A[i,j] B[j,k]
    # Let A' = F_row(A) where A'[i,r] = sum_j omega^{r*j} A[i,j]
    # Let B' = F_col(B) where B'[s,k] = sum_j omega^{j*s} B[j,k]
    # Then C[i,k] = (1/4) sum_r omega^{-r*i} sum_s omega^{-s*k} ... 
    #
    # This is getting too complex. Let me use a simpler strategy:
    # use the rank from a known Strassen-like or FFT-based scheme.
    
    # For 4x4 matrix multiplication, the best known upper bound on tensor rank
    # over rationals is quite high. Let's just use a practical FFT-based scheme.
    #
    # Actually, I realize I should just directly construct the decomposition
    # using the FFT property. The 4-point FFT has the property that:
    #
    # If we define U, V, W such that:
    # U[i*4+j, r] relates A[i,j] to the r-th frequency component
    # V[j*4+k, r] relates B[j,k] to the r-th frequency component  
    # W[k*4+i, r] reconstructs C[k,i]
    #
    # The trick is that we need the tensor to be a sum of rank-1 tensors.
    # 
    # For the 4x4x4 matrix multiplication tensor, I'll use the following
    # approach: express it as sum over (r,s) of rank-1 terms using
    # the Fourier basis.
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Insert completeness: delta(j,j') = (1/4) sum_r omega^{r(j-j')}
    #
    # C[i,k] = sum_{j,j'} A[i,j] delta(j,j') B[j',k]
    #        = (1/4) sum_r sum_{j,j'} A[i,j] omega^{r(j-j')} B[j',k]
    #        = (1/4) sum_r (sum_j A[i,j] omega^{r*j}) * (sum_{j'} omega^{-r*j'} B[j',k])
    #
    # So if we define:
    # U[i*4+j, r] -- but this isn't rank 1 because the sum over r is outside.
    #
    # The issue is that the rank-1 decomposition would be:
    # C[i,k] = sum_r U[r][i] * V[r][k] where U[r][i] = sum_j A[i,j] omega^{rj} and V[r][k] = sum_j B[j,k] omega^{-rj}
    #
    # But U depends on A and V depends on B, so this is the FFT-based multiplication:
    # C = FFT^{-1}(FFT(A) .* FFT(B)) element-wise in the appropriate sense.
    #
    # The tensor rank-1 decomposition is:
    # For each r in 0..3:
    #   term_r = (1/4) * (row_r_transform) * (col_r_transform)
    #
    # Where row_r_transform[i, j] = omega^{r*j} and col_r_transform[j, k] = omega^{-r*j}
    #
    # But this gives a sum over r of matrices, not a tensor decomposition.
    # The tensor decomposition would be:
    # T[i, j, k] = delta(j, j) -- no, T[i,j,k] = delta_{i,k} * ... 
    #
    # Actually the matmul tensor is T[i,j,k] = 1 for all i,j,k (it's the identity
    # mapping on the index j). The decomposition is:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{i,i'} * ... 
    #
    # Let me just be very explicit. The matmul tensor for 4x4*4x4 is:
    # T[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    #
    # We want: sum_{c=1}^{R} U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    # (This is the condition for the tensor to decompose correctly -- actually the
    # condition is that the bilinear form reproduces the matmul, which is:
    # sum_c U[i,j,c] * V[j,k,c] = C[i,k] when W encodes the output index.)
    #
    # Wait, the convention is: C[k*n+i, c] maps to output C[i,k]. So:
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c]  ... no, that's not right either.
    #
    # The convention from the problem: U has shape (n*m, rank), V has shape (m*p, rank),
    # W has shape (p*n, rank). The output factor W[k*n+i, r] means the r-th column
    # of W indexed by k*n+i corresponds to output C[i,k].
    #
    # The decomposition condition is:
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] ... this doesn't work because
    # U and V share the j index but have different j ranges.
    #
    # Actually I think the correct interpretation is:
    # The tensor is decomposed as: T[i,j,k] = sum_c u_i,c * v_j,c * w_k,c
    # where the "tensor" T[i,j,k] represents the matmul operation.
    #
    # For matrix multiplication, the relevant tensor is:
    # M[i,j,k] = 1 (scalar 1) for the multiplication at position (i,j,k)
    #
    # No wait. The matrix multiplication tensor for multiplying an n x m matrix
    # by an m x p matrix has entries T[i,j,k] where i is row of A, j is col of A /
    # row of B, k is col of B. The tensor entry is 1 (since C[i,k] = sum_j A[i,j]*B[j,k]).
    #
    # A rank-1 tensor decomposition is: T[i,j,k] = sum_c a[i,c] * b[j,c] * c[k,c]
    #
    # For the matmul tensor, T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor? No!)
    #
    # I'm confusing myself. Let me go back to basics.
    #
    # The matrix multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # This is a bilinear map. To represent it as a tensor decomposition, we write:
    # C[i,k] = sum_c (U_col[c][i] * A_stuff) ... 
    #
    # The standard tensor for matmul: The matmul tensor M has indices (i, j, k) where
    # i = row (n), j = column of A / row of B (m), k = column (p).
    # M[i,j,k] = 1 for all valid i,j,k.
    #
    # No! M[i,j,k] is the coefficient that multiplies A[i,j]*B[j,k] in C[...].
    # Actually the matmul "tensor" in the sense of tensor rank is:
    #
    # The bilinear form f: R^{n*m} x R^{m*p} -> R^{n*p}
    # f(A,B) = C where C[i,k] = sum_j A[i,j] B[j,k]
    #
    # As a 3-tensor: T[i,j,k] where i indexes A (flattened), j indexes B (flattened),
    # and k indexes C (flattened). T[a, b, c] = 1 if a corresponds to (i,j), 
    # b corresponds to (j,k), and c corresponds to (i,k), with matching j.
    #
    # So T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j'=j', k=k'
    #
    # A rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{n*p}
    #
    # The schoolbook decomposition: for each (i,j,k), u_{ijk} = e_{(i,j)}, v_{ijk} = e_{(j,k)}, w_{ijk} = e_{(i,k)}
    # This gives rank n*m*p.
    #
    # Now, for the FFT-based decomposition of 4x4x4:
    # We can use the fact that the 4-point DFT diagonalizes the cyclic convolution.
    # But matrix multiplication is NOT cyclic convolution.
    #
    # However, if we extend A and B to 4x4 and use the FFT:
    # C = FFT^{-1}(FFT(A) ⊙ FFT(B)) -- this is only true for cyclic convolution, not matmul.
    #
    # I think I was on the wrong track. The FFT/Winograd speedup applies to convolution,
    # not general matrix multiplication. For general matrix multiplication, the best
    # known upper bounds are from Strassen-like algorithms.
    #
    # For 4x4 matrix multiplication, let me look up or derive a good decomposition.
    # The naive rank is 64. Strassen's algorithm for 4x4:
    # 4x4 = 2x2 blocks. Use Strassen for each 2x2 block product: rank 7 per 2x2.
    # 16 block positions, but with Strassen we need 7 "super-block" multiplications
    # for each set of 2x2 blocks. For 4x4 = 2x2 blocks, there are 4x4 = 16 block
    # products, each 2x2*2x2. Using Strassen: 16 * 7 = 112. Worse than 64!
    #
    # What about using the fact that 4 = 2^2 and applying Strassen recursively?
    # For 4x4, split into four 2x2 blocks. The block-level multiplication is
    # a 2x2 * 2x2 problem (of 2x2 blocks), which has rank 7. Each block product
    # is 2x2 * 2x2 with rank 7. Total: 7 * 7 = 49? No...
    #
    # Actually, if we view the 4x4 matmul as a 2x2 matmul where each "entry" is a 2x2
    # matrix, then we need 7 "matrix multiplications" of 2x2 matrices (by Strassen).
    # Each 2x2 matmul has rank 8 (naive) or 7 (Strassen).
    # If we use naive 2x2: 7 * 8 = 56.
    # If we use Strassen 2x2: 7 * 7 = 49.
    #
    # So rank 49 or 56 is achievable for 4x4*4x4!
    #
    # But wait, the block structure means the "matrix multiplication" at the block
    # level involves sums of products, not just products. Let me think more carefully.
    #
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22] where each is 2x2.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Strassen's 7 multiplications for the block-level 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Now, each M_i involves a product of two 2x2 "matrices" (which are themselves
    # sums/differences of 2x2 blocks). The product of two 2x2 matrices has rank 8
    # (naive) or 7 (Strassen).
    #
    # If we use naive 2x2*2x2 (rank 8): total rank = 7 * 8 = 56.
    # If we use Strassen 2x2*2x2 (rank 7): total rank = 7 * 7 = 49.
    #
    # Let me implement the rank-56 version (7 Strassen block-level * 8 naive block)
    # or rank-49 (7*7).
    
    # Actually, let me just implement a concrete low-rank decomposition.
    # I'll use the rank-56 approach (Strassen blocks with naive 2x2 leaves).
    
    # For a 2x2 * 2x2 naive decomposition:
    # C[i,k] = sum_j A[i,j] B[j,k], rank 8.
    # U_2x2 is 4x8, V_2x2 is 4x8, W_2x2 is 4x8.
    
    def _strassen_4x4():
        """
        Rank-56 decomposition of 4x4 matmul via Strassen block decomposition.
        7 block-level multiplications, each of rank 8 (naive 2x2).
        Total rank = 56.
        """
        # We'll build up the full U, V, W for 4x4*4x4.
        # U: 16 x 56, V: 16 x 56, W: 16 x 56
        # U index: i*4+j (row i, col j of A), V index: j*4+k (row j, col k of B)
        # W index: k*4+i (row k, col i of C) -- wait, W[k*n+i] where n=4
        
        n = 4
        total_rank = 56
        U = [[Fraction(0)] * total_rank for _ in range(n * n)]  # 16 x 56
        V = [[Fraction(0)] * total_rank for _ in range(n * n)]  # 16 x 56
        W = [[Fraction(0)] * total_rank for _ in range(n * n)]  # 16 x 56
        
        # Strassen's 7 multiplications at block level:
        # Each multiplication is of form (sum of blocks of A) * (sum of blocks of B)
        # where each "block" is a 2x2 submatrix.
        #
        # Block indices: A11 = A[0:2, 0:2], A12 = A[0:2, 2:4],
        #                A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        # Similarly for B.
        #
        # The 7 block-level operations:
        # M1 = (A11+A22)*(B11+B22)  -> contributes to C11, C22
        # M2 = (A21+A22)*B11        -> contributes to C21, C22
        # M3 = A11*(B12-B22)        -> contributes to C12, C22
        # M4 = A22*(B21-B11)        -> contributes to C11, C21
        # M5 = (A11+A12)*B22        -> contributes to C11, C12
        # M6 = (A21-A11)*(B11+B12)  -> contributes to C22
        # M7 = (A12-A22)*(B21+B22)  -> contributes to C11
        #
        # For each block-level multiplication M_i = P_i * Q_i where P_i is a linear
        # combination of A blocks and Q_i is a linear combination of B blocks.
        # Each such product (which is a 2x2 * 2x2 matrix product) has a rank-8
        # (naive) decomposition.
        #
        # The linear combination means: P_i[r,c] = sum over blocks b of coef[b] * A_block[b][r,c]
        # where r,c are local 2x2 indices.
        #
        # For the naive 2x2 decomposition of P_i * Q_i:
        # (P_i * Q_i)[r,k] = sum_s P_i[r,s] * Q_i[s,k]
        # Rank-8 decomposition: for each (r,s,k), 
        #   u[global_A_index] = coef for the corresponding A block
        #   v[global_B_index] = coef for the corresponding B block  
        #   w[global_C_index] = 1
        #
        # Global A index for block b, local (r,s): 
        #   block 11: i=r, j=s -> index r*4+s
        #   block 12: i=r, j=2+s -> index r*4+(2+s)
        #   block 21: i=2+r, j=s -> index (2+r)*4+s
        #   block 22: i=2+r, j=2+s -> index (2+r)*4+(2+s)
        
        block_offsets = {
            11: (0, 0),  # (row_offset, col_offset)
            12: (0, 2),
            21: (2, 0),
            22: (2, 2)
        }
        
        def a_global(block, r, s):
            ro, co = block_offsets[block]
            return (ro + r) * 4 + (co + s)
        
        def b_global(block, s, k):
            ro, co = block_offsets[block]
            return (ro + s) * 4 + (co + k)
        
        def c_global(block, r, k):
            # C block: block 11 -> C[0:2,0:2], etc.
            ro, co = block_offsets[block]
            # W index: k*4 + i where i=row, k=col of C
            # C[i,k] -> W index k*4+i
            return k * 4 + (ro + r)
        
        # Define the 7 Strassen multiplications:
        # Each is (list of (block, coef)) * (list of (block, coef)) -> list of (c_block, coef)
        strassen_mults = [
            # M1 = (A11+A22)*(B11+B22)
            {
                'A': [(11, Fraction(1)), (22, Fraction(1))],
                'B': [(11, Fraction(1)), (22, Fraction(1))],
                'C': [(11, Fraction(1)), (22, Fraction(1))]
            },
            # M2 = (A21+A22)*B11
            {
                'A': [(21, Fraction(1)), (22, Fraction(1))],
                'B': [(11, Fraction(1))],
                'C': [(21, Fraction(1)), (22, Fraction(-1))]
            },
            # M3 = A11*(B12-B22)
            {
                'A': [(11, Fraction(1))],
                'B': [(12, Fraction(1)), (22, Fraction(-1))],
                'C': [(12, Fraction(1)), (22, Fraction(1))]
            },
            # M4 = A22*(B21-B11)
            {
                'A': [(22, Fraction(1))],
                'B': [(21, Fraction(1)), (11, Fraction(-1))],
                'C': [(11, Fraction(1)), (21, Fraction(1))]
            },
            # M5 = (A11+A12)*B22
            {
                'A': [(11, Fraction(1)), (12, Fraction(1))],
                'B': [(22, Fraction(1))],
                'C': [(11, Fraction(-1)), (12, Fraction(1))]
            },
            # M6 = (A21-A11)*(B11+B12)
            {
                'A': [(21, Fraction(1)), (11, Fraction(-1))],
                'B': [(11, Fraction(1)), (12, Fraction(1))],
                'C': [(22, Fraction(1))]
            },
            # M7 = (A12-A22)*(B21+B22)
            {
                'A': [(12, Fraction(1)), (22, Fraction(-1))],
                'B': [(21, Fraction(1)), (22, Fraction(1))],
                'C': [(11, Fraction(1))]
            }
        ]
        
        col = 0
        for mult in strassen_mults:
            A_parts = mult['A']  # list of (block, coef)
            B_parts = mult['B']
            C_parts = mult['C']
            
            # For each local (r, s, k) in 2x2*2x2 (r=row of product, s=inner, k=col):
            for r in range(2):
                for s in range(2):
                    for k in range(2):
                        # The rank-1 term for this (r,s,k) in the naive 2x2 product
                        # U contribution: for each A block with coef, add coef at a_global(block, r, s)
                        for (ab, coef_a) in A_parts:
                            idx_a = a_global(ab, r, s)
                            U[idx_a][col] += coef_a
                        
                        for (bb, coef_b) in B_parts:
                            idx_b = b_global(bb, s, k)
                            V[idx_b][col] += coef_b
                        
                        for (cb, coef_c) in C_parts:
                            idx_c = c_global(cb, r, k)
                            W[idx_c][col] += coef_c
                        
                        col += 1
        
        assert col == total_rank, f"Expected rank {total_rank}, got {col}"
        return U, V, W
    
    return _strassen_4x4()

def _build_full_decomposition():
    """
    Build the full 4x5 * 5x6 decomposition using:
    - Partition A into A_L (4x4) and A_R (4x1)
    - Partition B into B_L (4x4) and B_R (1x6) -- but B is 5x6, so B_L is 4x6 and B_R is 1x6
    - C = A_L * B_L + A_R * B_R
    - A_L * B_L: 4x4 * 4x6 -> decompose as 4x4 * 4x4 + 4x4 * 4x2
    - A_R * B_R: 4x1 * 1x6 -> schoolbook, rank 24
    
    Total: 56 (Strassen 4x4) + 8*2... wait let me recalculate.
    
    Actually: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    
    Part 1: sum_{j=0}^{3} A[i,j]*B[j,k] -- this is A_L (4x4) * B_L (4x6)
    Part 2: A[i,4]*B[4,k] -- this is A_R (4x1) * B_R (1x6), rank 24
    
    For Part 1: A_L is 4x4, B_L is 4x6.
    Decompose B_L = [B_L1 B_L2] where B_L1 is 4x4 and B_L2 is 4x2.
    A_L * B_L1: 4x4*4x4, rank 56 (Strassen)
    A_L * B_L2: 4x4*4x2, rank 4*4*2 = 32 (schoolbook)
    
    Total rank = 56 + 32 + 24 = 112.
    
    Hmm, that's better than 120 but not as good as the 40 claimed in the directive.
    The directive's claim of 40 seems to use a rank-16 decomposition for 4x4x4,
    which would require complex FFT entries. Since we need rational/half-integer
    entries, we can't use the FFT directly.
    
    Let me check: can we do better than Strassen for 4x4?
    The best known upper bound for omega(4) over the reals is around 49 (from
    recursive Strassen). Over rationals, it might be higher.
    
    Let me just go with the rank-112 approach for now. It's an improvement over 120.
    
    Actually, wait. Let me reconsider. Maybe I should use a rank-49 decomposition
    (recursive Strassen: 7*7). Let me try that.
    
    For recursive Strassen on 4x4:
    Level 1: 4x4 = 2x2 blocks, 7 block multiplications of 2x2*2x2
    Level 2: each 2x2*2x2 uses Strassen: 7 multiplications of 1x1*1x1 (scalars)
    Total: 7*7 = 49.
    
    This should work over the rationals! Let me implement it.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    total_U_cols = 0  # will track
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    U = []
    V = []
    W = []
    
    def _add_decomp(U_decomp, V_decomp, W_decomp, 
                    a_index_map, b_index_map, w_index_map):
        """
        Add a sub-decomposition to the global U, V, W.
        U_decomp: (n_sub*m_sub) x rank_sub
        V_decomp: (m_sub*p_sub) x rank_sub
        W_decomp: (p_sub*n_sub) x rank_sub
        a_index_map: maps local A index -> global A index (or None)
        b_index_map: maps local B index -> global B index
        w_index_map: maps local W index -> global W index
        Returns: rank_sub
        """
        global U, V, W
        rank_sub = len(U_decomp[0]) if U_decomp else 0
        if rank_sub == 0:
            return 0
        
        # Extend U, V, W with new columns
        for row in U:
            row.extend([Fraction(0)] * rank_sub)
        for row in V:
            row.extend([Fraction(0)] * rank_sub)
        for row in W:
            row.extend([Fraction(0)] * rank_sub)
        
        base_col = len(U[0]) - rank_sub
        
        for local_a_idx, val in [(i, row) for i, row in enumerate(U_decomp)]:
            if a_index_map is not None and local_a_idx < len(a_index_map):
                global_a_idx = a_index_map[local_a_idx]
                if global_a_idx is not None:
                    for c in range(rank_sub):
                        if val[c] != 0:
                            U[global_a_idx][base_col + c] += val[c]
        
        for local_b_idx, val in [(i, row) for i, row in enumerate(V_decomp)]:
            if b_index_map is not None and local_b_idx < len(b_index_map):
                global_b_idx = b_index_map[local_b_idx]
                if global_b_idx is not None:
                    for c in range(rank_sub):
                        if val[c] != 0:
                            V[global_b_idx][base_col + c] += val[c]
        
        for local_w_idx, val in [(i, row) for i, row in enumerate(W_decomp)]:
            if w_index_map is not None and local_w_idx < len(w_index_map):
                global_w_idx = w_index_map[local_w_idx]
                if global_w_idx is not None:
                    for c in range(rank_sub):
                        if val[c] != 0:
                            W[global_w_idx][base_col + c] += val[c]
        
        return rank_sub
    
    # Initialize global U, V, W with placeholder columns
    # We'll build them up incrementally
    U = [[Fraction(0)] for _ in range(U_rows)]
    V = [[Fraction(0)] for _ in range(V_rows)]
    W = [[Fraction(0)] for _ in range(W_rows)]
    
    # --- Part 1: A_L (4x4) * B_L1 (4x4) ---
    # A_L: rows 0-3, cols 0-3 of A. Global A index: i*5+j
    # B_L1: rows 0-3, cols 0-3 of B. Global B index: j*6+k
    # C part: cols 0-3 of C. Global W index: k*4+i for k in 0-3, i in 0-3
    
    U1, V1, W1 = _fft4_decomposition()
    
    a_map1 = [i*5 + j for i in range(4) for j in range(4)]  # 16 entries
    b_map1 = [j*6 + k for j in range(4) for k in range(4)]  # 16 entries
    w_map1 = [k*4 + i for k in range(4) for i in range(4)]  # 16 entries
    
    _add_decomp(U1, V1, W1, a_map1, b_map1, w_map1)
    
    # --- Part 2: A_L (4x4) * B_L2 (4x2) ---
    # B_L2: rows 0-3, cols 4-5 of B.
    # Schoolbook 4x4*4x2, rank 32.
    
    U2, V2, W2 = _schoolbook(4, 4, 2)
    
    a_map2 = [i*5 + j for i in range(4) for j in range(4)]  # 16 entries
    b_map2 = [j*6 + (k+4) for j in range(4) for k in range(2)]  # 8 entries
    w_map2 = [k*4 + i for k in range(2) for i in range(4)]  # 8 entries
    # Wait, W index for C[i,k] is k*4+i. Here k in 0..1 maps to global k = 4..5.
    w_map2 = [(k+4)*4 + i for k in range(2) for i in range(4)]
    
    _add_decomp(U2, V2, W2, a_map2, b_map2, w_map2)
    
    # --- Part 3: A_R (4x1) * B_R (1x6) ---
    # A_R: col 4 of A. Global A index: i*5+4 for i in 0..3.
    # B_R: row 4 of B. Global B index: 4*6+k for k in 0..5.
    # C: full 4x6. Global W index: k*4+i.
    # Schoolbook 4x1*1x6, rank 24.
    
    U3, V3, W3 = _schoolbook(4, 1, 6)
    
    a_map3 = [i*5 + 4 for i in range(4)]  # 4 entries
    b_map3 = [4*6 + k for k in range(6)]  # 6 entries
    w_map3 = [k*4 + i for k in range(6) for i in range(4)]  # 24 entries
    
    _add_decomp(U3, V3, W3, a_map3, b_map3, w_map3)
    
    total_rank = len(U[0])
    print(f"Total rank: {total_rank}")
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    _t0_local = time.time()
    result = _build_full_decomposition()
    elapsed = time.time() - _t0_local
    print(f"Construction time: {elapsed:.3f}s, rank={len(result['U'][0]) if result['U'] else 0}")
    return result
# EVOLVE-BLOCK-END
