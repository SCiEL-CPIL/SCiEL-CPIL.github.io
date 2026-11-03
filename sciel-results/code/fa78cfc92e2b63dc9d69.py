<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
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
=======
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 Strassen matrix multiplication.
    Each term is (u_entries, v_entries, w_entries) where:
    - u_entries: dict mapping (i,j) -> coefficient for A[i,j]
    - v_entries: dict mapping (j,k) -> coefficient for B[j,k]
    - w_entries: dict mapping (k,i) -> coefficient for C[i,k]
    
    The tensor decomposition: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    where the sum is over j for each r.
    
    Actually, for a rank-1 term, U is a vector in R^{n*m}, V in R^{m*p}, W in R^{p*n}.
    For 2x2x2: n=m=p=2, so U in R^4, V in R^4, W in R^4.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    
    The 7 Strassen terms:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00 and A11 have coeff 1
    # V: B00 and B11 have coeff 1
    # W: C00 gets M1, C11 gets M1
    # For the rank-1 term, we need to decompose this into the tensor form.
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # But in Strassen, this is ONE multiplication, so it's ONE rank-1 term in the 
    # "high-level" sense, but at the tensor level, we need to represent it properly.
    
    # Actually, I'm confusing myself. Let me reconsider.
    
    # The matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # A rank-1 decomposition is a set of (u, v, w) where:
    # u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    # and T = sum_r outer(u_r, v_r, w_r)
    
    # For the schoolbook 2x2x2: rank 8, with terms for each (i,j,k):
    # u has 1 at position i*2+j, v has 1 at position j*2+k, w has 1 at position k*2+i
    
    # For Strassen 2x2x2: rank 7. The 7 terms are:
    # Term 1: M1 = (A00+A11)(B00+B11)
    #   u: A00=1, A11=1 -> positions 0*2+0=0 and 1*2+1=3
    #   v: B00=1, B11=1 -> positions 0*2+0=0 and 1*2+1=3
    #   w: C00=1, C11=1 -> positions 0*2+0=0 and 1*2+1=3
    #   But wait, this single term doesn't directly give the right C entries.
    
    # I think the issue is that Strassen's algorithm combines multiple rank-1 terms.
    # Let me look up the explicit rank-7 decomposition.
    
    # The 7 rank-1 terms for 2x2x2 Strassen (in tensor form):
    # Each term (u, v, w) contributes to C[i,k] via:
    # contribution = sum_j u[i*2+j] * v[j*2+k] * w[k*2+i]
    
    # Term 1: u = [1,0,0,1], v = [1,0,0,1], w = [1,0,0,1]
    #   C00 += u[0]*v[0]*w[0] + u[1]*v[1]*w[0] = 1*1*1 + 0*0*1 = 1 (from A00*B00)
    #   C00 += u[0]*v[0]*w[0] ... wait, I need to be more careful.
    
    # For a given term (u, v, w):
    # C[i,k] += sum_j u[i*2+j] * v[j*2+k] * w[k*2+i]
    
    # Term 1: u = [1,0,0,1] (A00=1, A11=1), v = [1,0,0,1] (B00=1, B11=1), w = [1,0,0,1] (C00=1, C11=1)
    # C[0,0] += u[0*2+0]*v[0*2+0]*w[0*2+0] + u[0*2+1]*v[1*2+0]*w[0*2+0]
    #          = u[0]*v[0]*w[0] + u[1]*v[2]*w[0]
    #          = 1*1*1 + 0*0*1 = 1
    # C[0,1] += u[0*2+0]*v[0*2+1]*w[1*2+0] + u[0*2+1]*v[1*2+1]*w[1*2+0]
    #          = u[0]*v[1]*w[2] + u[1]*v[3]*w[2]
    #          = 1*0*1 + 0*1*1 = 0
    # C[1,0] += u[1*2+0]*v[0*2+0]*w[0*2+1] + u[1*2+1]*v[1*2+0]*w[0*2+1]
    #          = u[2]*v[0]*w[1] + u[3]*v[2]*w[1]
    #          = 0*1*0 + 1*0*0 = 0
    # C[1,1] += u[1*2+0]*v[0*2+1]*w[1*2+1] + u[1*2+1]*v[1*2+1]*w[1*2+1]
    #          = u[2]*v[1]*w[3] + u[3]*v[3]*w[3]
    #          = 0*0*1 + 1*1*1 = 1
    # So term 1 contributes 1 to C00 and 1 to C11.
    
    # This is getting very involved. Let me just implement the known 7-term decomposition.
    
    # From the literature, the 7 rank-1 terms for 2x2x2 Strassen are:
    # (using 0-indexed, positions in U: A00=0, A01=1, A10=2, A11=3)
    # (positions in V: B00=0, B01=1, B10=2, B11=3)
    # (positions in W: C00=0, C01=1, C10=2, C11=3)
    
    terms = [
        # M1 = (A00+A11)(B00+B11)
        # u: A00=1, A11=1
        # v: B00=1, B11=1
        # w: C00=1, C11=1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        
        # M2 = (A10+A11)B00
        # u: A10=1, A11=1
        # v: B00=1
        # w: C10=1
        ({2:1, 3:1}, {0:1}, {2:1}),
        
        # M3 = A00(B01-B11)
        # u: A00=1
        # v: B01=1, B11=-1
        # w: C01=1
        ({0:1}, {1:1, 3:-1}, {1:1}),
        
        # M4 = A11(B10-B00)
        # u: A11=1
        # v: B10=1, B00=-1
        # w: C10=1, C00=1
        ({3:1}, {2:1, 0:-1}, {2:1, 0:1}),
        
        # M5 = (A00+A01)B11
        # u: A00=1, A01=1
        # v: B11=1
        # w: C01=1, C00=-1
        ({0:1, 1:1}, {3:1}, {1:1, 0:-1}),
        
        # M6 = (A10-A00)(B00+B01)
        # u: A10=1, A00=-1
        # v: B00=1, B01=1
        # w: C11=1
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        
        # M7 = (A01-A11)(B10+B11)
        # u: A01=1, A11=-1
        # v: B10=1, B11=1
        # w: C00=1
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    return terms

def _build_4x4x4_strassen():
    """
    Build the 4x4x4 matrix multiplication tensor using recursive Strassen.
    Split 4x4 into four 2x2 blocks. Use 7 Strassen combinations at the block level,
    each involving 2x2x2 multiplications (which use the 7-term Strassen decomposition).
    
    This gives rank 7*7 = 49.
    """
    # The 4x4 matrix A is split into 2x2 blocks:
    # A = [[A00, A01],
    #      [A10, A11]]
    # where each Aij is 2x2.
    
    # Similarly for B and C.
    
    # The 7 Strassen block-level terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block-level term, we need to compute the product of two 2x2 matrices,
    # which itself requires 7 rank-1 terms (Strassen) or 4 (schoolbook).
    
    # Let's use schoolbook for the inner 2x2x2 to keep it simple: rank 4 per inner product.
    # Total rank = 7 * 4 = 28.
    
    # Actually, let me use Strassen for the inner too: 7*7 = 49. But 28 is better.
    # Let me use schoolbook for the inner 2x2x2.
    
    # For a 2x2x2 schoolbook, the 4 rank-1 terms are:
    # (i,j,k) for i,j,k in {0,1}
    
    # Let me build the 4x4x4 tensor.
    
    # U has shape (4*4, rank) = (16, rank)
    # V has shape (4*4, rank) = (16, rank)
    # W has shape (4*4, rank) = (16, rank)
    
    # For the block structure:
    # A[i,j] where i,j in 0..3. Block A[a,b] where a,b in {0,1}, and within block, indices (i',j') in {0,1}.
    # Global index: i = 2*a + i', j = 2*b + j'
    # So A[i,j] = A_block[a,b][i',j']
    
    # U index: i*4 + j = (2*a+i')*4 + (2*b+j') = 8*a + 4*i' + 4*b + 2*b + j' = 8*a + 4*b + 4*i' + 2*b + j'
    # Hmm, let me just use: U_index = i*4 + j where i = 2*a + i', j = 2*b + j'
    
    # For a block-level Strassen term, say M1 = (A00+A11)(B00+B11):
    # This involves A blocks A00 and A11, and B blocks B00 and B11.
    # A00 is at global positions (0,0), (0,1), (1,0), (1,1) -> U indices 0,1,4,5
    # A11 is at global positions (2,2), (2,3), (3,2), (3,3) -> U indices 10,11,14,15
    # B00 is at global positions (0,0), (0,1), (1,0), (1,1) -> V indices 0,1,4,5
    # B11 is at global positions (2,2), (2,3), (3,2), (3,3) -> V indices 10,11,14,15
    
    # The product (A00+A11)(B00+B11) is a 2x2 matrix. To compute it, we use 4 schoolbook terms.
    # Each schoolbook term for the inner 2x2x2:
    # For inner indices (i',j',k') in {0,1}^3:
    #   The contribution to C_block[0,0] (which is C00) is:
    #     A_block[i',j'] * B_block[j',k']
    
    # This is getting very complex. Let me just implement it step by step.
    
    terms = []
    
    # Block-level Strassen definitions:
    # Each entry: (name, A_blocks, B_blocks, C_blocks_with_signs)
    # A_blocks: dict {block_idx: coeff}
    # B_blocks: dict {block_idx: coeff}
    # C_blocks: dict {block_idx: coeff}
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    block_terms = [
        # M1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2
        ({2:1, 3:1}, {0:1}, {2:1}),
        # M3
        ({0:1}, {1:1, 3:-1}, {1:1}),
        # M4
        ({3:1}, {2:1, 0:-1}, {2:1, 0:1}),
        # M5
        ({0:1, 1:1}, {3:1}, {1:1, 0:-1}),
        # M6
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # M7
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    # For each block-level term, we need to compute the 2x2x2 matrix product.
    # Using schoolbook: 4 terms.
    
    # For a block-level term with A_blocks, B_blocks, C_blocks:
    # The contribution to C_block[c_block] is:
    #   (sum_{a_block} A_blocks[a_block] * A_block[a_block]) * (sum_{b_block} B_blocks[b_block] * B_block[b_block])
    #   * C_blocks[c_block]
    
    # But this is a product of two 2x2 matrices, so we need to expand it.
    
    # Let me think differently. For a block-level term like M1 = (A00+A11)(B00+B11):
    # The result is a 2x2 matrix R where:
    # R[i',k'] = sum_{j'} (A00[i',j'] + A11[i',j']) * (B00[j',k'] + B11[j',k'])
    
    # This expands to 4 terms (schoolbook for 2x2x2):
    # For each (i',j',k'):
    #   R[i',k'] += (A00[i',j'] + A11[i',j']) * (B00[j',k'] + B11[j',k'])
    
    # But in the tensor decomposition, each (i',j',k') gives a rank-1 term.
    
    # For the global tensor:
    # A00[i',j'] is at global position (2*0+i', 2*0+j') = (i', j') -> U index i'*4+j'
    # A11[i',j'] is at global position (2*1+i', 2*1+j') = (2+i', 2+j') -> U index (2+i')*4+(2+j')
    # B00[j',k'] is at global position (j', k') -> V index j'*4+k'
    # B11[j',k'] is at global position (2+j', 2+k') -> V index (2+j')*4+(2+k')
    # C00[i',k'] is at global position (i', k') -> W index k'*4+i'
    # C11[i',k'] is at global position (2+i', 2+k') -> W index (2+k')*4+(2+i')
    
    # So for M1, for each (i',j',k') in {0,1}^3:
    #   U has: 1 at i'*4+j' (for A00) and 1 at (2+i')*4+(2+j') (for A11)
    #   V has: 1 at j'*4+k' (for B00) and 1 at (2+j')*4+(2+k') (for B11)
    #   W has: 1 at k'*4+i' (for C00) and 1 at (2+k')*4+(2+i') (for C11)
    
    # This gives 4 rank-1 terms for M1.
    
    # Similarly for other block-level terms.
    
    # Let me implement this.
    
    rank = 0
    U = {}
    V = {}
    W = {}
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal rank
        col = rank
        for idx, val in u_dict.items():
            U.setdefault(idx, [0]*49)
            U[idx][col] = val
        for idx, val in v_dict.items():
            V.setdefault(idx, [0]*49)
            V[idx][col] = val
        for idx, val in w_dict.items():
            W.setdefault(idx, [0]*49)
            W[idx][col] = val
        rank += 1
    
    # For each block-level term, add 4 schoolbook terms for the inner 2x2x2.
    
    for a_blocks, b_blocks, c_blocks in block_terms:
        # a_blocks: dict {block: coeff}
        # b_blocks: dict {block: coeff}
        # c_blocks: dict {block: coeff}
        
        # For each (i', j', k') in {0,1}^3:
        for ip in range(2):
            for jp in range(2):
                for kp in range(2):
                    u_dict = {}
                    v_dict = {}
                    w_dict = {}
                    
                    for a_block, a_coeff in a_blocks.items():
                        # A_block[a_block][ip, jp] is at global position:
                        # a_block = (a_row, a_col) where a_block 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                        a_row, a_col = divmod(a_block, 2)
                        gi = 2*a_row + ip
                        gj = 2*a_col + jp
                        u_idx = gi*4 + gj
                        u_dict[u_idx] = u_dict.get(u_idx, 0) + a_coeff
                    
                    for b_block, b_coeff in b_blocks.items():
                        b_row, b_col = divmod(b_block, 2)
                        gj = 2*b_row + jp
                        gk = 2*b_col + kp
                        v_idx = gj*4 + gk
                        v_dict[v_idx] = v_dict.get(v_idx, 0) + b_coeff
                    
                    for c_block, c_coeff in c_blocks.items():
                        c_row, c_col = divmod(c_block, 2)
                        gi = 2*c_row + ip
                        gk = 2*c_col + kp
                        w_idx = gk*4 + gi
                        w_dict[w_idx] = w_dict.get(w_idx, 0) + c_coeff
                    
                    add_term(u_dict, v_dict, w_dict)
    
    # Convert to lists
    U_list = [[U.get(i, [0]*49)[r] for r in range(49)] for i in range(16)]
    V_list = [[V.get(i, [0]*49)[r] for r in range(49)] for i in range(16)]
    W_list = [[W.get(i, [0]*49)[r] for r in range(49)] for i in range(16)]
    
    return U_list, V_list, W_list, 49

def construct():
    n, m, p = DIMENSIONS
    # For 4x4x9, we can split the 9 columns into 4+4+1.
    # Compute A*B1 (4x4 x 4x4) using Strassen (rank 49)
    # Compute A*B2 (4x4 x 4x4) using Strassen (rank 49)
    # Compute A*B3 (4x4 x 4x1) using schoolbook (rank 4)
    # Total rank = 49 + 49 + 4 = 102
    
    # But wait, the Strassen 4x4x4 gives rank 49, which is worse than schoolbook 16.
    # So this approach is not helpful.
    
    # Let me reconsider. The schoolbook 4x4x4 has rank 16.
    # Strassen 2x2x2 has rank 7.
    # For 4x4x4, if I use Strassen at the block level with schoolbook inner, I get 7*4 = 28.
    # That's worse than 16.
    
    # So Strassen doesn't help for 4x4x4. It only helps for larger matrices.
    
    # For 4x4x9, the schoolbook rank is 144.
    # Can I do better?
    
    # One approach: use the fact that 9 = 4+4+1, and use Strassen for the 4x4x4 parts.
    # But Strassen 4x4x4 has rank 28 (with schoolbook inner) or 49 (with Strassen inner), both worse than 16.
    
    # So the best I can do with this approach is 16+16+4 = 36, which is much better than 144.
    
    # Let me implement the 4x4x4 schoolbook and 4x4x1 schoolbook.
    
    rank = 16 + 16 + 4  # 36
    U = [[0] * rank for _ in range(n * m)]  # (16, 36)
    V = [[0] * rank for _ in range(m * p)]  # (36, 36)
    W = [[0] * rank for _ in range(p * n)]  # (36, 36)
    
    col = 0
    
    # First 4x4x4 block: columns 0-3 of B
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Second 4x4x4 block: columns 4-7 of B
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Third 4x4x1 block: column 8 of B
    for i in range(n):
        for j in range(m):
            k = 8
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

Wait, I realize the issue. The Strassen approach for 4×4×4 doesn't improve over schoolbook (rank 16 vs 28 or 49). So the best I can do is split 9 into 4+4+1 and use schoolbook for each part, giving rank 16+16+4 = 36.

But actually, I can do even better. Let me think about whether there's a way to share terms between the blocks.

Actually, the above implementation with rank 36 is already a significant improvement over 144. Let me verify it's correct and implement it properly.

