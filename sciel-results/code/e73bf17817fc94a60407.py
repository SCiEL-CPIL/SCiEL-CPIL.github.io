"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _strassen_2x2x2_factors():
    """
    Return U, V, W factor matrices for 2x2x2 matrix multiplication using Strassen's rank-7 algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indices: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00 + A11
    V[0, 0] = 1; V[3, 0] = 1  # B00 + B11
    W[0, 0] = 1; W[3, 0] = 1  # C00 + C11
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10 + A11
    V[0, 1] = 1               # B00
    W[2, 1] = 1; W[3, 1] = -1  # C10 - C11
    
    # M3: A00(B01-B11)
    U[0, 2] = 1               # A00
    V[1, 2] = 1; V[3, 2] = -1  # B01 - B11
    W[1, 2] = 1; W[3, 2] = 1   # C01 + C11
    
    # M4: A11(B10-B00)
    U[3, 3] = 1               # A11
    V[2, 3] = 1; V[0, 3] = -1  # B10 - B00
    W[0, 3] = 1; W[2, 3] = 1   # C00 + C10
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00 + A01
    V[3, 4] = 1               # B11
    W[0, 4] = -1; W[1, 4] = 1  # -C00 + C01
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1  # A10 - A00
    V[0, 5] = 1; V[1, 5] = 1   # B00 + B01
    W[3, 5] = 1                # C11
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01 - A11
    V[2, 6] = 1; V[3, 6] = 1   # B10 + B11
    W[0, 6] = 1                # C00
    
    return U, V, W


def _winograd_2x4x4_factors():
    """
    Return U, V, W factor matrices for 2x4x4 matrix multiplication using Winograd's rank-20 algorithm.
    U has shape (8, 20), V has shape (16, 20), W has shape (8, 20).
    Indices: U[i*4+j, r] for A[i,j], V[j*4+k, r] for B[j,k], W[k*2+i, r] for C[i,k]
    """
    n = 4
    rank = n * n + n  # 20
    U = np.zeros((2 * n, rank), dtype=float)
    V = np.zeros((n * n, rank), dtype=float)
    W = np.zeros((n * 2, rank), dtype=float)
    
    # Winograd 2x4x4: 20 scalar multiplications
    # Products 0-15: for j in 0..3, k in 0..3
    #   Compute (A[0,j] + A[1,j]) * B[j,k], contributing to C[0,k] and C[1,k]
    for j in range(n):
        for k in range(n):
            r = j * n + k
            U[0, r] = 1.0
            U[1, r] = 1.0
            V[j * n + k, r] = 1.0
            W[k * 2 + 0, r] = 1.0
            W[k * 2 + 1, r] = 1.0
    
    # Products 16-19: for j in 0..3
    #   Compute (A[0,j] - A[1,j]) * (B[j,j] - B[j+1,j]) if j<3, else handle boundary
    # Actually, the correct Winograd pattern for the difference terms:
    # P_{16+j} = (A[0,j] - A[1,j]) * (B[j, j] - B[j, (j+1) % n])
    # This doesn't seem right. Let me use a different valid construction.
    
    # Alternative: Use the fact that we can express the 2x4x4 tensor as:
    # T = sum_{j,k} (e_0 + e_1) ⊗ e_j ⊗ e_k  * (A[0,j]+A[1,j]) * B[j,k]  [16 terms]
    #     + sum_j (e_0 - e_1) ⊗ e_j ⊗ (e_j - e_{j+1}) * (A[0,j]-A[1,j]) * (B[j,j]-B[j,j+1])  [4 terms]
    # But this doesn't decompose correctly.
    
    # Let me verify by checking if the above 16 products alone give the right answer:
    # sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*2+i, r]
    # For the 16 products: sum_{j',k'} U[i*4+j, j'*4+k'] * V[j*4+k, j'*4+k'] * W[k*2+i, j'*4+k']
    # U[i*4+j, j'*4+k'] is 1 only if j'=j, so: sum_{k'} 1 * V[j*4+k, j*4+k'] * W[k*2+i, j*4+k']
    # V[j*4+k, j*4+k'] is 1 only if k'=k, so: 1 * 1 * W[k*2+i, j*4+k]
    # W[k*2+i, j*4+k] is 1 for both i=0 and i=1.
    # So the sum is 1 for all i,j,k. That means the 16 products already give the correct answer!
    # But that would mean rank 16, which contradicts the known result.
    
    # Wait, I made an error. U[i*4+j, r] where r = j'*4+k'. For a given (i,j), 
    # U[i*4+j, j'*4+k'] = 1 if j'=j (regardless of k'). So:
    # sum_{k'} V[j*4+k, j*4+k'] * W[k*2+i, j*4+k']
    # V[j*4+k, j*4+k'] = 1 if k'=k, else 0.
    # So sum = W[k*2+i, j*4+k] = 1.
    # This gives 1 for all i,j,k, which is correct!
    
    # So actually, the 16 products with the "sum" pattern already work? That can't be right
    # because it would mean rank 16 for 2x4x4, but the known rank is 20.
    
    # I think I see the issue: the W matrix indices. W[k*2+i, r] where i is the output row.
    # For output C[i,k], we need W[k*2+i, r]. In my construction, W[k*2+0, r] = 1 and W[k*2+1, r] = 1
    # for all r in 0..15. So for output row 0, column k: sum_r U[0*4+j, r]*V[j*4+k, r]*W[k*2+0, r]
    # = sum_{r: U[4j,r]!=0} V[j*4+k, r] * 1
    # U[4j, r] is nonzero for r = j*4+k' for all k', i.e., r in {4j, 4j+1, 4j+2, 4j+3}
    # V[j*4+k, r] is nonzero only for r = j*4+k
    # So the sum is V[j*4+k, j*4+k] * W[k*2+0, j*4+k] = 1 * 1 = 1. Correct!
    
    # Similarly for output row 1. So it seems like rank 16 works? But that contradicts theory.
    
    # Let me double-check with a small example. For 2x2x2, the analogous construction:
    # 4 products: for j in 0..1, k in 0..1:
    #   U[0,j,r]=1, U[1,j,r]=1, V[j,k,r]=1, W[k,0,r]=1, W[k,1,r]=1
    # For output C[0,0]: sum_r U[0,r]*V[r,0]*W[0,r] where r indexes (j,k)
    # U[0,0]=1 (j=0,k=0), U[0,1]=0, U[0,2]=1 (j=1,k=0), U[0,3]=0
    # V[0,0]=1, V[1,0]=0, V[2,0]=0, V[3,0]=0  -- wait, V is indexed by [j*2+k, r]
    # V[j*2+k, r] where r is the product index. For product (j,k), V[j*2+k, j*2+k] = 1.
    # So V[0,0]=1, V[1,1]=1, V[2,2]=1, V[3,3]=1.
    # W[k*2+i, r]: for i=0, W[k, r]. W[0,0]=1, W[1,1]=1, W[2,2]=1, W[3,3]=1.
    # C[0,0] = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[1,0]*W[1,0] + U[0,2]*V[2,0]*W[2,0] + U[0,3]*V[3,0]*W[3,0]
    # = 1*1*1 + 0*0*0 + 1*0*0 + 0*0*0 = 1. Correct.
    # C[0,1] = U[0,0]*V[0,1]*W[1,0] + U[0,1]*V[1,1]*W[1,1] + U[0,2]*V[2,1]*W[1,2] + U[0,3]*V[3,1]*W[1,3]
    # = 1*0*0 + 1*1*1 + 0*0*0 + 0*0*0 = 1. Correct.
    # C[1,0] = U[1,0]*V[0,0]*W[2,0] + U[1,1]*V[1,0]*W[2,1] + U[1,2]*V[2,0]*W[2,2] + U[1,3]*V[3,0]*W[2,3]
    # = 1*1*1 + 0*0*0 + 1*0*0 + 0*0*0 = 1. Correct.
    # C[1,1] = U[1,0]*V[0,1]*W[3,0] + U[1,1]*V[1,1]*W[3,1] + U[1,2]*V[2,1]*W[3,2] + U[1,3]*V[3,1]*W[3,3]
    # = 1*0*0 + 1*1*1 + 0*0*0 + 0*0*0 = 1. Correct.
    
    # So for 2x2x2, this gives rank 4, but Strassen gives rank 7. That means my construction is WRONG.
    # The issue: I'm not accounting for the fact that U, V, W entries for different (i,j) pairs can overlap.
    
    # Let me re-examine. For 2x2x2 with my construction:
    # Product 0 (j=0,k=0): U[0,0]=1, U[1,0]=1, V[0,0]=1, W[0,0]=1, W[2,0]=1
    # Product 1 (j=0,k=1): U[0,1]=1, U[1,1]=1, V[1,1]=1, W[1,1]=1, W[3,1]=1
    # Product 2 (j=1,k=0): U[0,2]=1, U[1,2]=1, V[2,2]=1, W[0,2]=1, W[2,2]=1
    # Product 3 (j=1,k=1): U[0,3]=1, U[1,3]=1, V[3,3]=1, W[1,3]=1, W[3,3]=1
    #
    # C[0,0] = sum_r U[0,r]*V[r,0]*W[0,r]
    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 → 1
    # r=1: U[0,1]=1, V[1,0]=0, W[0,1]=0 → 0
    # r=2: U[0,2]=1, V[2,0]=0, W[0,2]=1 → 0
    # r=3: U[0,3]=1, V[3,0]=0, W[0,3]=0 → 0
    # Total: 1. Correct.
    #
    # But wait, V[r,0] where r is the product index. V has shape (4,4) for 2x2x2.
    # V[j*2+k, r]. For r=0 (product j=0,k=0): V[0,0]=1, V[1,0]=0, V[2,0]=0, V[3,0]=0.
    # For r=2 (product j=1,k=0): V[0,2]=0, V[1,2]=0, V[2,2]=1, V[3,2]=0.
    #
    # C[0,0] = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] + U[0,2]*V[0,2]*W[0,2] + U[0,3]*V[0,3]*W[0,3]
    # = 1*1*1 + 1*0*0 + 1*0*1 + 1*0*0 = 1. Correct.
    #
    # Hmm, it seems correct for 2x2x2. But the known rank is 7, not 4. I must be misunderstanding the tensor decomposition format.
    
    # OH WAIT. I think I see the issue. The tensor T[i,j,k] for matrix multiplication is:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # In my construction, for product r=(j',k'):
    # U[i, j, r] = 1 if j=j' (for both i=0 and i=1)
    # V[j, k, r] = 1 if j=j' and k=k'
    # W[k, i, r] = 1 if k=k' (for both i=0 and i=1)
    #
    # So T[i,j,k] = sum_{j',k'} [j=j'] * [j=j' and k=k'] * [k=k']
    # = sum_{j',k'} [j=j'] * [j=j'] * [k=k']  (since [j=j' and k=k'] = [j=j']*[k=k'])
    # = sum_{j',k'} [j=j'] * [k=k']
    # = 1 (when j'=j and k'=k)
    # 
    # Wait, that gives T[i,j,k] = 1 for all i,j,k. That's correct!
    # 
    # So why is the rank not 4 for 2x2x2? I think I'm confusing myself. Let me count the number of products: 2*2 = 4 products.
    # But Strassen uses 7. So either my construction is wrong, or Strassen is not optimal (which it is).
    
    # I think the error is: U[i,j,r] should depend on i as well! In my construction, U[0,j,r] = U[1,j,r] = 1 when j=j'.
    # But for the decomposition to work, we need:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # With my construction: sum_{j'=j} sum_{k'=k} 1*1*1 = 1. This works!
    
    # So it seems like rank 4 works for 2x2x2? That can't be right. Let me check the literature...
    # Actually, I think I'm making an error in the index convention. Let me re-read the problem.
    # 
    # "U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)"
    # "U[i*m+j, r] for A[i,j], V[j*p+k, r] for B[j,k], W[k*n+i, r] for C[i,k]"
    #
    # So for 2x2x2: U shape (4, rank), V shape (4, rank), W shape (4, rank)
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    #
    # My construction for product (j',k'):
    # U[i*2+j, r] = 1 if j=j' (for i=0,1)
    # V[j*2+k, r] = 1 if j=j' and k=k'
    # W[k*2+i, r] = 1 if k=k' (for i=0,1)
    #
    # T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    # = sum_{j',k'} [j=j'] * [j=j' and k=k'] * [k=k']
    # = [j=j] * [k=k] * 1 = 1. Correct!
    
    # So this construction gives a valid rank-4 decomposition for 2x2x2? That contradicts the known result.
    # 
    # I think the issue is that I'm allowing U, V, W to have arbitrary real values, and the "rank" is the number of terms in the sum.
    # With 4 terms, each term is a rank-1 tensor. The sum of 4 rank-1 tensors gives the matrix multiplication tensor.
    # But the known minimal rank is 7. So my construction must be wrong.
    
    # Let me test numerically:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # My 4 products:
    # P0 (j=0,k=0): (a+c) * e, contributes to C[0,0] and C[1,0]
    # P1 (j=0,k=1): (a+c) * f, contributes to C[0,1] and C[1,1]
    # P2 (j=1,k=0): (b+d) * g, contributes to C[0,0] and C[1,0]
    # P3 (j=1,k=1): (b+d) * h, contributes to C[0,1] and C[1,1]
    #
    # C[0,0] = P0 + P2 = (a+c)*e + (b+d)*g = ae + ce + bg + dg
    # But correct C[0,0] = ae + bg. Extra terms: ce + dg. WRONG!
    
    # I see the error! The W factor should distinguish between output rows. In my construction,
    # W[k*2+0, r] = W[k*2+1, r] = 1, so both output rows get the same contribution.
    # But we need C[0,0] = ae+bg and C[1,0] = ce+dg, which are different.
    
    # The fix: W should be 1 for only one output row per product, or use different coefficients.
    # This is why Strassen needs 7 products: to separate the contributions.
    
    # So my simple construction doesn't work. I need the actual Winograd factors.
    # Given the complexity, let me just keep the current Strassen-based approach and ensure it's correct.
    
    return None


def _build_strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for n x m times m x p matrix multiplication.
    Uses 2x2x2 Strassen blocks where possible, and schoolbook for remaining parts.
    """
    # We'll decompose into blocks of size 2x2x2 where possible
    # For 3x5x6: 
    # - Rows of A: 3 (can do one 2-block + one 1-row)
    # - Cols of A / rows of B: 5 (can do two 2-blocks + one 1-col)
    # - Cols of B: 6 (can do three 2-blocks)
    
    # Strategy: partition the computation into blocks
    # A is 3x5, B is 5x6, C is 3x6
    # 
    # Let's partition:
    # A rows: [0,1] and [2]
    # A cols/B rows: [0,1], [2,3], [4]
    # B cols: [0,1], [2,3], [4,5]
    
    # This gives us blocks:
    # Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (2x2x2, Strassen rank 7)
    # Block (0,1): A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]  (2x2x2, Strassen rank 7)
    # Block (0,2): A[0:2,4:5] * B[4:5,0:2] -> C[0:2,0:2]  (2x1x2, schoolbook rank 4)
    # Block (1,0): A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2]  (1x2x2, schoolbook rank 4)
    # Block (1,1): A[2:3,2:4] * B[2:4,0:2] -> C[2:3,0:2]  (1x2x2, schoolbook rank 4)
    # Block (1,2): A[2:3,4:5] * B[4:5,0:2] -> C[2:3,0:2]  (1x1x2, schoolbook rank 2)
    # Block (2,0): A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]  (2x2x2, Strassen rank 7)
    # Block (2,1): A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]  (2x2x2, Strassen rank 7)
    # Block (2,2): A[0:2,4:5] * B[4:5,2:4] -> C[0:2,2:4]  (2x1x2, schoolbook rank 4)
    # Block (3,0): A[2:3,0:2] * B[0:2,2:4] -> C[2:3,2:4]  (1x2x2, schoolbook rank 4)
    # Block (3,1): A[2:3,2:4] * B[2:4,2:4] -> C[2:3,2:4]  (1x2x2, schoolbook rank 4)
    # Block (3,2): A[2:3,4:5] * B[4:5,2:4] -> C[2:3,2:4]  (1x1x2, schoolbook rank 2)
    # Block (4,0): A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]  (2x2x2, Strassen rank 7)
    # Block (4,1): A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]  (2x2x2, Strassen rank 7)
    # Block (4,2): A[0:2,4:5] * B[4:5,4:6] -> C[0:2,4:6]  (2x1x2, schoolbook rank 4)
    # Block (5,0): A[2:3,0:2] * B[0:2,4:6] -> C[2:3,4:6]  (1x2x2, schoolbook rank 4)
    # Block (5,1): A[2:3,2:4] * B[2:4,4:6] -> C[2:3,4:6]  (1x2x2, schoolbook rank 4)
    # Block (5,2): A[2:3,4:5] * B[4:5,4:6] -> C[2:3,4:6]  (1x1x2, schoolbook rank 2)
    
    # Total rank: 6*7 + 6*4 + 6*2 = 42 + 24 + 12 = 78
    # Wait, let me recount:
    # Strassen blocks (2x2x2): 6 blocks * 7 = 42
    # Schoolbook 2x1x2 or 1x2x2: 6 blocks * 4 = 24
    # Schoolbook 1x1x2: 6 blocks * 2 = 12
    # Total: 78
    
    # Let's build this decomposition
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block
    def add_block(a_row_start, a_row_end, a_col_start, a_col_end, 
                  b_row_start, b_row_end, b_col_start, b_col_end,
                  c_row_start, c_row_end, c_col_start, c_col_end,
                  use_strassen=False):
        """Add a block to the decomposition."""
        a_rows = list(range(a_row_start, a_row_end))
        a_cols = list(range(a_col_start, a_col_end))
        b_rows = list(range(b_row_start, b_row_end))
        b_cols = list(range(b_col_start, b_col_end))
        c_rows = list(range(c_row_start, c_row_end))
        c_cols = list(range(c_col_start, c_col_end))
        
        ar = len(a_rows)
        ac = len(a_cols)
        br = len(b_rows)
        bc = len(b_cols)
        cr = len(c_rows)
        cc = len(c_cols)
        
        if use_strassen and ar == 2 and ac == 2 and br == 2 and bc == 2 and cr == 2 and cc == 2:
            # Use Strassen
            sU, sV, sW = _strassen_2x2x2_factors()
            for r in range(7):
                u_col = np.zeros(n * m, dtype=float)
                v_col = np.zeros(m * p, dtype=float)
                w_col = np.zeros(p * n, dtype=float)
                
                # Fill U
                for idx, ai in enumerate(a_rows):
                    for jdx, aj in enumerate(a_cols):
                        u_val = sU[idx * 2 + jdx, r]
                        if u_val != 0:
                            u_col[ai * m + aj] = u_val
                
                # Fill V
                for idx, bi in enumerate(b_rows):
                    for kdx, bk in enumerate(b_cols):
                        v_val = sV[idx * 2 + kdx, r]
                        if v_val != 0:
                            v_col[bi * p + bk] = v_val
                
                # Fill W
                for idx, ci in enumerate(c_rows):
                    for kdx, ck in enumerate(c_cols):
                        w_val = sW[idx * 2 + kdx, r]
                        if w_val != 0:
                            w_col[ck * n + ci] = w_val
                
                U_list.append(u_col)
                V_list.append(v_col)
                W_list.append(w_col)
        else:
            # Use schoolbook
            for ai in a_rows:
                for aj in a_cols:
                    for bk in b_cols:
                        # Note: b_rows should match a_cols for matrix multiplication
                        u_col = np.zeros(n * m, dtype=float)
                        v_col = np.zeros(m * p, dtype=float)
                        w_col = np.zeros(p * n, dtype=float)
                        
                        u_col[ai * m + aj] = 1
                        v_col[aj * p + bk] = 1
                        w_col[bk * n + ai] = 1
                        
                        U_list.append(u_col)
                        V_list.append(v_col)
                        W_list.append(w_col)
    
    # Define the blocks
    # A is 3x5, B is 5x6, C is 3x6
    # A rows: [0,1], [2]
    # A cols: [0,1], [2,3], [4]
    # B cols: [0,1], [2,3], [4,5]
    
    # Block (a_row_block, a_col_block, b_col_block)
    a_row_blocks = [(0, 2), (2, 3)]
    a_col_blocks = [(0, 2), (2, 4), (4, 5)]
    b_col_blocks = [(0, 2), (2, 4), (4, 6)]
    
    for ar_start, ar_end in a_row_blocks:
        for ac_start, ac_end in a_col_blocks:
            for bc_start, bc_end in b_col_blocks:
                ar = ar_end - ar_start
                ac = ac_end - ac_start
                bc = bc_end - bc_start
                cr = ar  # C rows match A rows
                cc = bc  # C cols match B cols
                
                use_strassen = (ar == 2 and ac == 2 and bc == 2)
                add_block(ar_start, ar_end, ac_start, ac_end,
                          ac_start, ac_end, bc_start, bc_end,
                          ar_start, ar_end, bc_start, bc_end,
                          use_strassen=use_strassen)
    
    # Convert to matrices
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    rank = U.shape[1]
    return U, V, W, rank


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    # Check: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k
    
    # Build the tensor from the decomposition
    rank = U.shape[1]
    
    # Check a few random entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    
    return True, None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # First, create the schoolbook decomposition as a fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try Strassen-based decomposition
    try:
        U_str, V_str, W_str, rank_str = _build_strassen_decomposition(n, m, p)
        
        # Verify
        valid, err = _verify_decomposition(U_str, V_str, W_str, n, m, p)
        
        if valid and rank_str < best_rank:
            best_rank = rank_str
            best_U = U_str
            best_V = V_str
            best_W = W_str
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
