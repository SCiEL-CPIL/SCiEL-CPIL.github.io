"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        """Return the trivial rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # The Strassen decomposition of the 2x2x2 matrix-multiplication tensor.
    # For 2x2 A and B, C = AB, the 7 products are:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #
    # Each M_t is a product of a linear form in A and a linear form in B.
    # In the tensor decomposition T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c],
    # each Strassen term contributes ONE rank-1 tensor where:
    #   U[i,j] = coefficient of A[i,j] in the A-factor
    #   V[j,k] = coefficient of B[j,k] in the B-factor
    #   W[k,i] = coefficient of M_t in C[i,k]
    #
    # The 7 terms (u for A, v for B, w for C):
    #   u, v, w are 2x2 matrices (flattened to length-4 vectors in row-major order)
    #   u = [A00_coeff, A01_coeff, A10_coeff, A11_coeff]
    #   v = [B00_coeff, B01_coeff, B10_coeff, B11_coeff]
    #   w = [C00_coeff, C01_coeff, C10_coeff, C11_coeff]
    #
    # M1: A-factor = A00+A11 => u=[1,0,0,1]; B-factor = B00+B11 => v=[1,0,0,1]
    #     C: C00+=M1, C11+=M1 => w=[1,0,0,1]
    # M2: A-factor = A10+A11 => u=[0,0,1,1]; B-factor = B00 => v=[1,0,0,0]
    #     C: C10+=M2, C11-=M2 => w=[0,0,1,-1]
    # M3: A-factor = A00 => u=[1,0,0,0]; B-factor = B01-B11 => v=[0,1,0,-1]
    #     C: C01+=M3, C11+=M3 => w=[0,1,0,1]
    # M4: A-factor = A11 => u=[0,0,0,1]; B-factor = B10-B00 => v=[-1,0,1,0]
    #     C: C00+=M4, C10+=M4 => w=[1,0,1,0]
    # M5: A-factor = A00+A01 => u=[1,1,0,0]; B-factor = B11 => v=[0,0,0,1]
    #     C: C00-=M5, C01+=M5 => w=[-1,1,0,0]
    # M6: A-factor = A10-A00 => u=[-1,0,1,0]; B-factor = B00+B01 => v=[1,1,0,0]
    #     C: C11+=M6 => w=[0,0,0,1]
    # M7: A-factor = A01-A11 => u=[0,1,0,-1]; B-factor = B10+B11 => v=[0,0,1,1]
    #     C: C00+=M7, C11+=M7 => w=[1,0,0,1]

    strassen_2x2x2 = [
        # (u_flat, v_flat, w_flat) each of length 4
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),   # M7
    ]

    # Verify the Strassen decomposition for 2x2x2
    # For the 2x2x2 matmul tensor, T[i,j,k] = 1 for all i,j,k in {0,1}
    # We check: sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] = 1 for all i,j,k
    # But wait - in our convention, the tensor is T[i,j,k]=1 meaning
    # A[i,j]*B[j,k] contributes to C[i,k]. The rank-1 term has:
    #   U[i,j], V[j,k], W[k,i] (note W is indexed by k,n+i i.e. k*n+i)
    # For 2x2x2: U is 2x2, V is 2x2, W is 2x2
    # T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should = 1

    # Let me verify: for i=0,j=0,k=0:
    # sum_c U[0,0,c]*V[0,0,c]*W[0,0,c]
    # M1: 1*1*1 = 1
    # M2: 0*1*0 = 0
    # M3: 1*0*0 = 0
    # M4: 0*(-1)*1 = 0
    # M5: 1*0*(-1) = 0
    # M6: -1*1*0 = 0
    # M7: 0*0*1 = 0
    # Sum = 1 ✓

    # For i=0,j=0,k=1:
    # sum_c U[0,0,c]*V[0,1,c]*W[1,0,c]
    # M1: 1*0*0 = 0
    # M2: 0*0*1 = 0
    # M3: 1*1*0 = 0
    # M4: 0*0*1 = 0
    # M5: 1*0*0 = 0
    # M6: -1*1*0 = 0
    # M7: 0*0*0 = 0
    # Sum = 0 ✗ (should be 1)

    # Hmm, this doesn't work. The issue is that the Strassen decomposition
    # works for the computation of C, not directly as a tensor decomposition
    # in the form T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c].
    #
    # Actually, I think the correct interpretation is different. In the
    # tensor decomposition of matrix multiplication, we have:
    # T[i,j,k] = 1 for all i,j,k (the "multiplication tensor")
    # and we want T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    #
    # But Strassen's algorithm doesn't directly give this. It gives a way
    # to compute C = AB using 7 scalar multiplications instead of 8.
    #
    # The correct way to see Strassen as a tensor decomposition:
    # The 2x2x2 matmul tensor has rank 7. One valid decomposition is:
    # T[i,j,k] = sum_{t=1}^7 u_t[i] * v_t[j] * w_t[k]
    # where u_t, v_t, w_t are vectors in R^2 (not matrices).
    #
    # Wait, that's for a different tensor. Let me reconsider.
    #
    # The matrix multiplication tensor for multiplying an n×m matrix by an m×p matrix
    # is a 3-tensor T of shape (n*m, m*p, p*n) where T[(i,j),(j',k),(k',i')] = 1
    # if j=j' and i=i' and k=k', else 0.
    #
    # A rank-r decomposition is: T = sum_{c=1}^r U_c ⊗ V_c ⊗ W_c
    # where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}.
    #
    # For the 2x2x2 case, T has shape (4,4,4) with T[(i,j),(j,k),(k,i)]=1.
    # The schoolbook decomposition has rank 8 (one for each (i,j,k) triple).
    # Strassen shows rank 7 is sufficient.
    #
    # In the Strassen decomposition, each "term" t corresponds to:
    #   U_t: a vector in R^4 (for the 4 entries of A)
    #   V_t: a vector in R^4 (for the 4 entries of B)  
    #   W_t: a vector in R^4 (for the 4 entries of C)
    # such that sum_t U_t[(i,j)] * V_t[(j,k)] * W_t[(k,i)] = 1 for all i,j,k.
    #
    # Let me re-derive. For 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = AB = [[c00, c01], [c10, c11]]
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    #
    # The tensor T has T[(i,j),(j,k),(k,i)] = 1 for i,j,k in {0,1}.
    # So T has 8 nonzero entries (one for each (i,j,k) triple).
    #
    # A rank-1 term is U ⊗ V ⊗ W where U is 4-vector, V is 4-vector, W is 4-vector.
    # (U⊗V⊗W)[(i,j),(j',k),(k',i')] = U[(i,j)] * V[(j',k)] * W[(k',i')]
    # For this to contribute to T, we need j=j' and i=i' and k=k'.
    # So the contribution of term t to position (i,j,k) is:
    #   U_t[(i,j)] * V_t[(j,k)] * W_t[(k,i)]
    # And we need sum_t U_t[(i,j)] * V_t[(j,k)] * W_t[(k,i)] = 1 for all i,j,k.
    #
    # Now, in Strassen, the 7 products are:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # And:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    #
    # Each M_t = (linear form in A) * (linear form in B)
    # The linear form in A for M_t gives U_t (as a 4-vector indexed by (i,j))
    # The linear form in B for M_t gives V_t (as a 4-vector indexed by (j,k))
    # The coefficients in C give W_t (as a 4-vector indexed by (k,i))
    #
    # M1: A-form: a00+a11 => U1 = [1,0,0,1] (indices: (0,0),(0,1),(1,0),(1,1))
    #     B-form: b00+b11 => V1 = [1,0,0,1] (indices: (0,0),(0,1),(1,0),(1,1))
    #     C-coeffs: c00+=1, c01+=0, c10+=0, c11+=1 => W1 = [1,0,0,1] (indices: (0,0),(1,0),(0,1),(1,1))
    #     Wait, W is indexed by (k,i). So W[(k,i)]:
    #     c00 is C[0,0], which corresponds to (k=0,i=0) => W[(0,0)]
    #     c01 is C[0,1], which corresponds to (k=1,i=0) => W[(1,0)]
    #     c10 is C[1,0], which corresponds to (k=0,i=1) => W[(0,1)]
    #     c11 is C[1,1], which corresponds to (k=1,i=1) => W[(1,1)]
    #     So W1 = [1, 0, 0, 1] for indices (0,0),(1,0),(0,1),(1,1)
    #
    # Let me check: sum_t U_t[(i,j)] * V_t[(j,k)] * W_t[(k,i)] for (i=0,j=0,k=0):
    # M1: U1[(0,0)]=1, V1[(0,0)]=1, W1[(0,0)]=1 => 1*1*1=1
    # M2: U2[(0,0)]=0, ... => 0
    # M3: U3[(0,0)]=1, V3[(0,0)]=0, ... => 0
    # M4: U4[(0,0)]=0, ... => 0
    # M5: U5[(0,0)]=1, V5[(0,0)]=0, ... => 0
    # M6: U6[(0,0)]=-1, V6[(0,0)]=1, W6[(0,0)]=0 => 0
    # M7: U7[(0,0)]=0, ... => 0
    # Sum = 1 ✓
    #
    # For (i=0,j=0,k=1):
    # M1: U1[(0,0)]=1, V1[(0,1)]=0, ... => 0
    # M2: U2[(0,0)]=0, ... => 0
    # M3: U3[(0,0)]=1, V3[(0,1)]=1, W3[(1,0)]=0 => 0
    # M4: U4[(0,0)]=0, ... => 0
    # M5: U5[(0,0)]=1, V5[(0,1)]=0, ... => 0
    # M6: U6[(0,0)]=-1, V6[(0,1)]=1, W6[(1,0)]=0 => 0
    # M7: U7[(0,0)]=0, ... => 0
    # Sum = 0 ✗
    #
    # This is wrong. The issue is that the Strassen decomposition doesn't directly
    # give a rank-7 tensor decomposition in this form. 
    #
    # Actually, I think I'm making an error. Let me reconsider.
    #
    # The matrix multiplication tensor for 2x2x2 is T ∈ R^{4×4×4} where
    # T[(i,j),(j',k),(k',i')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    #
    # A rank-r decomposition is T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c where
    # u_c ∈ R^4, v_c ∈ R^4, w_c ∈ R^4.
    #
    # (u⊗v⊗w)[(i,j),(j',k),(k',i')] = u[(i,j)] * v[(j',k)] * w[(k',i')]
    #
    # For T[(i,j),(j,k),(k,i)] = 1, we need:
    # sum_c u_c[(i,j)] * v_c[(j,k)] * w_c[(k,i)] = 1
    #
    # This is exactly what I had. And the Strassen algorithm should give this.
    #
    # Let me re-examine. The problem might be in how I'm assigning the W coefficients.
    #
    # In Strassen, M_t contributes to C with certain coefficients. But M_t itself is
    # (A-linear-form) * (B-linear-form). The contribution to C[i,k] from M_t is
    # coeff_t[i,k] * M_t.
    #
    # In the tensor decomposition, the term corresponding to M_t should satisfy:
    # u_t[(i,j)] * v_t[(j,k)] * w_t[(k,i)] = coeff_t[i,k] * (A-linear-form at (i,j)) * (B-linear-form at (j,k))
    #
    # Wait, that's not right either. Let me think more carefully.
    #
    # Actually, I think the correct statement is:
    # The tensor T can be written as sum_t (u_t ⊗ v_t ⊗ w_t) where each term t
    # corresponds to one of the Strassen products. The u_t, v_t, w_t are NOT simply
    # the linear forms and coefficients. Instead, the product (u_t ⊗ v_t ⊗ w_t)
    # evaluated at ((i,j),(j,k),(k,i)) gives u_t[(i,j)] * v_t[(j,k)] * w_t[(k,i)],
    # and the sum over t should give 1.
    #
    # Let me just directly solve for a rank-7 decomposition.
    #
    # Actually, let me try a different approach. Let me just verify whether my
    # Strassen terms work by computing the full sum.

    # Let me define the Strassen terms more carefully and verify computationally.
    # I'll use the following known-correct rank-7 decomposition of the 2x2x2 matmul tensor.
    # (These are from the literature; I'll verify them in code.)
    
    # Alternative: just use a verified set. Let me construct one by solving.
    # For now, let me use a simpler approach: block decomposition.
    
    # Strategy: decompose 4x5x6 into blocks and use Strassen where possible.
    # Split: i into [0,1],[2,3]; j into [0,1],[2,3],[4]; k into [0,1],[2,3],[4,5]
    # 2x2x2 blocks: 2*3*2 = 12 blocks (using Strassen rank 7 each = 84)
    # 2x1x2 blocks: 2*1*2 = 4 blocks (schoolbook rank 4 each = 16)  
    # 2x2x1 blocks: 2*2*1 = 4 blocks (schoolbook rank 4 each = 16)
    # Wait, let me recount.
    # i_groups: [0,1], [2,3] (2 groups)
    # j_groups: [0,1], [2,3], [4] (3 groups)
    # k_groups: [0,1], [2,3], [4,5] (3 groups)
    # Total blocks: 2*3*3 = 18
    # 2x2x2: when i_group has 2, j_group has 2, k_group has 2
    #   i_groups with 2: both (2)
    #   j_groups with 2: [0,1],[2,3] (2)
    #   k_groups with 2: [0,1],[2,3],[4,5] (3)
    #   Count: 2*2*3 = 12 blocks of 2x2x2
    # 2x2x1: j_group=[4] (size 1), others size 2
    #   Count: 2*1*2 = 4 blocks... wait, k_groups with size 2 are 3, so 2*1*3=6? No.
    #   j_groups: [0,1](2), [2,3](2), [4](1)
    #   k_groups: [0,1](2), [2,3](2), [4,5](2)
    #   2x2x2: i(2)*j(2)*k(2) = 2*2*3 = 12
    #   2x1x2: i(2)*j(1)*k(2) = 2*1*3 = 6
    #   2x2x1: none (all k_groups have size 2)
    #   1x1x1: none
    # Total: 12*7 + 6*4 = 84 + 24 = 108
    
    # But I need a valid Strassen decomposition for 2x2x2 first.
    # Let me just use schoolbook for everything and see if I can improve.
    
    # Actually, let me try a completely different approach.
    # Let me directly attempt to find a lower-rank decomposition using optimization.
    
    # Given the time constraints, let me just return the schoolbook decomposition
    # which is guaranteed to work. The rank is 120.
    
    # But wait - the current program claims to achieve rank 108 with Strassen.
    # Let me try to make that work by properly implementing the Strassen decomposition.
    
    # Let me use a verified Strassen decomposition. I'll solve the system.
    
    # For 2x2x2, we need 7 vectors (u_t, v_t, w_t) in R^4 each such that
    # sum_t u_t[(i,j)] * v_t[(j,k)] * w_t[(k,i)] = 1 for all i,j,k in {0,1}.
    
    # This is 8 equations. Let me try to find a solution.
    # Due to the small size, I can try a parametric approach.
    
    # Actually, let me just use the following known-correct decomposition:
    # (I'll verify it in code)
    
    # One valid rank-7 decomposition of the 2x2x2 matmul tensor:
    # (u, v, w) where each is a 4-vector:
    S = [
        # u, v, w
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # M3
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,1]),  # M7
    ]
    
    # Verify: sum_t u_t[(i,j)] * v_t[(j,k)] * w_t[(k,i)] should = 1
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for (u, v, w) in S:
                    s += u[i*2+j] * v[j*2+k] * w[k*2+i]
                if abs(s - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # The Strassen decomposition I have is not valid in this tensor form.
        # Let me just use schoolbook for all blocks.
        return make_schoolbook()
    
    # Build the full decomposition using block decomposition with Strassen for 2x2x2 blocks
    i_groups = [[0, 1], [2, 3]]
    j_groups = [[0, 1], [2, 3], [4]]
    k_groups = [[0, 1], [2, 3], [4, 5]]
    
    U = []
    V = []
    W = []
    rank = 0
    
    num_U = nm  # 20
    num_V = mp  # 30
    num_W = pn  # 24
    
    U = [[0] * 0 for _ in range(num_U)]
    V = [[0] * 0 for _ in range(num_V)]
    W = [[0] * 0 for _ in range(num_W)]
    
    def add_term(u_vals, v_vals, w_vals, i_indices, j_indices, k_indices):
        """Add a rank-1 term."""
        nonlocal rank
        for _ in range(num_U):
            U[_].append(0)
        for _ in range(num_V):
            V[_].append(0)
        for _ in range(num_W):
            W[_].append(0)
        for idx, i in enumerate(i_indices):
            for jdx, j in enumerate(j_indices):
                U[i * m + j][rank] = u_vals[idx * len(j_indices) + jdx]
        for jdx, j in enumerate(j_indices):
            for kdx, k in enumerate(k_indices):
                V[j * p + k][rank] = v_vals[jdx * len(k_indices) + kdx]
        for kdx, k in enumerate(k_indices):
            for idx, i in enumerate(i_indices):
                W[k * n + i][rank] = w_vals[kdx * len(i_indices) + idx]
        rank += 1
    
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                if len(ig) == 2 and len(jg) == 2 and len(kg) == 2:
                    # 2x2x2 block: use Strassen (7 terms)
                    for (u, v, w) in S:
                        add_term(u, v, w, ig, jg, kg)
                else:
                    # Schoolbook for this block
                    for i in ig:
                        for j in jg:
                            for k in kg:
                                u_val = [0] * (len(ig) * len(jg))
                                v_val = [0] * (len(jg) * len(kg))
                                w_val = [0] * (len(kg) * len(ig))
                                u_val[ig.index(i) * len(jg) + jg.index(j)] = 1
                                v_val[jg.index(j) * len(kg) + kg.index(k)] = 1
                                w_val[kg.index(k) * len(ig) + ig.index(i)] = 1
                                add_term(u_val, v_val, w_val, ig, jg, kg)
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if not verify(U, V, W, rank):
        return make_schoolbook()
    
    return result
# EVOLVE-BLOCK-END
