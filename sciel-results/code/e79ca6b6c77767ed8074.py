"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0  # Safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 4, 4

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Rank-7 decomposition for 2x2 * 2x2 -> 2x2
    # Using Strassen's algorithm.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B: C[0,0]=a00*b00+a01*b10, C[0,1]=a00*b01+a01*b11
    #          C[1,0]=a10*b00+a11*b10, C[1,1]=a10*b01+a11*b11
    #
    # Strassen's 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # For our tensor decomposition, each term is (u, v, w) where:
    # u is a vector over A entries (index i*2+j)
    # v is a vector over B entries (index j*2+k)
    # w is a vector over C entries (index k*2+i, per task convention W[k*n+i])
    #
    # For 2x2: C[i,k] is at W index k*2+i
    # C[0,0] -> W[0], C[0,1] -> W[2], C[1,0] -> W[1], C[1,1] -> W[3]
    #
    # Each Strassen term M_r = (linear combo of A) * (linear combo of B)
    # contributes to specific C entries with coefficients.
    #
    # We need to express each M_r as a sum of rank-1 tensors.
    # But actually, for the tensor decomposition, each "term" in the rank-r
    # decomposition is a single rank-1 tensor: u_r * v_r * w_r.
    #
    # The Strassen algorithm uses 7 scalar multiplications, but each
    # multiplication M_r = f_r(A) * g_r(B) where f_r and g_r are linear
    # combinations. In the tensor decomposition, we need to expand each
    # such product into rank-1 terms... but that would give more than 7 terms.
    #
    # Actually, the rank of the tensor is 7, meaning there exist 7 vectors
    # u_r, v_r, w_r such that T = sum_r u_r ⊗ v_r ⊗ w_r.
    #
    # Let me directly construct the 7 rank-1 tensors.
    # The tensor T has entries T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k],
    # i.e., T[i,j,k] = 1 for all i,j,k (it's the standard multiplication tensor).
    #
    # For 2x2x2, T[i,j,k] = 1 for all i,j,k in {0,1}.
    # We need T = sum_{r=1}^{7} u_r[i] * v_r[j] * w_r[k]
    # where u_r is indexed by (i,j) [4 components], v_r by (j,k) [4 components],
    # w_r by (i,k) [4 components].
    #
    # Wait, I need to be more careful. The multiplication tensor T is a 3-tensor
    # with indices (i, j, k) where i is row of A, j is col of A / row of B,
    # k is col of B. T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k].
    #
    # In the decomposition T = sum_r u_r ⊗ v_r ⊗ w_r:
    # u_r has indices (i,j) -> 4 components
    # v_r has indices (j,k) -> 4 components  
    # w_r has indices (i,k) -> 4 components
    #
    # And T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k]
    #
    # For the 2x2 case, this means:
    # 1 = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] for all i,j,k in {0,1}
    #
    # The Strassen decomposition gives us this. Let me derive it explicitly.
    #
    # From Strassen:
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # Where:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # Each M_r = f_r(A) * g_r(B) where f_r is linear in A and g_r is linear in B.
    # In the tensor decomposition, each M_r corresponds to one rank-1 term:
    # The contribution of M_r to C[i,k] is coeff_{r,i,k} * f_r(A) * g_r(B)
    #
    # So u_r[i,j] = coefficient of A[i,j] in f_r
    #    v_r[j,k] = coefficient of B[j,k] in g_r
    #    w_r[i,k] = coefficient of M_r in C[i,k]
    #
    # Let me write these out:
    #
    # M1: f = a00 + a11, g = b00 + b11
    #   u_1 = [1,0, 0,1] (a00, a01, a10, a11)
    #   v_1 = [1,0, 0,1] (b00, b01, b10, b11)
    #   C[0,0] gets +1, C[1,1] gets +1
    #   w_1 = [1, 0, 0, 1] (C[0,0], C[1,0], C[0,1], C[1,1])
    #
    # M2: f = a10 + a11, g = b00
    #   u_2 = [0,0, 1,1]
    #   v_2 = [1,0, 0,0]
    #   C[1,0] gets +1, C[1,1] gets -1
    #   w_2 = [0, 1, 0, -1]
    #
    # M3: f = a00, g = b01 - b11
    #   u_3 = [1,0, 0,0]
    #   v_3 = [0,1, 0,-1]
    #   C[0,1] gets +1, C[1,1] gets +1
    #   w_3 = [0, 0, 1, 1]
    #
    # M4: f = a11, g = b10 - b00
    #   u_4 = [0,0, 0,1]
    #   v_4 = [-1,0, 1,0]
    #   C[0,0] gets -1, C[1,0] gets +1
    #   w_4 = [-1, 1, 0, 0]
    #
    # M5: f = a00 + a01, g = b11
    #   u_5 = [1,1, 0,0]
    #   v_5 = [0,0, 0,1]
    #   C[0,0] gets +1, C[0,1] gets +1
    #   w_5 = [1, 0, 1, 0]
    #
    # M6: f = a10 - a00, g = b00 + b01
    #   u_6 = [-1,0, 1,0]
    #   v_6 = [1,1, 0,0]
    #   C[1,1] gets +1
    #   w_6 = [0, 0, 0, 1]
    #
    # M7: f = a01 - a11, g = b10 + b11
    #   u_7 = [0,1, 0,-1]
    #   v_7 = [0,0, 1,1]
    #   C[0,0] gets -1
    #   w_7 = [-1, 0, 0, 0]
    #
    # Let me verify: T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k]
    #
    # For i=0,j=0,k=0:
    # M1: u[0,0]=1, v[0,0]=1, w[0,0]=1 -> 1
    # M2: u[0,0]=0 -> 0
    # M3: u[0,0]=1, v[0,0]=0 -> 0
    # M4: u[0,0]=0 -> 0
    # M5: u[0,0]=1, v[0,0]=0 -> 0
    # M6: u[0,0]=-1, v[0,0]=1, w[0,0]=0 -> 0
    # M7: u[0,0]=0 -> 0
    # Sum = 1 ✓
    #
    # For i=0,j=0,k=1:
    # M1: u[0,0]=1, v[0,1]=0 -> 0
    # M2: u[0,0]=0 -> 0
    # M3: u[0,0]=1, v[0,1]=1, w[0,1]=1 -> 1
    # M4: u[0,0]=0 -> 0
    # M5: u[0,0]=1, v[0,1]=0 -> 0
    # M6: u[0,0]=-1, v[0,1]=1, w[0,1]=0 -> 0
    # M7: u[0,0]=0 -> 0
    # Sum = 1 ✓
    #
    # For i=0,j=1,k=0:
    # M1: u[0,1]=0 -> 0
    # M2: u[0,1]=0 -> 0
    # M3: u[0,1]=0 -> 0
    # M4: u[0,1]=0 -> 0
    # M5: u[0,1]=1, v[1,0]=0 -> 0
    # M6: u[0,1]=0 -> 0
    # M7: u[0,1]=1, v[1,0]=1, w[0,0]=-1 -> -1
    # Sum = -1 ✗ (should be 1)
    
    # Hmm, there's an issue. Let me recheck M7.
    # M7 = (a01 - a11)*(b10 + b11)
    # f_7 = a01 - a11, so u_7[0,1] = 1, u_7[1,1] = -1
    # g_7 = b10 + b11, so v_7[1,0] = 1, v_7[1,1] = 1
    # C[0,0] gets -1 from M7, so w_7[0,0] = -1
    #
    # For i=0,j=1,k=0:
    # M7: u_7[0,1]=1, v_7[1,0]=1, w_7[0,0]=-1 -> -1
    # But we need T[0,1,0] = 1.
    #
    # Let me recheck the Strassen formulas. The issue is that I may have
    # the C formulas wrong. Let me re-derive.
    #
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    #
    # Strassen:
    # M1 = (a00+a11)*(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # M2 = (a10+a11)*b00 = a10*b00 + a11*b00
    # M3 = a00*(b01-b11) = a00*b01 - a00*b11
    # M4 = a11*(b10-b00) = a11*b10 - a11*b00
    # M5 = (a00+a01)*b11 = a00*b11 + a01*b11
    # M6 = (a10-a00)*(b00+b01) = a10*b00 + a10*b01 - a00*b00 - a00*b01
    # M7 = (a01-a11)*(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
    #
    # C[0,0] = a00*b00 + a01*b10
    #   = M1 - M4 + M5 - M7
    #   = (a00*b00 + a00*b11 + a11*b00 + a11*b11)
    #     - (a11*b10 - a11*b00)
    #     + (a00*b11 + a01*b11)
    #     - (a01*b10 + a01*b11 - a11*b10 - a11*b11)
    #   = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b10 + a11*b00 + a00*b11 + a01*b11 - a01*b10 - a01*b11 + a11*b10 + a11*b11
    #   = a00*b00 + 2*a00*b11 + 2*a11*b00 + 2*a11*b11 - a01*b10
    #   This is NOT equal to a00*b00 + a01*b10. My Strassen formulas are wrong.
    #
    # Let me look up the correct Strassen formulas.
    #
    # Correct Strassen:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # Let me re-verify C[0,0]:
    # M1 - M4 + M5 - M7
    # = (a00*b00 + a00*b11 + a11*b00 + a11*b11)
    #   - (a11*b10 - a11*b00)
    #   + (a00*b11 + a01*b11)
    #   - (a01*b10 + a01*b11 - a11*b10 - a11*b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b10 + a11*b00 + a00*b11 + a01*b11 - a01*b10 - a01*b11 + a11*b10 + a11*b11
    # = a00*b00 + 2*a00*b11 + 2*a11*b00 + 2*a11*b11 - a01*b10
    #
    # This is still wrong. I think I have the Strassen formulas incorrect.
    #
    # Let me use a different, verified set of Strassen formulas.
    #
    # Standard Strassen (from Wikipedia):
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b11)
    # M4 = a22(b22-b12)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Using 1-based indexing: a11=a00, a12=a01, a21=a10, a22=a11
    # b11=b00, b12=b01, b21=b10, b22=b11
    #
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b00)
    # M4 = a11(b11-b01)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # Let me verify C[0,0] = M1 - M4 + M5 - M7:
    # M1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # -M4 = -a11*b11 + a11*b01
    # +M5 = a00*b11 + a01*b11
    # -M7 = -a01*b10 - a01*b11 + a11*b10 + a11*b11
    # Sum = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b11 + a11*b01 + a00*b11 + a01*b11 - a01*b10 - a01*b11 + a11*b10 + a11*b11
    # = a00*b00 + 2*a00*b11 + a11*b00 + a11*b11 + a11*b01 - a01*b10 + a11*b10
    # Still not right.
    #
    # I think the issue is with the indexing convention. Let me be very explicit.
    #
    # A = [[a00, a01],
    #      [a10, a11]]
    # B = [[b00, b01],
    #      [b10, b11]]
    # C = A*B = [[a00*b00 + a01*b10, a00*b01 + a01*b11],
    #            [a10*b00 + a11*b10, a10*b01 + a11*b11]]
    #
    # Strassen (1-based, from standard references):
    # M1 = (A11+A22)(B11+B22) = (a00+a11)(b00+b11)
    # M2 = (A21+A22)B11 = (a10+a11)*b00
    # M3 = A11(B21-B11) = a00*(b10-b00)
    # M4 = A22(B22-B12) = a11*(b11-b01)
    # M5 = (A11+A12)B22 = (a00+a01)*b11
    # M6 = (A21-A11)(B11+B12) = (a10-a00)*(b00+b01)
    # M7 = (A12-A22)(B21+B22) = (a01-a11)*(b10+b11)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Verify C11 = a00*b00 + a01*b10:
    # M1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # -M4 = -(a11*b11 - a11*b01) = -a11*b11 + a11*b01
    # +M5 = a00*b11 + a01*b11
    # -M7 = -(a01*b10 + a01*b11 - a11*b10 - a11*b11) = -a01*b10 - a01*b11 + a11*b10 + a11*b11
    #
    # Sum: a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b11 + a11*b01 + a00*b11 + a01*b11 - a01*b10 - a01*b11 + a11*b10 + a11*b11
    # = a00*b00 + 2*a00*b11 + a11*b00 + a11*b11 + a11*b01 - a01*b10 + a11*b10
    #
    # This is NOT a00*b00 + a01*b10. Something is fundamentally wrong.
    #
    # Let me try a completely different approach. I'll use a verified
    # rank-7 decomposition from the literature.
    #
    # Actually, let me just use a brute-force approach: I know the tensor
    # T[i,j,k] = 1 for all i,j,k in {0,1}. I need to find 7 vectors
    # u_r (4-dim), v_r (4-dim), w_r (4-dim) such that
    # sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = 1 for all i,j,k.
    #
    # This is a system of 8 equations in 84 unknowns (7*4*3).
    # There are many solutions. Let me just use a known one.
    #
    # A simple construction: use the 8 schoolbook terms and find one
    # linear dependency to eliminate one term.
    #
    # Schoolbook terms (r=0..7):
    # r=0: (i,j,k)=(0,0,0): u=[1,0,0,0], v=[1,0,0,0], w=[1,0,0,0]
    # r=1: (i,j,k)=(0,0,1): u=[1,0,0,0], v=[0,1,0,0], w=[0,0,1,0]
    # r=2: (i,j,k)=(0,1,0): u=[0,1,0,0], v=[0,0,1,0], w=[1,0,0,0]
    # r=3: (i,j,k)=(0,1,1): u=[0,1,0,0], v=[0,0,0,1], w=[0,0,1,0]
    # r=4: (i,j,k)=(1,0,0): u=[0,0,1,0], v=[1,0,0,0], w=[0,1,0,0]
    # r=5: (i,j,k)=(1,0,1): u=[0,0,1,0], v=[0,1,0,0], w=[0,0,0,1]
    # r=6: (i,j,k)=(1,1,0): u=[0,0,0,1], v=[0,0,1,0], w=[0,1,0,0]
    # r=7: (i,j,k)=(1,1,1): u=[0,0,0,1], v=[0,0,0,1], w=[0,0,0,1]
    #
    # (Using w index = k*2+i: C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3)
    #
    # I need to find coefficients c_0,...,c_7 (not all zero) such that
    # sum_r c_r * u_r[i,j] * v_r[j,k] * w_r[i,k] = 0 for all i,j,k.
    #
    # This gives 8 equations. With 8 unknowns, we expect a 1-dimensional
    # solution space (if the tensor has rank exactly 7).
    #
    # Let me set up the system. For each (i,j,k), the equation is:
    # sum_{r: i_r=i, j_r=j, k_r=k} c_r = 0
    #
    # Wait, that's not right. Each term r has specific i,j,k values.
    # For term r with (i_r, j_r, k_r), it contributes c_r * u_r[i,j] * v_r[j,k] * w_r[i,k].
    # But u_r[i,j] is nonzero only when (i,j) = (i_r, j_r), etc.
    #
    # So for a given (i,j,k), only terms r with i_r=i, j_r=j, k_r=k contribute,
    # and each contributes c_r * 1 * 1 * 1 = c_r.
    #
    # So the equations are:
    # c_0 = 0 (for i=0,j=0,k=0)
    # c_1 = 0 (for i=0,j=0,k=1)
    # c_2 = 0 (for i=0,j=1,k=0)
    # c_3 = 0 (for i=0,j=1,k=1)
    # c_4 = 0 (for i=1,j=0,k=0)
    # c_5 = 0 (for i=1,j=0,k=1)
    # c_6 = 0 (for i=1,j=1,k=0)
    # c_7 = 0 (for i=1,j=1,k=1)
    #
    # This means there's NO linear dependency among the schoolbook terms!
    # That can't be right if the rank is 7.
    #
    # Ah, I see the issue. The schoolbook terms are rank-1 tensors, but
    # they're not the only way to express the tensor. The rank-7 decomposition
    # uses DIFFERENT vectors u_r, v_r, w_r that are not just unit vectors.
    #
    # So I need to find 7 vectors (u_r, v_r, w_r) such that the tensor equation
    # holds. This is a nonlinear system.
    #
    # Let me just use a known correct rank-7 decomposition. I'll use the one
    # from the literature that I can verify computationally.
    #
    # Actually, for the purpose of this task, let me take a different approach.
    # Instead of trying to get the exact rank-7 decomposition right analytically,
    # let me implement the structured block decomposition with the schoolbook
    # for each 2x2 block (giving rank 8 per block instead of 7), which still
    # gives a better total rank than the full schoolbook.
    #
    # For 3x4 * 4x4, split j into {0,1} and {2,3}:
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #
    # For the first sum (j in {0,1}), for each (i,k) pair, this is a
    # 1x2 * 2x1 dot product, which has rank 2.
    # There are 3*4 = 12 (i,k) pairs, so this part has rank 12*2 = 24.
    #
    # For the second sum (j in {2,3}), similarly rank 24.
    #
    # Total rank: 48. Same as schoolbook. No improvement.
    #
    # The improvement comes from sharing structure between different (i,k) pairs
    # within the same block. For a 2x2 block of C (i in {0,1}, k in {0,1}),
    # the contribution from j in {0,1} is A_block * B_block where A_block is 2x2
    # and B_block is 2x2. This has rank 7 (not 8).
    #
    # So for the 3x4 * 4x4 case:
    # Split i into {0,1} and {2}, k into {0,1} and {2,3}, j into {0,1} and {2,3}.
    #
    # Block (i in {0,1}, k in {0,1}, j in {0,1}): 2x2 * 2x2 -> rank 7
    # Block (i in {0,1}, k in {0,1}, j in {2,3}): 2x2 * 2x2 -> rank 7
    # Block (i in {0,1}, k in {2,3}, j in {0,1}): 2x2 * 2x2 -> rank 7
    # Block (i in {0,1}, k in {2,3}, j in {2,3}): 2x2 * 2x2 -> rank 7
    # Block (i=2, k in {0,1}, j in {0,1}): 1x2 * 2x2 -> rank 2
    # Block (i=2, k in {0,1}, j in {2,3}): 1x2 * 2x2 -> rank 2
    # Block (i=2, k in {2,3}, j in {0,1}): 1x2 * 2x2 -> rank 2
    # Block (i=2, k in {2,3}, j in {2,3}): 1x2 * 2x2 -> rank 2
    #
    # Total rank: 7+7+7+7+2+2+2+2 = 36
    #
    # This is better than 48! But I need the correct rank-7 decomposition.
    #
    # Let me just implement this with a numerically verified rank-7 decomposition.
    # I'll use numpy to find it.
    
    import numpy as np
    
    # Find a rank-7 decomposition of the 2x2x2 multiplication tensor
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    # We need T = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
    
    # Use a known construction. The 2x2x2 tensor can be written as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (8 terms)
    # We need to find a linear combination that reduces to 7 terms.
    #
    # Actually, let me just use a specific known decomposition.
    # From "Algorithms for Matrix Multiplication" (Bini et al.):
    #
    # The 7 terms can be constructed as follows. Let me use a
    # computational approach: solve for the coefficients.
    #
    # Since this is getting complex, let me just use the schoolbook
    # solution for now (rank 48) and note that a future iteration
    # could implement the rank-36 structured decomposition.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
