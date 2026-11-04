import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110

    def build_bcl_333():
        """
        BCL rank-19 decomposition of the 3x3x3 multiplication tensor.
        T[i,j,k] = 1 if i==k (C[i,k] = sum_j A[i,j]*B[j,k])
        U indices: i*3+j (0..8)
        V indices: j*3+k (0..8)
        W indices: k*3+i (0..8)
        Returns list of (u_dict, v_dict, w_dict) where each dict maps index->coeff
        """
        # BCL 3x3x3 decomposition (19 rank-1 terms)
        # Using the standard Bini-Capovani-De Falco-Ladisi construction
        # 
        # The 19 products for C = A*B (3x3):
        # Let aij = A[i,j], bkj = B[k,j] (note: B is stored as B[j,k] so V index j*3+k)
        #
        # We need: C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i,j,k] = delta(i,k)
        # U index = i*3+j, V index = j*3+k, W index = k*3+i
        
        terms = []
        
        # Term 1: a00 * b00 -> C[0,0]
        terms.append(({0:1}, {0:1}, {0:1}))  # U[0*3+0]=0, V[0*3+0]=0, W[0*3+0]=0
        
        # Term 2: a01 * b10 -> C[0,0]
        terms.append(({1:1}, {3:1}, {0:1}))  # U[0*3+1]=1, V[1*3+0]=3, W[0*3+0]=0
        
        # Term 3: a02 * b20 -> C[0,0]
        terms.append(({2:1}, {6:1}, {0:1}))  # U[0*3+2]=2, V[2*3+0]=6, W[0*3+0]=0
        
        # Term 4: a10 * b01 -> C[1,1]
        terms.append(({3:1}, {1:1}, {4:1}))  # U[1*3+0]=3, V[0*3+1]=1, W[1*3+1]=4
        
        # Term 5: a11 * b11 -> C[1,1]
        terms.append(({4:1}, {4:1}, {4:1}))  # U[1*3+1]=4, V[1*3+1]=4, W[1*3+1]=4
        
        # Term 6: a12 * b21 -> C[1,1]
        terms.append(({5:1}, {7:1}, {4:1}))  # U[1*3+2]=5, V[2*3+1]=7, W[1*3+1]=4
        
        # Term 7: a20 * b02 -> C[2,2]
        terms.append(({6:1}, {2:1}, {8:1}))  # U[2*3+0]=6, V[0*3+2]=2, W[2*3+2]=8
        
        # Term 8: a21 * b12 -> C[2,2]
        terms.append(({7:1}, {5:1}, {8:1}))  # U[2*3+1]=7, V[1*3+2]=5, W[2*3+2]=8
        
        # Term 9: a22 * b22 -> C[2,2]
        terms.append(({8:1}, {8:1}, {8:1}))  # U[2*3+2]=8, V[2*3+2]=8, W[2*3+2]=8
        
        # Now we need 10 more terms to handle the off-diagonal entries
        # C[0,1] = a00*b01 + a01*b11 + a02*b21
        # C[0,2] = a00*b02 + a01*b12 + a02*b22
        # C[1,0] = a10*b00 + a11*b10 + a12*b20
        # C[1,2] = a10*b02 + a11*b12 + a12*b22
        # C[2,0] = a20*b00 + a21*b10 + a22*b20
        # C[2,1] = a20*b01 + a21*b11 + a22*b21
        
        # BCL uses clever combinations. Let me use the known 19-term construction.
        # Actually the above 9 terms are just schoolbook. BCL needs to be different.
        
        # Let me use the proper BCL construction:
        terms = []
        
        # BCL 3x3x3: 19 multiplications
        # Using the formulation where we compute:
        # p1 = (a00+a01)*(b00+b10)
        # p2 = (a00+a02)*(b00+b20)  
        # p3 = (a01+a02)*(b10+b20)
        # p4 = a00*b00
        # p5 = a01*b10
        # p6 = a02*b20
        # ... etc for other rows/columns
        
        # Actually, let me use a verified explicit BCL construction.
        # The standard BCL 3x3x3 uses these 19 rank-1 tensors:
        
        # I'll use the construction from the literature with explicit coefficients.
        # For the 3x3x3 tensor T[i,j,k] = delta(i,k):
        
        # Using BCL's algorithm:
        # Split into 2x2 block (top-left) + extensions
        
        # 7 terms for the 2x2x2 Strassen on indices {0,1} x {0,1} x {0,1}:
        # S1: (a00+a10)(b00+b01) -> contributes to C[0,0],C[0,1],C[1,0],C[1,1]
        # S2: (a00+a10)(b10+b11) -> ...
        # S3: (a00+a01)(b00+b10) -> ...
        # S4: (a01+a11)(b01+b10) -> ...
        # S5: a00*b00
        # S6: a01*b10
        # S7: a10*b01
        
        # This is getting too complex to derive by hand. Let me use a computational approach.
        
        return None  # Placeholder - will use alternative approach

    def build_rank23_bcl():
        """
        Build rank-23 decomposition: BCL 3x3x3 (rank 19) + schoolbook 3x1x4 (rank 4)
        
        The 3x4 x 4x4 multiplication:
        A is 3x4, B is 4x4, C is 3x4
        C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
        
        Split B into B_left (4x3) and B_right (4x1):
        C_left = A * B_left (3x3 result) -> use BCL rank 19
        C_right = A * B_right (3x1 result) -> schoolbook rank 4 (one per row)
        
        Wait, C_right is 3x1, so we need 3*1 = 3 multiplications? No.
        C_right[i,0] = sum_j A[i,j] * B[j,3] for i=0,1,2
        That's 3 outputs, each needing 4 products = 12 schoolbook terms.
        
        Hmm, that gives 19 + 12 = 31, not 23.
        
        Let me reconsider. The directive says "rank-4 schoolbook decomposition for the remaining 3x1x4 strip".
        A 3x1x4 tensor has 3*1*4 = 12 entries, but rank can be less.
        Actually the tensor T[i,j,k] for the strip where k=3 only:
        T[i,j,3] = delta(i,3)... no that doesn't make sense.
        
        Let me re-read: "appending a rank-4 schoolbook decomposition for the remaining 3x1x4 strip"
        
        I think the idea is:
        - Use BCL for the 3x3x3 part (first 3 columns of B)
        - For the 4th column of B (making it 3x1x4 in the sense of 3 rows of A, 1 column of output, 4 columns of B):
          C[i,3] = sum_j A[i,j] * B[j,3]
          This is a 3x4 x 4x1 multiplication.
          Schoolbook: 3*4*1 = 12 terms.
          
        But the directive says rank-4. Maybe it means something different.
        
        Actually wait - maybe the "3x1x4 strip" refers to the tensor with dimensions (3,1,4) meaning:
        3 rows of A, 1 column of output, 4 columns of B.
        The tensor is T[i, j, k] where i in {0,1,2}, j in {0,1,2,3}, k in {3} (only output col 3).
        But that's not a 3x1x4 tensor in the standard sense.
        
        I think the intended decomposition is:
        - BCL 3x3x3 for C[:,0:3] = A[:,0:3] * B[:,0:3] (but A is 3x4, B is 4x4)
        
        Hmm, this doesn't quite work because A has 4 columns but we only use 3.
        
        Let me reconsider the problem structure.
        A: 3x4, B: 4x4, C: 3x4
        C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
        
        We can split:
        C[:,0:3] = A[:,0:3] * B[:,0:3] + A[:,3:4] * B[:,3:4][:,0:3]
        C[:,3:4] = A[:,0:3] * B[:,0:3][:,3:4] + A[:,3:4] * B[:,3:4][:,3:4]
        
        This is getting complicated. Let me try a different split.
        
        Actually, I think the correct interpretation is:
        - Take the 3x3x3 sub-tensor using A[0:3,0:3] and B[0:3,0:3] -> C[0:3,0:3]
        - The remaining terms involve A[:,3] (4th column of A) and B[3,:] (4th row of B)
        
        C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k] + A[i,3]*B[3,k]
        
        The first sum for i,k in {0,1,2} is the 3x3x3 tensor (BCL rank 19).
        But we also need C[i,3] for i in {0,1,2} and C[3,k]... wait, C is only 3x4, so i is only 0,1,2.
        
        C[i,3] = sum_{j=0}^{2} A[i,j]*B[j,3] + A[i,3]*B[3,3] for i=0,1,2
        
        So we need additional terms for:
        1. C[i,3] for i=0,1,2: involves A[i,0:3]*B[0:3,3] and A[i,3]*B[3,3]
        2. The BCL only covers C[i,k] for i,k in {0,1,2} using A[i,0:3]*B[0:3,k]
        
        Additional terms needed:
        - A[i,3]*B[3,k] for i=0,1,2 and k=0,1,2,3: that's 4*3 = 12 terms (schoolbook)
        - A[i,0:3]*B[0:3,3] for i=0,1,2: that's 3*3 = 9 terms (schoolbook)
        - A[i,3]*B[3,3] is already counted above
        
        Total: 19 + 12 + 9 = 40. Not 23.
        
        The directive must mean something else. Let me re-read:
        "taking the rank-19 Bini-Capovani-De Falco-Ladisi decomposition of the 3x3x3 multiplication tensor, appending a rank-4 schoolbook decomposition for the remaining 3x1x4 strip"
        
        Maybe "3x1x4" means the tensor has shape (3,1,4) in some specific indexing.
        Or maybe the BCL decomposition is applied differently.
        
        Actually, I wonder if the BCL 3x3x3 can be applied to compute C[0:3, 0:3] = A[0:3, 0:4] * B[0:4, 0:3] by treating the 4th column specially.
        
        I'm going to take a different approach. Let me just implement the BCL 3x3x3 decomposition properly and combine it with the necessary extra terms, then verify.
        
        For now, let me use a known explicit BCL 3x3x3 decomposition.
        """
        
        # BCL 3x3x3 decomposition (19 terms)
        # T[i,j,k] = 1 if i==k
        # U index = i*3+j, V index = j*3+k, W index = k*3+i
        # 
        # Using the standard BCL construction:
        # We compute 19 products and combine them to get all 9 entries of C.
        
        # Let me use a verified explicit construction.
        # The 19 rank-1 terms (u, v, w) where each is a dict {index: coeff}:
        
        bcl_terms = []
        
        # I'll use the BCL algorithm as described in the literature.
        # The key insight is that we can compute 3x3 matrix multiplication with 19 multiplications.
        
        # Standard BCL 3x3x3 (from Bini et al. 1978):
        # Products:
        # p1 = (a11 + a12) * (b11 + b21)
        # p2 = (a11 + a21) * (b11 + b12)  
        # p3 = (a12 + a22) * (b12 + b22)
        # p4 = (a21 + a22) * (b21 + b22)
        # p5 = a11 * b11
        # p6 = a12 * b21
        # p7 = a21 * b12
        # p8 = a22 * b22
        # p9 = a13 * b31
        # p10 = a31 * b13
        # p11 = a33 * b33
        # p12 = (a13 + a33) * (b13 + b31)
        # p13 = a11 * b13
        # p14 = a13 * b11
        # p15 = a31 * b33
        # p16 = a33 * b31
        # p17 = (a11 + a33) * (b33 + b11)
        # p18 = a31 * b11
        # p19 = a13 * b33
        
        # Wait, I need to be careful with indexing. Let me use 0-based.
        # A[i,j], B[j,k], C[i,k]
        # V index for B[j,k] is j*3+k
        
        # Let me just hardcode a known correct BCL decomposition.
        # I'll verify it computationally.
        
        # Using 0-based indexing:
        # A[0,0]=a00, A[0,1]=a01, A[0,2]=a02
        # A[1,0]=a10, A[1,1]=a11, A[1,2]=a12
        # A[2,0]=a20, A[2,1]=a21, A[2,2]=a22
        # B[0,0]=b00, B[0,1]=b01, B[0,2]=b02
        # B[1,0]=b10, B[1,1]=b11, B[1,2]=b12
        # B[2,0]=b20, B[2,1]=b21, B[2,2]=b22
        
        # C[0,0] = a00*b00 + a01*b10 + a02*b20
        # C[0,1] = a00*b01 + a01*b11 + a02*b21
        # C[0,2] = a00*b02 + a01*b12 + a02*b22
        # C[1,0] = a10*b00 + a11*b10 + a12*b20
        # C[1,1] = a10*b01 + a11*b11 + a12*b21
        # C[1,2] = a10*b02 + a11*b12 + a12*b22
        # C[2,0] = a20*b00 + a21*b10 + a22*b20
        # C[2,1] = a20*b01 + a21*b11 + a22*b21
        # C[2,2] = a20*b02 + a21*b12 + a22*b22
        
        # BCL 19-term decomposition (verified from literature):
        # I'll use the explicit construction and verify it.
        
        # Term definitions: (u_entries, v_entries, w_entries)
        # u_entries: list of (u_index, coeff)
        # v_entries: list of (v_index, coeff)  
        # w_entries: list of (w_index, coeff)
        
        # U index = i*3+j, V index = j*3+k, W index = k*3+i
        
        bcl_terms = []
        
        # I'll use a computational search approach to find the BCL terms.
        # For now, let me use a known explicit construction.
        
        # After careful derivation, here are the 19 BCL terms:
        
        # Using the construction from Bini, Capovani, De Falco, Ladisi (1978)
        # with 0-based indexing:
        
        # p1 = (a00+a01)*(b00+b10)
        bcl_terms.append(
            {0:1, 1:1},   # A[0,0]+A[0,1]
            {0:1, 3:1},   # B[0,0]+B[1,0]
            {0:1}          # C[0,0]
        )
        
        # p2 = (a00+a02)*(b00+b20)
        bcl_terms.append(
            {0:1, 2:1},
            {0:1, 6:1},
            {0:1}
        )
        
        # p3 = (a01+a02)*(b10+b20)
        bcl_terms.append(
            {1:1, 2:1},
            {3:1, 6:1},
            {0:1}
        )
        
        # p4 = a00*b00
        bcl_terms.append(
            {0:1},
            {0:1},
            {0:1}
        )
        
        # p5 = a01*b10
        bcl_terms.append(
            {1:1},
            {3:1},
            {0:1}
        )
        
        # p6 = a02*b20
        bcl_terms.append(
            {2:1},
            {6:1},
            {0:1}
        )
        
        # p7 = (a10+a11)*(b00+b01)
        bcl_terms.append(
            {3:1, 4:1},
            {0:1, 1:1},
            {4:1}  # C[1,1]
        )
        
        # p8 = (a10+a12)*(b00+b02)
        bcl_terms.append(
            {3:1, 5:1},
            {0:1, 2:1},
            {4:1}
        )
        
        # p9 = (a11+a12)*(b01+b02)
        bcl_terms.append(
            {4:1, 5:1},
            {1:1, 2:1},
            {4:1}
        )
        
        # p10 = a10*b00
        bcl_terms.append(
            {3:1},
            {0:1},
            {4:1}
        )
        
        # p11 = a11*b01
        bcl_terms.append(
            {4:1},
            {1:1},
            {4:1}
        )
        
        # p12 = a12*b02
        bcl_terms.append(
            {5:1},
            {2:1},
            {4:1}
        )
        
        # p13 = (a20+a21)*(b00+b10)
        bcl_terms.append(
            {6:1, 7:1},
            {0:1, 3:1},
            {8:1}  # C[2,2]
        )
        
        # p14 = (a20+a22)*(b00+b20)
        bcl_terms.append(
            {6:1, 8:1},
            {0:1, 6:1},
            {8:1}
        )
        
        # p15 = (a21+a22)*(b10+b20)
        bcl_terms.append(
            {7:1, 8:1},
            {3:1, 6:1},
            {8:1}
        )
        
        # p16 = a20*b00
        bcl_terms.append(
            {6:1},
            {0:1},
            {8:1}
        )
        
        # p17 = a21*b10
        bcl_terms.append(
            {7:1},
            {3:1},
            {8:1}
        )
        
        # p18 = a22*b20
        bcl_terms.append(
            {8:1},
            {6:1},
            {8:1}
        )
        
        # p19 = ???
        # I only have 18 terms above and they don't cover all entries.
        # This approach is not working. Let me use a different method.
        
        return None  # This approach is flawed

    def build_rank23():
        """
        Build rank-23 decomposition using BCL 3x3x3 + extra terms.
        
        Strategy:
        - Use BCL rank-19 for the 3x3x3 core
        - Add terms for the 4th column/row interactions
        
        Actually, let me use a simpler but correct approach:
        - Strassen 2x2x2 for top-left 2x2 block: 7 terms
        - Schoolbook for remaining parts
        
        Or better: use the BCL 3x3x3 properly.
        
        Let me use a verified BCL construction.
        """
        
        # I'll use the explicit BCL 3x3x3 decomposition from the literature.
        # After research, here's a correct 19-term decomposition:
        
        # The BCL algorithm computes C = A*B (3x3) with 19 multiplications.
        # Using 0-based indexing, the 19 rank-1 tensors are:
        
        # I'll define them as (u_dict, v_dict, w_dict)
        # where u_dict maps A-index to coefficient, etc.
        
        # A index: i*3+j
        # B index: j*3+k  
        # C index: k*3+i
        
        bcl = []
        
        # Verified BCL 3x3x3 terms (from Bini et al. 1978, 0-based):
        
        # Group 1: For computing C[0,*] using A[0,*]
        # p1 = (a00+a01+a02)*(b00+b10+b20) -> partial C[0,0]
        # This is getting too complex. Let me use a different verified construction.
        
        # I'll use the "3x3x3 = 19" construction from the online literature.
        # One explicit construction:
        
        # Let me use the construction where we have:
        # 7 terms for a 2x2x2 Strassen on rows/cols {0,1}
        # Plus 12 more terms for the 3rd row/col interactions
        
        # Actually, 7 + 12 = 19. That works!
        
        # Strassen 2x2x2 on A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # This uses 7 multiplications.
        
        # Then we need to handle:
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
        # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
        
        # That's 15 more terms (schoolbook), total 7+15=22. Not 19.
        
        # The BCL algorithm is more clever. Let me just use a computational approach.
        # I'll set up the tensor and find a rank-19 decomposition using known methods.
        
        # For now, let me fall back to a simpler construction that I can verify.
        # I'll use Strassen for 2x2x2 blocks and schoolbook for the rest.
        
        # 3x4 x 4x4:
        # Split A into A_top (2x4) and A_bot (1x4)
        # Split B into B_left (4x3) and B_right (4x1)
        
        # C_top_left (2x3) = A_top * B_left: use 2x3x4 = ? 
        # This is getting complicated. Let me just use the Strassen+schoolbook approach
        # that gives rank 44 (current) and see if I can improve.
        
        # Actually, let me try a different approach: use the fact that 3x4x4 can be
        # decomposed as 3x3x4 + 3x1x4.
        # 3x3x4 = 3x3x3 + 3x3x1 = 19 + 9 = 28 (if BCL works)
        # 3x1x4 = 3x1x3 + 3x1x1 = 9 + 3 = 12
        # Total: 28 + 12 = 40. Still not 23.
        
        # The directive's claim of rank 23 may be based on a different decomposition strategy.
        # Let me just implement what I can verify and return the best I can do.
        
        # For now, return the rank-44 decomposition as fallback.
        return None

    def build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]

        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col

        col = add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
        col = add_strassen([2,3,6,7], [8,9,12,13], [0,3,1,4], col)
        col = add_strassen([0,1,4,5], [2,3,6,7], [6,9,7,10], col)
        col = add_strassen([2,3,6,7], [10,11,14,15], [6,9,7,10], col)

        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1

        return U, V, W

    def verify(U, V, W, n, m, p):
        U_int = np.round(np.array(U) * 2).astype(np.int64)
        V_int = np.round(np.array(V) * 2).astype(np.int64)
        W_int = np.round(np.array(W) * 2).astype(np.int64)

        T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 8

        T_actual = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        return np.array_equal(T_actual, T_expected)

    def fast_verify(U, V, W):
        """Faster verification using float arithmetic with tolerance."""
        U_f = np.array(U, dtype=np.float64)
        V_f = np.array(V, dtype=np.float64)
        W_f = np.array(W, dtype=np.float64)

        T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 1.0

        T_actual = np.einsum('ir,jr,kr->ijk', U_f, V_f, W_f)
        return np.allclose(T_actual, T_expected, atol=1e-9)

    # Try BCL-based rank-23 decomposition first
    # Using explicit BCL 3x3x3 + schoolbook extras
    
    def build_bcl_rank23():
        """
        Rank-23 decomposition: BCL 3x3x3 (rank 19) + 4 schoolbook terms.
        
        The key insight: we can compute C[0:3, 0:3] = A[0:3, 0:3] * B[0:3, 0:3] 
        with BCL (19 terms), and then handle the 4th column of B and 4th column of A
        with additional terms.
        
        But 19 + extra > 23 unless the extra is only 4 terms.
        
        Alternative interpretation: The 3x4x4 tensor can be seen as having a 
        3x3x3 sub-tensor plus a 3x1x4 "strip" that has rank 4.
        
        A 3x1x4 tensor T[i,j,k] where the "1" dimension means something specific.
        If it's T[i, j, 3] = A[i,j] * B[j,3] for the output column 3,
        then T[i,j,3] = delta(i,3)... no.
        
        Actually, I think the correct decomposition is:
        - BCL 3x3x3 for the core 3x3x3 sub-problem
        - The "3x1x4 strip" has rank 4, meaning 4 rank-1 terms suffice
        
        Let me just implement a correct decomposition and verify it.
        """
        
        rank = 23
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        
        # BCL 3x3x3 decomposition (19 terms)
        # Using explicit construction from Bini et al. (1978)
        # 
        # I'll use the following verified 19-term decomposition for 3x3x3:
        # T[i,j,k] = 1 if i==k
        # U_idx = i*3+j, V_idx = j*3+k, W_idx = k*3+i
        
        # The 19 rank-1 terms (u_entries, v_entries, w_entries):
        bcl_terms = [
            # p1
            ({0:1, 1:1}, {0:1, 3:1}, {0:1, 1:1, 4:1}),
            # p2  
            ({0:1, 2:1}, {0:1, 6:1}, {0:1, 2:1, 4:1}),
            # p3
            ({1:1, 2:1}, {3:1, 6:1}, {0:1, 1:1, 2:1}),
            # p4
            ({0:1}, {0:1}, {0:1}),
            # p5
            ({1:1}, {3:1}, {0:1}),
            # p6
            ({2:1}, {6:1}, {0:1}),
            # p7
            ({3:1, 4:1}, {0:1, 1:1}, {1:1, 4:1}),
            # p8
            ({3:1, 5:1}, {0:1, 2:1}, {2:1, 4:1}),
            # p9
            ({4:1, 5:1}, {1:1, 2:1}, {1:1, 2:1}),
            # p10
            ({3:1}, {0:1}, {1:1}),
            # p11
            ({4:1}, {1:1}, {4:1}),
            # p12
            ({5:1}, {2:1}, {2:1}),
            # p13
            ({6:1, 7:1}, {0:1, 3:1}, {0:1, 3:1, 8:1}),
            # p14
            ({6:1, 8:1}, {0:1, 6:1}, {0:1, 6:1, 8:1}),
            # p15
            ({7:1, 8:1}, {3:1, 6:1}, {3:1, 6:1}),
            # p16
            ({6:1}, {0:1}, {0:1}),
            # p17
            ({7:1}, {3:1}, {3:1}),
            # p18
            ({8:1}, {6:1}, {6:1}),
            # p19
            ({6:1, 7:1, 8:1}, {0:1, 3:1, 6:1}, {0:1, 3:1, 6:1, 8:1}),
        ]
        
        # This is a guess - I need to verify it. Let me use a different approach.
        # I'll construct the BCL decomposition computationally.
        
        # For now, let me use a simpler but verifiable approach.
        # I'll use Strassen for 2x2x2 blocks and optimize the rest.
        
        return None

    # Build the rank-44 decomposition as baseline
    U, V, W = build_rank44()
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # First verify the base decomposition is correct
    if not verify(U_np, V_np, W_np, n, m, p):
        rank = 48
        U = [[0]*rank for _ in range(dim_u)]
        V = [[0]*rank for _ in range(dim_v)]
        W = [[0]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank by checking linear dependence and removing redundant columns
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = 44

    # Iteratively try to remove columns
    removed_count = 0
    max_removal_attempts = 50

    for attempt in range(max_removal_attempts):
        if time.time() > deadline:
            break
        
        if current_rank <= 1:
            break

        M = np.vstack([current_U, current_V, current_W])
        
        try:
            s_vals = np.linalg.svd(M, full_matrices=False, compute_uv=False)
            tol = max(M.shape) * np.finfo(float).eps * (s_vals[0] if len(s_vals) > 0 else 1)
            num_rank = int(np.sum(s_vals > tol))
        except:
            num_rank = current_rank

        if num_rank >= current_rank:
            break

        found = False
        for r_idx in range(current_rank):
            if time.time() > deadline:
                break
            
            keep_indices = [i for i in range(current_rank) if i != r_idx]
            
            U_test = current_U[:, keep_indices]
            V_test = current_V[:, keep_indices]
            W_test = current_W[:, keep_indices]
            
            if verify(U_test, V_test, W_test, n, m, p):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                removed_count += 1
                found = True
                break
        
        if not found:
            break

    # Final verification
    if current_rank < 44:
        if verify(current_U, current_V, current_W, n, m, p):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}

    return {"U": U, "V": V, "W": W}
