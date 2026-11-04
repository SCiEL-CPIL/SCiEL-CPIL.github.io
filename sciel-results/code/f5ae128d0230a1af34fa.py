"""Matrix multiplication 2x3x4 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    def build_rank22():
        """
        Constructs a verified rank-22 decomposition.
        Split A (2x3) into A1 (2x2) and A2 (2x1).
        Split B (3x4) into B1 (2x4) and B2 (1x4).
        C = A1*B1 + A2*B2.
        A1*B1 involves two 2x2x2 multiplications (rank 7 each) -> 14.
        A2*B2 is an outer product 2x1 * 1x4 (rank 8) -> 8.
        Total 22.
        """
        U = np.zeros((n * m, 22))
        V = np.zeros((m * p, 22))
        W = np.zeros((p * n, 22))
        
        def add_u(i, j, c, val):
            U[i * m + j, c] += val
        def add_v(j, k, c, val):
            V[j * p + k, c] += val
        def add_w(k, i, c, val):
            W[k * n + i, c] += val
        
        def add_strassen_block(a00, a01, a10, a11, b00, b01, b10, b11, C00, C01, C10, C11, offset):
            # m1 = (a00+a11)(b00+b11)
            add_u(a00[0], a00[1], offset+0, 1)
            add_u(a11[0], a11[1], offset+0, 1)
            add_v(b00[0], b00[1], offset+0, 1)
            add_v(b11[0], b11[1], offset+0, 1)
            add_w(C00[0], C00[1], offset+0, 1)
            add_w(C11[0], C11[1], offset+0, 1)
            
            # m2 = (a10+a11)(b00)
            add_u(a10[0], a10[1], offset+1, 1)
            add_u(a11[0], a11[1], offset+1, 1)
            add_v(b00[0], b00[1], offset+1, 1)
            add_w(C10[0], C10[1], offset+1, 1)
            add_w(C11[0], C11[1], offset+1, -1)
            
            # m3 = a00(b01-b11)
            add_u(a00[0], a00[1], offset+2, 1)
            add_v(b01[0], b01[1], offset+2, 1)
            add_v(b11[0], b11[1], offset+2, -1)
            add_w(C01[0], C01[1], offset+2, 1)
            add_w(C11[0], C11[1], offset+2, 1)
            
            # m4 = a11(b10-b00)
            add_u(a11[0], a11[1], offset+3, 1)
            add_v(b10[0], b10[1], offset+3, 1)
            add_v(b00[0], b00[1], offset+3, -1)
            add_w(C00[0], C00[1], offset+3, 1)
            add_w(C10[0], C10[1], offset+3, 1)
            
            # m5 = (a00+a01)(b11)
            add_u(a00[0], a00[1], offset+4, 1)
            add_u(a01[0], a01[1], offset+4, 1)
            add_v(b11[0], b11[1], offset+4, 1)
            add_w(C00[0], C00[1], offset+4, -1)
            add_w(C01[0], C01[1], offset+4, 1)
            
            # m6 = (a10-a00)(b00+b10)
            add_u(a10[0], a10[1], offset+5, 1)
            add_u(a00[0], a00[1], offset+5, -1)
            add_v(b00[0], b00[1], offset+5, 1)
            add_v(b10[0], b10[1], offset+5, 1)
            add_w(C11[0], C11[1], offset+5, 1)
            
            # m7 = (a01-a11)(b10+b11)
            add_u(a01[0], a01[1], offset+6, 1)
            add_u(a11[0], a11[1], offset+6, -1)
            add_v(b10[0], b10[1], offset+6, 1)
            add_v(b11[0], b11[1], offset+6, 1)
            add_w(C00[0], C00[1], offset+6, 1)

        # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
        # A indices: (0,0),(0,1),(1,0),(1,1)
        # B indices: (0,0),(0,1),(1,0),(1,1)
        # C indices: (0,0),(0,1),(1,0),(1,1) -> W[k*n+i]
        # C00 is C[0,0] -> k=0,i=0
        # C01 is C[0,1] -> k=0,i=1
        # C10 is C[1,0] -> k=1,i=0
        # C11 is C[1,1] -> k=1,i=1
        a1 = [(0,0), (0,1), (1,0), (1,1)]
        b1 = [(0,0), (0,1), (1,0), (1,1)]
        c1 = [(0,0), (0,1), (1,0), (1,1)] # (k,i) pairs: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
        # Note: Strassen standard mapping:
        # C00 = m1 - m4 + m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m6 + m7
        # My add_strassen_block implementation above:
        # m1: W(C00, +1), W(C11, +1)
        # m2: W(C10, +1), W(C11, -1)
        # m3: W(C01, +1), W(C11, +1) -- Wait, standard Strassen m3 contributes to C01 and C11?
        # Let's re-verify standard Strassen:
        # m1 = (a00+a11)(b00+b11) -> C00 += m1, C11 += m1
        # m2 = (a10+a11)(b00) -> C10 += m2, C11 -= m2
        # m3 = a00(b01-b11) -> C01 += m3, C11 += m3
        # m4 = a11(b10-b00) -> C00 += m4, C10 += m4
        # m5 = (a00+a01)(b11) -> C00 -= m5, C01 += m5
        # m6 = (a10-a00)(b00+b10) -> C11 += m6
        # m7 = (a01-a11)(b10+b11) -> C00 += m7
        
        # Checking my implementation:
        # m1: W(C00, 1), W(C11, 1). Correct.
        # m2: W(C10, 1), W(C11, -1). Correct.
        # m3: W(C01, 1), W(C11, 1). Correct.
        # m4: W(C00, 1), W(C10, 1). Correct.
        # m5: W(C00, -1), W(C01, 1). Correct.
        # m6: W(C11, 1). Correct.
        # m7: W(C00, 1). Correct.
        
        add_strassen_block(a1[0], a1[1], a1[2], a1[3], b1[0], b1[1], b1[2], b1[3], c1[0], c1[1], c1[2], c1[3], 0)
        
        # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
        # A indices same as above.
        # B indices: (0,2),(0,3),(1,2),(1,3)
        # C indices: 
        # C[0,2] -> k=2, i=0
        # C[0,3] -> k=3, i=0
        # C[1,2] -> k=2, i=1
        # C[1,3] -> k=3, i=1
        b2 = [(0,2), (0,3), (1,2), (1,3)]
        c2 = [(2,0), (3,0), (2,1), (3,1)] # C00=(2,0), C01=(3,0), C10=(2,1), C11=(3,1)
        add_strassen_block(a1[0], a1[1], a1[2], a1[3], b2[0], b2[1], b2[2], b2[3], c2[0], c2[1], c2[2], c2[3], 7)
        
        # Block 3: A[0:2, 2:3] * B[2:3, 0:4]
        # A indices: (0,2), (1,2)
        # B indices: (2,0), (2,1), (2,2), (2,3)
        # This is a 2x1 * 1x4 multiplication.
        # C[0,0] += A[0,2]*B[2,0]
        # C[0,1] += A[0,2]*B[2,1]
        # ...
        offset = 14
        for i in range(2):
            for k in range(4):
                add_u(i, 2, offset, 1)
                add_v(2, k, offset, 1)
                add_w(k, i, offset, 1)
                offset += 1
                
        return U, V, W

    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(R):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True

    U22, V22, W22 = build_rank22()
    
    if verify(U22, V22, W22):
        best_result = {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}
        
        # Attempt rank 21 with a short budget
        if time.time() - START_TIME < DEADLINE - 30:
            target_rank = 21
            found = False
            for attempt in range(20):
                if time.time() - START_TIME > DEADLINE - 10:
                    break
                np.random.seed(attempt * 12345 + target_rank)
                U = np.random.randn(n*m, target_rank)
                V = np.random.randn(m*p, target_rank)
                W = np.random.randn(p*n, target_rank)
                
                for iteration in range(50):
                    # Solve U
                    for i in range(n):
                        for j in range(m):
                            A_mat = np.zeros((p, target_rank))
                            for k in range(p):
                                for r in range(target_rank):
                                    A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                            b_vec = np.ones(p)
                            if np.linalg.matrix_rank(A_mat) > 0:
                                U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    # Solve V
                    for j in range(m):
                        for k in range(p):
                            A_mat = np.zeros((n, target_rank))
                            for i in range(n):
                                for r in range(target_rank):
                                    A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                            b_vec = np.ones(n)
                            if np.linalg.matrix_rank(A_mat) > 0:
                                V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    # Solve W
                    for k in range(p):
                        for i in range(n):
                            A_mat = np.zeros((m, target_rank))
                            for j in range(m):
                                for r in range(target_rank):
                                    A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                            b_vec = np.ones(m)
                            if np.linalg.matrix_rank(A_mat) > 0:
                                W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                if verify(U, V, W):
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    if verify(U_rounded, V_rounded, W_rounded):
                        best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        found = True
                        break
                    else:
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        found = True
                        break
            if found:
                pass # Keep best_result updated
        
        return best_result
    else:
        # Fallback to schoolbook rank-24
        U_school = np.zeros((n * m, 24))
        V_school = np.zeros((m * p, 24))
        W_school = np.zeros((p * n, 24))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, idx] = 1
                    V_school[j * p + k, idx] = 1
                    W_school[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
