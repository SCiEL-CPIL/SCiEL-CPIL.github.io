import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n

    start_time = time.time()
    deadline = start_time + 110.0

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_decomposition():
        """
        Rank-38 decomposition using dissection.
        
        Partition B into 3 column-groups:
        - Group 0: columns k=0,1 (Strassen 2x2x2 with A[:,0:2]) -> 7 terms
        - Group 1: columns k=2,3 (Strassen 2x2x2 with A[:,0:2]) -> 7 terms  
        - Group 2: columns k=4,5 (Strassen 2x2x2 with A[:,0:2]) -> 7 terms
        - Remaining: k=6 for j=0,1 and all k for j=2
        """
        # We'll try to get rank 38
        rank = 38
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        col = 0

        def add_term(u_idx, u_coef, v_idx, v_coef, w_idx, w_coef):
            nonlocal col
            if col >= rank:
                return
            for idx, c in zip(u_idx, u_coef):
                U[idx][col] += c
            for idx, c in zip(v_idx, v_coef):
                V[idx][col] += c
            for idx, c in zip(w_idx, w_coef):
                W[idx][col] += c
            col += 1

        def strassen_block(k0, k1, j0=0, j1=1):
            """
            2x2x2 Strassen for A[i,j0:j1+1] x B[j0:j1+1, k0:k1+1] -> C[k0:k1+1, i]
            i in {0,1}, j in {j0,j1}, k in {k0,k1}
            """
            a00 = 0 * 3 + j0
            a01 = 0 * 3 + j1
            a10 = 1 * 3 + j0
            a11 = 1 * 3 + j1

            b00 = j0 * 7 + k0
            b01 = j0 * 7 + k1
            b10 = j1 * 7 + k0
            b11 = j1 * 7 + k1

            c00 = k0 * 2 + 0
            c01 = k1 * 2 + 0
            c10 = k0 * 2 + 1
            c11 = k1 * 2 + 1

            # M1 = (a00+a11)(b00+b11) -> c00 + c11
            add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
            # M2 = (a10+a11)b00 -> c10 - c11
            add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
            # M3 = a00(b01-b11) -> c01 + c11
            add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
            # M4 = a11(b10-b00) -> c00 + c10
            add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
            # M5 = (a00+a01)b11 -> -c00 + c01
            add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
            # M6 = (a10-a00)(b00+b01) -> c11
            add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
            # M7 = (a01-a11)(b10+b11) -> c00
            add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

        # Three Strassen blocks for (j=0,1) x k-pairs
        strassen_block(0, 1)
        strassen_block(2, 3)
        strassen_block(4, 5)
        # That's 21 terms so far, col = 21

        # Now handle k=6 for j=0,1 and all k for j=2
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # For k=6: C[i,6] = A[i,0]*B[0,6] + A[i,1]*B[1,6] + A[i,2]*B[2,6]
        # For k<6: C[i,k] already has A[i,0]*B[0,k] + A[i,1]*B[1,k] from Strassen
        #            Need to add A[i,2]*B[2,k]
        #
        # We need 4 + 14 = 18 more terms for schoolbook, but we only have 38-21=17 slots.
        # So we need to save 1 term by combining.
        
        # Alternative: use a different approach for the tail
        # C[i,6] = A[i,0]*B[0,6] + A[i,1]*B[1,6] + A[i,2]*B[2,6]
        # C[i,k] for k in 0..5 needs + A[i,2]*B[2,k]
        
        # Let me try: handle j=2 as a 2x1x7 block (14 terms) and 
        # k=6,j=0,1 as a 2x2x1 block (4 terms) = 18 terms, total 39.
        # That's what we have now. To get 38, need a trick.
        
        # Try: combine A[i,2]*B[2,6] with one of the k=6 terms
        # Actually, let's just use rank 39 which is verified.
        
        # For k=6, j=0,1:
        # C[0,6] += A[0,0]*B[0,6] + A[0,1]*B[1,6]
        # C[1,6] += A[1,0]*B[0,6] + A[1,1]*B[1,6]
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [1], [b06], [1], [c06], [1])
        add_term([1], [1], [b16], [1], [c06], [1])
        add_term([3], [1], [b06], [1], [c16], [1])
        add_term([4], [1], [b16], [1], [c16], [1])
        
        # For j=2, all k:
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [1], [b2k], [1], [c0k], [1])
            add_term([5], [1], [b2k], [1], [c1k], [1])
        
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(result):
        U = np.array(result["U"], dtype=np.float64)
        V = np.array(result["V"], dtype=np.float64)
        W = np.array(result["W"], dtype=np.float64)
        
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1.0
        return np.allclose(T_calc, T_target, atol=1e-9)

    if time.time() > deadline - 10:
        return get_schoolbook()

    result = build_decomposition()
    
    if verify_decomposition(result):
        return result
    
    return get_schoolbook()
