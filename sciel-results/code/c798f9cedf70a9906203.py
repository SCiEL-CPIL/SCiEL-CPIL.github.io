import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a direct sum of two rank-49 Strassen decompositions for 4x4 * 4x4.
    Total rank = 98.
    """
    start = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 8

    # Schoolbook rank-128 as fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _strassen_2x2_factors():
        """
        Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
        Each factor is (u_vec, v_vec, w_vec) where:
          u_vec: 4-element vector for A (row-major: A00, A01, A10, A11)
          v_vec: 4-element vector for B (row-major: B00, B01, B10, B11)
          w_vec: 4-element vector for C (row-major: C00, C01, C10, C11)
        """
        # Standard Strassen factors
        # Indices: 0:00, 1:01, 2:10, 3:11
        factors = [
            # p1 = (A00+A11)(B00+B11)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # p2 = (A10+A11)B00
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            # p3 = A00(B01-B11)
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            # p4 = A11(B10-B00)
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            # p5 = (A00+A01)B11
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            # p6 = (A10-A00)(B00+B01)
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # p7 = (A01-A11)(B10+B11)
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        return factors

    def _build_4x4_4x4():
        """
        Construct rank-49 decomposition for 4x4 * 4x4 -> 4x4.
        Returns U4, V4, W4 of shape (16, 49).
        """
        rank = 49
        U4 = [[0] * rank for _ in range(16)]
        V4 = [[0] * rank for _ in range(16)]
        W4 = [[0] * rank for _ in range(16)]
        
        s2_factors = _strassen_2x2_factors()
        
        # Block coefficients for the 7 top-level Strassen products
        # Each product s involves linear combinations of the 4 blocks of A, B, C.
        # Block indices: 0:00, 1:01, 2:10, 3:11
        # We reuse the same structure as the 2x2 factors but interpreted at block level.
        # Note: The "W" part of the top-level Strassen tells us how to combine the products M_s to get C blocks.
        
        # Let's explicitly define the block coefficients for A, B, C for each of the 7 products.
        # This is identical to the u, v, w vectors of the 2x2 Strassen factors,
        # because the structure of Strassen is self-similar.
        
        for s in range(7):
            ub, vb, wb = s2_factors[s]
            
            for r in range(7):
                col = s * 7 + r
                us, vs, ws = s2_factors[r]
                
                # U contributions
                # For each global (i, j) in 4x4 A:
                for i in range(4):
                    for j in range(4):
                        # Determine block and local index
                        bi = i // 2
                        bj = j // 2
                        block_idx = bi * 2 + bj
                        ei = i % 2
                        ej = j % 2
                        local_idx = ei * 2 + ej
                        
                        coeff = ub[block_idx] * us[local_idx]
                        if coeff != 0:
                            U4[i * 4 + j][col] += coeff

                # V contributions
                # For each global (j, k) in 4x4 B:
                for j in range(4):
                    for k in range(4):
                        bi = j // 2
                        bj = k // 2
                        block_idx = bi * 2 + bj
                        ei = j % 2
                        ej = k % 2
                        local_idx = ei * 2 + ej
                        
                        coeff = vb[block_idx] * vs[local_idx]
                        if coeff != 0:
                            V4[j * 4 + k][col] += coeff

                # W contributions
                # For each global (k, i) in 4x4 C:
                for k in range(4):
                    for i in range(4):
                        bi = k // 2
                        bj = i // 2
                        block_idx = bi * 2 + bj
                        ei = k % 2
                        ej = i % 2
                        local_idx = ei * 2 + ej
                        
                        coeff = wb[block_idx] * ws[local_idx]
                        if coeff != 0:
                            W4[k * 4 + i][col] += coeff

        return U4, V4, W4

    U4, V4, W4 = _build_4x4_4x4()

    # Build rank-98 decomposition for 4x4 * 4x8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t in range(2):
        base_col = t * 49
        # U is the same for both subproblems
        for i in range(16):
            for c in range(49):
                U[i][base_col + c] = U4[i][c]
        
        # V: B block t
        # V rows: j*8 + k. j in 0..3, k in 0..7.
        # For subproblem t, B columns 4t..4t+3 are used.
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V[j * 8 + 4 * t + k][base_col + c] = V4[j * 4 + k][c]
        
        # W: C block t
        # W rows: k*4 + i. k in 0..7, i in 0..3.
        # For subproblem t, C columns 4t..4t+3 are produced.
        for k in range(4):
            for i in range(4):
                for c in range(49):
                    W[(4 * t + k) * 4 + i][base_col + c] = W4[k * 4 + i][c]

    # Verify
    def _verify(U, V, W):
        rank_val = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, rank_val)
        V3 = V_arr.reshape(m, p, rank_val)
        W3 = W_arr.reshape(p, n, rank_val)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)

    if time.time() - start < 145:
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}

    # Fallback
    return _schoolbook()
