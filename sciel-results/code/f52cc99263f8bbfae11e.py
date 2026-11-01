"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def get_rank14_factors():
        """Return the known rank-14 Strassen decomposition factors."""
        R = 14
        U = np.zeros((4, R))
        V = np.zeros((8, R))
        W = np.zeros((8, R))

        # Left block (Strassen for 2x2 * 2x2 -> cols 0,1 of B)
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[5,0]=1
        W[0,0]=1; W[3,0]=1

        # r=1: M2 = (a10+a11)b00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[1,1]=1; W[3,1]=-1

        # r=2: M3 = a00(b01-b11)
        U[0,2]=1
        V[1,2]=1; V[5,2]=-1
        W[2,2]=1; W[3,2]=1

        # r=3: M4 = a11(b10-b00)
        U[3,3]=1
        V[4,3]=1; V[0,3]=-1
        W[0,3]=1; W[1,3]=1

        # r=4: M5 = (a00+a01)b11
        U[0,4]=1; U[1,4]=1
        V[5,4]=1
        W[0,4]=-1; W[2,4]=1

        # r=5: M6 = (a10-a00)(b00+b01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1

        # r=6: M7 = (a01-a11)(b10+b11)
        U[1,6]=1; U[3,6]=-1
        V[4,6]=1; V[5,6]=1
        W[0,6]=1

        # Right block (Strassen for 2x2 * 2x2 -> cols 2,3 of B)
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0,7]=1; U[3,7]=1
        V[2,7]=1; V[7,7]=1
        W[4,7]=1; W[7,7]=1

        # r=8: N2 = (a10+a11)b02
        U[2,8]=1; U[3,8]=1
        V[2,8]=1
        W[5,8]=1; W[7,8]=-1

        # r=9: N3 = a00(b03-b13)
        U[0,9]=1
        V[3,9]=1; V[7,9]=-1
        W[6,9]=1; W[7,9]=1

        # r=10: N4 = a11(b12-b02)
        U[3,10]=1
        V[6,10]=1; V[2,10]=-1
        W[4,10]=1; W[5,10]=1

        # r=11: N5 = (a00+a01)b13
        U[0,11]=1; U[1,11]=1
        V[7,11]=1
        W[4,11]=-1; W[6,11]=1

        # r=12: N6 = (a10-a00)(b02+b03)
        U[2,12]=1; U[0,12]=-1
        V[2,12]=1; V[3,12]=1
        W[7,12]=1

        # r=13: N7 = (a01-a11)(b12+b13)
        U[1,13]=1; U[3,13]=-1
        V[6,13]=1; V[7,13]=1
        W[4,13]=1

        return U, V, W

    def try_peeling_rank13():
        """
        Partial peeling: For each of the 14 rank-1 terms in the Strassen decomposition,
        check if it can be expressed as a linear combination of the other 13 terms.
        If so, the total tensor can be written with only 13 terms.
        
        Key insight: If t_k = sum_{j != k} c_j * t_j, then
        T_total = sum_{j != k} (1 + c_j) * t_j  (if we're replacing t_k)
        Wait, that's not right. Let me think again.
        
        T_total = sum_{all j} t_j = t_k + sum_{j != k} t_j
        If t_k = sum_{j != k} d_j * t_j, then
        T_total = sum_{j != k} t_j + sum_{j != k} d_j * t_j = sum_{j != k} (1 + d_j) * t_j
        This is still 13 rank-1 terms! Each (1+d_j)*t_j is still rank-1.
        
        So we need: t_k in span{t_j : j != k}
        """
        if time.time() > deadline - 30:
            return None

        U14, V14, W14 = get_rank14_factors()
        R14 = 14

        # Flatten each rank-1 tensor
        # Tensor shape: (4, 8, 8) -> flattened to (256,)
        tensors = np.zeros((R14, 4*8*8))
        for r in range(R14):
            tensors[r] = np.einsum('i,j,k->ijk', U14[:,r], V14[:,r], W14[:,r]).flatten()

        # For each k, check if tensors[k] is in the span of {tensors[j] : j != k}
        for k in range(R14):
            if time.time() > deadline - 15:
                break

            # Build the matrix of other 13 tensors
            other_indices = [j for j in range(R14) if j != k]
            A = tensors[other_indices]  # shape (13, 256)
            b = tensors[k]  # shape (256,)

            # Check if b is in the column space of A^T (i.e., row space of A)
            # Equivalently: solve A^T * x = b
            # A^T has shape (256, 13), b has shape (256,)
            # This is an underdetermined system if 256 > 13, which it is.
            # We need to check if b is in the column space of A^T.
            # Equivalent to: rank([A^T | b]) == rank(A^T)
            # Or: the residual of the least-squares solution A^T x = b is zero.

            # Use least squares: minimize ||A^T x - b||
            # A^T is (256, 13), so we solve for x in R^13
            x, residuals, rank_A, sv = np.linalg.lstsq(A.T, b, rcond=1e-10)

            # Check if the residual is essentially zero
            residual = A.T @ x - b
            residual_norm = np.linalg.norm(residual)

            if residual_norm < 1e-8:
                # t_k is in the span of the other 13 terms
                # T_total = sum_{j != k} (1 + d_j) * t_j where d_j are the coefficients
                # But wait: t_k = sum_{j != k} x_j * t_j (where x_j corresponds to index j)
                # So T_total = t_k + sum_{j != k} t_j = sum_{j != k} x_j * t_j + sum_{j != k} t_j
                #         = sum_{j != k} (1 + x_j) * t_j
                # Each (1+x_j)*t_j is rank-1, so we have a rank-13 decomposition!

                # Build the new factors
                U13 = np.zeros((4, 13))
                V13 = np.zeros((8, 13))
                W13 = np.zeros((8, 13))

                for new_r, old_j in enumerate(other_indices):
                    coeff = 1.0 + x[new_r]
                    U13[:, new_r] = coeff * U14[:, old_j]
                    V13[:, new_r] = coeff * V14[:, old_j]
                    W13[:, new_r] = coeff * W14[:, old_j]

                # Verify
                T_check = np.einsum('ir,jr,kr->ijk', U13, V13, W13)
                T_mult = np.zeros((4, 8, 8))
                for i in range(n):
                    for j in range(m):
                        for k2 in range(p):
                            T_mult[i*m+j, j*p+k2, k2*n+i] = 1.0

                if np.linalg.norm(T_check - T_mult) < 1e-7:
                    U_list = [[float(U13[i,r]) for r in range(13)] for i in range(4)]
                    V_list = [[float(V13[j,r]) for r in range(13)] for j in range(8)]
                    W_list = [[float(W13[k2,r]) for r in range(13)] for k2 in range(8)]
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}

        return None

    def try_direct_rank13_search():
        """
        Alternative: try to find a rank-13 decomposition by exploiting the structure.
        
        For 2x2 * 2x4, we can think of it as computing C = A*B where:
        A = [[a00, a01], [a10, a11]]  (2x2)
        B = [[b00, b01, b02, b03], [b10, b11, b12, b13]]  (2x4)
        C = [[c00, c01, c02, c03], [c10, c11, c12, c13]]  (2x4)
        
        The key structural observation: the left 2x2 block (cols 0,1) and right 2x2 block
        (cols 2,3) both use the same A. Can we share one multiplication?
        
        In standard Strassen for 2x2*2x2, we use 7 multiplications. For two independent
        2x2*2x2 blocks, that's 14. But if we can share one, we get 13.
        
        The shared term must be of the form:
        u * v * w where:
        - u is a linear combination of {a00, a01, a10, a11}
        - v is a linear combination of {b00, b01, b02, b03, b10, b11, b12, b13}
        - w is a linear combination of {c00, c01, c02, c03, c10, c11, c12, c13}
        
        And this term must contribute to BOTH the left and right output blocks.
        
        Let's try: the term (a00+a11)(b00+b11+b02+b13) would contribute to both blocks,
        but we need to check if the remaining 12 terms can complete the computation.
        
        Actually, let's try a more systematic approach: use the fact that
        M1_left = (a00+a11)(b00+b11) and M1_right = (a00+a11)(b02+b13)
        share the factor (a00+a11).
        
        We can compute (a00+a11) once, then multiply by (b00+b11) and (b02+b13).
        But that's still two multiplications involving (a00+a11).
        
        However, if we use:
        M_shared = (a00+a11)(b00+b11+b02+b13)
        Then we need:
        M1_left = M_shared - (a00+a11)(b02+b13)  -- but this is not rank-1!
        
        So simple sharing doesn't work. Let's try a different approach.
        
        What if we use:
        M1 = (a00+a11)(b00+b11)  -- for left block
        N1 = (a00+a11)(b02+b13)  -- for right block
        
        These are two separate rank-1 terms. Can we replace them with a single
        rank-1 term plus adjustments to other terms?
        
        This is getting complex. Let's try a numerical search instead.
        """
        if time.time() > deadline - 20:
            return None

        # Build the true multiplication tensor
        T_mult = np.zeros((4, 8, 8))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_mult[i*m+j, j*p+k, k*n+i] = 1.0

        # Try random initializations for rank 13
        best_norm = np.inf
        best_result = None

        np.random.seed(123)
        for attempt in range(50):
            if time.time() > deadline - 10:
                break

            U = np.random.randn(4, 13) * 0.5
            V = np.random.randn(8, 13) * 0.5
            W = np.random.randn(8, 13) * 0.5

            # Gradient descent
            lr = 0.05
            for step in range(200):
                if time.time() > deadline - 5:
                    break

                T_curr = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = T_curr - T_mult
                norm = np.linalg.norm(residual)

                if norm < best_norm:
                    best_norm = norm
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()

                if norm < 1e-8:
                    break

                dU = np.einsum('ijk,jr,kr->ir', residual, V, W)
                dV = np.einsum('ijk,ir,kr->jr', residual, U, W)
                dW = np.einsum('ijk,ir,jr->kr', residual, U, V)

                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

        if best_norm < 1e-7:
            U_list = [[float(best_U[i,r]) for r in range(13)] for i in range(4)]
            V_list = [[float(best_V[j,r]) for r in range(13)] for j in range(8)]
            W_list = [[float(best_W[k,r]) for r in range(13)] for k in range(8)]
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}

        return None

    def try_strassen_rectangular():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]

        # Left block
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1

        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1

        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1

        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1

        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1

        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1

        # Right block
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1

        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1

        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1

        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1

        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1

        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1

        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1

        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            return None

    # Try peeling first (fast, deterministic)
    result = try_peeling_rank13()
    if result is not None:
        return result

    # Try direct search (slower, probabilistic)
    result = try_direct_rank13_search()
    if result is not None:
        return result

    # Fall back to rank-14
    result = try_strassen_rectangular()
    if result is not None:
        return result

    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
