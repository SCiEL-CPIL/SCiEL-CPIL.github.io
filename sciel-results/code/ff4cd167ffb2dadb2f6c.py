import time
import numpy as np
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0


def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7

    def elapsed():
        return time.time() - start_time

    def check_valid(U, V, W, tol=1e-8):
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2_2x2():
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        # p1 = (a11+a22)(b11+b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # p2 = (a21+a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # p3 = a11(b12-b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # p4 = a22(b21-b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # p5 = (a11+a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # p6 = (a21-a11)(b11+b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # p7 = (a12-a22)(b21+b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def kronecker_decompose_4x4x4():
        U2, V2, W2 = build_strassen_2x2_2x2()
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                if val != 0:
                                    U4[i * 4 + j][r] += val
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                if val != 0:
                                    V4[j * 4 + k][r] += val
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                                if val != 0:
                                    W4[k * 4 + i][r] += val
        return U4, V4, W4

    def optimize_2x2x3(start_rank=12, target_rank=8, max_iters=500, rng=None):
        """
        Optimizes a 2x2x3 tensor decomposition to reduce rank.
        Returns U, V, W as lists of lists, or None if fails.
        """
        if rng is None:
            rng = random.Random(42)
        
        # Initialize with schoolbook rank 12
        r = start_rank
        U = np.zeros((4, r))
        V = np.zeros((6, r))
        W = np.zeros((6, r))
        
        # Schoolbook init for 2x2x3
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[i * 2 + j, col] = 1
                    V[j * 3 + k, col] = 1
                    W[k * 2 + i, col] = 1
                    col += 1
        
        # ALS Optimization
        for it in range(max_iters):
            # Update U
            for c in range(r):
                # V[:, c] and W[:, c] are fixed
                # Residual for each (i,j,k)
                # We want U[i,j,c] such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
                # This is a linear system for U[:, c]
                # Let A_c be the matrix of coefficients for U[:, c]
                # Actually, for fixed c, U[:, c] is a vector of size 4.
                # The contribution of term c to the tensor is U[:, c] \otimes V[:, c] \otimes W[:, c]
                # We want to minimize || T - sum_{c'} U[:, c'] \otimes V[:, c'] \otimes W[:, c'] ||
                # This is not simple least squares on U alone because V and W are coupled.
                # Standard ALS: Fix V, W. Solve for U.
                # For each (i,j), the value is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
                # This is not a standard matrix multiplication.
                
                # Let's use gradient descent step for simplicity and speed
                pass
                
        # Since ALS is complex to implement correctly for 3-way tensors quickly,
        # we will use a simpler "Random Merge" but much faster version.
        # Or just return the schoolbook if optimization fails.
        
        # Actually, let's just try to drop columns and re-solve with least squares.
        # This is what the current code does, but we can make it faster by using
        # numpy operations instead of python loops in the Jacobian.
        
        def residual_vec(U, V, W):
            res = np.zeros(12)
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = 0.0
                        for c in range(r):
                            s += U[i * 2 + j, c] * V[j * 3 + k, c] * W[k * 2 + i, c]
                        res[(i * 2 + j) * 3 + k] = s - 1.0
            return res

        def build_jacobian(U, V, W):
            J = np.zeros((12, 4 * r + 6 * r + 6 * r))
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        row = (i * 2 + j) * 3 + k
                        for c in range(r):
                            uv = V[j * 3 + k, c] * W[k * 2 + i, c]
                            vw = U[i * 2 + j, c] * W[k * 2 + i, c]
                            wu = U[i * 2 + j, c] * V[j * 3 + k, c]
                            J[row, i * 2 * r + j * r + c] = uv
                            J[row, 4 * r + j * 3 * r + k * r + c] = vw
                            J[row, 4 * r + 6 * r + k * 2 * r + i * r + c] = wu
            return J

        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_res = np.linalg.norm(residual_vec(U, V, W))
        
        for _ in range(50):
            if r <= target_rank:
                break
            # Try to drop 1 column
            cols = list(range(r))
            rng.shuffle(cols)
            for dc in cols:
                new_r = r - 1
                kept = [c for c in range(r) if c != dc]
                U_new = U[:, kept]
                V_new = V[:, kept]
                W_new = W[:, kept]
                
                res = residual_vec(U_new, V_new, W_new)
                if np.linalg.norm(res) < 1e-6:
                    U, V, W = U_new, V_new, W_new
                    r = new_r
                    break
                J = build_jacobian(U_new, V_new, W_new)
                try:
                    delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
                    x_new = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()]) + delta
                    U2, V2, W2 = x_new[:4*new_r].reshape(4, new_r), x_new[4*new_r:4*new_r+6*new_r].reshape(6, new_r), x_new[4*new_r+6*new_r:].reshape(6, new_r)
                    res2 = residual_vec(U2, V2, W2)
                    if np.linalg.norm(res2) < 1e-6:
                        U, V, W = U2, V2, W2
                        r = new_r
                        break
                except:
                    pass
            if r == target_rank:
                break
                
        if r == target_rank and np.linalg.norm(residual_vec(U, V, W)) < 1e-6:
            return U.tolist(), V.tolist(), W.tolist()
        return None

    def build_4x4x7_decomposition():
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        
        # Build 4x4x3 part using 4 blocks of 2x2x3
        # A is 4x4, B is 4x3.
        # Split A into 2x2 blocks: A00, A01, A10, A11
        # Split B into 2x3 blocks: B00, B01, B10, B11 (rows 0-1, 2-3)
        # C is 4x3, split into 2x3 blocks: C00, C01, C10, C11
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        
        # This decomposition is not a direct sum of independent 2x2x3 tensors
        # because A00 appears in both C00 and C01.
        # So we cannot just use 4 independent 2x2x3 decompositions.
        # We must use the full 4x4x3 tensor.
        
        # Let's stick to the schoolbook for the 3 columns for now, 
        # but try to optimize the whole 4x4x3 tensor.
        
        r3 = 48
        U3 = np.zeros((16, r3))
        V3 = np.zeros((28, r3))
        W3 = np.zeros((28, r3))
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U3[i * 4 + j, col] = 1
                    V3[j * 7 + 4 + k, col] = 1 # B indices: j (0-3), k (4-6) -> j*7 + (4+k)
                    W3[(4 + k) * 4 + i, col] = 1 # C indices: k (4-6), i (0-3) -> (4+k)*4 + i
                    col += 1
                    
        # Combine
        r_total = r4 + r3
        U = np.zeros((16, r_total))
        V = np.zeros((28, r_total))
        W = np.zeros((28, r_total))
        
        U[:, :r4] = U4
        V[:, :r4] = V4
        W[:, :r4] = W4
        
        U[:, r4:] = U3
        V[:, r4:] = V3
        W[:, r4:] = W3
        
        return U.tolist(), V.tolist(), W.tolist()

    # Main flow
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p

    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_decomposition()
            r_kron = len(U_kron[0])
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception:
            pass

    return best_result
