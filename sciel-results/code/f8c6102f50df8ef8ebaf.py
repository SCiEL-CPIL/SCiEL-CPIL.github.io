"""Combined program for matrix multiplication 2x3x4 tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 140  # Leave 10s safety margin before 150s kill

    def verify_np(U, V, W, n, m, p, rank, tol=1e-9):
        """Vectorized verification."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_exact(U, V, W, n, m, p, rank):
        """Exact verification using fractions."""
        # Convert to Fractions, assuming entries are close to half-integers
        Uf = [[Fraction(int(round(x * 2)), 2) for x in row] for row in U]
        Vf = [[Fraction(int(round(x * 2)), 2) for x in row] for row in V]
        Wf = [[Fraction(int(round(x * 2)), 2) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(Uf[i*m+j][t] * Vf[j*p+k][t] * Wf[k*n+i][t] for t in range(rank))
                    if s != 1:
                        return False
        return True

    def build_rank_22():
        """Constructs the known rank 22 decomposition."""
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        # A indices: 00:0, 01:1, 10:3, 11:4
        # B indices: 00:0, 01:1, 10:4, 11:5
        # C indices (W): 00:0, 01:2, 10:1, 11:3
        
        # M1: (A00+A11)(B00+B11) -> C00+C11
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        # M2: (A10+A11)B00 -> C10-C11
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        # M3: A00(B01-B11) -> C01+C11
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        # M4: A11(B10-B00) -> C00+C10
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        # M5: (A00+A01)B11 -> -C00+C01
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        # B indices: 02:2, 03:3, 12:6, 13:7
        # C indices (W): 02:4, 03:6, 12:5, 13:7
        
        # M1': (A00+A11)(B02+B13) -> C02+C13
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        # M2': (A10+A11)B02 -> C12-C13
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        # M3': A00(B03-B13) -> C03+C13
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        # M4': A11(B12-B02) -> C02+C12
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        # M5': (A00+A01)B13 -> -C02+C03
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        # M6': (A10-A00)(B02+B03) -> C13
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        # M7': (A01-A11)(B12+B13) -> C02
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        # A indices: 02:2, 12:5
        # B indices: 20:8, 21:9, 22:10, 23:11
        # C indices: 00:0, 01:2, 02:4, 03:6 for i=0
        #            10:1, 11:3, 12:5, 13:7 for i=1
        
        # A02 * B20 -> C00
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        # A02 * B21 -> C01
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        # A02 * B22 -> C02
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        # A02 * B23 -> C03
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        # A12 * B20 -> C10
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        # A12 * B21 -> C11
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        # A12 * B22 -> C12
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        # A12 * B23 -> C13
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    # 1. Construct and verify the known Rank 22 solution
    U22, V22, W22, rank22 = build_rank_22()
    U22_np = np.array(U22, dtype=float)
    V22_np = np.array(V22, dtype=float)
    W22_np = np.array(W22, dtype=float)
    
    best_result = None
    best_rank = rank22
    
    if verify_np(U22_np, V22_np, W22_np, n, m, p, rank22):
        if verify_exact(U22_np, V22_np, W22_np, n, m, p, rank22):
            best_result = {"U": U22, "V": V22, "W": W22}
        else:
            # Fallback to float representation if exact fraction check fails due to precision issues in conversion
            # But the entries are integers, so this should pass.
            best_result = {"U": [[float(x) for x in row] for row in U22], 
                           "V": [[float(x) for x in row] for row in V22], 
                           "W": [[float(x) for x in row] for row in W22]}
    else:
        # If verification fails (should not happen), fall back to schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
        return best_result

    # 2. Attempt to find a lower rank using ALS and rational polishing
    def try_als(rank, max_iter=500, tol=1e-14, seed=42, init_method='random'):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        if init_method == 'random':
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
        else:  # 'structured' based on rank 22 sparsified
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            # Use first 'rank' columns of the rank 22 solution as init
            U[:, :min(rank, rank22)] = U22_np[:, :min(rank, rank22)]
            V[:, :min(rank, rank22)] = V22_np[:, :min(rank, rank22)]
            W[:, :min(rank, rank22)] = W22_np[:, :min(rank, rank22)]
            U += rng.randn(n * m, rank) * 0.1
            V += rng.randn(m * p, rank) * 0.1
            W += rng.randn(p * n, rank) * 0.1

        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                if verify_np(U, V, W, n, m, p, rank, tol=1e-10):
                    return U, V, W
        
        if verify_np(U, V, W, n, m, p, rank, tol=1e-10):
            return U, V, W
        
        return None

    def rational_polish(U, V, W, rank, max_iters=50, seed=0):
        """Try to find a rational (half-integer) decomposition near the given solution."""
        rng = np.random.RandomState(seed)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_np(U_r, V_r, W_r, n, m, p, rank, tol=1e-8):
                if verify_exact(U_r, V_r, W_r, n, m, p, rank):
                    return U_r, V_r, W_r
            
            # Perturb and retry
            scale = max(0.1, 1.0 - it / max_iters)
            U = U_r + rng.randn(n * m, rank) * 0.05 * scale
            V = V_r + rng.randn(m * p, rank) * 0.05 * scale
            W = W_r + rng.randn(p * n, rank) * 0.05 * scale
            
        return None

    # Try ranks 20 and 21
    for target_rank in [20, 21]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for init_method in ['structured', 'random']:
            if time.time() > deadline - 5:
                break
            
            for seed in range(50):
                if time.time() > deadline - 3:
                    break
                
                result = try_als(target_rank, max_iter=200, tol=1e-12, seed=seed, init_method=init_method)
                if result is not None:
                    U, V, W = result
                    U_r, V_r, W_r = rational_polish(U, V, W, target_rank, max_iters=20, seed=seed)
                    
                    if U_r is not None:
                        U_list = [[float(x) for x in row] for row in U_r]
                        V_list = [[float(x) for x in row] for row in V_r]
                        W_list = [[float(x) for x in row] for row in W_r]
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
