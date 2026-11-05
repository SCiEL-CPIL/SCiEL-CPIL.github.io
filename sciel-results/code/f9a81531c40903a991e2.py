import numpy as np
import time
from itertools import product, combinations
from fractions import Fraction
import math

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    def get_tensor_error(U, V, W, n, m, p, rank):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        S = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        err = np.sum((S - 1.0)**2)
        return err, S

    def verify(U, V, W, n, m, p, rank, tol=1e-8):
        err, S = get_tensor_error(U, V, W, n, m, p, rank)
        return err < tol

    def build_rank_22():
        """Construct a rank-22 decomposition using Strassen blocks + outer products."""
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
        
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W

    def verify_lists(U, V, W, rank):
        """Verify decomposition with list-of-lists format."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank_als(rank, seed, max_iter=100):
        """Alternating Least Squares with half-integer rounding"""
        if time.time() > deadline - 15:
            return None
            
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        best_err = float('inf')
        best_state = None
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V3[j, k, :] * W3[k, i, :]
                    b = np.ones(p)
                    try:
                        U3[i, j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        U3[i, j, :] = rng.randn(rank) * 0.1
            U = U3.reshape(n*m, rank)
            
            # Solve for V
            V3 = V.reshape(m, p, rank)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U3[i, j, :] * W3[k, i, :]
                    b = np.ones(n)
                    try:
                        V3[j, k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        V3[j, k, :] = rng.randn(rank) * 0.1
            V = V3.reshape(m*p, rank)
            
            # Solve for W
            W3 = W.reshape(p, n, rank)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U3[i, j, :] * V3[j, k, :]
                    b = np.ones(m)
                    try:
                        W3[k, i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        W3[k, i, :] = rng.randn(rank) * 0.1
            W = W3.reshape(p*n, rank)
            
            # Check error
            err, S = get_tensor_error(U, V, W, n, m, p, rank)
            if err < best_err:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r, n, m, p, rank, tol=1e-6):
                    return U_r, V_r, W_r
        
        # Try rounding the best state found
        if best_state is not None:
            U_b, V_b, W_b = best_state
            U_r = np.round(U_b * 2) / 2
            V_r = np.round(V_b * 2) / 2
            W_r = np.round(W_b * 2) / 2
            if verify(U_r, V_r, W_r, n, m, p, rank, tol=1e-6):
                return U_r, V_r, W_r
        
        return None

    def try_rank_random_search(rank, seed):
        """Random search with local refinement on half-integers"""
        if time.time() > deadline - 15:
            return None
            
        rng = np.random.RandomState(seed)
        
        for attempt in range(30):
            if time.time() > deadline - 10:
                return None
            
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            err, S = get_tensor_error(U, V, W, n, m, p, rank)
            if err < 1e-8:
                return U, V, W
            
            # Local refinement
            for flip_round in range(30):
                if time.time() > deadline - 10:
                    return None
                
                improved = False
                for _ in range(150):
                    mat_choice = rng.randint(3)
                    if mat_choice == 0:
                        idx = rng.randint(n*m*rank)
                        row, col = idx // rank, idx % rank
                        old_val = U[row, col]
                        # Try all half-integer values
                        for new_val in [-1, -0.5, 0, 0.5, 1]:
                            if new_val == old_val:
                                continue
                            U[row, col] = new_val
                            err_new, _ = get_tensor_error(U, V, W, n, m, p, rank)
                            if err_new < err - 1e-12:
                                err = err_new
                                improved = True
                                break
                            U[row, col] = old_val
                    
                    elif mat_choice == 1:
                        idx = rng.randint(m*p*rank)
                        row, col = idx // rank, idx % rank
                        old_val = V[row, col]
                        for new_val in [-1, -0.5, 0, 0.5, 1]:
                            if new_val == old_val:
                                continue
                            V[row, col] = new_val
                            err_new, _ = get_tensor_error(U, V, W, n, m, p, rank)
                            if err_new < err - 1e-12:
                                err = err_new
                                improved = True
                                break
                            V[row, col] = old_val
                    
                    else:
                        idx = rng.randint(p*n*rank)
                        row, col = idx // rank, idx % rank
                        old_val = W[row, col]
                        for new_val in [-1, -0.5, 0, 0.5, 1]:
                            if new_val == old_val:
                                continue
                            W[row, col] = new_val
                            err_new, _ = get_tensor_error(U, V, W, n, m, p, rank)
                            if err_new < err - 1e-12:
                                err = err_new
                                improved = True
                                break
                            W[row, col] = old_val
                
                if not improved:
                    break
                
                if err < 1e-8:
                    return U, V, W
        
        return None

    # First, establish a valid baseline with rank 22
    U_base, V_base, W_base = build_rank_22()
    if verify_lists(U_base, V_base, W_base, 22):
        best_result = {
            "U": U_base,
            "V": V_base,
            "W": W_base
        }
        current_rank = 22
    else:
        # Fallback schoolbook rank 24
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
        current_rank = 24

    # Try to find lower rank decompositions
    for rank in range(14, current_rank):
        if time.time() > deadline - 25:
            break
        
        # Try ALS first
        found = False
        for seed in range(8):
            if time.time() > deadline - 20:
                break
            res = try_rank_als(rank, seed)
            if res is not None:
                U, V, W = res
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                current_rank = rank
                found = True
                break
        
        if found:
            continue
        
        # Try random search
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            res = try_rank_random_search(rank, seed)
            if res is not None:
                U, V, W = res
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                current_rank = rank
                break
        
        # If we found a lower rank, we can stop since we're going from low to high
        # and we want the minimum rank
        if current_rank < 22:
            break

    return best_result
