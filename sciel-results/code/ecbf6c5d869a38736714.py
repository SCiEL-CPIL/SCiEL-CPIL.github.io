import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    _start = time.time()
    _deadline = _start + 115  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    def build_rank22():
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
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    def build_rank20():
        """Construct a rank-20 decomposition using shared Strassen terms."""
        rank = 20
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

        # A is 2x3: indices 0-5 (row-major: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5)
        # B is 3x4: indices 0-11 (row-major: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7, B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11)
        # C is 2x4: W[k*n+i] so C[0,0]=W[0], C[0,1]=W[1], C[0,2]=W[2], C[0,3]=W[3], C[1,0]=W[4], C[1,1]=W[5], C[1,2]=W[6], C[1,3]=W[7]
        
        # Use shared Strassen-like terms that work across both B blocks
        # Define shared products:
        # P0 = (A00+A10) * (B00+B10)  -> contributes to C[0,0], C[1,0], C[0,1], C[1,1]
        # P1 = A00 * B00
        # P2 = A10 * B10
        # P3 = (A00+A10) * B00
        # P4 = A00 * (B00+B10)
        # P5 = A10 * (B00+B10)
        # P6 = (A00+A10) * B10
        # P7 = A00 * B01
        # P8 = A10 * B11
        # P9 = (A00+A10) * B01
        # P10 = A00 * (B01+B11)
        # P11 = A10 * (B01+B11)
        # P12 = (A00+A10) * B11
        # P13 = A02 * B00
        # P14 = A12 * B10
        # P15 = A02 * B01
        # P16 = A12 * B11
        # P17 = A02 * B02
        # P18 = A12 * B12
        # P19 = A02 * B03
        # P20 = A12 * B13
        
        # That's 21 terms. Let me try to merge some.
        
        # Actually, let me use a different approach: use the fact that for the 2x3 x 3x4 case,
        # we can use 7 Strassen terms for the 2x2 x 2x2 subproblem (covering B columns 0,1)
        # and handle B columns 2,3 with direct terms, but share some computations.
        
        # Strassen for A[0:2, 0:2] * B[0:2, 0:2] (7 terms, columns 0-1 of B, rows 0-1 of B)
        add_term([(0, 1), (3, 1)], [(0, 1), (4, 1)], [(0, 1), (4, 1)], 0)   # (A00+A10)*(B00+B10)
        add_term([(3, 1), (0, -1)], [(0, 1)], [(0, 1), (4, -1)], 1)          # A10*B00
        add_term([(0, 1)], [(4, 1), (0, -1)], [(0, -1), (4, 1)], 2)          # A00*(B10-B00)
        add_term([(3, 1)], [(4, 1)], [(4, 1)], 3)                             # A10*B10
        add_term([(0, 1), (3, -1)], [(0, 1), (4, 1)], [(0, 1)], 4)           # (A00-A10)*(B00+B10)
        add_term([(0, 1)], [(0, 1), (4, 1)], [(0, 1), (4, 1)], 5)            # A00*(B00+B10)
        add_term([(3, 1)], [(4, 1)], [(0, 1)], 6)                             # A10*B10 (duplicate, need to fix)
        
        # This is getting messy. Let me just return None and keep rank 22.
        return None

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    def is_half_integer(val):
        return val == int(val) or abs(val * 2 - round(val * 2)) < 1e-10

    def check_valid(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not is_half_integer(val):
                        return False
        return True

    # Start with rank-22
    U, V, W, rank = build_rank22()

    if not verify(U, V, W, rank):
        raise RuntimeError("Rank-22 decomposition failed verification")

    if not check_valid(U, V, W):
        raise RuntimeError("Rank-22 has non-half-integer entries")

    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def compute_residual(U_arr, V_arr, W_arr, rank_val):
        """Compute residual tensor T - sum of rank-1 terms.
        U_arr, V_arr, W_arr are (n*m, rank_val), (m*p, rank_val), (p*n, rank_val)"""
        res = np.ones((n, m, p))
        for r in range(rank_val):
            u = U_arr[:, r]
            v = V_arr[:, r]
            w = W_arr[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] -= u[i*m+j] * v[j*p+k] * w[k*n+i]
        return res

    def try_remove_term(U_arr, V_arr, W_arr, rank_val, term_idx):
        """Try to remove term_idx and see if the rest still decomposes exactly.
        Returns (new_rank, new_U, new_V, new_W) if successful, else None."""
        if time.time() > _deadline - 15:
            return None
        
        # Remove the term
        U_new = [U_arr[:, r].copy() for r in range(rank_val) if r != term_idx]
        V_new = [V_arr[:, r].copy() for r in range(rank_val) if r != term_idx]
        W_new = [W_arr[:, r].copy() for r in range(rank_val) if r != term_idx]
        
        new_rank = rank_val - 1
        if new_rank == 0:
            return None
            
        U_new_arr = np.array(U_new, dtype=float).T  # shape (new_rank, n*m)
        V_new_arr = np.array(V_new, dtype=float).T  # shape (new_rank, m*p)
        W_new_arr = np.array(W_new, dtype=float).T  # shape (new_rank, p*n)
        
        # Check if the remaining terms exactly decompose T
        res = compute_residual(U_new_arr, V_new_arr, W_new_arr, new_rank)
        if np.max(np.abs(res)) < 1e-10:
            # Success! Convert back to list format
            U_list = [[float(x) for x in U_new_arr[r]] for r in range(new_rank)]
            V_list = [[float(x) for x in V_new_arr[r]] for r in range(new_rank)]
            W_list = [[float(x) for x in W_new_arr[r]] for r in range(new_rank)]
            return new_rank, U_list, V_list, W_list
        
        return None

    # Try removing each term one by one
    for r_remove in range(rank):
        if time.time() > _deadline - 15:
            break
        
        result = try_remove_term(U_np, V_np, W_np, rank, r_remove)
        if result is not None:
            new_rank, new_U, new_V, new_W = result
            if verify(new_U, new_V, new_W, new_rank) and check_valid(new_U, new_V, new_W):
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U, best_V, best_W = new_U, new_V, new_W
                    U_np = np.array(best_U, dtype=float).T
                    V_np = np.array(best_V, dtype=float).T
                    W_np = np.array(best_W, dtype=float).T
                    rank = new_rank
                    r_remove = -1  # restart loop with new rank

    # Try a more sophisticated approach: modify terms to enable rank reduction
    # Use local optimization to adjust terms slightly, then snap to half-integers
    if time.time() < _deadline - 20 and best_rank > 15:
        for attempt in range(5):
            if time.time() > _deadline - 15:
                break
            
            np.random.seed(42 + attempt)
            U_try = np.array(U_np, dtype=float).copy()
            V_try = np.array(V_np, dtype=float).copy()
            W_try = np.array(W_np, dtype=float).copy()
            
            # Add small perturbations
            U_try += np.random.randn(*U_try.shape) * 0.01
            V_try += np.random.randn(*V_try.shape) * 0.01
            W_try += np.random.randn(*W_try.shape) * 0.01
            
            # Snap to half-integers
            U_snap = np.round(U_try * 2) / 2
            V_snap = np.round(V_try * 2) / 2
            W_snap = np.round(W_try * 2) / 2
            
            # Check if snapped version still decomposes
            res = compute_residual(U_snap, V_snap, W_snap, rank)
            if np.max(np.abs(res)) < 1e-10:
                # Valid decomposition at same rank, but maybe different structure
                # Try removing terms from this new configuration
                for r_remove in range(rank):
                    if time.time() > _deadline - 15:
                        break
                    result = try_remove_term(U_snap, V_snap, W_snap, rank, r_remove)
                    if result is not None:
                        new_rank, new_U, new_V, new_W = result
                        if verify(new_U, new_V, new_W, new_rank) and check_valid(new_U, new_V, new_W):
                            if new_rank < best_rank:
                                best_rank = new_rank
                                best_U, best_V, best_W = new_U, new_V, new_W
                                U_np = np.array(best_U, dtype=float).T
                                V_np = np.array(best_V, dtype=float).T
                                W_np = np.array(best_W, dtype=float).T
                                rank = new_rank
                                break

    # Final verification
    if not verify(best_U, best_V, best_W, best_rank):
        # Fall back to original rank-22
        best_U, best_V, best_W, best_rank = build_rank22()

    return {"U": best_U, "V": best_V, "W": best_W}
