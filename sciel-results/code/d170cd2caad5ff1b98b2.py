"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

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

    def _build_rank7_2x2():
        """Return list of 7 (u, v, w) dicts for 2x2 matrix multiplication (Strassen)."""
        terms = []
        # p1 = (a11+a22)*(b11+b22)
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        # p2 = (a21+a22)*b11
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
        # p3 = a11*(b12-b22)
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
        # p4 = a22*(b21-b11)
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
        # p5 = (a11+a12)*b22
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
        # p6 = (a21-a11)*(b11+b12)
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        # p7 = (a12-a22)*(b21+b22)
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    rank7_terms = _build_rank7_2x2()

    # Build the full 3x4 * 4x4 decomposition.
    # Split strategy:
    #   Rows of A: top 2 (i=0,1), bottom 1 (i=2)
    #   Cols of B / rows of C: left 2 (k=0,1), right 2 (k=2,3)
    #   Middle dim j: j=0,1 and j=2,3
    #
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]  -> 7+7=14
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]  -> 7+7=14
    # C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]  -> 2+2=4
    # C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]  -> 2+2=4
    # Total: 36

    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 44

    def add_rank7_block(a_row_offset, b_col_offset, j_offset, c_row_offset, c_col_offset, col_start):
        """Add a rank-7 2x2*2x2 block.
        a_row_offset: row offset in A (0 or 1)
        b_col_offset: col offset in B (0 or 2)
        j_offset: row offset in B / col offset in A (0 or 2)
        c_row_offset: row offset in C (0 or 1)
        c_col_offset: col offset in C (0 or 2)
        """
        for r in range(7):
            u_dict, v_dict, w_dict = rank7_terms[r]
            col = col_start + r
            for (ai, aj), val in u_dict.items():
                U[(a_row_offset + ai) * m + (j_offset + aj)][col] += val
            for (bj, bk), val in v_dict.items():
                V[(j_offset + bj) * p + (b_col_offset + bk)][col] += val
            for (ci, ck), val in w_dict.items():
                W[(c_col_offset + ck) * n + (c_row_offset + ci)][col] += val
        return col_start + 7

    def add_rank4_block_1x2x2x2(a_row, b_col_offset, j_offset, c_col_offset, col_start):
        """Add a rank-4 1x2 * 2x2 -> 1x2 block (bottom row of A, 2 cols of B)."""
        for r in range(4):
            col = col_start + r
            j_idx = j_offset + (r % 2)
            k_idx = b_col_offset + (r // 2)
            U[a_row * m + j_idx][col] = 1
            V[j_idx * p + k_idx][col] = 1
            W[k_idx * n + a_row][col] = 1
        return col_start + 4

    col = 0
    # Top 2 rows of A, all 4x4 of B: 4 rank-7 blocks
    col = add_rank7_block(0, 0, 0, 0, 0, col)
    col = add_rank7_block(0, 0, 2, 0, 0, col)
    col = add_rank7_block(0, 2, 0, 0, 2, col)
    col = add_rank7_block(0, 2, 2, 0, 2, col)
    # Bottom row of A, all 4x4 of B: 4 rank-4 blocks
    col = add_rank4_block_1x2x2x2(2, 0, 0, 0, col)
    col = add_rank4_block_1x2x2x2(2, 0, 2, 0, col)
    col = add_rank4_block_1x2x2x2(2, 2, 0, 2, col)
    col = add_rank4_block_1x2x2x2(2, 2, 2, 2, col)

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verify the decomposition
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        C_computed = np.zeros((n, p))
        for r in range(total_rank):
            a_r = sum(U_arr[i*m+j, r] * A_test[i,j] for i in range(n) for j in range(m))
            b_r = sum(V_arr[j*p+k, r] * B_test[j,k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_computed[i,k] += W_arr[k*n+i, r] * a_r * b_r

        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-9:
            print(f"Verification failed: max error = {max_err}")
            return {"U": U_school, "V": V_school, "W": W_school}
    except Exception as e:
        print(f"Verification error: {e}")
        return {"U": U_school, "V": V_school, "W": W_school}

    # Try to find a lower-rank decomposition by searching for sparse solutions
    # We'll use a simple randomized approach: start with the rank-44 solution and
    # try to eliminate terms by finding linear dependencies.
    
    def _try_reduce_rank(U_base, V_base, W_base, max_rank, time_limit):
        """Try to find a decomposition of rank <= max_rank by searching for sparse solutions."""
        start_time = time.time()
        U_arr = np.array(U_base, dtype=float)
        V_arr = np.array(V_base, dtype=float)
        W_arr = np.array(W_base, dtype=float)
        
        # Build the target tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        
        # For each rank r from max_rank down to 36, try to find a solution
        # We'll use a simple approach: try to eliminate columns one at a time
        
        best_rank = total_rank
        best_U = U_arr.copy()
        best_V = V_arr.copy()
        best_W = W_arr.copy()
        
        # Try to eliminate terms by finding which terms can be removed
        # while maintaining the decomposition
        
        # For each term r, check if removing it still gives the correct result
        # with the remaining terms (possibly with adjusted weights)
        
        # This is a complex search, so we'll use a simpler heuristic:
        # Try to find linear dependencies among the terms
        
        for target_rank in range(min(max_rank, total_rank - 1), 35, -1):
            if time.time() > time_limit:
                break
            
            # Try to find a subset of terms of size target_rank that works
            # We'll use a greedy approach: try to remove one term at a time
            
            current_U = best_U.copy()
            current_V = best_V.copy()
            current_W = best_W.copy()
            current_rank = best_rank
            
            for trial in range(100):  # Limit trials
                if time.time() > time_limit:
                    break
                
                # Try to remove a random term
                if current_rank <= target_rank:
                    break
                
                term_to_remove = np.random.randint(current_rank)
                
                # Check if removing this term still gives a valid decomposition
                # with the remaining terms
                test_U = np.delete(current_U, term_to_remove, axis=1)
                test_V = np.delete(current_V, term_to_remove, axis=1)
                test_W = np.delete(current_W, term_to_remove, axis=1)
                
                # Verify with random test
                np.random.seed(trial)
                A_test = np.random.randn(n, m)
                B_test = np.random.randn(m, p)
                C_expected = A_test @ B_test
                
                C_computed = np.zeros((n, p))
                for r in range(test_U.shape[1]):
                    a_r = sum(test_U[i*m+j, r] * A_test[i,j] for i in range(n) for j in range(m))
                    b_r = sum(test_V[j*p+k, r] * B_test[j,k] for j in range(m) for k in range(p))
                    for i in range(n):
                        for k in range(p):
                            C_computed[i,k] += test_W[k*n+i, r] * a_r * b_r
                
                max_err = np.max(np.abs(C_computed - C_expected))
                if max_err < 1e-9:
                    # Found a valid lower-rank decomposition
                    best_U = test_U
                    best_V = test_V
                    best_W = test_W
                    best_rank = current_rank - 1
                    current_U = test_U
                    current_V = test_V
                    current_W = test_W
                    current_rank = best_rank
                    break
        
        if best_rank < total_rank:
            return best_rank, best_U, best_V, best_W
        
        return total_rank, U_arr, V_arr, W_arr
    
    # Try to reduce the rank
    if time.time() < deadline - 30:
        try:
            reduced_rank, reduced_U, reduced_V, reduced_W = _try_reduce_rank(
                U, V, W, min(total_rank - 1, 43), deadline - 30
            )
            if reduced_rank < total_rank:
                # Convert back to lists
                U = reduced_U.tolist()
                V = reduced_V.tolist()
                W = reduced_W.tolist()
                total_rank = reduced_rank
                print(f"Reduced rank to {total_rank}")
        except Exception as e:
            print(f"Rank reduction failed: {e}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
