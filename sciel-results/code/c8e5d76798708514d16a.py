# EVOLVE-BLOCK-START
import numpy as np
import time
import collections


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115.0

    # Step 1: Initial assignment using rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # Compute initial cost
    def compute_cost(perm):
        b_perm = b[np.ix_(perm, perm)]
        return np.sum(a * b_perm)

    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()

    # Precompute for fast delta evaluation
    a_rows = a
    a_cols = a.T
    b_matrix = b

    def compute_delta(i, j, perm):
        li = perm[i]
        lj = perm[j]
        if li == lj:
            return 0

        diff_row = b_matrix[lj, :] - b_matrix[li, :]
        diff_col = b_matrix[:, lj] - b_matrix[:, li]

        s1 = np.dot(a_rows[i, :], diff_row[perm])
        s1 -= a_rows[i, i] * diff_row[li]
        s1 -= a_rows[i, j] * diff_row[lj]

        s2 = -np.dot(a_rows[j, :], diff_row[perm])
        s2 += a_rows[j, i] * diff_row[li]
        s2 += a_rows[j, j] * diff_row[lj]

        s3 = np.dot(a_cols[i, :], diff_col[perm])
        s3 -= a_cols[i, i] * diff_col[li]
        s3 -= a_cols[i, j] * diff_col[lj]

        s4 = -np.dot(a_cols[j, :], diff_col[perm])
        s4 += a_cols[j, i] * diff_col[li]
        s4 += a_cols[j, j] * diff_col[lj]

        cross = (a_rows[j, i] - a_rows[i, j]) * (b_matrix[li, lj] - b_matrix[lj, li])

        return s1 + s2 + s3 + s4 + cross

    # Tabu Search Parameters
    tabu_tenure = n // 2  # 50 iterations
    tabu_queue = collections.deque()
    tabu_set = set()

    # Main Loop
    while time.time() - start_time < time_budget - 2.0:
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Iterate over all pairs
        for i in range(n):
            for j in range(i + 1, n):
                # Check time
                if time.time() - start_time > time_budget - 2.0:
                    break
                
                delta = compute_delta(i, j, p)
                
                if delta == 0:
                    continue

                move = (i, j)
                is_tabu = move in tabu_set
                
                # Aspiration Criterion: Accept if it improves global best
                if is_tabu:
                    if current_cost + delta < best_cost:
                        # Allow
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_j = j
                else:
                    # Not tabu, accept if it's the best non-tabu move
                    # Note: Standard TS picks the best non-tabu move, even if it's worse than current (if no improving move exists)
                    # However, to be safe and fast, we often just pick the best move overall that is not tabu.
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if time.time() - start_time > time_budget - 2.0:
                break

        if best_i == -1:
            break

        # Apply move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta

        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()

        # Update Tabu List
        move = (best_i, best_j)
        tabu_queue.append(move)
        tabu_set.add(move)
        
        if len(tabu_queue) > tabu_tenure:
            old_move = tabu_queue.popleft()
            tabu_set.discard(old_move)

    return best_p.tolist()
# EVOLVE-BLOCK-END
