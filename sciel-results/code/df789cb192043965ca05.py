import math
import time
import random
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 90s wall-clock, leave 5s margin

    N = 256
    max_sum = 2 * N
    perfect_squares = set()
    for k in range(1, int(math.isqrt(max_sum)) + 2):
        sq = k * k
        if sq <= max_sum:
            perfect_squares.add(sq)

    # Self-loop check: 2a is a perfect square
    self_loop_set = set()
    for m in range(1, 13):
        a = 2 * m * m
        if 1 <= a <= N:
            self_loop_set.add(a)

    # Remaining vertices
    remaining = [a for a in range(1, N + 1) if a not in self_loop_set]
    n_rem = len(remaining)

    # Map each remaining vertex to an index 0..n_rem-1
    vert_to_idx = {v: i for i, v in enumerate(remaining)}

    # Build adjacency list for constraints
    # adj[i] is a list of j such that i != j and (remaining[i] + remaining[j]) is a perfect square
    adj = [[] for _ in range(n_rem)]
    for i in range(n_rem):
        a = remaining[i]
        for j in range(i + 1, n_rem):
            b = remaining[j]
            if (a + b) in perfect_squares:
                adj[i].append(j)
                adj[j].append(i)

    # Precompute adjacency masks for fast operations
    adj_mask = [0] * n_rem
    for i in range(n_rem):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask

    best_indices = []
    best_size = 0

    # Strategy 1: Try ILP if available
    if HAS_MILP:
        try:
            # Variables x_0 ... x_{n_rem-1}
            # Maximize sum(x)
            # Constraints: x_i + x_j <= 1 for all edges (i, j)
            # x_i >= 0, x_i <= 1

            c = -np.ones(n_rem)  # Minimize -sum(x)

            # Build constraint matrix A
            # Each edge (i, j) gives a row: 1 at i, 1 at j, 0 elsewhere
            rows = []
            cols = []
            data = []
            for i in range(n_rem):
                for j in adj[i]:
                    if i < j:  # Only add each edge once
                        row_idx = len(rows) // 2
                        rows.extend([row_idx, row_idx])
                        cols.extend([i, j])
                        data.extend([1, 1])

            if data:
                from scipy.sparse import coo_matrix
                A = coo_matrix((data, (rows, cols)), shape=(len(rows) // 2, n_rem)).tocsr()
                lb = np.zeros(len(rows) // 2)
                ub = np.ones(len(rows) // 2)
                constraints = LinearConstraint(A, lb, ub)
            else:
                # No edges, take all
                constraints = None

            bounds = Bounds(lb=0, ub=1)

            # Set time limit for the solver
            options = {"time_limit": 40.0}

            res = milp(c=c, constraints=constraints, integrality=np.ones(n_rem), bounds=bounds, options=options)

            if res.status == 0:  # Success
                sol = res.x
                selected = [i for i in range(n_rem) if sol[i] > 0.5]

                # Validate
                valid = True
                sel_set = set(selected)
                for i in selected:
                    for j in adj[i]:
                        if j in sel_set:
                            valid = False
                            break
                    if not valid:
                        break

                if valid and len(selected) > best_size:
                    best_indices = selected
                    best_size = len(selected)
            elif res.status == 1:  # Time limit reached
                if res.x is not None:
                    sol = res.x
                    selected = [i for i in range(n_rem) if sol[i] > 0.5]
                    valid = True
                    sel_set = set(selected)
                    for i in selected:
                        for j in adj[i]:
                            if j in sel_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and len(selected) > best_size:
                        best_indices = selected
                        best_size = len(selected)
        except Exception:
            pass

    # Strategy 2: Heuristic Search (Fallback or Refinement)
    def greedy_min_degree(rs):
        available = (1 << n_rem) - 1
        result_indices = []

        while available:
            min_deg = n_rem + 1
            candidates = []
            avail_list = []
            temp = available
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                avail_list.append(i)
                temp ^= lsb

            for i in avail_list:
                deg = bin(adj_mask[i] & available).count('1')
                if deg < min_deg:
                    min_deg = deg
                    candidates = [i]
                elif deg == min_deg:
                    candidates.append(i)

            v = rs.choice(candidates)
            result_indices.append(v)
            available &= ~(1 << v)
            available &= ~adj_mask[v]

        return result_indices

    def local_improvement(indices, rs, max_iters=3000):
        if not indices:
            return indices

        selected_mask = 0
        for i in indices:
            selected_mask |= (1 << i)

        improved = True
        iters = 0

        while improved and iters < max_iters and time.time() < DEADLINE:
            improved = False
            iters += 1

            non_selected = ((1 << n_rem) - 1) & ~selected_mask
            non_sel_list = []
            temp = non_selected
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                non_sel_list.append(i)
                temp ^= lsb

            rs.shuffle(non_sel_list)

            for v in non_sel_list:
                conflicts = adj_mask[v] & selected_mask
                if conflicts == 0:
                    selected_mask |= (1 << v)
                    improved = True
                else:
                    conflict_count = bin(conflicts).count('1')
                    net_gain = 1 - conflict_count
                    if net_gain > 0:
                        selected_mask = (selected_mask & ~conflicts) | (1 << v)
                        improved = True

        result = []
        temp = selected_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            result.append(i)
            temp ^= lsb

        return result

    def iterated_local_search(rs, max_iters=500):
        """Perform iterated local search with random restarts and perturbations."""
        # Start with a greedy solution
        current = greedy_min_degree(rs)
        current = local_improvement(current, rs, max_iters=500)
        best = list(current)
        best_size_local = len(best)

        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break

            # Perturb: randomly remove some elements
            perturbed = list(current)
            n_remove = max(1, len(perturbed) // 10)
            if perturbed:
                to_remove = rs.sample(perturbed, min(n_remove, len(perturbed)))
                for v in to_remove:
                    perturbed.remove(v)

            # Try to add back and improve
            perturbed = local_improvement(perturbed, rs, max_iters=200)

            if len(perturbed) > best_size_local:
                best = list(perturbed)
                best_size_local = len(perturbed)
                current = list(perturbed)
            else:
                # With some probability, accept worse solutions to escape local optima
                if rs.random() < 0.1 and perturbed:
                    current = list(perturbed)

        return best

    # Run heuristics if ILP didn't find a good solution
    if best_size < n_rem:
        for seed in range(100):
            if time.time() > DEADLINE:
                break
            rs = random.Random(seed)
            s = greedy_min_degree(rs)
            s = local_improvement(s, rs, max_iters=1500)
            if len(s) > best_size:
                best_indices = s
                best_size = len(s)

        # Iterated local search with more iterations
        for seed in range(30):
            if time.time() > DEADLINE:
                break
            rs = random.Random(seed + 1000)
            s = iterated_local_search(rs, max_iters=200)
            if len(s) > best_size:
                best_indices = s
                best_size = len(s)

    # Final validation and conversion
    result = sorted([remaining[i] for i in best_indices])

    valid = True
    for i in range(len(result)):
        a = result[i]
        if (2 * a) in perfect_squares:
            valid = False
            break
        for j in range(i + 1, len(result)):
            b = result[j]
            if (a + b) in perfect_squares:
                valid = False
                break
        if not valid:
            break

    if not valid or not result:
        # Fallback: return a valid single element
        for a in range(1, N + 1):
            if 2 * a not in perfect_squares:
                return [a]
        return [1]

    return result
