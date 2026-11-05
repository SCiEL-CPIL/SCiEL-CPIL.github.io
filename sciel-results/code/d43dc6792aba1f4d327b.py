"""Combined approach: MIS-style greedy row selection + SA refinement for DTS(4,7)."""
import random
import time
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    def get_max_entry(rows):
        return max(max(row) for row in rows)

    def get_row_diffs_bitset(row):
        """Compute bitset of all positive differences in a row."""
        diffs = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                diffs |= (1 << d)
        return diffs

    def get_row_diffs_set(row):
        """Compute set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_solution(rows):
        """Check if a full solution is valid."""
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_row(max_val, global_used):
        """Greedily build a single row up to max_val avoiding global_used diffs."""
        row = [0]
        current_diffs = 0
        while len(row) < k + 1:
            last = row[-1]
            found = False
            for v in range(last + 1, max_val + 1):
                conflict = False
                new_diffs = 0
                for u in row:
                    d = v - u
                    if (current_diffs >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if conflict:
                    continue
                if new_diffs & global_used:
                    continue
                row.append(v)
                current_diffs |= new_diffs
                found = True
                break
            if not found:
                return None
        return row

    def pure_greedy():
        """Pure greedy baseline."""
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_diffs = 0
            v = 1
            while len(row) < k + 1:
                conflict = False
                new_diffs = 0
                for u in row:
                    d = v - u
                    if (current_diffs >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                if not conflict and not (new_diffs & used):
                    row.append(v)
                    current_diffs |= new_diffs
                v += 1
            rows.append(row)
            used |= current_diffs
        return rows

    # --- MIS-style: generate candidate rows, then select disjoint subset ---
    def generate_candidate_rows(M, num_candidates, seed=None):
        """Generate many candidate Golomb rulers of scope <= M."""
        if seed is not None:
            random.seed(seed)
        candidates = []
        # Generate via greedy with randomization
        for _ in range(num_candidates):
            row = [0]
            current_diffs = 0
            used = 0
            success = True
            for _ in range(k):
                last = row[-1]
                # Randomize candidate selection
                candidates_v = list(range(last + 1, M + 1))
                random.shuffle(candidates_v)
                found = False
                for v in candidates_v:
                    conflict = False
                    new_diffs = 0
                    for u in row:
                        d = v - u
                        if (current_diffs >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    if conflict:
                        continue
                    if new_diffs & used:
                        continue
                    row.append(v)
                    current_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    success = False
                    break
            if success and len(row) == k + 1:
                candidates.append(row)
        return candidates

    def mis_greedy_select(candidates, n_rows_needed):
        """
        Weighted greedy maximal independent set heuristic.
        Nodes = candidate rows, edges = overlapping difference sets.
        Weight = negative max entry (prefer compact rows).
        """
        if not candidates:
            return None

        # Compute diff bitsets for all candidates
        diff_bitsets = [get_row_diffs_bitset(c) for c in candidates]
        max_entries = [c[-1] for c in candidates]

        # Sort by weight (prefer smaller max entry), with random tie-breaking
        indices = list(range(len(candidates)))
        random.shuffle(indices)
        indices.sort(key=lambda i: (max_entries[i], random.random()))

        selected = []
        selected_diffs = 0
        used_indices = set()

        for idx in indices:
            if len(selected) == n_rows_needed:
                break
            if idx in used_indices:
                continue
            # Check if this candidate conflicts with already selected
            if diff_bitsets[idx] & selected_diffs:
                continue
            selected.append(candidates[idx])
            selected_diffs |= diff_bitsets[idx]
            used_indices.add(idx)

        if len(selected) == n_rows_needed:
            return selected
        return None

    # --- SA-based refinement ---
    def compute_cost(rows, max_limit):
        """Compute cost for SA: max_entry + conflict penalties."""
        used_diffs = {}
        max_entry = 0
        for row in rows:
            if row:
                m = max(row)
                if m > max_entry:
                    max_entry = m
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        used_diffs[d] += 1
                    else:
                        used_diffs[d] = 1
        num_conflicts = 0
        for count in used_diffs.values():
            if count > 1:
                num_conflicts += count - 1
        conflict_penalty = 10000 * num_conflicts
        limit_penalty = 0
        if max_entry > max_limit:
            limit_penalty = 1000 * (max_entry - max_limit)
        return max_entry + conflict_penalty + limit_penalty

    def sa_refine(initial_rows, best_max, deadline_t, max_iters=50000):
        """Simulated annealing to refine a solution."""
        current_rows = [list(r) for r in initial_rows]
        T_start = 50.0
        T_end = 0.5
        cooling_rate = 0.9995
        T = T_start
        current_cost = compute_cost(current_rows, best_max - 1)
        best_rows_sa = [list(r) for r in current_rows]
        best_cost = current_cost

        iters = 0
        while T > T_end and time.time() < deadline_t and iters < max_iters:
            iters += 1
            # Pick random row and position to modify
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)

            min_val = current_rows[r_idx][c_idx - 1] + 1
            max_val = best_max - 1

            if c_idx < k:
                max_val = min(max_val, current_rows[r_idx][c_idx + 1] - 1)

            if min_val > max_val:
                T *= cooling_rate
                continue

            new_val = random.randint(min_val, max_val)
            if new_val == current_rows[r_idx][c_idx]:
                T *= cooling_rate
                continue

            old_val = current_rows[r_idx][c_idx]
            current_rows[r_idx][c_idx] = new_val

            new_cost = compute_cost(current_rows, best_max - 1)
            delta = new_cost - current_cost

            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_rows_sa = [list(r) for r in current_rows]
            else:
                current_rows[r_idx][c_idx] = old_val

            T *= cooling_rate

        # Check if we found a valid solution
        if is_valid_solution(best_rows_sa):
            curr_max = get_max_entry(best_rows_sa)
            if curr_max < best_max:
                return best_rows_sa, curr_max
        return None, best_max

    # --- Local search: rebuild individual rows to reduce max ---
    def local_search_polish(rows, deadline_t, max_iter=200):
        """Try to reduce max entry by rebuilding individual rows."""
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)

        for iteration in range(max_iter):
            if time.time() > deadline_t:
                break

            # Find the row with the largest entry
            max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
            target_row = rows[max_row_idx]
            current_max = target_row[-1]

            if current_max <= lower_bound:
                break

            # Compute diffs used by all other rows
            global_used_except = 0
            for i in range(n_rows):
                if i != max_row_idx:
                    global_used_except |= get_row_diffs_bitset(rows[i])

            # Try to rebuild this row with a smaller max
            target = current_max - 1
            new_row = None
            for attempt in range(15):
                row = [0]
                current_diffs = 0
                success = True
                for _ in range(k):
                    last = row[-1]
                    candidates = list(range(last + 1, target + 1))
                    random.shuffle(candidates)
                    found = False
                    for v in candidates:
                        conflict = False
                        new_diffs = 0
                        for u in row:
                            d = v - u
                            if (current_diffs >> d) & 1:
                                conflict = True
                                break
                            new_diffs |= (1 << d)
                        if conflict:
                            continue
                        if new_diffs & global_used_except:
                            continue
                        row.append(v)
                        current_diffs |= new_diffs
                        found = True
                        break
                    if not found:
                        success = False
                        break
                if success and len(row) == k + 1:
                    new_row = row
                    break

            if new_row is not None:
                rows[max_row_idx] = new_row
                new_max = get_max_entry(rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in rows]

        return best_rows, best_max

    # --- Main search ---
    # Phase 0: Get a baseline
    best_rows = pure_greedy()
    best_max = get_max_entry(best_rows)

    search_deadline = deadline - 5.0

    # Phase 1: MIS-style search with decreasing M
    # Try to find solutions with smaller and smaller max entry
    M_start = best_max - 1
    M_min = max(lower_bound, M_start - 30)

    for M in range(M_start, M_min - 1, -1):
        if time.time() > search_deadline:
            break

        remaining_time = search_deadline - time.time()
        if remaining_time < 1.0:
            break

        # Number of candidates to generate
        num_candidates = min(200, max(50, int(remaining_time * 20)))
        time_slice = min(remaining_time / 10, 3.0)
        deadline_for_this = time.time() + time_slice

        # Generate candidates and try MIS selection
        found = False
        for restart in range(3):
            if time.time() > deadline_for_this:
                break
            seed = M * 100 + restart
            candidates = generate_candidate_rows(M, num_candidates, seed=seed)
            if not candidates:
                continue
            solution = mis_greedy_select(candidates, n_rows)
            if solution is not None:
                curr_max = get_max_entry(solution)
                if curr_max < best_max:
                    best_max = curr_max
                    best_rows = solution
                    found = True
                    break

        if found:
            # Try to improve further with local search
            if time.time() < search_deadline:
                improved_rows, improved_max = local_search_polish(best_rows, search_deadline, max_iter=30)
                if improved_max < best_max:
                    best_max = improved_max
                    best_rows = improved_rows

    # Phase 2: SA refinement on the best solution found
    if time.time() < search_deadline - 2.0:
        sa_deadline = min(search_deadline, time.time() + 40.0)
        for restart in range(5):
            if time.time() > sa_deadline:
                break
            # Perturb the best solution
            perturbed = [list(r) for r in best_rows]
            for _ in range(random.randint(1, 3)):
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, k)
                min_val = perturbed[r_idx][c_idx - 1] + 1
                max_val = best_max
                if c_idx < k:
                    max_val = min(max_val, perturbed[r_idx][c_idx + 1] - 1)
                if min_val <= max_val:
                    perturbed[r_idx][c_idx] = random.randint(min_val, max_val)

            sa_rows, sa_max = sa_refine(perturbed, best_max, sa_deadline, max_iters=20000)
            if sa_rows is not None and sa_max < best_max:
                best_max = sa_max
                best_rows = sa_rows

    # Phase 3: Final local search polish
    if time.time() < deadline - 3.0:
        final_rows, final_max = local_search_polish(best_rows, deadline - 3.0, max_iter=100)
        if final_max < best_max:
            best_max = final_max
            best_rows = final_rows

    # Final validation
    if not is_valid_solution(best_rows):
        # Fallback to pure greedy
        best_rows = pure_greedy()
        best_max = get_max_entry(best_rows)

    return best_rows

# EVOLVE-BLOCK-END
