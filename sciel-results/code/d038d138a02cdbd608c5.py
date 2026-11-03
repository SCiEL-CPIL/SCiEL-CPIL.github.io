import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]

    rows = []
    cols = []
    data = []

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)

    # Bitmasks for fast intersection checks in local search
    # block_mask[j] = bitmask of t-subsets covered by block j
    # subset_mask[i] = bitmask of blocks covering t-subset i
    block_mask = [0] * n_k
    subset_mask = [0] * n_t
    for j in range(n_k):
        m = 0
        for s in subsets_of_block[j]:
            m |= (1 << s)
        block_mask[j] = m
    for i in range(n_t):
        m = 0
        for b in blocks_of_subset[i]:
            m |= (1 << b)
        subset_mask[i] = m

    # Conflict mask for packing LB: two t-subsets conflict if some block covers both
    # conflict_mask[i] = bitmask of t-subsets that share a block with t-subset i
    conflict_mask = [0] * n_t
    for j in range(n_k):
        m = block_mask[j]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            conflict_mask[idx] |= m
            m ^= lsb

    def packing_lb(uncovered_set):
        """Greedy maximal independent set size as lower bound on blocks needed."""
        count = 0
        remaining = set(uncovered_set)
        while remaining:
            # Pick element with smallest conflict set among remaining
            best = None
            best_conf = None
            for s in remaining:
                conf = conflict_mask[s] & (1 << s)  # include self
                # Count how many remaining elements conflict
                # Use bit operations for speed
                cnt = 0
                for r in remaining:
                    if conf >> r & 1:
                        cnt += 1
                if best is None or cnt < best_conf:
                    best = s
                    best_conf = cnt
                    if cnt == 1:
                        break
            count += 1
            # Remove all conflicts of best
            conf = conflict_mask[best]
            remaining = {s for s in remaining if not (conf >> s) & 1}
        return count

    best_solution = None
    best_size = float('inf')

    # 2. LP Relaxation for duals
    duals = None
    try:
        if time.time() > deadline - 5:
            raise TimeoutError()
        M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
        best_duals = None
        best_lp_obj = float('inf')
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            c = np.ones(n_k)
            if attempt > 0:
                c += np.random.uniform(0, 0.1, n_k)
            A_ub = -M.T.tocsr()
            b_ub = -np.ones(n_t)
            bounds = [(0, 1)] * n_k
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                duals_candidate = np.abs(res.ineqlin.marginals)
                if np.sum(duals_candidate) > 0:
                    if best_duals is None or res.fun < best_lp_obj:
                        best_duals = duals_candidate
                        best_lp_obj = res.fun
    except Exception:
        best_duals = None

    if best_duals is not None:
        duals = best_duals
    else:
        duals = np.ones(n_t)

    # 3. Multi-restart heap-based greedy with LP duals
    num_restarts = 12
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break

        duals_variant = duals.copy()
        if restart % 3 == 1:
            for s_idx in range(n_t):
                if len(blocks_of_subset[s_idx]) < 10:
                    duals_variant[s_idx] *= 2.0
        elif restart % 3 == 2:
            duals_variant += np.random.uniform(0, 0.05, n_t)

        # Vectorized initial block weights
        # block_weights[j] = sum of duals_variant[s] for s in subsets_of_block[j]
        block_weights = np.zeros(n_k)
        for j in range(n_k):
            if subsets_of_block[j]:
                block_weights[j] = np.sum(duals_variant[subsets_of_block[j]])

        if restart > 0:
            block_weights += np.random.uniform(0, 0.005 + 0.002 * restart, n_k)

        heap = [(-block_weights[j], j) for j in range(n_k)]
        heapq.heapify(heap)

        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        solution_blocks = []

        while heap:
            if time.time() > deadline - 10:
                break
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            if picked[j]:
                continue
            if w > block_weights[j] + 1e-9:
                continue
            if w <= 1e-9:
                break

            picked[j] = True
            solution_blocks.append(j)

            for s_idx in subsets_of_block[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    y_s = duals_variant[s_idx]
                    if y_s > 0:
                        for b_prime in blocks_of_subset[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                heapq.heappush(heap, (-block_weights[b_prime], b_prime))

            if np.all(covered):
                break

        if len(solution_blocks) < best_size and np.all(covered):
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()

    # Fallback
    if best_solution is None:
        uncovered = set(range(n_t))
        res_blocks = []
        while uncovered:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = sum(1 for s in subsets_of_block[j] if s in uncovered)
                if cnt > max_cov:
                    max_cov = cnt
                    best_b = j
            if best_b == -1 or max_cov == 0:
                break
            res_blocks.append(best_b)
            for s in subsets_of_block[best_b]:
                if s in uncovered:
                    uncovered.remove(s)
        best_solution = res_blocks
        best_size = len(best_solution)

    # 4. Deletion Pass
    if best_solution is not None:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                can_remove = True
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)
            best_solution = new_solution
            best_size = len(best_solution)

    # 5. Local Search with bitset-accelerated candidate finding
    if best_solution is not None and time.time() < deadline - 8:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1

        improved = True
        max_iters = 20
        iter_count = 0
        while improved and time.time() < deadline - 8 and iter_count < max_iters:
            improved = False
            iter_count += 1
            new_solution = best_solution.copy()
            solution_set = set(best_solution)

            # Try 1-for-1 replacement
            for i, b_idx in enumerate(best_solution):
                if time.time() > deadline - 8:
                    break
                subs_b = subsets_of_block[b_idx]
                unique_subs = [s for s in subs_b if cover_count[s] == 1]

                if not unique_subs:
                    cover_count[subs_b] -= 1
                    new_solution.pop(i)
                    solution_set.discard(b_idx)
                    improved = True
                    cover_count = np.zeros(n_t, dtype=np.int32)
                    for b_idx2 in new_solution:
                        for s_idx in subsets_of_block[b_idx2]:
                            cover_count[s_idx] += 1
                    break
                else:
                    # Use bitset intersection to find candidate blocks
                    # Start with most constrained unique subset
                    unique_subs_sorted = sorted(unique_subs, key=lambda s: len(blocks_of_subset[s]))
                    candidate_mask = subset_mask[unique_subs_sorted[0]]
                    for s in unique_subs_sorted[1:]:
                        candidate_mask &= subset_mask[s]
                        if candidate_mask == 0:
                            break

                    if candidate_mask == 0:
                        continue

                    # Remove current block from candidates
                    candidate_mask &= ~(1 << b_idx)

                    if candidate_mask == 0:
                        continue

                    # Try candidates (limit to first few for speed)
                    best_replacement = -1
                    cm = candidate_mask
                    while cm:
                        lsb = cm & -cm
                        b_new = lsb.bit_length() - 1
                        if b_new not in solution_set:
                            best_replacement = b_new
                            break
                        cm ^= lsb

                    if best_replacement != -1:
                        cover_count[subs_b] -= 1
                        new_solution[i] = best_replacement
                        solution_set.discard(b_idx)
                        solution_set.add(best_replacement)
                        for s_idx in subsets_of_block[best_replacement]:
                            cover_count[s_idx] += 1
                        improved = True
                        break

            # Try 2-for-1 with packing LB pruning
            if not improved and len(best_solution) >= 2:
                for i in range(len(best_solution)):
                    if time.time() > deadline - 8:
                        break
                    b_i = best_solution[i]
                    subs_i = subsets_of_block[b_i]
                    unique_i = [s for s in subs_i if cover_count[s] == 1]

                    for j in range(i + 1, len(best_solution)):
                        b_j = best_solution[j]
                        subs_j = subsets_of_block[b_j]
                        unique_j = [s for s in subs_j if cover_count[s] == 1]

                        combined_unique = set(unique_i) | set(unique_j)
                        if not combined_unique:
                            continue

                        # Packing LB: if we need more than 1 block to cover combined_unique, skip
                        if len(combined_unique) > 1:
                            lb = packing_lb(combined_unique)
                            if lb > 1:
                                continue

                        # Bitset intersection for candidate
                        candidate_mask = None
                        for s in combined_unique:
                            if candidate_mask is None:
                                candidate_mask = subset_mask[s]
                            else:
                                candidate_mask &= subset_mask[s]
                            if candidate_mask == 0:
                                break

                        if candidate_mask is None or candidate_mask == 0:
                            continue

                        candidate_mask &= ~(1 << b_i)
                        candidate_mask &= ~(1 << b_j)

                        if candidate_mask == 0:
                            continue

                        best_replacement = -1
                        cm = candidate_mask
                        while cm:
                            lsb = cm & -cm
                            b_new = lsb.bit_length() - 1
                            if b_new not in solution_set:
                                best_replacement = b_new
                                break
                            cm ^= lsb

                        if best_replacement != -1:
                            cover_count[subs_i] -= 1
                            cover_count[subs_j] -= 1
                            new_solution.pop(j)
                            new_solution.pop(i)
                            solution_set.discard(b_i)
                            solution_set.discard(b_j)
                            new_solution.append(best_replacement)
                            solution_set.add(best_replacement)
                            for s_idx in subsets_of_block[best_replacement]:
                                cover_count[s_idx] += 1
                            improved = True
                            break

                    if improved:
                        break

            if improved:
                best_solution = new_solution
                best_size = len(best_solution)

    # 6. Final deletion pass
    if best_solution is not None and time.time() < deadline - 2:
        for pass_num in range(5):
            if time.time() > deadline - 2:
                break
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in best_solution:
                for s_idx in subsets_of_block[b_idx]:
                    cover_count[s_idx] += 1
            changed = True
            while changed and time.time() < deadline - 2:
                changed = False
                new_solution = []
                for b_idx in best_solution:
                    subs_b = subsets_of_block[b_idx]
                    can_remove = True
                    for s_idx in subs_b:
                        if cover_count[s_idx] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        cover_count[subs_b] -= 1
                        changed = True
                    else:
                        new_solution.append(b_idx)
                best_solution = new_solution
                best_size = len(best_solution)

    return [list(k_subsets[idx]) for idx in best_solution]
