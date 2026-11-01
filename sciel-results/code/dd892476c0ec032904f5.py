import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)

    # Incidence: for each block, which t-subsets does it cover
    block_cover = []
    for b in all_k_blocks:
        covered = []
        for ts in itertools.combinations(b, t):
            covered.append(ts_index[ts])
        block_cover.append(covered)

    # For each t-subset, which blocks cover it
    ts_to_blocks = [[] for _ in range(num_t)]
    for j, covered in enumerate(block_cover):
        for r in covered:
            ts_to_blocks[r].append(j)

    def C(n, r):
        if r < 0 or r > n:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        res = 1
        for i in range(1, r + 1):
            res = res * (n - i + 1) // i
        return res

    # ===================== Phase 1: Greedy initialization =====================
    def greedy_init():
        gain = np.full(num_k, C(k, t), dtype=np.int32)
        covered = np.zeros(num_t, dtype=bool)
        selected = np.zeros(num_k, dtype=bool)
        sol = []
        while True:
            if time.time() > deadline - 10.0:
                break
            best = np.argmax(gain)
            if gain[best] <= 0:
                break
            sol.append(list(all_k_blocks[best]))
            selected[best] = True
            gain[best] = -1
            for r in block_cover[best]:
                if not covered[r]:
                    covered[r] = True
                    for b in ts_to_blocks[r]:
                        if not selected[b]:
                            gain[b] -= 1
        return sol

    best_sol = greedy_init()
    best_len = len(best_sol)

    # ===================== Phase 2: Remove redundant blocks =====================
    def remove_redundant(blocks):
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index[ts]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index[ts]] == 1:
                        redundant = False
                        break
                if redundant:
                    rem = blocks.pop(i)
                    for ts in itertools.combinations(rem, t):
                        cover_count[ts_index[ts]] -= 1
                    changed = True
                    break
        return blocks

    best_sol = remove_redundant(best_sol)
    best_len = len(best_sol)

    # ===================== Phase 3: Simulated Annealing on block selection =====================
    # Represent solution as a set of block indices. We try to remove blocks (and possibly swap)
    # to reduce the count while maintaining full coverage.

    def coverage_check(blocks):
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
        return cc

    def is_valid(blocks):
        cc = coverage_check(blocks)
        return np.all(cc >= 1)

    def try_remove(blocks, target_idx):
        """Try removing blocks[target_idx]. Returns (success, new_blocks)."""
        if target_idx >= len(blocks):
            return False, blocks
        cc = coverage_check(blocks)
        rem = blocks[target_idx]
        for ts in itertools.combinations(rem, t):
            if cc[ts_index[ts]] == 1:
                return False, blocks
        new_blocks = blocks[:target_idx] + blocks[target_idx + 1:]
        return True, new_blocks

    def random_swap(blocks):
        """Randomly replace one block with another. Returns new blocks or None."""
        if len(blocks) == 0:
            return None
        i = random.randint(0, len(blocks) - 1)
        # Pick a random block not already in solution
        cur_set = set(blocks)
        candidates = [b for b in all_k_blocks if b not in cur_set]
        if not candidates:
            return None
        new_b = random.choice(candidates)
        new_blocks = blocks[:i] + [new_b] + blocks[i + 1:]
        return new_blocks

    def repair(blocks):
        """Greedily add blocks to cover any uncovered t-subsets."""
        cc = coverage_check(blocks)
        uncovered = np.where(cc < 1)[0]
        for r in uncovered:
            if time.time() > deadline - 3.0:
                break
            # Find a block covering this t-subset not already in solution
            cur_set = set(blocks)
            found = False
            for b in ts_to_blocks[r]:
                if b not in cur_set:
                    blocks.append(list(all_k_blocks[b]))
                    cur_set.add(all_k_blocks[b])
                    for ts2 in itertools.combinations(all_k_blocks[b], t):
                        cc[ts_index[ts2]] += 1
                    found = True
                    break
            if not found:
                # All blocks covering this t-subset are already in solution
                # This shouldn't happen if coverage is possible
                break
        return blocks

    # Simulated Annealing
    def simulated_annealing(initial_blocks, max_iters=20000):
        current = list(initial_blocks)
        current_len = len(current)
        best = list(current)
        best_len_sa = current_len
        cc = coverage_check(current)

        T = 1.0
        cooling = 0.99995
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            # Random move: try to remove a random block
            if random.random() < 0.7:
                if len(current) == 0:
                    continue
                i = random.randint(0, len(current) - 1)
                rem = current[i]
                # Check if removal is safe
                safe = True
                for ts in itertools.combinations(rem, t):
                    if cc[ts_index[ts]] == 1:
                        safe = False
                        break
                if safe:
                    # Accept removal
                    for ts in itertools.combinations(rem, t):
                        cc[ts_index[ts]] -= 1
                    current.pop(i)
                    current_len -= 1
                    if current_len < best_len_sa:
                        best_len_sa = current_len
                        best = list(current)
                else:
                    # With probability exp(-delta/T), accept even if unsafe
                    # Here delta = number of t-subsets that would become uncovered
                    delta = 0
                    for ts in itertools.combinations(rem, t):
                        if cc[ts_index[ts]] == 1:
                            delta += 1
                    if delta > 0 and random.random() < np.exp(-delta / max(T, 1e-10)):
                        for ts in itertools.combinations(rem, t):
                            cc[ts_index[ts]] -= 1
                        current.pop(i)
                        current_len -= 1
                        # Repair
                        current = repair(current)
                        cc = coverage_check(current)
                        current_len = len(current)
                        if current_len < best_len_sa:
                            best_len_sa = current_len
                            best = list(current)
                        else:
                            # Revert
                            current = best  # This is expensive; let's do it properly
                            cc = coverage_check(current)
                            current_len = len(current)
            else:
                # Try a swap
                new_b = random_swap(current)
                if new_b is None:
                    continue
                new_cc = coverage_check(new_b)
                new_len = len(new_b)
                # If new solution is valid and shorter, accept
                if np.all(new_cc >= 1) and new_len < current_len:
                    current = new_b
                    cc = new_cc
                    current_len = new_len
                    if current_len < best_len_sa:
                        best_len_sa = current_len
                        best = list(current)
                elif np.all(new_cc >= 1):
                    # Same length, valid - accept with some probability for diversity
                    if random.random() < 0.3:
                        current = new_b
                        cc = new_cc
                else:
                    # Invalid - repair and see if it's shorter
                    repaired = repair(new_b)
                    rcc = coverage_check(repaired)
                    if np.all(rcc >= 1) and len(repaired) < current_len:
                        current = repaired
                        cc = rcc
                        current_len = len(current)
                        if current_len < best_len_sa:
                            best_len_sa = current_len
                            best = list(current)

            T *= cooling

        return best

    # Run simulated annealing
    if time.time() < deadline - 10.0:
        sa_result = simulated_annealing(best_sol, max_iters=15000)
        sa_result = remove_redundant(sa_result)
        if len(sa_result) < best_len:
            best_len = len(sa_result)
            best_sol = sa_result

    # ===================== Phase 4: Local search - block swaps =====================
    def local_search_swap(blocks, max_rounds=50):
        cc = coverage_check(blocks)
        sol_set = set(blocks)
        improved = False
        for _ in range(max_rounds):
            if time.time() > deadline - 3.0:
                break
            any_swap = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    break
                # Find t-subsets uniquely covered by blocks[i]
                unique_ts = []
                for ts in itertools.combinations(blocks[i], t):
                    if cc[ts_index[ts]] == 1:
                        unique_ts.append(ts_index[ts])
                if not unique_ts:
                    # Block is redundant, remove it
                    rem = blocks[i]
                    for ts in itertools.combinations(rem, t):
                        cc[ts_index[ts]] -= 1
                    blocks.pop(i)
                    sol_set.discard(rem)
                    improved = True
                    any_swap = True
                    break
                # Find candidate blocks that cover all unique_ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                valid = [c for c in candidates if all_k_blocks[c] not in sol_set]
                if valid:
                    new_block = list(all_k_blocks[random.choice(valid)])
                    old_block = blocks[i]
                    # Update coverage
                    for ts in itertools.combinations(old_block, t):
                        cc[ts_index[ts]] -= 1
                    for ts in itertools.combinations(new_block, t):
                        cc[ts_index[ts]] += 1
                    blocks[i] = new_block
                    sol_set.discard(old_block)
                    sol_set.add(new_block)
                    improved = True
                    any_swap = True
                    break
            if not any_swap:
                break
        return blocks, improved

    # Run local search
    for _ in range(10):
        if time.time() > deadline - 3.0:
            break
        best_sol, improved = local_search_swap(best_sol)
        if improved:
            best_sol = remove_redundant(best_sol)
            best_len = len(best_sol)
        else:
            break

    # ===================== Phase 5: Multiple randomized restarts with greedy + local search =====================
    for restart in range(15):
        if time.time() > deadline - 10.0:
            break
        # Randomized greedy: perturb the gain
        random.seed(restart * 1000 + 42)
        gain = np.full(num_k, C(k, t), dtype=np.int32)
        covered = np.zeros(num_t, dtype=bool)
        selected = np.zeros(num_k, dtype=bool)
        sol = []
        while True:
            if time.time() > deadline - 5.0:
                break
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
            # Add small random perturbation to break ties
            perturbed = gain + np.random.uniform(0, 0.5, size=num_k)
            best = np.argmax(perturbed)
            if gain[best] <= 0:
                break
            sol.append(list(all_k_blocks[best]))
            selected[best] = True
            gain[best] = -1
            for r in block_cover[best]:
                if not covered[r]:
                    covered[r] = True
                    for b in ts_to_blocks[r]:
                        if not selected[b]:
                            gain[b] -= 1
        sol = remove_redundant(sol)
        # Local search
        sol, _ = local_search_swap(sol, max_rounds=20)
        sol = remove_redundant(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol

    # Final cleanup
    best_sol = remove_redundant(best_sol)
    return best_sol
