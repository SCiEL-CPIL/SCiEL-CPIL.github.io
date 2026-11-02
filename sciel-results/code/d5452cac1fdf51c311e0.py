import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # Build incidence: for each block, which t-subsets it covers
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for block in all_k_blocks:
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(len(block_to_ts))
        block_to_ts.append(ts_list)

    # Convert to numpy arrays for speed
    block_to_ts_np = [np.array(ts, dtype=np.int32) for ts in block_to_ts]

    # Greedy initialization
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)

    solution_blocks_idx = []
    while True:
        if time.time() > deadline - 10.0:
            break
        max_gain = gain.max()
        if max_gain <= 0:
            break
        candidates = np.where(gain == max_gain)[0]
        max_idx = candidates[0]
        solution_blocks_idx.append(max_idx)
        is_selected[max_idx] = True
        gain[max_idx] = -1
        for ts_idx in block_to_ts[max_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1

    best_solution_idx = solution_blocks_idx[:]
    best_len = len(solution_blocks_idx)

    def get_cover_counts(x):
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for j in range(num_k_blocks):
            if x[j]:
                counts[block_to_ts_np[j]] += 1
        return counts

    x = np.zeros(num_k_blocks, dtype=bool)
    for j in best_solution_idx:
        x[j] = True
    cover_count = get_cover_counts(x)

    # Density-driven block generation: given current uncovered set, find best new blocks
    def find_density_blocks(uncovered_mask, num_to_find=20):
        """Find blocks that maximize coverage of currently uncovered t-subsets."""
        # For each block, compute how many uncovered t-subsets it covers
        scores = np.zeros(num_k_blocks, dtype=np.int32)
        for j in range(num_k_blocks):
            ts_arr = block_to_ts_np[j]
            scores[j] = np.sum(uncovered_mask[ts_arr])
        
        # Get top scoring blocks
        top_indices = np.argsort(scores)[::-1][:num_to_find]
        return [(j, scores[j]) for j in top_indices if scores[j] > 0]

    # Local search with removals
    def local_search_removal(x, cover_count, max_iters=3000):
        x = x.copy()
        cover_count = cover_count.copy()
        current_cost = int(np.sum(x))
        best_cost = current_cost
        best_x = x.copy()
        lb = (C(v, t) + C(k, t) - 1) // C(k, t)
        
        active_blocks = np.where(x)[0].tolist()
        no_improvement = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if not active_blocks:
                break
            improved = False
            random.shuffle(active_blocks)
            for j in active_blocks:
                if np.all(cover_count[block_to_ts_np[j]] > 1):
                    x[j] = False
                    cover_count[block_to_ts_np[j]] -= 1
                    current_cost -= 1
                    active_blocks.remove(j)
                    improved = True
                    break
            if improved:
                no_improvement = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
                    if best_cost == lb:
                        return best_x, best_cost
            else:
                no_improvement += 1
                if no_improvement > 30:
                    break
        return best_x, best_cost

    # Local search with swaps
    def local_search_swap(x, cover_count, max_iters=2000):
        x = x.copy()
        cover_count = cover_count.copy()
        current_cost = int(np.sum(x))
        best_cost = current_cost
        best_x = x.copy()
        lb = (C(v, t) + C(k, t) - 1) // C(k, t)
        
        active_blocks = np.where(x)[0].tolist()
        inactive_blocks = np.where(~x)[0].tolist()
        
        # Precompute sets for fast subset checks
        block_to_ts_set = [set(block_to_ts_np[j].tolist()) for j in range(num_k_blocks)]
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if not active_blocks:
                break
            
            # Try removals first
            removed_any = False
            random.shuffle(active_blocks)
            for j in active_blocks:
                if np.all(cover_count[block_to_ts_np[j]] > 1):
                    x[j] = False
                    cover_count[block_to_ts_np[j]] -= 1
                    current_cost -= 1
                    active_blocks.remove(j)
                    inactive_blocks.append(j)
                    removed_any = True
                    break
            if removed_any:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
                    if best_cost == lb:
                        return best_x, best_cost
                continue
            
            # Try 1-swap
            if not inactive_blocks:
                break
            j = random.choice(active_blocks)
            unique_ts_j = set(ts_idx for ts_idx in block_to_ts_np[j] if cover_count[ts_idx] == 1)
            
            if not unique_ts_j:
                continue
            
            sample_size = min(80, len(inactive_blocks))
            sample = random.sample(inactive_blocks, sample_size)
            found = False
            for j_prime in sample:
                if unique_ts_j.issubset(block_to_ts_set[j_prime]):
                    x[j] = False
                    cover_count[block_to_ts_np[j]] -= 1
                    x[j_prime] = True
                    cover_count[block_to_ts_np[j_prime]] += 1
                    if np.min(cover_count) >= 1:
                        active_blocks.remove(j)
                        active_blocks.append(j_prime)
                        inactive_blocks.remove(j_prime)
                        inactive_blocks.append(j)
                        found = True
                        break
                    else:
                        x[j_prime] = False
                        cover_count[block_to_ts_np[j_prime]] -= 1
                        x[j] = True
                        cover_count[block_to_ts_np[j]] += 1
            if found:
                continue
            
            # Try 2-exchange
            if len(active_blocks) < 2:
                break
            critical_blocks = [j for j in active_blocks if not np.all(cover_count[block_to_ts_np[j]] > 1)]
            if len(critical_blocks) < 2:
                continue
            j1 = random.choice(critical_blocks)
            j2_candidates = [j for j in critical_blocks if j != j1]
            if not j2_candidates:
                continue
            j2 = random.choice(j2_candidates)
            
            unique_ts_12 = set()
            for ts_idx in block_to_ts_np[j1]:
                if cover_count[ts_idx] == 1:
                    unique_ts_12.add(ts_idx)
            for ts_idx in block_to_ts_np[j2]:
                if cover_count[ts_idx] == 1:
                    unique_ts_12.add(ts_idx)
            
            if not unique_ts_12:
                continue
            sample_size = min(120, len(inactive_blocks))
            sample = random.sample(inactive_blocks, sample_size)
            found_swap = False
            for j3 in sample:
                if unique_ts_12.issubset(block_to_ts_set[j3]):
                    x[j1] = False
                    cover_count[block_to_ts_np[j1]] -= 1
                    x[j2] = False
                    cover_count[block_to_ts_np[j2]] -= 1
                    x[j3] = True
                    cover_count[block_to_ts_np[j3]] += 1
                    if np.min(cover_count) >= 1:
                        active_blocks.remove(j1)
                        active_blocks.remove(j2)
                        active_blocks.append(j3)
                        inactive_blocks.remove(j3)
                        inactive_blocks.append(j1)
                        inactive_blocks.append(j2)
                        current_cost -= 1
                        found_swap = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_x = x.copy()
                            if best_cost == lb:
                                return best_x, best_cost
                        break
                    else:
                        x[j3] = False
                        cover_count[block_to_ts_np[j3]] -= 1
                        x[j2] = True
                        cover_count[block_to_ts_np[j2]] += 1
                        x[j1] = True
                        cover_count[block_to_ts_np[j1]] += 1
                        current_cost += 1
        return best_x, best_cost

    # Density-driven reconstruction: rebuild solution using density heuristic
    def density_reconstruct(max_time=10.0):
        """Rebuild a solution from scratch using density-driven block selection."""
        end_time = time.time() + max_time
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        selected = []
        
        while time.time() < end_time:
            # Find blocks with highest density of uncovered t-subsets
            uncovered_mask = is_uncovered
            scores = np.zeros(num_k_blocks, dtype=np.int32)
            for j in range(num_k_blocks):
                ts_arr = block_to_ts_np[j]
                scores[j] = np.sum(uncovered_mask[ts_arr])
            
            max_score = scores.max()
            if max_score <= 0:
                break
            
            # Pick a block with max score (random tie-break for diversity)
            candidates = np.where(scores == max_score)[0]
            j = random.choice(candidates)
            selected.append(j)
            
            # Mark covered t-subsets
            for ts_idx in block_to_ts[j]:
                is_uncovered[ts_idx] = False
            
            if not is_uncovered.any():
                break
        
        if is_uncovered.any():
            # Fallback: add greedy blocks to cover remaining
            gain = np.zeros(num_k_blocks, dtype=np.int32)
            for j in range(num_k_blocks):
                gain[j] = np.sum(is_uncovered[block_to_ts_np[j]])
            while is_uncovered.any():
                if time.time() > end_time:
                    break
                max_g = gain.max()
                if max_g <= 0:
                    break
                candidates = np.where(gain == max_g)[0]
                j = candidates[0]
                selected.append(j)
                for ts_idx in block_to_ts[j]:
                    if is_uncovered[ts_idx]:
                        is_uncovered[ts_idx] = False
                gain[j] = -1
                for ts_idx in block_to_ts[j]:
                    if not is_uncovered[ts_idx]:
                        for b in ts_to_blocks[ts_idx]:
                            if b not in selected:
                                gain[b] -= 1
        
        return selected

    # Run the search
    current_x = x
    current_cc = cover_count

    # Phase 1: Hill climbing removals
    current_x, current_cost = local_search_removal(current_x, current_cc, max_iters=3000)
    if current_cost < best_len:
        best_len = current_cost
        best_solution_idx = np.where(current_x)[0].tolist()
        current_cc = get_cover_counts(current_x)

    # Phase 2: Multiple restarts with swaps
    for restart in range(15):
        if time.time() > deadline - 5.0:
            break
        random.seed(restart * 12345 + 67890)
        new_x, new_cost = local_search_swap(current_x, current_cc, max_iters=800)
        if new_cost < best_len:
            best_len = new_cost
            best_solution_idx = np.where(new_x)[0].tolist()
            current_x = new_x
            current_cc = get_cover_counts(new_x)
            current_x, current_cost = local_search_removal(current_x, current_cc, max_iters=1000)
            if current_cost < best_len:
                best_len = current_cost
                best_solution_idx = np.where(current_x)[0].tolist()
                current_cc = get_cover_counts(current_x)

    # Phase 3: Density-driven reconstruction attempts
    if time.time() < deadline - 20.0:
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
            random.seed(attempt * 99999 + 1)
            density_sel = density_reconstruct(max_time=min(8.0, deadline - 10.0 - time.time()))
            if len(density_sel) < best_len:
                # Polish with local search
                dx = np.zeros(num_k_blocks, dtype=bool)
                for j in density_sel:
                    dx[j] = True
                dcc = get_cover_counts(dx)
                dx, dcost = local_search_removal(dx, dcc, max_iters=1500)
                if dcost < best_len:
                    best_len = dcost
                    best_solution_idx = np.where(dx)[0].tolist()
                    current_x = dx
                    current_cc = get_cover_counts(dx)
                dx, dcost = local_search_swap(dx, get_cover_counts(dx), max_iters=500)
                if dcost < best_len:
                    best_len = dcost
                    best_solution_idx = np.where(dx)[0].tolist()
                    current_x = dx
                    current_cc = get_cover_counts(dx)

    # Final polish
    if time.time() < deadline - 5.0:
        final_x = np.zeros(num_k_blocks, dtype=bool)
        for j in best_solution_idx:
            final_x[j] = True
        final_cc = get_cover_counts(final_x)
        final_x, final_cost = local_search_removal(final_x, final_cc, max_iters=2000)
        if final_cost < best_len:
            best_len = final_cost
            best_solution_idx = np.where(final_x)[0].tolist()

    final_blocks = [list(all_k_blocks[j]) for j in best_solution_idx]
    return final_blocks
