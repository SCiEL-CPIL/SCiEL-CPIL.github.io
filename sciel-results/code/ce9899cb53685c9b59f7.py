import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    # Build reverse index
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i in range(num_kblocks):
        for j in range(num_tsubs):
            if M[i, j] == 1:
                tsub_blocks[j].append(i)
    tsub_blocks = [np.array(lst, dtype=np.int64) if lst else np.array([], dtype=np.int64) for lst in tsub_blocks]

    rng = random.Random(42)

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    # Phase 1: Vectorized Greedy Construction
    uncovered = np.ones(num_tsubs, dtype=np.uint8)
    counts = np.zeros(num_tsubs, dtype=np.int32)
    selected_mask = np.zeros(num_kblocks, dtype=bool)
    selected_indices = []

    while np.any(uncovered):
        if time.monotonic() > deadline - 10.0:
            break
        gains = M @ uncovered
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        selected_mask[best_idx] = True
        row = M[best_idx]
        covered_indices = np.where(row == 1)[0]
        uncovered[covered_indices] = 0
        counts[covered_indices] += 1

    if not selected_indices:
        return []

    # Phase 2: Local Search Polish with ILS
    current_indices = selected_indices[:]
    current_counts = counts.copy()
    current_mask = selected_mask.copy()

    def try_remove_redundant():
        removed = False
        changed = True
        while changed:
            changed = False
            for b_idx in current_indices[:]:
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                if np.all(current_counts[covered_indices] > 1):
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    removed = True
                    changed = True
        return removed

    def try_swaps():
        improved = False
        idx_list = current_indices[:]
        rng.shuffle(idx_list)

        for b_idx in idx_list:
            if time.monotonic() > deadline - 2.0:
                break
            row = M[b_idx]
            covered_indices = np.where(row == 1)[0]
            critical_mask = current_counts == 1
            critical_indices = covered_indices[critical_mask[covered_indices]]

            if len(critical_indices) == 0:
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                current_counts[covered_indices] -= 1
                improved = True
                break

            if len(critical_indices) == 1:
                crit = critical_indices[0]
                candidates = tsub_blocks[crit]
                candidates = candidates[~current_mask[candidates]]
                if len(candidates) == 0:
                    continue
                best_cand = None
                best_score = -1
                for cand in candidates[:50]:
                    cand_row = M[cand]
                    score = np.sum(cand_row * (current_counts == 1))
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                if best_cand is not None:
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    current_mask[best_cand] = True
                    current_indices.append(best_cand)
                    cand_row = M[best_cand]
                    cand_covered = np.where(cand_row == 1)[0]
                    current_counts[cand_covered] += 1
                    improved = True
                    break
            elif len(critical_indices) <= 3:
                common = None
                for crit in critical_indices:
                    blocks_c = tsub_blocks[crit]
                    if common is None:
                        common = set(blocks_c.tolist())
                    else:
                        common &= set(blocks_c.tolist())
                if common:
                    candidates = np.array(list(common), dtype=np.int64)
                    candidates = candidates[~current_mask[candidates]]
                    if len(candidates) > 0:
                        best_cand = None
                        best_score = -1
                        for cand in candidates[:30]:
                            cand_row = M[cand]
                            score = np.sum(cand_row * (current_counts == 1))
                            if score > best_score:
                                best_score = score
                                best_cand = cand
                        if best_cand is not None:
                            current_mask[b_idx] = False
                            current_indices.remove(b_idx)
                            current_counts[covered_indices] -= 1
                            current_mask[best_cand] = True
                            current_indices.append(best_cand)
                            cand_row = M[best_cand]
                            cand_covered = np.where(cand_row == 1)[0]
                            current_counts[cand_covered] += 1
                            improved = True
                            break

        return improved

    def perturb():
        # Calculate marginal gains (how many unique subsets each block covers)
        # unique_mask is 1 where count is exactly 1
        unique_mask = (current_counts == 1).astype(np.uint8)
        gains = M @ unique_mask
        
        n_remove = min(5 + rng.randint(0, 5), max(1, len(current_indices) // 3))
        
        # Select blocks to remove: mix of low-gain and random
        # Low gain blocks are "safer" to remove as they contribute less to coverage
        sorted_indices = np.argsort(gains)
        low_gain_candidates = sorted_indices[:max(1, n_remove * 2)]
        
        # Filter to only those currently in the solution
        valid_low_gain = [i for i in low_gain_candidates if current_mask[i]]
        
        to_remove = set()
        
        # Pick some low gain
        if valid_low_gain:
            pick_low = rng.sample(valid_low_gain, min(len(valid_low_gain), n_remove // 2 + 1))
            to_remove.update(pick_low)
            
        # Pick some random from current
        remaining = [i for i in current_indices if i not in to_remove]
        if remaining:
            pick_rand = rng.sample(remaining, min(len(remaining), n_remove - len(to_remove)))
            to_remove.update(pick_rand)
            
        # If we still need more (unlikely), just take random
        if len(to_remove) < n_remove:
            remaining = [i for i in current_indices if i not in to_remove]
            if remaining:
                pick_extra = rng.sample(remaining, n_remove - len(to_remove))
                to_remove.update(pick_extra)

        to_remove_list = list(to_remove)
        
        for b_idx in to_remove_list:
            if b_idx in current_indices:
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                current_counts[covered_indices] -= 1

        uncovered_mask = (current_counts == 0).astype(np.uint8)
        while np.any(uncovered_mask):
            if time.monotonic() > deadline - 5.0:
                break
            gains_greedy = M @ uncovered_mask
            best_idx = np.argmax(gains_greedy)
            if gains_greedy[best_idx] == 0:
                break
            current_mask[best_idx] = True
            current_indices.append(best_idx)
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            current_counts[covered_indices] += 1
            uncovered_mask = (current_counts == 0).astype(np.uint8)

    # Main local search loop with Simulated Annealing
    best_len = len(current_indices)
    best_solution = current_indices[:]
    best_counts = current_counts.copy()
    best_mask = current_mask.copy()

    # SA Parameters
    T = 5.0
    cooling = 0.95
    min_T = 0.1
    stagnation = 0
    max_stagnation = 20 # Trigger kick more often

    while time.monotonic() < deadline - 3.0:
        improved = False
        old_len = len(current_indices)

        if try_remove_redundant():
            improved = True

        if not improved and try_swaps():
            improved = True

        if improved:
            stagnation = 0
            T = max(T * cooling, min_T)
            if len(current_indices) < best_len:
                best_len = len(current_indices)
                best_solution = current_indices[:]
                best_counts = current_counts.copy()
                best_mask = current_mask.copy()
        else:
            stagnation += 1
            # Kick condition: stagnation or random chance if T is high
            if stagnation >= max_stagnation or (T > min_T and rng.random() < 0.2):
                # Save state for potential revert
                save_indices = current_indices[:]
                save_counts = current_counts.copy()
                save_mask = current_mask.copy()
                save_len = len(current_indices)

                perturb()
                try_remove_redundant()
                
                new_len = len(current_indices)
                delta = new_len - save_len
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    # Accept to explore landscape
                    accept = True 
                else:
                    # Worse, accept with prob exp(-delta/T)
                    p_accept = np.exp(-delta / T)
                    if rng.random() < p_accept:
                        accept = True

                if accept:
                    T = max(T * cooling, min_T)
                    stagnation = 0
                    if new_len < best_len:
                        best_len = new_len
                        best_solution = current_indices[:]
                        best_counts = current_counts.copy()
                        best_mask = current_mask.copy()
                else:
                    # Revert
                    current_indices = save_indices
                    current_counts = save_counts
                    current_mask = save_mask
                    # Reheat slightly to encourage more exploration
                    T = min(T * 1.1, 10.0)
                    stagnation = 0
            
            # Decay temperature
            T = max(T * cooling, min_T)

    final_indices = best_solution
    final_counts = best_counts.copy()
    final_mask = best_mask.copy()

    for b_idx in final_indices[:]:
        if not final_mask[b_idx]:
            continue
        row = M[b_idx]
        covered_indices = np.where(row == 1)[0]
        if np.all(final_counts[covered_indices] > 1):
            final_mask[b_idx] = False
            final_indices.remove(b_idx)
            final_counts[covered_indices] -= 1

    verify_counts = np.zeros(num_tsubs, dtype=np.int32)
    for b_idx in final_indices:
        row = M[b_idx]
        covered_indices = np.where(row == 1)[0]
        verify_counts[covered_indices] += 1

    if not np.all(verify_counts >= 1):
        final_indices = selected_indices

    return get_blocks(final_indices)
