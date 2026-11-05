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
        n_remove = min(3 + rng.randint(0, 3), max(1, len(current_indices) // 4))
        to_remove = rng.sample(current_indices, n_remove)
        for b_idx in to_remove:
            current_mask[b_idx] = False
            current_indices.remove(b_idx)
            row = M[b_idx]
            covered_indices = np.where(row == 1)[0]
            current_counts[covered_indices] -= 1

        uncovered_mask = (current_counts == 0).astype(np.uint8)
        while np.any(uncovered_mask):
            if time.monotonic() > deadline - 5.0:
                break
            gains = M @ uncovered_mask
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            current_mask[best_idx] = True
            current_indices.append(best_idx)
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            current_counts[covered_indices] += 1
            uncovered_mask = (current_counts == 0).astype(np.uint8)

    # Main local search loop
    best_len = len(current_indices)
    best_solution = current_indices[:]
    best_counts = current_counts.copy()
    best_mask = current_mask.copy()

    stagnation = 0
    max_stagnation = 30

    while time.monotonic() < deadline - 3.0:
        improved = False

        if try_remove_redundant():
            improved = True

        if not improved and try_swaps():
            improved = True

        if improved:
            stagnation = 0
            if len(current_indices) < best_len:
                best_len = len(current_indices)
                best_solution = current_indices[:]
                best_counts = current_counts.copy()
                best_mask = current_mask.copy()
        else:
            stagnation += 1
            if stagnation >= max_stagnation:
                perturb()
                try_remove_redundant()
                if len(current_indices) < best_len:
                    best_len = len(current_indices)
                    best_solution = current_indices[:]
                    best_counts = current_counts.copy()
                    best_mask = current_mask.copy()
                stagnation = 0

    final_indices = best_solution
    final_counts = best_counts.copy()
    final_mask = best_mask.copy()

    # Final high-precision polish: multiple passes of removal and swap
    for polish_pass in range(5):
        if time.monotonic() > deadline - 1.0:
            break
        # Remove redundant
        changed = True
        while changed:
            changed = False
            for b_idx in final_indices[:]:
                if not final_mask[b_idx]:
                    continue
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                if np.all(final_counts[covered_indices] > 1):
                    final_mask[b_idx] = False
                    final_indices.remove(b_idx)
                    final_counts[covered_indices] -= 1
                    changed = True
        # Try swaps
        idx_list = final_indices[:]
        rng.shuffle(idx_list)
        for b_idx in idx_list:
            if time.monotonic() > deadline - 1.0:
                break
            if not final_mask[b_idx]:
                continue
            row = M[b_idx]
            covered_indices = np.where(row == 1)[0]
            critical_mask = final_counts == 1
            critical_indices = covered_indices[critical_mask[covered_indices]]
            
            if len(critical_indices) == 0:
                final_mask[b_idx] = False
                final_indices.remove(b_idx)
                final_counts[covered_indices] -= 1
                continue
            
            if len(critical_indices) == 1:
                crit = critical_indices[0]
                candidates = tsub_blocks[crit]
                candidates = candidates[~final_mask[candidates]]
                if len(candidates) == 0:
                    continue
                best_cand = None
                best_score = -1
                for cand in candidates[:80]:
                    cand_row = M[cand]
                    score = np.sum(cand_row * (final_counts == 1))
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                if best_cand is not None:
                    final_mask[b_idx] = False
                    final_indices.remove(b_idx)
                    final_counts[covered_indices] -= 1
                    final_mask[best_cand] = True
                    final_indices.append(best_cand)
                    cand_row = M[best_cand]
                    cand_covered = np.where(cand_row == 1)[0]
                    final_counts[cand_covered] += 1

    verify_counts = np.zeros(num_tsubs, dtype=np.int32)
    for b_idx in final_indices:
        row = M[b_idx]
        covered_indices = np.where(row == 1)[0]
        verify_counts[covered_indices] += 1

    if not np.all(verify_counts >= 1):
        final_indices = selected_indices

    return get_blocks(final_indices)
