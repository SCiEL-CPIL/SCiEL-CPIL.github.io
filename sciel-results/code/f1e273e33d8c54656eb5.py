import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n:
            return 0
        if r == 0:
            return 1
        num = 1
        den = 1
        for i in range(r):
            num *= (n - i)
            den *= (i + 1)
        return num // den

    t_count = nCr(k, t)

    # Precompute: for each block, which t-subsets it covers
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]

    # Precompute: for each triple index, which blocks contain it
    triple_to_blocks = defaultdict(list)
    for bi in range(n_ks):
        for ti in block_t_indices[bi]:
            triple_to_blocks[ti].append(bi)

    def greedy(seed):
        """Greedy set cover with rarity-weighted tie-breaking."""
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        blocks = []
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        remaining = n_ts

        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                break

            # Compute gain: number of new t-subsets covered
            gains = (~covered[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]

            if len(candidates) > 1:
                # Rarity-weighted scoring: prefer blocks covering rare triples
                best_score = -float('inf')
                best_idx = candidates[0]
                for c_idx in candidates:
                    subs = block_t_indices[c_idx]
                    newly = ~covered[subs]
                    if np.any(newly):
                        # Sum of inverse (1 + cover_freq) for newly covered
                        bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly]]))
                    else:
                        bonus = 0.0
                    score = max_gain + 0.1 * bonus + rng.uniform(0, 0.1)
                    if score > best_score:
                        best_score = score
                        best_idx = c_idx
            else:
                best_idx = candidates[0]

            blocks.append(best_idx)
            subs = block_t_indices[best_idx]
            newly = ~covered[subs]
            remaining -= newly.sum()
            cover_freq[subs] += 1
            covered[subs] = True

        return blocks

    def greedy_diverse(seed):
        """Greedy that starts by forcing coverage of the most uncovered triples first."""
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        blocks = []
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        remaining = n_ts

        # Start with a random triple to force diversity
        if n_ts > 0:
            first_triple_idx = rng.randint(0, n_ts - 1)
            first_block_candidates = triple_to_blocks[first_triple_idx]
            if first_block_candidates:
                first_block = first_block_candidates[rng.randint(0, len(first_block_candidates) - 1)]
                blocks.append(first_block)
                subs = block_t_indices[first_block]
                newly = ~covered[subs]
                remaining -= newly.sum()
                cover_freq[subs] += 1
                covered[subs] = True

        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                break

            gains = (~covered[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]

            if len(candidates) > 1:
                best_score = -float('inf')
                best_idx = candidates[0]
                for c_idx in candidates:
                    subs = block_t_indices[c_idx]
                    newly = ~covered[subs]
                    if np.any(newly):
                        bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly]]))
                    else:
                        bonus = 0.0
                    score = max_gain + 0.1 * bonus + rng.uniform(0, 0.1)
                    if score > best_score:
                        best_score = score
                        best_idx = c_idx
            else:
                best_idx = candidates[0]

            blocks.append(best_idx)
            subs = block_t_indices[best_idx]
            newly = ~covered[subs]
            remaining -= newly.sum()
            cover_freq[subs] += 1
            covered[subs] = True

        return blocks

    def build_coverage(blocks_list):
        """Build coverage count array."""
        cov = np.zeros(n_ts, dtype=np.int32)
        for b_idx in blocks_list:
            cov[block_t_indices[b_idx]] += 1
        return cov

    def aggressive_deletion(blocks_list, coverage_count, time_budget):
        """Multi-pass aggressive block removal."""
        current = list(blocks_list)
        deadline = time.time() + time_budget
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_current = []
            for b_idx in current:
                subs = block_t_indices[b_idx]
                other_covers = coverage_count[subs] - 1
                if np.all(other_covers > 0):
                    improved = True
                    coverage_count[subs] -= 1
                else:
                    new_current.append(b_idx)
            current = new_current
        return current

    def local_search_remove(blocks_list):
        """Remove redundant blocks iteratively."""
        current = list(blocks_list)
        coverage_count = build_coverage(current)
        return aggressive_deletion(current, coverage_count, 5.0)

    def local_search_replace(blocks_list, time_budget):
        """Try to replace pairs of blocks with a single block (2-for-1 swap)."""
        current = list(blocks_list)
        deadline = time.time() + time_budget
        coverage_count = build_coverage(current)

        while time.time() < deadline:
            n_blocks = len(current)
            found_swap = False

            for i in range(n_blocks):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_blocks):
                    if time.time() > deadline:
                        break

                    b1 = current[i]
                    b2 = current[j]
                    subs1 = block_t_indices[b1]
                    subs2 = block_t_indices[b2]

                    # Critical triples: those that would become uncovered if we remove both
                    critical = set()
                    for s in subs1:
                        if coverage_count[s] == 1:
                            critical.add(s)
                    for s in subs2:
                        if coverage_count[s] == 1:
                            critical.add(s)
                    set1 = set(subs1.tolist())
                    set2 = set(subs2.tolist())
                    for s in set1 & set2:
                        if coverage_count[s] == 2:
                            critical.add(s)

                    if not critical:
                        continue

                    # Find a single block that covers all critical triples
                    crit_triples = [all_ts[c] for c in critical]
                    all_points_needed = set()
                    for ct in crit_triples:
                        all_points_needed.update(ct)
                    if len(all_points_needed) > k:
                        continue

                    first_tri = crit_triples[0]
                    first_tri_idx = idx_map[first_tri]
                    candidate_blocks = triple_to_blocks[first_tri_idx]

                    for cb in candidate_blocks:
                        cb_subs = set(block_t_indices[cb].tolist())
                        if all(c in cb_subs for c in critical):
                            # Valid replacement
                            new_current = [b for idx, b in enumerate(current) if idx != i and idx != j]
                            if cb not in new_current:
                                new_current.append(cb)
                            current = new_current
                            coverage_count = build_coverage(current)
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    break

            if not found_swap:
                break

        return current

    def local_search_single(blocks_list, time_budget):
        """Try to replace one block with another that covers all its unique triples."""
        current = list(blocks_list)
        deadline = time.time() + time_budget
        coverage_count = build_coverage(current)

        while time.time() < deadline:
            n_blocks = len(current)
            found = False

            for i in range(n_blocks):
                if time.time() > deadline:
                    break
                b = current[i]
                subs = block_t_indices[b]
                unique = [s for s in subs if coverage_count[s] == 1]
                if not unique:
                    continue

                first_tri = all_ts[unique[0]]
                first_tri_idx = idx_map[first_tri]
                candidates = triple_to_blocks[first_tri_idx]

                for cb in candidates:
                    if cb == b:
                        continue
                    cb_subs = set(block_t_indices[cb].tolist())
                    if all(s in cb_subs for s in unique):
                        new_current = current[:i] + current[i+1:]
                        if cb not in new_current:
                            new_current.append(cb)
                        current = new_current
                        coverage_count = build_coverage(current)
                        found = True
                        break
                if found:
                    break

            if not found:
                break

        return current

    def simulated_annealing(blocks_list, time_budget, rng):
        """Simulated annealing with add/remove/swap moves."""
        current = list(blocks_list)
        deadline = time.time() + time_budget

        coverage = build_coverage(current)
        uncovered = np.sum(coverage == 0)
        energy = uncovered + 0.01 * len(current)

        best_energy = energy
        best_state = list(current)

        T = 3.0
        T_end = 0.01
        cooling = 0.997

        current_set = set(current)

        while time.time() < deadline and T > T_end:
            T *= cooling

            move_type = rng.random()

            if move_type < 0.25 and len(current) < n_ks:
                # Add a random block
                new_block = rng.randint(0, n_ks - 1)
                if new_block in current_set:
                    continue
                subs = block_t_indices[new_block]
                delta_uncovered = -np.sum(coverage[subs] == 0)
                delta_energy = delta_uncovered + 0.01
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / T):
                    current.append(new_block)
                    current_set.add(new_block)
                    coverage[subs] += 1
                    energy += delta_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_state = list(current)

            elif move_type < 0.50 and len(current) > 1:
                # Remove a random block
                idx = rng.randint(0, len(current) - 1)
                b_idx = current[idx]
                subs = block_t_indices[b_idx]
                delta_uncovered = np.sum(coverage[subs] == 1)
                delta_energy = delta_uncovered - 0.01
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / T):
                    current.pop(idx)
                    current_set.discard(b_idx)
                    coverage[subs] -= 1
                    energy += delta_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_state = list(current)

            else:
                # Swap
                if len(current) < 1:
                    continue
                idx = rng.randint(0, len(current) - 1)
                old_block = current[idx]
                new_block = rng.randint(0, n_ks - 1)
                if new_block == old_block or new_block in current_set:
                    continue

                subs_old = block_t_indices[old_block]
                subs_new = block_t_indices[new_block]

                # Compute delta carefully
                delta_uncovered = 0
                # Removing old_block
                for s in subs_old:
                    if coverage[s] == 1:
                        delta_uncovered += 1
                # Adding new_block (after removing old)
                for s in subs_new:
                    if s in subs_old:
                        pass  # net zero for overlap
                    else:
                        if coverage[s] == 0:
                            delta_uncovered -= 1

                delta_energy = delta_uncovered
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / T):
                    current[idx] = new_block
                    current_set.discard(old_block)
                    current_set.add(new_block)
                    coverage[subs_old] -= 1
                    coverage[subs_new] += 1
                    energy += delta_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_state = list(current)

        return best_state

    # Main loop
    best_blocks = None
    best_count = float('inf')
    restart_deadline = time_limit * 0.45

    seed = 42
    while time.time() - start_time < restart_deadline:
        # Alternate between standard greedy and diverse greedy
        if seed % 2 == 0:
            greedy_blocks = greedy(seed)
        else:
            greedy_blocks = greedy_diverse(seed)
        
        if not greedy_blocks:
            seed += 1
            continue

        # Local search pipeline
        refined = local_search_remove(greedy_blocks)
        if time.time() - start_time < time_limit - 15:
            refined = local_search_replace(refined, min(1.0, max(0.1, (time_limit - (time.time() - start_time)) * 0.03)))
        if time.time() - start_time < time_limit - 12:
            refined = local_search_remove(refined)

        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = refined

        seed += 1
        if best_count <= 70:
            break

    # Final polish
    if best_blocks is not None and time.time() - start_time < time_limit - 5:
        polished = local_search_remove(best_blocks)
        if time.time() - start_time < time_limit - 4:
            polished = local_search_replace(polished, min(4.0, max(0.5, time_limit - (time.time() - start_time) - 4.0)))
        if time.time() - start_time < time_limit - 3:
            polished = local_search_remove(polished)

        # Simulated annealing
        if time.time() - start_time < time_limit - 2:
            sa_time = min(6.0, max(0.5, time_limit - (time.time() - start_time) - 2.0))
            sa_rng = random.Random(12345)
            sa_result = simulated_annealing(polished, sa_time, sa_rng)
            if len(sa_result) < len(polished):
                polished = sa_result

        if time.time() - start_time < time_limit - 1:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 0.5:
            polished = local_search_single(polished, min(2.0, max(0.2, time_limit - (time.time() - start_time) - 0.5)))
        best_blocks = polished

    final_blocks = [list(all_ks[i]) for i in best_blocks]
    return final_blocks
