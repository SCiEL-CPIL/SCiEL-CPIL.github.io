import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s wall-clock limit

    # Precomputation
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

    # block_t_indices[i] = array of t-subset indices contained in k-subset i
    # Use only the first t_count entries, rest are padding
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
    
    # Precompute valid indices per block (all are valid since k choose t = t_count)
    block_valid_mask = np.ones((n_ks, t_count), dtype=bool)

    # Precompute: for each triple index, which blocks contain it
    # triple_to_blocks[tri] = list of block indices containing that triple
    triple_to_blocks = defaultdict(list)
    for bi in range(n_ks):
        for ti in block_t_indices[bi]:
            triple_to_blocks[ti].append(bi)

    def greedy(seed):
        """Greedy set cover with smooth scoring and randomized tie-breaking."""
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        blocks = []
        point_load = np.zeros(v, dtype=np.int32)

        while True:
            uncovered_flags = ~covered[block_t_indices]
            gains = uncovered_flags.sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                # Smooth scoring: use log-sum-exp style soft-max for tie-breaking
                # Prefer blocks with lower average coverage and balanced point loads
                scores = np.zeros(len(candidates), dtype=np.float64)
                for ci, c_idx in enumerate(candidates):
                    block = all_ks[c_idx]
                    load = float(np.sum(point_load[list(block)]))
                    avg_cov = float(np.mean(coverage_count[block_t_indices[c_idx]]))
                    # Smooth penalty: exponential decay to avoid sharp discontinuities
                    scores[ci] = -0.5 * load - 1.0 * avg_cov + rng.gauss(0, 0.2)
                best_local = np.argmax(scores)
                best_idx = candidates[best_local]
            else:
                best_idx = candidates[0]

            blocks.append(best_idx)
            block = all_ks[best_idx]
            subs_in_block = block_t_indices[best_idx]
            covered[subs_in_block] = True
            coverage_count[subs_in_block] += 1
            for p in block:
                point_load[p] += 1

            if np.all(covered):
                break

        return blocks

    def local_search_remove(blocks_list):
        """Remove redundant blocks iteratively (optimized)."""
        current = list(blocks_list)
        improved = True
        while improved:
            improved = False
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1

            new_current = []
            for b_idx in current:
                subs = block_t_indices[b_idx]
                # Check if all triples in this block have coverage > 1
                if np.all(coverage_count[subs] > 1):
                    improved = True
                    coverage_count[subs] -= 1
                else:
                    new_current.append(b_idx)
            current = new_current
        return current

    def local_search_replace(blocks_list, time_budget):
        """Try to replace pairs of blocks with a single block (2-for-1 swap).
        Uses a smooth surrogate: prefer replacements that minimize the soft-min
        of coverage counts (log-sum-exp) over affected triples, breaking ties
        by maximum additional coverage."""
        current = list(blocks_list)
        deadline = time.time() + time_budget

        while time.time() < deadline:
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1

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

                    # Critical triples: those that would become uncovered if we remove both b1 and b2
                    critical_set = set()
                    for s in subs1:
                        if coverage_count[s] == 1:
                            critical_set.add(int(s))
                    for s in subs2:
                        if coverage_count[s] == 1:
                            critical_set.add(int(s))
                    # Triples in both with cov==2
                    set1 = set(subs1.tolist())
                    set2 = set(subs2.tolist())
                    for s in set1 & set2:
                        if coverage_count[s] == 2:
                            critical_set.add(int(s))

                    if not critical_set:
                        continue

                    # Find a single block that covers all critical triples
                    all_points_needed = set()
                    for c in critical_set:
                        all_points_needed.update(all_ts[c])
                    if len(all_points_needed) > k:
                        continue

                    # Use the first critical triple to narrow down candidates
                    first_tri_idx = list(critical_set)[0]
                    candidate_blocks = triple_to_blocks[first_tri_idx]

                    best_cb = None
                    best_score = float('-inf')
                    for cb in candidate_blocks:
                        cb_subs = set(block_t_indices[cb].tolist())
                        if critical_set.issubset(cb_subs):
                            # Smooth surrogate: compute soft-min of coverage after swap
                            # Triples affected: those in subs1 or subs2 that are not in cb_subs
                            affected = set()
                            for s in subs1:
                                si = int(s)
                                if si not in cb_subs:
                                    affected.add(si)
                            for s in subs2:
                                si = int(s)
                                if si not in cb_subs:
                                    affected.add(si)
                            # Compute new coverage for affected triples
                            new_covs = np.array([max(0, int(coverage_count[si]) - 1) for si in affected], dtype=np.float64)
                            if len(new_covs) > 0:
                                # Soft-min via log-sum-exp (temperature 0.5)
                                temp = 0.5
                                shifted = new_covs / temp
                                max_val = np.max(shifted)
                                log_sum_exp = max_val + np.log(np.sum(np.exp(shifted - max_val)))
                                soft_min = -temp * log_sum_exp
                            else:
                                soft_min = 0.0
                            # Additional coverage gain: triples in cb not in subs1 ∪ subs2
                            union_12 = set1 | set2
                            extra = len(cb_subs - union_12)
                            # Combined score: minimize soft_min (maximize -soft_min), maximize extra
                            score = -soft_min + 0.3 * extra
                            if score > best_score:
                                best_score = score
                                best_cb = cb

                    if best_cb is not None:
                        new_current = [b for idx, b in enumerate(current) if idx != i and idx != j]
                        if best_cb not in new_current:
                            new_current.append(best_cb)
                        current = new_current
                        found_swap = True
                        break
                if found_swap:
                    break

            if not found_swap:
                break

        return current

    def local_search_single(blocks_list, time_budget):
        """Try to replace one block with another that covers all its unique triples.
        Uses smooth surrogate: minimize soft-min of coverage over affected triples."""
        current = list(blocks_list)
        deadline = time.time() + time_budget

        while time.time() < deadline:
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1

            n_blocks = len(current)
            found = False

            for i in range(n_blocks):
                if time.time() > deadline:
                    break
                b = current[i]
                subs = block_t_indices[b]
                # Triples uniquely covered by b (coverage_count == 1)
                unique = [int(s) for s in subs if coverage_count[s] == 1]
                if not unique:
                    continue

                # Find a block that covers all unique triples
                first_tri_idx = unique[0]
                candidates = triple_to_blocks[first_tri_idx]
                unique_set = set(unique)

                best_cb = None
                best_score = float('-inf')
                for cb in candidates:
                    if cb == b:
                        continue
                    cb_subs = set(block_t_indices[cb].tolist())
                    if unique_set.issubset(cb_subs):
                        # Triples affected: in subs but not in cb_subs
                        affected = set()
                        for s in subs:
                            si = int(s)
                            if si not in cb_subs:
                                affected.add(si)
                        # Compute new coverage for affected triples
                        new_covs = np.array([max(0, int(coverage_count[si]) - 1) for si in affected], dtype=np.float64)
                        if len(new_covs) > 0:
                            temp = 0.5
                            shifted = new_covs / temp
                            max_val = np.max(shifted)
                            log_sum_exp = max_val + np.log(np.sum(np.exp(shifted - max_val)))
                            soft_min = -temp * log_sum_exp
                        else:
                            soft_min = 0.0
                        # Additional coverage gain
                        extra = len(cb_subs - set(subs.tolist()))
                        score = -soft_min + 0.3 * extra
                        if score > best_score:
                            best_score = score
                            best_cb = cb

                if best_cb is not None:
                    new_current = current[:i] + current[i+1:]
                    if best_cb not in new_current:
                        new_current.append(best_cb)
                    current = new_current
                    found = True
                    break

            if not found:
                break

        return current

    # Main loop: multiple restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    restart_deadline = time_limit * 0.40  # Leave more time for final local search

    seed = 42
    while time.time() - start_time < restart_deadline:
        random.seed(seed)
        np.random.seed(seed)
        greedy_blocks = greedy(seed)
        # Local search: remove then replace
        refined = local_search_remove(greedy_blocks)
        if time.time() - start_time < time_limit - 15:
            remaining = time_limit - (time.time() - start_time) - 15
            refined = local_search_replace(refined, min(2.0, max(0.1, remaining * 0.08)))
        if time.time() - start_time < time_limit - 12:
            refined = local_search_remove(refined)

        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = refined

        seed += 1
        if best_count <= 70:  # Good enough, stop early
            break

    # Final polish on the best solution
    if best_blocks is not None and time.time() - start_time < time_limit - 8:
        polished = local_search_remove(best_blocks)
        if time.time() - start_time < time_limit - 6:
            remaining = time_limit - (time.time() - start_time) - 6
            polished = local_search_replace(polished, min(8.0, max(0.5, remaining * 0.4)))
        if time.time() - start_time < time_limit - 5:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 4:
            remaining = time_limit - (time.time() - start_time) - 4
            polished = local_search_single(polished, min(5.0, max(0.3, remaining * 0.5)))
        if time.time() - start_time < time_limit - 3:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 2:
            remaining = time_limit - (time.time() - start_time) - 2
            polished = local_search_replace(polished, min(3.0, max(0.2, remaining * 0.3)))
        if time.time() - start_time < time_limit - 1:
            polished = local_search_remove(polished)
        best_blocks = polished

    # Convert to final format
    final_blocks = [list(all_ks[i]) for i in best_blocks]
    return final_blocks
