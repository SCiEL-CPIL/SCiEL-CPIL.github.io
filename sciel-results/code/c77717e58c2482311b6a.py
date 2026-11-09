"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin before 150s kill
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # For each point, list of t-subset indices containing that point
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        idx = tsub_to_idx[ts]
        for p in ts:
            point_tsubs[p].append(idx)

    # Precompute for each (t-1)-subset, which t-subsets extend it
    if t >= 2:
        tminus1_to_tsubs = {}
        for ts in all_tsubs:
            ts_set = set(ts)
            for rm in range(t):
                sub = tuple(sorted(ts_set - {ts[rm]}))
                if sub not in tminus1_to_tsubs:
                    tminus1_to_tsubs[sub] = []
                tminus1_to_tsubs[sub].append(tsub_to_idx[ts])
    else:
        tminus1_to_tsubs = {}

    def greedy_cover_with_prob_init():
        """Randomized greedy set cover with probabilistic Johnson-scheme-inspired init."""
        coverage = [0] * n_tsubs
        blocks = []
        block_set_list = []

        while True:
            if time.time() > deadline:
                return None

            uncovered_idx = None
            for i in range(n_tsubs):
                if coverage[i] == 0:
                    uncovered_idx = i
                    break
            if uncovered_idx is None:
                break

            seed_tsub = all_tsubs[uncovered_idx]
            block_set = set(seed_tsub)

            for _ in range(k - t):
                if time.time() > deadline:
                    return None

                best_point = None
                best_gain = -1
                best_gain_list = []
                candidates = [p for p in range(v) if p not in block_set]

                for p in candidates:
                    gain = 0
                    if t >= 2:
                        if t == 2:
                            for x in block_set:
                                pair = tuple(sorted((p, x)))
                                if pair in tsub_to_idx:
                                    idx = tsub_to_idx[pair]
                                    if coverage[idx] == 0:
                                        gain += 1
                        else:
                            for sub in itertools.combinations(sorted(block_set), t - 1):
                                ts_candidate = tuple(sorted(sub + (p,)))
                                if ts_candidate in tsub_to_idx:
                                    idx = tsub_to_idx[ts_candidate]
                                    if coverage[idx] == 0:
                                        gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                        best_gain_list = [p]
                    elif gain == best_gain:
                        best_gain_list.append(p)

                if best_point is not None:
                    if len(best_gain_list) > 1:
                        weights = []
                        for p in best_gain_list:
                            w = sum(1 for ti in point_tsubs[p] if coverage[ti] == 0)
                            weights.append(w + 1)
                        total_w = sum(weights)
                        r = rng.random() * total_w
                        cum = 0
                        chosen = best_gain_list[0]
                        for p, w in zip(best_gain_list, weights):
                            cum += w
                            if r <= cum:
                                chosen = p
                                break
                        block_set.add(chosen)
                    else:
                        block_set.add(best_point)

            block = sorted(block_set)
            blocks.append(block)
            block_set_list.append(block_set)

            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage[idx] += 1

        return blocks

    def local_search_remove(blocks):
        """Try to remove redundant blocks."""
        if not blocks:
            return blocks

        coverage_count = [0] * n_tsubs
        block_tsubs = []

        for block in blocks:
            tsubs_in_block = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
                tsubs_in_block.append(idx)
            block_tsubs.append(tsubs_in_block)

        keep = [True] * len(blocks)

        for i in range(len(blocks)):
            if not keep[i]:
                continue
            can_remove = True
            for tsub_idx in block_tsubs[i]:
                if coverage_count[tsub_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                keep[i] = False
                for tsub_idx in block_tsubs[i]:
                    coverage_count[tsub_idx] -= 1

        return [blocks[i] for i in range(len(blocks)) if keep[i]]

    def targeted_repair_local_search(blocks, max_rounds=5):
        """Targeted local search that optimizes block overlap for critical triples.
        
        Strategy: For each block, identify its critical (uniquely covered) t-subsets.
        Try to replace the block with a new block that:
        1. Covers all critical t-subsets
        2. Maximizes overlap with already-covered t-subsets (making other blocks redundant)
        3. Uses a soft-max scoring to balance overlap vs. new coverage potential
        """
        if not blocks:
            return blocks

        coverage_count = [0] * n_tsubs
        block_tsubs = []

        for block in blocks:
            tsubs_in_block = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
                tsubs_in_block.append(idx)
            block_tsubs.append(tsubs_in_block)

        improved = True
        round_num = 0

        while improved and round_num < max_rounds:
            improved = False
            round_num += 1

            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks

                # Identify t-subsets uniquely covered by block i
                unique_tsubs = [ts_idx for ts_idx in block_tsubs[i] if coverage_count[ts_idx] == 1]

                if not unique_tsubs:
                    # Block i is redundant
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    # Rebuild
                    coverage_count = [0] * n_tsubs
                    block_tsubs = []
                    for block in blocks:
                        tsubs_in_block = []
                        for ts in itertools.combinations(block, t):
                            idx = tsub_to_idx[ts]
                            coverage_count[idx] += 1
                            tsubs_in_block.append(idx)
                        block_tsubs.append(tsubs_in_block)
                    break

                # Collect required points from unique t-subsets
                required_points = set()
                for ts_idx in unique_tsubs:
                    required_points.update(all_tsubs[ts_idx])

                if len(required_points) > k:
                    continue

                # Try multiple random extensions with soft-max scoring
                best_new_block = None
                best_score = -float('inf')
                num_trials = min(30, max(5, int((deadline - time.time()) * 2)))

                for trial in range(num_trials):
                    new_block_set = set(required_points)

                    # Greedily add points with soft-max scoring
                    for _ in range(k - len(new_block_set)):
                        scores = []
                        candidates = [p for p in range(v) if p not in new_block_set]
                        rng.shuffle(candidates)

                        for p in candidates:
                            # Score: weighted combination of overlap and potential
                            overlap_score = 0
                            potential_score = 0
                            test_set = new_block_set | {p}
                            for ts in itertools.combinations(sorted(test_set), t):
                                idx = tsub_to_idx.get(ts, -1)
                                if idx >= 0:
                                    if coverage_count[idx] > 1:
                                        overlap_score += 2  # already well-covered
                                    elif coverage_count[idx] == 1:
                                        overlap_score += 1
                                    else:
                                        potential_score += 1

                            # Soft-max style: prefer overlap but allow exploration
                            score = overlap_score + 0.5 * potential_score + rng.random() * 0.3
                            scores.append((score, p))

                        if scores:
                            scores.sort(reverse=True)
                            # Pick from top few with probability
                            top_n = min(3, len(scores))
                            if rng.random() < 0.7:
                                chosen_p = scores[0][1]
                            else:
                                chosen_p = scores[rng.randint(0, top_n - 1)][1]
                            new_block_set.add(chosen_p)

                    # Verify coverage of unique t-subsets
                    valid = True
                    for ts_idx in unique_tsubs:
                        ts_set = set(all_tsubs[ts_idx])
                        if not ts_set.issubset(new_block_set):
                            valid = False
                            break

                    if not valid:
                        continue

                    # Calculate final score
                    overlap = 0
                    new_coverage = 0
                    for ts in itertools.combinations(sorted(new_block_set), t):
                        idx = tsub_to_idx.get(ts, -1)
                        if idx >= 0:
                            if coverage_count[idx] > 1:
                                overlap += 1
                            elif coverage_count[idx] == 0:
                                new_coverage += 1

                    final_score = overlap * 2 + new_coverage * 0.5 + rng.random() * 0.1

                    if final_score > best_score:
                        best_score = final_score
                        best_new_block = sorted(new_block_set)

                if best_new_block is not None:
                    # Temporarily replace
                    old_block_tsubs = block_tsubs[i]

                    for ts_idx in old_block_tsubs:
                        coverage_count[ts_idx] -= 1

                    new_block_tsubs = []
                    for ts in itertools.combinations(best_new_block, t):
                        idx = tsub_to_idx[ts]
                        coverage_count[idx] += 1
                        new_block_tsubs.append(idx)

                    # Check if any other blocks become redundant
                    additional_removals = 0
                    new_keep = [True] * len(blocks)

                    for j in range(len(blocks)):
                        if j == i or not new_keep[j]:
                            continue
                        can_remove = True
                        for ts_idx in block_tsubs[j]:
                            if coverage_count[ts_idx] <= 1:
                                can_remove = False
                                break
                        if can_remove:
                            new_keep[j] = False
                            additional_removals += 1
                            for ts_idx in block_tsubs[j]:
                                coverage_count[ts_idx] -= 1

                    if additional_removals > 0:
                        blocks[i] = best_new_block
                        block_tsubs[i] = new_block_tsubs

                        for j in range(len(blocks) - 1, -1, -1):
                            if not new_keep[j]:
                                blocks.pop(j)
                                block_tsubs.pop(j)

                        improved = True
                        coverage_count = [0] * n_tsubs
                        block_tsubs = []
                        for block in blocks:
                            tsubs_in_block = []
                            for ts in itertools.combinations(block, t):
                                idx = tsub_to_idx[ts]
                                coverage_count[idx] += 1
                                tsubs_in_block.append(idx)
                            block_tsubs.append(tsubs_in_block)
                        break
                    else:
                        for ts_idx in old_block_tsubs:
                            coverage_count[ts_idx] += 1
                        for ts_idx in new_block_tsubs:
                            coverage_count[ts_idx] -= 1

        return blocks

    def final_polish(blocks):
        """Final high-precision polish pass: try pairwise block replacements."""
        if not blocks:
            return blocks

        coverage_count = [0] * n_tsubs
        block_tsubs = []

        for block in blocks:
            tsubs_in_block = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
                tsubs_in_block.append(idx)
            block_tsubs.append(tsubs_in_block)

        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks

                unique_to_i = set(ts_idx for ts_idx in block_tsubs[i] if coverage_count[ts_idx] == 1)

                if not unique_to_i:
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    coverage_count = [0] * n_tsubs
                    block_tsubs = []
                    for block in blocks:
                        tsubs_in_block = []
                        for ts in itertools.combinations(block, t):
                            idx = tsub_to_idx[ts]
                            coverage_count[idx] += 1
                            tsubs_in_block.append(idx)
                        block_tsubs.append(tsubs_in_block)
                    break

                for j in range(len(blocks)):
                    if i == j:
                        continue
                    if time.time() > deadline:
                        return blocks

                    unique_to_j = set(ts_idx for ts_idx in block_tsubs[j] if coverage_count[ts_idx] == 1)

                    req_i = set()
                    for ts_idx in unique_to_i:
                        req_i.update(all_tsubs[ts_idx])

                    req_j = set()
                    for ts_idx in unique_to_j:
                        req_j.update(all_tsubs[ts_idx])

                    combined_req = req_i | req_j
                    if len(combined_req) <= k:
                        new_block_set = set(combined_req)
                        for _ in range(k - len(new_block_set)):
                            best_p = None
                            best_score = -1
                            for p in range(v):
                                if p in new_block_set:
                                    continue
                                score = 0
                                test_set = new_block_set | {p}
                                for ts in itertools.combinations(sorted(test_set), t):
                                    idx = tsub_to_idx.get(ts, -1)
                                    if idx >= 0 and coverage_count[idx] > 0:
                                        score += 1
                                if score > best_score:
                                    best_score = score
                                    best_p = p
                            if best_p is not None:
                                new_block_set.add(best_p)

                        new_block = sorted(new_block_set)

                        valid = True
                        for ts_idx in unique_to_i | unique_to_j:
                            if not set(all_tsubs[ts_idx]).issubset(new_block_set):
                                valid = False
                                break

                        if valid:
                            for ts_idx in block_tsubs[i]:
                                coverage_count[ts_idx] -= 1
                            for ts_idx in block_tsubs[j]:
                                coverage_count[ts_idx] -= 1

                            for ts in itertools.combinations(new_block, t):
                                idx = tsub_to_idx[ts]
                                coverage_count[idx] += 1

                            all_covered = all(coverage_count[idx] > 0 for idx in range(n_tsubs))

                            if all_covered:
                                blocks.pop(j)
                                block_tsubs.pop(j)
                                blocks[i] = new_block
                                block_tsubs[i] = [tsub_to_idx[ts] for ts in itertools.combinations(new_block, t)]
                                improved = True
                                coverage_count = [0] * n_tsubs
                                block_tsubs = []
                                for block in blocks:
                                    tsubs_in_block = []
                                    for ts in itertools.combinations(block, t):
                                        idx = tsub_to_idx[ts]
                                        coverage_count[idx] += 1
                                        tsubs_in_block.append(idx)
                                    block_tsubs.append(tsubs_in_block)
                                break
                            else:
                                for ts_idx in block_tsubs[i]:
                                    coverage_count[ts_idx] += 1
                                for ts_idx in block_tsubs[j]:
                                    coverage_count[ts_idx] += 1
                                for ts in itertools.combinations(new_block, t):
                                    idx = tsub_to_idx[ts]
                                    coverage_count[idx] -= 1

                    if improved:
                        break

                if improved:
                    break

        return blocks

    best_blocks = None
    best_count = float('inf')

    # Run greedy with randomization and local search
    num_restarts = 4
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        blocks = greedy_cover_with_prob_init()
        if blocks is None:
            continue

        blocks = local_search_remove(blocks)

        if time.time() < deadline - 25:
            blocks = targeted_repair_local_search(blocks)
            blocks = local_search_remove(blocks)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()

    if best_blocks is not None and time.time() < deadline - 15:
        improved_blocks = final_polish(best_blocks.copy())
        if improved_blocks is not None and len(improved_blocks) < best_count:
            best_count = len(improved_blocks)
            best_blocks = improved_blocks

    if best_blocks is not None:
        return best_blocks

    blocks = greedy_cover_with_prob_init()
    if blocks is not None:
        return blocks

    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
# EVOLVE-BLOCK-END
