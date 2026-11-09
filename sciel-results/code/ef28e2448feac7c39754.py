"""Improved covering design C(24,7,3) solver with efficient gain computation and enhanced local search."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin before 150s kill
    rng = random.Random(42)

    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # Precompute: for each (t-1)-subset, list of t-subset indices that extend it
    # For t=3: pair -> list of triples containing that pair
    tminus1_to_tsubs = {}
    if t >= 2:
        for ts in all_tsubs:
            idx = tsub_to_idx[ts]
            for rm in range(t):
                sub = tuple(sorted(tuple(x for x in ts if x != ts[rm])))
                if sub not in tminus1_to_tsubs:
                    tminus1_to_tsubs[sub] = []
                tminus1_to_tsubs[sub].append(idx)

    def fast_gain(block_set, p, coverage):
        """Compute gain of adding point p to block_set: number of newly covered t-subsets."""
        gain = 0
        if t == 2:
            for x in block_set:
                pair = (min(p, x), max(p, x))
                if pair in tsub_to_idx:
                    idx = tsub_to_idx[pair]
                    if coverage[idx] == 0:
                        gain += 1
        elif t == 3:
            # For each pair in block_set, check if (pair, p) forms an uncovered triple
            for a in block_set:
                for b in block_set:
                    if a >= b:
                        continue
                    # Form triple with p
                    triple = tuple(sorted((a, b, p)))
                    if triple in tsub_to_idx:
                        idx = tsub_to_idx[triple]
                        if coverage[idx] == 0:
                            gain += 1
        else:
            # General case
            for sub in itertools.combinations(sorted(block_set), t - 1):
                ts_candidate = tuple(sorted(sub + (p,)))
                if ts_candidate in tsub_to_idx:
                    idx = tsub_to_idx[ts_candidate]
                    if coverage[idx] == 0:
                        gain += 1
        return gain

    def greedy_cover(seed_tsub=None, temperature=0.0):
        """Greedy set cover. If seed_tsub is given, start with that t-subset in the first block.
        temperature > 0 adds randomness to tie-breaking."""
        coverage = [0] * n_tsubs
        blocks = []

        while True:
            if time.time() > deadline:
                return None

            # Find an uncovered t-subset
            uncovered_idx = None
            if seed_tsub is not None:
                # Use the seed t-subset if still uncovered
                if seed_tsub in tsub_to_idx and coverage[tsub_to_idx[seed_tsub]] == 0:
                    uncovered_idx = tsub_to_idx[seed_tsub]
                seed_tsub = None  # only use once

            if uncovered_idx is None:
                for i in range(n_tsubs):
                    if coverage[i] == 0:
                        uncovered_idx = i
                        break
            if uncovered_idx is None:
                break

            # Build a block starting from this t-subset
            block_set = set(all_tsubs[uncovered_idx])

            for _ in range(k - t):
                if time.time() > deadline:
                    return None

                # Find best point to add
                best_gain = -1
                best_points = []
                candidates = [p for p in range(v) if p not in block_set]

                for p in candidates:
                    gain = fast_gain(block_set, p, coverage)
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [p]
                    elif gain == best_gain:
                        best_points.append(p)

                if not best_points:
                    break

                # Select from best_points with optional randomness
                if len(best_points) == 1:
                    chosen = best_points[0]
                else:
                    if temperature > 0:
                        # Soft-max style selection
                        weights = []
                        for p in best_points:
                            # Weight by number of uncovered t-subsets containing p
                            w = sum(1 for idx in range(n_tsubs) if coverage[idx] == 0 and p in all_tsubs[idx])
                            weights.append(w + 1)
                        total_w = sum(weights)
                        r = rng.random() * total_w
                        cum = 0
                        chosen = best_points[0]
                        for p, w in zip(best_points, weights):
                            cum += w
                            if r <= cum:
                                chosen = p
                                break
                    else:
                        # Deterministic: pick the one with most uncovered t-subsets
                        max_w = -1
                        chosen = best_points[0]
                        for p in best_points:
                            w = sum(1 for idx in range(n_tsubs) if coverage[idx] == 0 and p in all_tsubs[idx])
                            if w > max_w:
                                max_w = w
                                chosen = p

                block_set.add(chosen)

            block = sorted(block_set)
            blocks.append(block)

            # Update coverage
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage[idx] += 1

        return blocks

    def local_search_remove(blocks):
        """Remove redundant blocks."""
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

    def enhanced_local_search(blocks):
        """Enhanced local search: replace blocks to maximize overlap and enable removals."""
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
        max_iterations = 15
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

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

                # Try multiple random extensions to find best replacement
                best_new_block = None
                best_score = -1

                num_trials = min(15, max(1, int((deadline - time.time()) / 3)))
                for trial in range(num_trials):
                    new_block_set = set(required_points)

                    # Greedily extend to size k, maximizing overlap with existing coverage
                    for _ in range(k - len(new_block_set)):
                        best_p = None
                        best_p_score = -1
                        candidates = [p for p in range(v) if p not in new_block_set]
                        rng.shuffle(candidates)

                        for p in candidates:
                            # Score: count t-subsets in new_block_set ∪ {p} that are already covered
                            score = 0
                            test_set = new_block_set | {p}
                            for ts in itertools.combinations(sorted(test_set), t):
                                idx = tsub_to_idx.get(ts, -1)
                                if idx >= 0 and coverage_count[idx] > 0:
                                    score += 1
                            if score > best_p_score:
                                best_p_score = score
                                best_p = p
                        if best_p is not None:
                            new_block_set.add(best_p)

                    # Verify coverage of unique t-subsets
                    valid = True
                    for ts_idx in unique_tsubs:
                        ts_set = set(all_tsubs[ts_idx])
                        if not ts_set.issubset(new_block_set):
                            valid = False
                            break
                    if not valid:
                        continue

                    # Compute score: overlap with other blocks (t-subsets covered by >1 block)
                    overlap = 0
                    for ts in itertools.combinations(sorted(new_block_set), t):
                        idx = tsub_to_idx.get(ts, -1)
                        if idx >= 0 and coverage_count[idx] > 1:
                            overlap += 1

                    if overlap > best_score:
                        best_score = overlap
                        best_new_block = sorted(new_block_set)

                if best_new_block is not None:
                    # Temporarily replace block i
                    old_tsubs = block_tsubs[i]
                    for ts_idx in old_tsubs:
                        coverage_count[ts_idx] -= 1

                    new_tsubs = []
                    for ts in itertools.combinations(best_new_block, t):
                        idx = tsub_to_idx[ts]
                        coverage_count[idx] += 1
                        new_tsubs.append(idx)

                    # Check if any other blocks become removable
                    removals = 0
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
                            removals += 1
                            for ts_idx in block_tsubs[j]:
                                coverage_count[ts_idx] -= 1

                    if removals > 0:
                        blocks[i] = best_new_block
                        block_tsubs[i] = new_tsubs
                        for j in range(len(blocks) - 1, -1, -1):
                            if not new_keep[j]:
                                blocks.pop(j)
                                block_tsubs.pop(j)
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
                    else:
                        # Revert
                        for ts_idx in old_tsubs:
                            coverage_count[ts_idx] += 1
                        for ts_idx in new_tsubs:
                            coverage_count[ts_idx] -= 1

        return blocks

    def final_polish(blocks):
        """Final polish: try pairwise merges."""
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
                    if i == j or time.time() > deadline:
                        continue

                    unique_to_j = set(ts_idx for ts_idx in block_tsubs[j] if coverage_count[ts_idx] == 1)

                    req_i = set()
                    for ts_idx in unique_to_i:
                        req_i.update(all_tsubs[ts_idx])
                    req_j = set()
                    for ts_idx in unique_to_j:
                        req_j.update(all_tsubs[ts_idx])

                    combined = req_i | req_j
                    if len(combined) <= k:
                        new_block_set = set(combined)
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
                            # Try merge
                            for ts_idx in block_tsubs[i]:
                                coverage_count[ts_idx] -= 1
                            for ts_idx in block_tsubs[j]:
                                coverage_count[ts_idx] -= 1
                            for ts in itertools.combinations(new_block, t):
                                idx = tsub_to_idx[ts]
                                coverage_count[idx] += 1

                            if all(coverage_count[idx] > 0 for idx in range(n_tsubs)):
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

    # Multiple restarts with different strategies
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Vary temperature and seed selection
        temp = 0.0 if restart < 4 else 0.3
        seed = None
        if restart >= 4:
            # Pick a random t-subset as seed
            seed = all_tsubs[rng.randint(0, n_tsubs - 1)]

        blocks = greedy_cover(seed_tsub=seed, temperature=temp)
        if blocks is None:
            continue

        blocks = local_search_remove(blocks)

        if time.time() < deadline - 30:
            blocks = enhanced_local_search(blocks)
            blocks = local_search_remove(blocks)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()

    # Final polish
    if best_blocks is not None and time.time() < deadline - 20:
        improved = final_polish(best_blocks.copy())
        if improved is not None and len(improved) < best_count:
            best_count = len(improved)
            best_blocks = improved

    if best_blocks is not None:
        return best_blocks

    # Fallback
    blocks = greedy_cover()
    if blocks is not None:
        return blocks

    # Last resort
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
