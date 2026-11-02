"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import heapq
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute: for each point, which t-subsets contain it
    point_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subsets):
        for pt in sub:
            point_tsubs[pt].append(i)

    # Precompute all k-subsets and their t-subset memberships for faster lookup
    # For gain computation: given a partial block S and candidate x,
    # gain = count of uncovered t-subsets in C(S ∪ {x}, t)
    # = count of uncovered t-subsets in C(S, t) [already covered by S]
    #   + count of uncovered t-subsets containing x and t-1 elements from S
    # We need: for each (t-1)-subset of S, and each x not in S,
    #   the t-subset = (t-1)-subset ∪ {x}

    # Precompute: for each (t-1)-combination, map to list of (point, t_sub_idx)
    # i.e., for a (t-1)-tuple R, for each point x not in R,
    #   t_sub = R ∪ {x}, idx = t_sub_to_idx[t_sub]
    # This is expensive to precompute fully, so we'll compute on the fly but cache

    best_blocks = None
    best_count = float('inf')

    n_restarts = 60

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(12345 + restart * 7919)

        covered = [False] * n_t
        covered_count = 0
        blocks = []

        # Dynamic greedy with improved gain computation
        while covered_count < n_t and time_ok():
            # Find uncovered t-subsets
            uncovered_indices = [i for i in range(n_t) if not covered[i]]
            if not uncovered_indices:
                break

            # Score uncovered t-subsets by scarcity (inverse point frequency)
            point_uncovered_count = [0] * v
            for idx in uncovered_indices:
                for pt in all_t_subsets[idx]:
                    point_uncovered_count[pt] += 1

            # Pick seed candidates using weighted sampling
            if len(uncovered_indices) <= 30:
                candidates = uncovered_indices
            else:
                weights = []
                for idx in uncovered_indices:
                    w = sum(1.0 / max(1, point_uncovered_count[pt]) for pt in all_t_subsets[idx])
                    weights.append(w)
                total_w = sum(weights)
                if total_w == 0:
                    candidates = [random.choice(uncovered_indices)]
                else:
                    n_samples = min(15, len(uncovered_indices))
                    candidates = []
                    for _ in range(n_samples):
                        r = random.random() * total_w
                        cum = 0
                        for i, w in enumerate(weights):
                            cum += w
                            if cum >= r:
                                candidates.append(uncovered_indices[i])
                                break
                    if not candidates:
                        candidates = [random.choice(uncovered_indices)]

            best_block = None
            best_gain = -1

            for seed_idx in candidates:
                if not time_ok():
                    break
                seed_subset = all_t_subsets[seed_idx]
                block_set = set(seed_subset)

                # Extend block greedily
                while len(block_set) < k and time_ok():
                    block_list = sorted(block_set)
                    candidates_pts = [x for x in all_points if x not in block_set]

                    # Compute gain for each candidate point
                    best_x = None
                    best_x_gain = -1

                    # Precompute (t-1)-subsets of current block
                    t_minus_1_subs = list(itertools.combinations(block_list, t - 1))

                    for x in candidates_pts:
                        gain = 0
                        for rest in t_minus_1_subs:
                            sub = tuple(sorted(rest + (x,)))
                            idx = t_sub_to_idx[sub]
                            if not covered[idx]:
                                gain += 1
                        # Small random tiebreaker for diversity
                        tie = random.random() * 0.005
                        if gain + tie > best_x_gain:
                            best_x_gain = gain + tie
                            best_x = x

                    if best_x is None:
                        break
                    block_set.add(best_x)

                if len(block_set) < k:
                    remaining = [x for x in all_points if x not in block_set]
                    random.shuffle(remaining)
                    for x in remaining[:k - len(block_set)]:
                        block_set.add(x)

                block = tuple(sorted(block_set))

                # Calculate total gain of this complete block
                gain = 0
                for sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[sub]
                    if not covered[idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_block = block

            if best_block is None or best_gain <= 0:
                block_set = set(random.sample(all_points, k))
                best_block = tuple(sorted(block_set))
                best_gain = sum(1 for sub in itertools.combinations(best_block, t)
                                if not covered[t_sub_to_idx[sub]])

            if best_gain <= 0:
                break

            # Apply the block
            for sub in itertools.combinations(best_block, t):
                idx = t_sub_to_idx[sub]
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
            blocks.append(best_block)

        # Deletion phase: iterative removal of redundant blocks
        if time_ok() and covered_count == n_t:
            tsub_block_count = [0] * n_t
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[sub]
                    tsub_block_count[idx] += 1

            # Priority queue: blocks with most redundancy (fewest unique t-subsets) first
            # For each block, compute number of t-subsets it uniquely covers
            def unique_count(block_idx):
                b = blocks[block_idx]
                uc = 0
                for sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[sub]
                    if tsub_block_count[idx] == 1:
                        uc += 1
                return uc

            # Build initial priority queue
            pq = []
            for i in range(len(blocks)):
                uc = unique_count(i)
                heapq.heappush(pq, (uc, i))

            changed = True
            while changed and time_ok():
                changed = False
                # Try to find a removable block
                while pq:
                    uc, i = heapq.heappop(pq)
                    if i >= len(blocks):
                        continue
                    # Check if still removable
                    b = blocks[i]
                    removable = True
                    for sub in itertools.combinations(b, t):
                        idx = t_sub_to_idx[sub]
                        if tsub_block_count[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        # Remove block i
                        for sub in itertools.combinations(b, t):
                            idx = t_sub_to_idx[sub]
                            tsub_block_count[idx] -= 1
                        blocks.pop(i)
                        changed = True
                        # Rebuild priority queue (expensive but correct)
                        pq = []
                        for j in range(len(blocks)):
                            uc2 = unique_count(j)
                            heapq.heappush(pq, (uc2, j))
                        break
                if not pq:
                    break

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)

        # Early stop if we've done very well
        if best_count <= 145:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
