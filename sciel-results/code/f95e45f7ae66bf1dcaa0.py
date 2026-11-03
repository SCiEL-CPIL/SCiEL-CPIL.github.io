# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Uses a scarcity-weighted greedy algorithm with local improvement.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Precompute all t-subsets for fast lookup
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t_subsets = len(all_t_subsets)

    # Map each t-subset to an index for fast array operations
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # Precompute for each point, the list of t-subsets that contain it
    # This is used for faster gain computation
    point_to_subsets = defaultdict(list)
    for i, sub in enumerate(all_t_subsets):
        for x in sub:
            point_to_subsets[x].append(i)

    def greedy_cover_scarcity(seed_offset=0):
        """
        Greedy covering with scarcity-weighted heuristic.
        Prioritizes covering triples that are 'harder' to cover (appear in fewer potential blocks).
        """
        t0 = time.time()
        rng = random.Random(42 + seed_offset)

        # Track uncovered subsets using a boolean numpy array
        uncovered_mask = np.ones(total_t_subsets, dtype=bool)
        blocks = []

        # Track how many potential blocks contain each triple
        # A triple {a,b,c} is contained in C(v-3, k-3) blocks
        # We use this as a static weight: triples with fewer potential blocks are 'harder'
        # For C(24,7,3): each triple is in C(21,4) = 5985 blocks
        # But we can dynamically track which triples are still uncovered and prioritize
        # those that appear in fewer *remaining* potential blocks

        # For simplicity and speed, we use a dynamic scarcity measure:
        # For each uncovered triple, count how many of its potential blocks have not been "used" yet
        # This is expensive, so we use an approximation: prioritize triples that contain
        # points that appear in fewer existing blocks (more "isolated" points)

        while np.any(uncovered_mask):
            if time.time() > deadline:
                return None

            # Pick a random uncovered subset to start the block, but weight by scarcity
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break

            # Score each uncovered triple by how "hard" it is to cover
            # Use the number of points in the triple that appear in fewer blocks
            # This encourages covering triples with rare points first
            if len(uncovered_indices) > 200:
                # For performance, sample a subset of uncovered triples
                sample_size = 200
                sample_indices = rng.sample(uncovered_indices.tolist(), sample_size)
            else:
                sample_indices = uncovered_indices.tolist()

            # Compute scarcity score for each sampled triple
            # A triple is "hard" if its points appear in few existing blocks
            point_block_count = np.zeros(v, dtype=np.int32)
            for block in blocks:
                for x in block:
                    point_block_count[x] += 1

            best_idx = None
            best_score = -1
            for idx in sample_indices:
                triple = all_t_subsets[idx]
                # Score = sum of (1 / (count + 1)) for each point in triple
                # Points with fewer blocks get higher score
                score = 0.0
                for x in triple:
                    score += 1.0 / (point_block_count[x] + 1)
                # Add small random tie-breaker
                score += rng.random() * 0.01
                if score > best_score:
                    best_score = score
                    best_idx = idx

            rand_idx = best_idx
            start_subset = list(all_t_subsets[rand_idx])
            block = list(start_subset)

            # Add points one by one to maximize coverage of uncovered subsets
            while len(block) < k:
                if time.time() > deadline:
                    return None

                best_x = None
                best_gain = -1

                # Precompute (t-1)-subsets of the current block
                block_subsets = list(itertools.combinations(block, t - 1))

                for x in range(v):
                    if x in block:
                        continue

                    # Calculate gain: how many uncovered t-subsets would be covered
                    # A t-subset is covered if it's a subset of block + {x}
                    # i.e., it contains t-1 elements from block and x
                    indices_to_check = []
                    for rest in block_subsets:
                        candidate = tuple(sorted(rest + (x,)))
                        idx = subset_to_idx.get(candidate, -1)
                        if idx != -1:
                            indices_to_check.append(idx)

                    if indices_to_check:
                        # Vectorized sum of uncovered status for these indices
                        gain = int(np.sum(uncovered_mask[indices_to_check]))
                    else:
                        gain = 0

                    # Secondary criterion: prefer points that are "scarcer" (appear in fewer blocks)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and gain > 0:
                        # Tie-break by scarcity
                        if point_block_count[x] < point_block_count.get(best_x, v):
                            best_x = x
                        elif rng.random() < 0.3:
                            best_x = x

                if best_x is None:
                    break

                block.append(best_x)

            if len(block) < k:
                break

            block.sort()

            # Remove covered subsets from uncovered
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx.get(sub, -1)
                if idx != -1:
                    uncovered_mask[idx] = False

            blocks.append(block)

        return blocks

    def _verify_covering(blocks, v, k, t):
        """Verify that every t-subset is covered by at least one block."""
        covered = set()
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered.add(sub)

        # Check if all t-subsets are covered
        for sub in itertools.combinations(range(v), t):
            if sub not in covered:
                return False
        return True

    def _local_improve(blocks, v, k, t, deadline):
        """
        Try to improve the covering by removing redundant blocks and swapping.
        """
        if not blocks:
            return blocks

        blocks_copy = list(blocks)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False

            # First pass: try to remove redundant blocks
            for i in range(len(blocks_copy)):
                if time.time() > deadline - 5:
                    break

                # Check if block i is redundant
                without_i = blocks_copy[:i] + blocks_copy[i+1:]
                if _verify_covering(without_i, v, k, t):
                    blocks_copy = without_i
                    changed = True
                    break

            if changed:
                continue

            # Second pass: try to swap blocks to reduce count
            # Try replacing one block with another that covers the same unique triples
            for i in range(len(blocks_copy)):
                if time.time() > deadline - 5:
                    break

                # Find triples only covered by block i
                unique_triples = set()
                for sub in itertools.combinations(blocks_copy[i], t):
                    only_in_i = True
                    for j, block in enumerate(blocks_copy):
                        if j == i:
                            continue
                        if sub in set(itertools.combinations(block, t)):
                            only_in_i = False
                            break
                    if only_in_i:
                        unique_triples.add(sub)

                if not unique_triples:
                    # Block i is redundant
                    blocks_copy = blocks_copy[:i] + blocks_copy[i+1:]
                    changed = True
                    break

                # Try to find a replacement block that covers all unique_triples
                # and has more total coverage
                best_replacement = None
                best_score = -1

                # Sample some candidate blocks
                candidates = list(itertools.combinations(range(v), k))
                rng_local = random.Random(42)
                sample_size = min(500, len(candidates))
                sample_candidates = rng_local.sample(candidates, sample_size)

                for cand in sample_candidates:
                    cand_set = set(cand)
                    # Check if cand covers all unique_triples
                    covers_all = True
                    for triple in unique_triples:
                        if not set(triple).issubset(cand_set):
                            covers_all = False
                            break
                    if not covers_all:
                        continue

                    # Score: number of new triples covered
                    new_triples = 0
                    for sub in itertools.combinations(cand, t):
                        is_covered = False
                        for j, block in enumerate(blocks_copy):
                            if j == i:
                                continue
                            if sub in set(itertools.combinations(block, t)):
                                is_covered = True
                                break
                        if not is_covered:
                            new_triples += 1

                    if new_triples > best_score:
                        best_score = new_triples
                        best_replacement = list(cand)

                if best_replacement is not None and best_score > 0:
                    blocks_copy[i] = best_replacement
                    changed = True
                    break

        return blocks_copy

    # Try multiple restarts and keep the best
    best_blocks = None
    best_len = float('inf')

    restart_count = 0
    while time.time() < deadline - 10:
        blocks = greedy_cover_scarcity(seed_offset=restart_count)
        restart_count += 1

        if blocks is not None and len(blocks) > 0:
            # Verify the covering is valid
            if time.time() < deadline - 5:
                if _verify_covering(blocks, v, k, t):
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = blocks

                        # Try to improve with local search
                        if time.time() < deadline - 15:
                            improved = _local_improve(blocks, v, k, t, deadline)
                            if improved is not None and len(improved) < len(best_blocks):
                                # Verify improved solution
                                if _verify_covering(improved, v, k, t):
                                    best_blocks = improved
                                    best_len = len(best_blocks)

    if best_blocks is None:
        # Fallback: use the first greedy result even if not fully verified
        blocks = greedy_cover_scarcity(seed_offset=0)
        if blocks is not None:
            return blocks
        # Last resort: return a trivial (invalid) result
        return []

    return best_blocks
# EVOLVE-BLOCK-END
