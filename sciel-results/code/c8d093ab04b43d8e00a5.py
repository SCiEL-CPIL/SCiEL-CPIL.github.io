import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    # Precompute for each point, the list of triple indices containing it
    point_to_triples = [[] for _ in range(v)]
    for i, tr in enumerate(triples):
        for p in tr:
            point_to_triples[p].append(i)

    # Precompute for each pair, the list of triple indices containing both
    pair_to_triples = {}
    for i, tr in enumerate(triples):
        for j in range(t):
            for m in range(j+1, t):
                pair = (tr[j], tr[m])
                if pair not in pair_to_triples:
                    pair_to_triples[pair] = []
                pair_to_triples[pair].append(i)

    # Precompute for each triple, the set of points that can extend it to a k-block
    # Actually, any point not in the triple can be added.

    def compute_block_triples(block):
        """Return list of triple indices covered by this block."""
        b_triples = []
        for triple in itertools.combinations(block, t):
            b_triples.append(triple_to_idx[triple])
        return b_triples

    def greedy_cover(uncovered_mask=None, random_state=0, scarcity_weighted=True):
        rng = random.Random(random_state)
        if uncovered_mask is None:
            uncovered = np.ones(num_triples, dtype=np.uint8)
            num_uncovered = num_triples
        else:
            uncovered = uncovered_mask.copy()
            num_uncovered = int(np.sum(uncovered))

        blocks = []
        points = list(range(v))

        # Compute scarcity: for each point, how many uncovered triples contain it
        # Points with fewer uncovered triples are "scarcer" for covering remaining triples
        if scarcity_weighted and num_uncovered > 0:
            point_scarcity = np.zeros(v, dtype=np.int32)
            for p in range(v):
                for idx in point_to_triples[p]:
                    if uncovered[idx]:
                        point_scarcity[p] += 1

        while num_uncovered > 0:
            # Pick the uncovered triple with the most "pressure"
            # Use scarcity-weighted selection: prefer triples whose points are scarce
            uncovered_indices = np.where(uncovered > 0)[0]
            if len(uncovered_indices) == 0:
                break

            if scarcity_weighted and num_uncovered > 0:
                # Score each uncovered triple by sum of scarcity of its points
                # Lower scarcity sum = harder to cover = prioritize
                best_idx = uncovered_indices[0]
                best_score = float('inf')
                # Sample a subset if too many
                sample_size = min(len(uncovered_indices), 50)
                sample = rng.sample(uncovered_indices.tolist(), sample_size)
                for idx in sample:
                    tr = triples[idx]
                    score = sum(point_scarcity[p] for p in tr)
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                start_idx = best_idx
            else:
                start_idx = uncovered_indices[0]

            if uncovered[start_idx] == 0:
                break

            target_triple = triples[start_idx]
            block = list(target_triple)
            block_set = set(block)

            # Mark target triple as covered
            uncovered[start_idx] = 0
            num_uncovered -= 1

            # Update scarcity
            if scarcity_weighted:
                for p in block:
                    point_scarcity[p] -= 1

            while len(block) < k and num_uncovered > 0:
                candidates = [x for x in points if x not in block_set]
                if not candidates:
                    break

                best_gain = -1
                best_point = -1

                for x in candidates:
                    gain = 0
                    # Count uncovered triples that would be covered by adding x
                    for p1 in block:
                        for p2 in block:
                            if p1 == p2:
                                continue
                            pair = (min(p1, p2), max(p1, p2))
                            if pair in pair_to_triples:
                                for idx in pair_to_triples[pair]:
                                    if x in triples[idx] and uncovered[idx]:
                                        gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point == -1 or best_gain <= 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

                # Update uncovered
                for p1 in block:
                    for p2 in block:
                        if p1 == p2 or p1 == best_point or p2 == best_point:
                            continue
                        pair = (min(p1, p2), max(p1, p2))
                        if pair in pair_to_triples:
                            for idx in pair_to_triples[pair]:
                                if best_point in triples[idx] and uncovered[idx]:
                                    uncovered[idx] = 0
                                    num_uncovered -= 1
                                    if scarcity_weighted:
                                        for p in triples[idx]:
                                            point_scarcity[p] -= 1

            blocks.append(tuple(sorted(block)))

        return blocks, uncovered, num_uncovered

    def trim_blocks(blocks):
        if not blocks:
            return blocks

        counts = np.zeros(num_triples, dtype=np.int32)
        block_triples = []

        for block in blocks:
            b_triples = compute_block_triples(block)
            for idx in b_triples:
                counts[idx] += 1
            block_triples.append(b_triples)

        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                essential = False
                for idx in block_triples[i]:
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    changed = True
                    for idx in block_triples[i]:
                        counts[idx] -= 1

        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def local_search_trim(blocks, max_iters=30):
        if not blocks:
            return blocks

        counts = np.zeros(num_triples, dtype=np.int32)
        block_triples = []
        for block in blocks:
            b_triples = compute_block_triples(block)
            for idx in b_triples:
                counts[idx] += 1
            block_triples.append(b_triples)

        active = [True] * len(blocks)
        best_len = len(blocks)

        for _ in range(max_iters):
            improved = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                temp_counts = counts.copy()
                for idx in block_triples[i]:
                    temp_counts[idx] -= 1
                if np.all(temp_counts >= 0):
                    uncovered_after = np.where(temp_counts == 0)[0]
                    if len(uncovered_after) == 0:
                        active[i] = False
                        counts = temp_counts
                        best_len -= 1
                        improved = True
                        break
                    else:
                        for j in range(len(blocks)):
                            if not active[j] or j == i:
                                continue
                            covers_all = True
                            for uidx in uncovered_after:
                                if uidx not in block_triples[j]:
                                    covers_all = False
                                    break
                            if covers_all:
                                active[i] = False
                                counts = temp_counts
                                best_len -= 1
                                improved = True
                                break
                        if improved:
                            break
            if not improved:
                break

        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def simulated_annealing(blocks, max_time=None):
        """Simulated annealing to minimize block count."""
        if not blocks:
            return blocks

        current_blocks = list(blocks)
        current_len = len(current_blocks)

        counts = np.zeros(num_triples, dtype=np.int32)
        block_triples = []
        for block in current_blocks:
            b_triples = compute_block_triples(block)
            for idx in b_triples:
                counts[idx] += 1
            block_triples.append(b_triples)

        best_blocks = list(current_blocks)
        best_len = current_len

        if max_time is None:
            max_time = deadline - time.time() - 5

        sa_start = time.time()
        T = 1.0
        T_min = 0.01
        cooling = 0.999

        while time.time() - sa_start < max_time and T > T_min:
            # Pick a random block to modify
            i = random.randrange(len(current_blocks))

            # Option 1: Remove block i if possible
            if random.random() < 0.5:
                temp_counts = counts.copy()
                for idx in block_triples[i]:
                    temp_counts[idx] -= 1
                if np.all(temp_counts >= 0):
                    uncovered_after = np.where(temp_counts == 0)[0]
                    if len(uncovered_after) == 0:
                        # Remove block i
                        current_blocks.pop(i)
                        block_triples.pop(i)
                        counts = temp_counts
                        current_len -= 1
                        if current_len < best_len:
                            best_len = current_len
                            best_blocks = list(current_blocks)
                        T *= cooling
                        continue
                    else:
                        # Try to find a block that covers all uncovered_after
                        for j in range(len(current_blocks)):
                            if j == i:
                                continue
                            covers_all = True
                            for uidx in uncovered_after:
                                if uidx not in block_triples[j]:
                                    covers_all = False
                                    break
                            if covers_all:
                                current_blocks.pop(i)
                                block_triples.pop(i)
                                counts = temp_counts
                                current_len -= 1
                                if current_len < best_len:
                                    best_len = current_len
                                    best_blocks = list(current_blocks)
                                T *= cooling
                                break
                        if not covers_all:
                            T *= cooling
                            continue

            # Option 2: Swap block i with a new random block
            else:
                # Generate a random block
                new_block = tuple(sorted(random.sample(range(v), k)))
                new_block_triples = compute_block_triples(new_block)

                # Check if this swap is valid (all triples still covered)
                temp_counts = counts.copy()
                for idx in block_triples[i]:
                    temp_counts[idx] -= 1
                for idx in new_block_triples:
                    temp_counts[idx] += 1

                if np.all(temp_counts >= 0):
                    # Accept the swap
                    current_blocks[i] = new_block
                    block_triples[i] = new_block_triples
                    counts = temp_counts
                    T *= cooling
                else:
                    # Reject
                    T *= cooling

        return best_blocks

    best_blocks = None
    best_len = float('inf')

    seed_counter = 0

    while time.time() < deadline - 15:
        # Phase 1: Fresh greedy
        blocks, uncovered, num_uncovered = greedy_cover(random_state=seed_counter, scarcity_weighted=True)
        trimmed = trim_blocks(blocks)

        if len(trimmed) < best_len:
            best_len = len(trimmed)
            best_blocks = list(trimmed)

        if best_len <= 69:
            break

        # Phase 2: Simulated annealing
        if time.time() < deadline - 8:
            sa_time = min(5.0, deadline - time.time() - 8)
            sa_blocks = simulated_annealing(trimmed, max_time=sa_time)
            if len(sa_blocks) < best_len:
                best_len = len(sa_blocks)
                best_blocks = list(sa_blocks)

        # Phase 3: Perturbation and repair
        if time.time() < deadline - 5:
            rng_pert = random.Random(seed_counter + 1000)

            strategies = []

            if len(trimmed) > 5:
                num_kicks = rng_pert.randint(1, min(3, len(trimmed) - 1))
                kick_indices = rng_pert.sample(range(len(trimmed)), num_kicks)
                remaining = [b for i, b in enumerate(trimmed) if i not in kick_indices]
                strategies.append(remaining)

            for remaining_blocks in strategies:
                if time.time() >= deadline - 5:
                    break

                covered_mask = np.zeros(num_triples, dtype=np.uint8)
                for block in remaining_blocks:
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        covered_mask[idx] = 1

                uncovered_mask = 1 - covered_mask
                num_uncovered = int(np.sum(uncovered_mask))

                if num_uncovered > 0:
                    repaired_blocks, _, _ = greedy_cover(uncovered_mask=uncovered_mask, random_state=seed_counter + 200, scarcity_weighted=True)
                    combined = remaining_blocks + repaired_blocks
                else:
                    combined = remaining_blocks

                trimmed_combined = trim_blocks(combined)
                trimmed_combined = local_search_trim(trimmed_combined, max_iters=15)

                if len(trimmed_combined) < best_len:
                    best_len = len(trimmed_combined)
                    best_blocks = list(trimmed_combined)

        seed_counter += 1

    # Final polish
    if best_blocks and time.time() < deadline - 2:
        best_blocks = local_search_trim(best_blocks, max_iters=50)

    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
