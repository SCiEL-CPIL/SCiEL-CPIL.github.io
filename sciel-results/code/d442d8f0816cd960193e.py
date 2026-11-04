"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Enumerate all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # For each block (k-subset), precompute which t-subsets it covers
    # We'll build this on-the-fly during greedy

    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = bytearray(n_t)  # 0 = uncovered, 1 = covered
        blocks = []

        # Precompute for each point, the list of t-subsets containing it
        point_t_subs = [[] for _ in range(v)]
        for i, sub in enumerate(all_t_subs):
            for p in sub:
                point_t_subs[p].append(i)

        while True:
            # Find first uncovered t-subset
            first_uncovered = -1
            for i in range(n_t):
                if not covered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                break

            # Check deadline
            if time.time() > deadline:
                break

            # Start block with the points from the first uncovered t-subset
            block = list(all_t_subs[first_uncovered])

            # Build coverage of current partial block
            # A t-subset is covered by partial block if it's a subset of block
            # For adding point x, gain = number of uncovered t-subsets that contain x
            # and are subsets of block + {x}

            while len(block) < k:
                # For each candidate point, compute how many new t-subsets it covers
                # A t-subset s is newly covered if:
                #   - s contains x
                #   - s is subset of block ∪ {x}
                #   - s is not yet covered
                # Since s has size t and block has size b, s ⊆ block∪{x} means
                # s ⊆ block ∪ {x}, so s \ {x} ⊆ block and s ∩ block has size t-1 (if x∉block)

                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count uncovered t-subsets containing x where the other t-1 elements are in block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        idx = t_sub_to_idx.get(s)
                        if idx is not None and not covered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x == -1 or best_gain <= 0:
                    # Can't improve, just add a random point
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        best_x = random.choice(candidates)
                    else:
                        break

                block.append(best_x)

            block.sort()

            # Mark all t-subsets of this block as covered
            for sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[sub]
                covered[idx] = 1

            blocks.append(block)

        return blocks

    # Run greedy with a fixed seed first
    blocks = greedy_cover(seed=42)

    # Check if complete
    def is_complete(blocks):
        covered = bytearray(n_t)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered[t_sub_to_idx[sub]] = 1
        return all(covered)

    if not is_complete(blocks):
        # Try different seeds
        for seed in range(1, 20):
            if time.time() > deadline - 5:
                break
            b = greedy_cover(seed=seed)
            if is_complete(b) and len(b) < len(blocks):
                blocks = b

    # Reduction pass: try to remove redundant blocks
    def try_reduce(blocks):
        if not blocks:
            return blocks
        n = len(blocks)
        # For each block, check if removing it still covers everything
        # Precompute coverage per block
        block_cover = []
        for block in blocks:
            cov = set()
            for sub in itertools.combinations(block, t):
                cov.add(t_sub_to_idx[sub])
            block_cover.append(cov)

        # Total coverage
        total_cov = set()
        for cov in block_cover:
            total_cov.update(cov)

        changed = True
        while changed:
            changed = False
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline - 2:
                    return blocks
                # Check if block i is redundant
                # A t-subset is covered by block i and no other block if it's in block_cover[i] but not in union of others
                # Faster: for each t-subset in block_cover[i], check if any other block covers it
                redundant = True
                for idx in block_cover[i]:
                    # Check if any other block covers this t-subset
                    found = False
                    for j in range(n):
                        if j == i:
                            continue
                        if idx in block_cover[j]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    block_cover.pop(i)
                    changed = True
                    break  # restart loop since indices changed
        return blocks

    if time.time() < deadline - 10:
        blocks = try_reduce(blocks)

    # Final verification - if incomplete, fall back to a fresh run
    if not is_complete(blocks):
        # Emergency: run simple greedy without deadline
        covered = bytearray(n_t)
        blocks = []
        while True:
            first_uncovered = -1
            for i in range(n_t):
                if not covered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                break
            block = list(all_t_subs[first_uncovered])
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        idx = t_sub_to_idx.get(s)
                        if idx is not None and not covered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    candidates = [x for x in range(v) if x not in block]
                    best_x = candidates[0] if candidates else 0
                block.append(best_x)
            block.sort()
            for sub in itertools.combinations(block, t):
                covered[t_sub_to_idx[sub]] = 1
            blocks.append(block)

    return blocks
# EVOLVE-BLOCK-END
