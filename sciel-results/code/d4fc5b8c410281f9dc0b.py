"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    # Bitmask-based greedy covering with deadline guard and deterministic repair.
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Precompute all t-subsets as bitmasks and index them
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {tuple(c): i for i, c in enumerate(all_tsubs)}

    # uncovered_counts[i] = 1 if t-subset i is still uncovered, 0 otherwise
    uncovered_counts = bytearray(len(all_tsubs))
    for i in range(len(all_tsubs)):
        uncovered_counts[i] = 1

    # For each point x, precompute which t-subsets contain x
    # tsub_mask[x] is a bytearray of length len(all_tsubs) with 1 if x in t-subset
    # Actually, let's use a different approach: for each point x, store list of t-subset indices containing x
    point_to_tsubs = [[] for _ in range(v)]
    for i, c in enumerate(all_tsubs):
        for x in c:
            point_to_tsubs[x].append(i)

    # For efficient gain computation: when building a block, we track which t-subsets
    # are "covered by current partial block" (i.e., all but one element of the t-subset
    # is in the block, and that one element is the candidate x)
    # gain(x) = number of uncovered t-subsets S such that |S \ {x}| ⊆ block and x ∉ block

    blocks = []
    remaining = len(all_tsubs)  # number of uncovered t-subsets

    while remaining > 0:
        if time.time() > deadline:
            break

        # Find the first uncovered t-subset to start a new block
        start_idx = -1
        for i in range(len(all_tsubs)):
            if uncovered_counts[i]:
                start_idx = i
                break
        if start_idx == -1:
            break

        block = list(all_tsubs[start_idx])

        while len(block) < k:
            best_x = -1
            best_gain = -1
            block_set = set(block)

            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets where all other t-1 elements are in block
                gain = 0
                # Iterate over t-subsets containing x
                for idx in point_to_tsubs[x]:
                    if uncovered_counts[idx]:
                        # Check if all other elements of this t-subset are in block
                        c = all_tsubs[idx]
                        ok = True
                        for y in c:
                            if y != x and y not in block_set:
                                ok = False
                                break
                        if ok:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1:
                # No point helps; just pick any unused point
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
            block.append(best_x)

        block.sort()
        blocks.append(block)

        # Mark all t-subsets within this block as covered
        block_set = set(block)
        for i, c in enumerate(all_tsubs):
            if uncovered_counts[i] and all(x in block_set for x in c):
                uncovered_counts[i] = 0
                remaining -= 1

    # Deterministic repair: if time ran out, finish covering with naive greedy
    if remaining > 0:
        uncovered = set()
        for i, c in enumerate(all_tsubs):
            if uncovered_counts[i]:
                uncovered.add(c)
        while uncovered:
            block = list(min(uncovered))
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    for x in range(v):
                        if x not in block_set:
                            best_x = x
                            break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            block_set = set(block)
            uncovered.difference_update(itertools.combinations(block, t))

    # Remove duplicate blocks if any
    seen = set()
    unique_blocks = []
    for b in blocks:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            unique_blocks.append(b)
    return unique_blocks
# EVOLVE-BLOCK-END
