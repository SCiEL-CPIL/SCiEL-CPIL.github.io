"""Covering design C(19,8,5) - randomized greedy with bitmask acceleration and multi-restart."""
import itertools
import time
import random
import sys


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Precompute all t-subsets and encode as bitmasks
    all_t_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_t_subs)

    # Map each t-subset to a bitmask of its elements
    sub_bitmasks = [0] * n_subs
    for i, sub in enumerate(all_t_subs):
        mask = 0
        for x in sub:
            mask |= (1 << x)
        sub_bitmasks[i] = mask

    # Map bitmask -> index (for fast lookup)
    bitmask_to_idx = {sub_bitmasks[i]: i for i in range(n_subs)}

    # For each point, precompute which t-subsets contain it
    point_to_subs = [[] for _ in range(v)]
    for i, mask in enumerate(sub_bitmasks):
        for x in range(v):
            if mask & (1 << x):
                point_to_subs[x].append(i)

    best_blocks = None
    best_count = n_subs + 1  # worst case

    # Multi-restart randomized greedy
    max_restarts = 200
    rng = random.Random(42)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Track uncovered t-subsets using a boolean array
        uncovered = bytearray(b'\x01') * n_subs  # all covered initially... wait, we want uncovered=1
        # Actually: uncovered[i] = 1 means subset i is NOT yet covered
        uncovered = bytearray([1] * n_subs)
        uncovered_count = n_subs

        blocks = []

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            # Pick a random uncovered t-subset to start the block
            # Use a simple approach: scan for an uncovered one
            # For efficiency, maintain a list of uncovered indices
            # But that's expensive to maintain; instead, pick randomly and retry
            # Actually, let's just pick the first uncovered one for determinism within a restart
            # but randomize the order by permuting indices
            
            # Find an uncovered subset
            # For speed, we'll pick a random index and check
            found = False
            for _ in range(200):  # try up to 200 random picks
                idx = rng.randint(0, n_subs - 1)
                if uncovered[idx]:
                    found = True
                    break
            if not found:
                # Fallback: scan
                for idx in range(n_subs):
                    if uncovered[idx]:
                        found = True
                        break
            if not found:
                break  # Should not happen

            # Start block with elements of this t-subset
            start_sub = all_t_subs[idx]
            block_set = set(start_sub)
            block_mask = sub_bitmasks[idx]

            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block_set) < k:
                if time.time() > deadline:
                    break

                best_gain = -1
                best_point = None

                for x in range(v):
                    if x in block_set:
                        continue
                    # Compute how many uncovered t-subsets would be covered by adding x
                    # A t-subset is covered if all its elements are in the block
                    # When we add x to the block, a t-subset S is covered iff
                    # S \ {x} ⊆ block_set and x ∈ S
                    # So we need: S contains x, and S \ {x} is a subset of block_set
                    # i.e., (S_mask & ~(block_mask | (1 << x))) == 0 and (S_mask & (1 << x)) != 0

                    gain = 0
                    for s_idx in point_to_subs[x]:
                        if not uncovered[s_idx]:
                            continue
                        sm = sub_bitmasks[s_idx]
                        # Check if sm contains x and sm without x is subset of block_mask
                        if sm & (1 << x):
                            if sm & ~(block_mask | (1 << x)) == 0:
                                gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point is None or best_gain <= 0:
                    # No point adds coverage; pick a random one to complete the block
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining)
                    else:
                        break

                block_set.add(best_point)
                block_mask |= (1 << best_point)

            # Convert block to sorted list
            block = sorted(block_set)

            # Check if this block is a duplicate
            block_tuple = tuple(block)
            if block_tuple in {tuple(b) for b in blocks}:
                continue

            blocks.append(block)

            # Update uncovered: mark all t-subsets contained in this block as covered
            for i in range(n_subs):
                if uncovered[i]:
                    if sub_bitmasks[i] & ~block_mask == 0:
                        uncovered[i] = 0
                        uncovered_count -= 1

            if uncovered_count == 0:
                break

        # Check if we have a complete covering
        if uncovered_count == 0 and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

            # If we're close to the known best, we can stop early
            if best_count <= 335:
                break

    if best_blocks is None:
        # Fallback: return a trivial (but likely huge) covering
        # Use the deterministic greedy as fallback
        uncovered_set = set(range(n_subs))
        blocks = []
        for idx in sorted(uncovered_set):
            if time.time() > deadline:
                break
            block = list(all_t_subs[idx])
            blocks.append(block)
        best_blocks = blocks

    return best_blocks
