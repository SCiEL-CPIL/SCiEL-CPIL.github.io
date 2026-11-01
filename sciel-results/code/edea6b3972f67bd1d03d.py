"""Covering design C(19,8,5) - optimized greedy with bitmask acceleration and multi-restart."""
import itertools
import time
import random
import sys


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit

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

    # For each point, precompute which t-subsets contain it
    point_to_subs = [[] for _ in range(v)]
    for i, mask in enumerate(sub_bitmasks):
        for x in range(v):
            if mask & (1 << x):
                point_to_subs[x].append(i)

    # Precompute for each point x and each subset of size t-1 containing x,
    # the list of t-subsets that would be covered if x is added to a block containing that (t-1)-subset.
    # Actually, let's precompute: for each point x, for each (t-1)-subset S not containing x,
    # the list of t-subsets equal to S ∪ {x}.
    # But that's just one t-subset per (S, x) pair. So we can just check directly.

    # Better: for gain computation, when block has mask B and we consider adding x,
    # a t-subset S is newly covered iff:
    #   x ∈ S, S \ {x} ⊆ B, and S was uncovered.
    # Equivalently: S_mask & ~(B | (1<<x)) == 0 and S_mask & (1<<x) != 0.
    # Since S_mask & (1<<x) != 0 is always true for s_idx in point_to_subs[x],
    # we just need: sub_bitmasks[s_idx] & ~B == (1<<x) or 0... wait.
    # S_mask & ~(B | (1<<x)) == 0 means all bits in S_mask are in B or x.
    # Since x is in S_mask, this means S_mask & ~B ⊆ {x}, i.e., S_mask & ~B is either 0 or (1<<x).
    # But since x ∈ S, S_mask & (1<<x) != 0, so S_mask & ~B must have x as a bit.
    # So condition is: (S_mask & ~B) == (1<<x) or (S_mask & ~B) == 0? No.
    # S_mask & ~(B | (1<<x)) == 0 means no bit of S_mask is outside B∪{x}.
    # Since x ∈ S_mask, the bits of S_mask outside B are a subset of {x}.
    # So S_mask & ~B is either 0 or (1<<x). But x ∈ S_mask, so if x ∉ B, then S_mask & ~B has x.
    # If x ∈ B, then S_mask & ~B = 0.
    # So the condition simplifies to: (S_mask & ~B) & ~(1<<x) == 0, i.e., S_mask & ~B is subset of {x}.
    # Since we know x ∈ S, this is equivalent to: S_mask & ~(B | (1<<x)) == 0.

    best_blocks = None
    best_count = n_subs + 1  # worst case

    # Multi-restart randomized greedy
    max_restarts = 300
    rng = random.Random(42)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Track uncovered t-subsets using a boolean array
        uncovered = bytearray([1] * n_subs)
        uncovered_count = n_subs

        blocks = []
        block_set_cache = set()

        while uncovered_count > 0:
            if time.time() > deadline:
                break

            # Find an uncovered subset - use weighted random selection favoring "hard" subsets
            # For now, just pick randomly with fallback scan
            found = False
            for _ in range(50):
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
                break

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
                second_best_gain = -1
                second_best_point = None

                for x in range(v):
                    if x in block_set:
                        continue

                    gain = 0
                    for s_idx in point_to_subs[x]:
                        if not uncovered[s_idx]:
                            continue
                        sm = sub_bitmasks[s_idx]
                        # Check if adding x would cover this subset
                        if sm & ~(block_mask | (1 << x)) == 0:
                            gain += 1

                    if gain > best_gain:
                        second_best_gain = best_gain
                        second_best_point = best_point
                        best_gain = gain
                        best_point = x
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = x

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
            block_tuple = tuple(block)

            # Check if this block is a duplicate
            if block_tuple in block_set_cache:
                continue

            block_set_cache.add(block_tuple)
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
        uncovered_set = set(range(n_subs))
        blocks = []
        for idx in sorted(uncovered_set):
            if time.time() > deadline:
                break
            block = list(all_t_subs[idx])
            blocks.append(block)
        best_blocks = blocks

    return best_blocks
