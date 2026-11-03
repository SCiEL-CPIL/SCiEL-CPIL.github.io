import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []

    # Precompute: for each (t-1)-subset, which points complete it to a t-subset
    # (t-1)-subsets that appear in uncovered triples
    # For efficiency, map each (t-1)-subset to a frozenset of points that complete it

    # Actually, let's use a different approach:
    # For a partial block B (size < k), gain(x) = number of uncovered triples containing x and (t-1) elements of B
    # = sum over (t-1)-subsets S of B of |{uncovered triples containing S ∪ {x}}|

    # Precompute for each (t-1)-subset: the set of points that, when added, form an uncovered triple
    # But uncovered changes over time, so we need dynamic computation.

    # Alternative: maintain a count array. For each point x and each (t-1)-subset S,
    # count[x][S] = number of uncovered triples containing S ∪ {x}
    # This is too large.

    # Better: for each uncovered triple, it contributes to gain(x) for each x in the triple,
    # and for each (t-1)-subset of the triple.
    # So gain(x) for a partial block B = number of uncovered triples T such that T ∩ B has size >= t-1 and x ∈ T.

    # Let me think differently. For a partial block B of size b, and candidate x not in B:
    # Adding x to B covers an uncovered triple T if T ⊆ B ∪ {x}, i.e., T ∩ B has size t-1 and x ∈ T.
    # Equivalently, T contains exactly t-1 elements from B and x.

    # So gain(x) = |{T ∈ uncovered : T ⊆ B ∪ {x} and x ∈ T}|
    #            = |{T ∈ uncovered : (T \ {x}) ⊆ B and |T ∩ B| = t-1}|

    # For small t (t=3), this is: for each pair (a,b) in B, count uncovered triples containing {a,b,x}.
    # We can precompute a dictionary: pair_to_points = {(a,b): set of x such that (a,b,x) is uncovered}

    # Let's maintain: for each (t-1)-subset S, a set of points x such that S ∪ {x} is uncovered.
    # When we cover a triple, we remove x from the set for each (t-1)-subset of that triple.

    # This is feasible for t=3: there are C(24,2) = 276 pairs, each with up to 22 points.

    # Build initial structure
    # For each (t-1)-subset, store a set of completing points
    from collections import defaultdict

    # (t-1)-subset -> set of points
    t_minus_1 = t - 1
    # Map: frozenset of (t-1) points -> set of points that complete to uncovered t-subset
    completing = defaultdict(set)
    for triple in uncovered:
        for i in range(t):
            # (t-1)-subset by removing element i
            sub = tuple(x for j, x in enumerate(triple) if j != i)
            removed = triple[i]
            completing[frozenset(sub)].add(removed)

    def add_block_to_completing(block):
        """Remove all t-subsets of block from uncovered, update completing."""
        for tsub in itertools.combinations(block, t):
            if tsub in uncovered:
                uncovered.discard(tsub)
                for i in range(t):
                    sub = frozenset(x for j, x in enumerate(tsub) if j != i)
                    removed = tsub[i]
                    if sub in completing:
                        completing[sub].discard(removed)

    while uncovered:
        if time.time() > deadline:
            break

        # Start with the lexicographically smallest uncovered triple
        first_triple = min(uncovered)
        block = list(first_triple)

        while len(block) < k:
            if time.time() > deadline:
                break

            block_set = set(block)
            best_x = -1
            best_gain = -1

            # For each candidate x not in block, compute gain
            # gain(x) = sum over (t-1)-subsets S of block of |completing[S] ∩ {x}|
            #         = sum over (t-1)-subsets S of block of (1 if x in completing[S] else 0)

            # For t=3, (t-1)=2, so we iterate over pairs in block
            candidates = [x for x in range(v) if x not in block_set]

            # Vectorized approach for t=3:
            # For each pair in block, get the set of completing points
            # Then for each candidate, count how many pairs have it in their completing set
            if t_minus_1 == 2:
                # Build a count array
                gain_counts = np.zeros(v, dtype=np.int32)
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        pair = frozenset((block[i], block[j]))
                        if pair in completing:
                            for x in completing[pair]:
                                gain_counts[x] += 1

                # Find best candidate
                for x in candidates:
                    g = gain_counts[x]
                    if g > best_gain:
                        best_gain = g
                        best_x = x
            else:
                # General case: iterate over all (t-1)-subsets of block
                for sub in itertools.combinations(block, t_minus_1):
                    fs = frozenset(sub)
                    if fs in completing:
                        for x in completing[fs]:
                            if x not in block_set:
                                # This x gets +1 gain for this sub
                                pass
                # Fallback: compute gain for each candidate
                for x in candidates:
                    g = 0
                    for sub in itertools.combinations(block, t_minus_1):
                        if x in completing.get(frozenset(sub), set()):
                            g += 1
                    if g > best_gain:
                        best_gain = g
                        best_x = x

            if best_x == -1:
                # No gain possible, just pick any remaining point
                best_x = candidates[0] if candidates else -1
                if best_x == -1:
                    break

            block.append(best_x)

        block.sort()
        blocks.append(block)
        add_block_to_completing(block)

    # Post-optimization: try to improve by swapping points in blocks
    # For each block, try swapping one point out and another in if it covers more uncovered triples
    if uncovered and time.time() < deadline - 5:
        # Rebuild completing structure for post-optimization
        # Actually, we can just try swaps by checking coverage directly
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for bi, block in enumerate(blocks):
                if time.time() > deadline - 5:
                    break
                block_set = set(block)
                # Try each point in block as the one to remove
                for out_idx in range(k):
                    out_point = block[out_idx]
                    # Try each point not in block as the one to add
                    for in_point in range(v):
                        if in_point in block_set:
                            continue
                        new_block = block[:out_idx] + [in_point] + block[out_idx + 1:]
                        new_block_set = set(new_block)
                        # Count how many uncovered triples are in new_block but not in block
                        # And how many uncovered triples are in block but not in new_block
                        gain = 0
                        loss = 0
                        for tsub in itertools.combinations(new_block, t):
                            if tsub in uncovered:
                                # Check if it was covered by old block
                                if tsub not in set(itertools.combinations(block, t)):
                                    gain += 1
                        for tsub in itertools.combinations(block, t):
                            if tsub in uncovered:
                                if tsub not in set(itertools.combinations(new_block, t)):
                                    loss += 1
                        if gain > loss:
                            blocks[bi] = sorted(new_block)
                            # Update uncovered
                            for tsub in itertools.combinations(new_block, t):
                                uncovered.discard(tsub)
                            improved = True
                            break
                    if improved:
                        break

    return blocks
