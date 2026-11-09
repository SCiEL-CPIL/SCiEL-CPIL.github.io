"""Vectorized greedy covering with deadline guard."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 115.0  # seconds, safety margin before 150s kill

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        return False
    return time.time() > _DEADLINE


def construct(v, k, t):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    # All t-subsets as tuples
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)

    # Map each t-subset to an index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # uncovered: boolean array, True means still uncovered
    uncovered = np.ones(n_t, dtype=bool)
    n_uncovered = n_t

    blocks = []

    while n_uncovered > 0:
        if _check_deadline():
            break

        # Pick the first uncovered t-subset to seed the block
        first_idx = int(np.argmax(uncovered))
        seed = list(all_t_subs[first_idx])

        # Build block greedily: start with seed points
        block = set(seed)

        # For each remaining point, compute how many uncovered t-subsets it would cover
        # A point x covers t-subset S if x in S and S is uncovered
        # We need to count, for each candidate x, the number of uncovered t-subsets
        # that contain x and are "completable" with the current block.
        #
        # More precisely: if we add point x to block, the new t-subsets covered are
        # those t-subsets S such that:
        #   - S is uncovered
        #   - x in S
        #   - S \ {x} is a subset of block (i.e., all other elements of S are already in block)
        #
        # So for a given block B and candidate x not in B:
        #   gain(x) = number of uncovered t-subsets S with x in S and S\{x} ⊆ B
        #
        # We can precompute for each t-subset its elements. For efficiency,
        # let's think of it differently.
        #
        # For the current block B (size b), when we add x, the new t-subsets covered
        # are t-subsets S where S = (B ∩ S) ∪ {x} and |B ∩ S| = t-1, x ∉ B.
        # Equivalently, S\{x} ⊆ B and |S\{x}| = t-1.
        #
        # So gain(x) = number of (t-1)-subsets U of B such that U ∪ {x} is an uncovered t-subset.

        while len(block) < k:
            if _check_deadline():
                break

            # For each candidate x not in block, compute gain
            candidates = [x for x in range(v) if x not in block]

            # Precompute: for each (t-1)-subset U of block, find all x such that U∪{x} is uncovered
            # Map: for each (t-1)-subset U, the set of x where U∪{x} is a t-subset and is uncovered
            # gain[x] = count of U in C(block, t-1) such that sorted(U ∪ {x}) is uncovered

            # Efficient approach: iterate over all (t-1)-subsets of block
            # For each such subset U, for each x not in U (and not in block), check if U∪{x} is uncovered
            # But this could be expensive. Let's use a different approach.

            # Alternative: for each candidate x, gain(x) = number of uncovered t-subsets S
            # such that x ∈ S and S\{x} ⊆ block.
            # S\{x} ⊆ block means S ⊆ block ∪ {x} and x ∈ S.
            # So S is a t-subset of block ∪ {x} that contains x.
            # Number of such S = C(|block|, t-1) if |block| >= t-1.

            # But we only count those that are uncovered. So:
            # gain(x) = |{S : S ⊆ block∪{x}, x∈S, |S|=t, S uncovered}|

            # Let's compute this by iterating over (t-1)-subsets of block.
            block_list = list(block)
            b = len(block_list)

            if b < t - 1:
                # Not enough points in block to form any (t-1)-subset
                # Just pick the first candidate
                best_x = candidates[0]
            else:
                # For each (t-1)-subset U of block, for each x not in block:
                #   if sorted(U + [x]) is a valid t-subset and uncovered, increment gain[x]
                gains = {x: 0 for x in candidates}

                # Iterate over (t-1)-subsets of block
                for U in itertools.combinations(block_list, t - 1):
                    U_set = set(U)
                    for x in candidates:
                        if x not in U_set:
                            # Form the t-subset
                            t_sub = tuple(sorted(U + (x,)))
                            idx = t_sub_to_idx.get(t_sub)
                            if idx is not None and uncovered[idx]:
                                gains[x] += 1

                # Pick the candidate with max gain; tie-break by smallest x
                best_x = max(candidates, key=lambda x: (gains[x], -x))

            block.add(best_x)

        # Block is complete; sort and add
        block_sorted = sorted(block)
        blocks.append(block_sorted)

        # Mark all t-subsets of this block as covered
        for t_sub in itertools.combinations(block_sorted, t):
            idx = t_sub_to_idx[t_sub]
            if uncovered[idx]:
                uncovered[idx] = False
                n_uncovered -= 1

    return blocks
# EVOLVE-BLOCK-END
