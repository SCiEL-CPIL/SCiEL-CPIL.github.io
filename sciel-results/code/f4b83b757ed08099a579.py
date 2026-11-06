import time
import random
from itertools import combinations

def construct():
    """
    Construct a maximal packing of 6-sets on 12 points with no repeated 5-subset.
    Target: 132 blocks (Steiner system S(5,6,12)).
    """
    start_time = time.time()
    DEADLINE = 100  # seconds; leave margin before 150s kill

    def now():
        return time.time() - start_time

    # Precompute all 5-subsets of each 6-block
    # Represent 5-subsets as frozensets for fast membership
    all_blocks = list(combinations(range(12), 6))
    # Map each block to its 5-subsets
    block_5subs = []
    for b in all_blocks:
        subs = frozenset(combinations(b, 5))
        block_5subs.append(subs)

    # Greedy lexicographic packing (same as seed)
    covered = set()
    blocks = []
    for i, b in enumerate(all_blocks):
        subs = block_5subs[i]
        if not any(s in covered for s in subs):
            blocks.append(list(b))
            covered.update(subs)

    if len(blocks) == 132:
        return blocks

    # Try to improve via local search: swap one block for another to increase count
    # This is unlikely to help since greedy is already optimal for this structure,
    # but let's try randomized restarts with different orderings.

    # Try random restarts with shuffled order
    best_blocks = blocks[:]
    for _ in range(20):
        if now() > DEADLINE:
            break
        rng = random.Random()
        order = list(range(len(all_blocks)))
        rng.shuffle(order)
        covered = set()
        blocks = []
        for idx in order:
            if now() > DEADLINE:
                break
            b = all_blocks[idx]
            subs = block_5subs[idx]
            if not any(s in covered for s in subs):
                blocks.append(list(b))
                covered.update(subs)
        if len(blocks) > len(best_blocks):
            best_blocks = blocks[:]

    return best_blocks
