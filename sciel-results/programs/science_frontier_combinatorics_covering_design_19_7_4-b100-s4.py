"""Covering design C(v, k, t) – deterministic greedy with local search."""
import itertools
import time
import random
from collections import defaultdict

_DEADLINE_S = 115.0  # Safety margin from 120s evaluator limit
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _count_tsubs_in_block(block, v, t):
    """Count number of t-subsets in a block."""
    return len(list(itertools.combinations(block, t)))


def _greedy_construct(v, k, t, seed=42):
    """
    Deterministic greedy construction using frequency-weighted scoring.
    Each candidate block is scored by sum of 1/freq[x] for each uncovered t-subset it covers,
    where freq[x] is how many times point x has been used in previous blocks.
    This encourages spreading points evenly.
    """
    random.seed(seed)
    all_tsubs = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    block_set = set()
    point_freq = [0] * v  # frequency of each point in chosen blocks

    while uncovered and _time_left() > 1.0:
        # Find best block: maximize weighted coverage
        best_block = None
        best_score = -1
        best_weighted_gain = -1

        # Sample candidate blocks if too many to enumerate
        # For v=19, k=7: C(19,7) = 50388, feasible to enumerate
        candidates = list(itertools.combinations(range(v), k))
        # Shuffle for diversity
        random.shuffle(candidates)

        for cand in candidates:
            cand_set = set(cand)
            # Compute weighted gain: for each t-subset in cand, if uncovered, add 1/sqrt(1+sum(freq[x] for x in tsub))
            gain = 0.0
            for tsub in itertools.combinations(cand, t):
                if tsub in uncovered:
                    # Weight by inverse frequency of points in this t-subset
                    weight = 1.0
                    for x in tsub:
                        weight /= (1.0 + point_freq[x])
                    gain += weight

            if gain > best_weighted_gain or (gain == best_weighted_gain and len(cand_set) > 0):
                best_weighted_gain = gain
                best_block = cand

        if best_block is None or best_weighted_gain <= 0:
            # No improvement possible with greedy; break
            break

        best_block = tuple(sorted(best_block))
        if best_block in block_set:
            continue
        block_set.add(best_block)
        blocks.append(list(best_block))

        # Update point frequencies
        for x in best_block:
            point_freq[x] += 1

        # Remove covered t-subsets
        for tsub in itertools.combinations(best_block, t):
            uncovered.discard(tsub)

        if _time_left() < 1.0:
            break

    return blocks, uncovered


def _local_search(blocks, v, k, t, max_iter=200):
    """
    Local search: try to replace blocks with alternatives that maintain coverage
    but allow removal of redundant blocks, or swap to improve structure.
    Returns improved block list.
    """
    if not blocks:
        return blocks

    # Build coverage map: for each t-subset, which blocks cover it
    all_tsubs = set(itertools.combinations(range(v), t))
    tsub_to_blocks = defaultdict(list)
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            tsub_to_blocks[tsub].append(i)

    current_blocks = [list(b) for b in blocks]
    best_len = len(current_blocks)
    best_blocks = [list(b) for b in current_blocks]

    for iteration in range(max_iter):
        if _time_left() < 2.0:
            break

        # Try removing each block and see if coverage is still complete
        improved = False
        for i in range(len(current_blocks)):
            if _time_left() < 2.0:
                break
            # Check if block i is redundant: all its t-subsets are covered by other blocks
            redundant = True
            block_i = current_blocks[i]
            for tsub in itertools.combinations(block_i, t):
                # Check if any other block covers this t-subset
                covered_by_other = False
                for j in range(len(current_blocks)):
                    if j == i:
                        continue
                    if tsub in set(itertools.combinations(current_blocks[j], t)):
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break

            if redundant:
                # Remove block i
                new_blocks = [b for j, b in enumerate(current_blocks) if j != i]
                current_blocks = new_blocks
                improved = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = [list(b) for b in current_blocks]
                break  # Restart check after removal

        if not improved:
            # Try swapping: replace one block with a different one that covers same or more
            # For simplicity, try random replacement and check validity
            if random.random() < 0.3 and _time_left() > 3.0:
                i = random.randrange(len(current_blocks))
                old_block = current_blocks[i]
                # Generate a random new block
                new_block = sorted(random.sample(range(v), k))
                new_block = tuple(new_block)
                # Check if new block covers at least all t-subsets that old block uniquely covered
                old_tsubs = set(itertools.combinations(old_block, t))
                new_tsubs = set(itertools.combinations(new_block, t))
                # A t-subset is uniquely covered by old block if no other block covers it
                unique_old = set()
                for tsub in old_tsubs:
                    is_unique = True
                    for j in range(len(current_blocks)):
                        if j == i:
                            continue
                        if tsub in set(itertools.combinations(current_blocks[j], t)):
                            is_unique = False
                            break
                    if is_unique:
                        unique_old.add(tsub)

                if unique_old.issubset(new_tsubs):
                    # Safe swap
                    current_blocks[i] = list(new_block)
                    if len(set(map(tuple, current_blocks))) < len(current_blocks):
                        # Duplicate created, revert
                        current_blocks[i] = list(old_block)
                    else:
                        if len(current_blocks) < best_len:
                            best_len = len(current_blocks)
                            best_blocks = [list(b) for b in current_blocks]

    return best_blocks


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using deterministic greedy with local search.
    Returns a list of blocks (each a list of k distinct integers in 0..v-1).
    """
    # Try multiple seeds for greedy construction, keep best result
    best_blocks = None
    best_len = float('inf')

    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718]
    for seed in seeds:
        if _time_left() < 5.0:
            break

        blocks, uncovered = _greedy_construct(v, k, t, seed=seed)

        if not blocks:
            continue

        # Apply local search to improve
        improved_blocks = _local_search(blocks, v, k, t, max_iter=100)

        # Verify coverage
        all_tsubs = set(itertools.combinations(range(v), t))
        covered = set()
        for block in improved_blocks:
            for tsub in itertools.combinations(block, t):
                covered.add(tsub)

        if covered == all_tsubs:
            if len(improved_blocks) < best_len:
                best_len = len(improved_blocks)
                best_blocks = improved_blocks
                if best_len <= 151:
                    break

    # Fallback: if no valid covering found, do a pure greedy without local search
    if best_blocks is None:
        blocks, uncovered = _greedy_construct(v, k, t, seed=0)
        if blocks:
            best_blocks = blocks

    # Final fallback: exhaustive greedy (should not be needed)
    if best_blocks is None or _time_left() < 1.0:
        all_tsubs = set(itertools.combinations(range(v), t))
        uncovered = set(all_tsubs)
        blocks = []
        while uncovered and _time_left() > 0.5:
            target = min(uncovered)
            # Greedily build a block containing target
            block = list(target)
            block_set = set(block)
            while len(block) < k and _time_left() > 0.3:
                best_pt = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(combo + (x,)))
                        if sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pt = x
                if best_pt is None:
                    candidates = [x for x in range(v) if x not in block_set]
                    if not candidates:
                        break
                    best_pt = random.choice(candidates)
                block.append(best_pt)
                block_set.add(best_pt)
            if len(block) < k:
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
            blocks.append(sorted(block))
            for tsub in itertools.combinations(sorted(block), t):
                uncovered.discard(tsub)
        best_blocks = blocks

    return best_blocks
