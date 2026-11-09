import itertools
import random
import time
from collections import defaultdict

def _bitmask_of_set(s, v):
    m = 0
    for x in s:
        m |= (1 << x)
    return m

def _all_4subsets(v):
    return list(itertools.combinations(range(v), 4))

def _build_4subsets_index(v):
    """For each point, list the 4-subsets containing it (as frozensets)."""
    idx = defaultdict(list)
    for c in itertools.combinations(range(v), 4):
        for x in c:
            idx[x].append(c)
    return idx

def _greedy_cover(v, k, t, seed):
    """Fast greedy using bitmask coverage tracking. Returns list of blocks (each a frozenset)."""
    rng = random.Random(seed)
    all_tsubs = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index
    tsub_to_idx = {c: i for i, c in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # uncovered bitmask: bit i set means t-subset i is uncovered
    uncovered_mask = (1 << n_tsubs) - 1  # all uncovered
    # But n_tsubs can be large (C(19,4)=3876), so use a Python int (arbitrary precision)
    
    blocks = []
    # Precompute for each point: which 4-subsets contain it
    point_to_tsubs = defaultdict(list)
    for i, c in enumerate(all_tsubs):
        for x in c:
            point_to_tsubs[x].append(i)
    
    # Precompute: for a partial block (set of points), the bitmask of t-subsets it covers
    # We'll compute incrementally
    
    while uncovered_mask:
        # Pick a random uncovered t-subset to start the block
        # Find one set bit
        # Get lowest set bit index
        low_bit = uncovered_mask & (-uncovered_mask)
        idx = low_bit.bit_length() - 1
        # Randomize: pick a random set bit
        # For speed, just use the lowest (or randomize among first few)
        start_tsub = all_tsubs[idx]
        
        # Build block starting with one random point from start_tsub
        block_points = [rng.choice(start_tsub)]
        
        # Incrementally track covered t-subsets for this block
        # covered_so_far: bitmask of t-subsets covered by current partial block
        covered_so_far = 0
        for p in block_points:
            for si in point_to_tsubs[p]:
                # Check if all points of all_tsubs[si] are in block_points
                # This is expensive; instead, build up the set of points in block
                pass
        
        # Better approach: track the set of points in block, and compute which t-subsets are covered
        # For a block with points {p1,...,pm}, a t-subset is covered if all its t points are in {p1,...,pm}
        # We need an efficient way. Let's use a different strategy:
        # For each candidate point x, compute gain = number of uncovered t-subsets that would be covered
        # when x is added to the current block.
        
        # A t-subset c is covered by block B if c ⊆ B.
        # When we add x to B, the new t-subsets covered are those c where c ⊆ B∪{x} but c ⊄ B.
        # These are c where x ∈ c and c\{x} ⊆ B.
        
        # So for current block B (set of points), gain(x) = number of uncovered t-subsets c
        # such that x ∈ c and c\{x} ⊆ B.
        
        # Precompute for each pair (x, subset_of_B_of_size_t-1): whether c = (x ∪ subset) is a t-subset
        # This is getting complex. Let's use a simpler but still fast approach:
        
        # For each x not in block, gain(x) = count of uncovered t-subsets c where x ∈ c and c\{x} ⊆ block
        # We can precompute for each (t-1)-subset s and point x not in s: whether s∪{x} is a t-subset
        # and its index.
        
        # Precompute: for each (t-1)-tuple, map to the t-subsets that extend it
        # Actually, let's just do it directly but efficiently.
        
        block_set = frozenset(block_points)
        
        while len(block_points) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets c where x ∈ c and c\{x} ⊆ block_set
                gain = 0
                for si in point_to_tsubs[x]:
                    c = all_tsubs[si]
                    # Check if c\{x} ⊆ block_set
                    rest_ok = True
                    for y in c:
                        if y != x and y not in block_set:
                            rest_ok = False
                            break
                    if rest_ok and (uncovered_mask >> si) & 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block_points.append(best_x)
            block_set = frozenset(block_points)
        
        block_set = frozenset(block_points)
        # Update uncovered_mask: remove all t-subsets covered by this block
        for si, c in enumerate(all_tsubs):
            if all(y in block_set for y in c):
                uncovered_mask &= ~(1 << si)
        
        blocks.append(block_set)
    
    return blocks

def _local_search_reduce(blocks, v, k, t, seed, time_limit):
    """Try to reduce the number of blocks via local search. Returns improved blocks or original."""
    rng = random.Random(seed)
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {c: i for i, c in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Build coverage: for each t-subset, which blocks cover it
    # block i covers t-subset j if all_tsubs[j] ⊆ blocks[i]
    coverage = [0] * n_tsubs  # bitmask of blocks covering t-subset j
    for i, b in enumerate(blocks):
        for j, c in enumerate(all_tsubs):
            if all(y in b for y in c):
                coverage[j] |= (1 << i)
    
    n_blocks = len(blocks)
    best_blocks = list(blocks)
    best_count = n_blocks
    start_time = time.time()
    
    # Try removing each block and see if we can re-cover with fewer
    for trial in range(200):
        if time.time() - start_time > time_limit:
            break
        if n_blocks <= best_count:
            break
        
        # Pick a random block to try removing
        ri = rng.randrange(n_blocks)
        
        # Check if removing block ri leaves any t-subset uncovered
        # A t-subset j is still covered if coverage[j] has a bit other than ri
        uncovered_after_removal = []
        for j in range(n_tsubs):
            if coverage[j] & (1 << ri):
                # This t-subset is covered by block ri
                if coverage[j] & ~(1 << ri) == 0:
                    # Only covered by block ri
                    uncovered_after_removal.append(j)
        
        if not uncovered_after_removal:
            # Block ri is redundant! Remove it.
            blocks.pop(ri)
            n_blocks -= 1
            # Rebuild coverage
            coverage = [0] * n_tsubs
            for i, b in enumerate(blocks):
                for j, c in enumerate(all_tsubs):
                    if all(y in b for y in c):
                        coverage[j] |= (1 << i)
            continue
        
        # Try to re-cover the uncovered t-subsets with a new block
        # Greedily build a block that covers as many uncovered t-subsets as possible
        # while also covering some already-covered ones (for efficiency)
        
        # Start with a random uncovered t-subset
        start_j = uncovered_after_removal[rng.randrange(len(uncovered_after_removal))]
        start_tsub = all_tsubs[start_j]
        block_points = [rng.choice(start_tsub)]
        block_set = frozenset(block_points)
        
        # Uncovered t-subsets to cover (indices)
        to_cover = set(uncovered_after_removal)
        
        while len(block_points) < k and to_cover:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for j in to_cover:
                    c = all_tsubs[j]
                    if x in c and all(y in block_set or y == x for y in c):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain <= 0:
                break
            block_points.append(best_x)
            block_set = frozenset(block_points)
            # Update to_cover
            to_cover = {j for j in to_cover if not all(y in block_set for y in all_tsubs[j])}
        
        if to_cover:
            # Can't fully cover; skip this trial
            continue
        
        # Replace block ri with the new block
        new_block = block_set
        # Check: does this new block cover all of uncovered_after_removal?
        for j in uncovered_after_removal:
            if not all(y in new_block for y in all_tsubs[j]):
                break
        else:
            # Good replacement
            blocks[ri] = new_block
            # Update coverage
            for j in range(n_tsubs):
                coverage[j] &= ~(1 << ri)  # Clear old coverage from block ri
                if all(y in new_block for y in all_tsubs[j]):
                    coverage[j] |= (1 << ri)
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts
    seed = 42
    restart = 0
    while time.time() - start_time < deadline:
        # Phase 1: Greedy
        blocks = _greedy_cover(v, k, t, seed)
        count = len(blocks)
        
        if count < best_count:
            best_count = count
            best_blocks = list(blocks)
        
        # Phase 2: Local search to reduce
        remaining = deadline - (time.time() - start_time)
        if remaining > 5:  # Only do local search if we have time
            ls_blocks = _local_search_reduce(blocks, v, k, t, seed + 1000, min(remaining - 5, 30))
            ls_count = len(ls_blocks)
            if ls_count < best_count:
                best_count = ls_count
                best_blocks = list(ls_blocks)
        
        seed += 12345
        restart += 1
        # After a few restarts, check time more carefully
        if restart % 5 == 0:
            if time.time() - start_time > deadline - 10:
                break
    
    # Convert frozensets to sorted lists
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = _greedy_cover(v, k, t, 0)
    
    result = [sorted(b) for b in best_blocks]
    return result
