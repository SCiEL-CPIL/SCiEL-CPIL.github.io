# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _fast_greedy(v, k, t, rng, deadline):
    """
    Fast greedy covering design construction using NumPy vectorization.
    Returns list of blocks (each a sorted list of k integers).
    """
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each point x, precompute the list of t-subset indices that contain x
    # point_tsub_indices[x] = array of t-subset indices containing x
    point_tsub_indices = [[] for _ in range(v)]
    for ts in all_tsubs:
        for x in ts:
            point_tsub_indices[x].append(tsub_to_idx[ts])
    for x in range(v):
        point_tsub_indices[x] = np.array(point_tsub_indices[x], dtype=np.int32)
    
    # For each pair of points (x,y), precompute t-subsets containing both
    # This is used when building a block: when we have a partial block, we want to know
    # which uncovered t-subsets would be covered by adding a new point.
    # Actually, let's use a different approach:
    # uncovered is a boolean array of size n_tsubs
    # When we add point x to a partial block B, the t-subsets newly covered are those
    # that contain x and (t-1) elements from B.
    # For a partial block B of size s, the t-subsets covered by B ∪ {x} that weren't covered by B
    # are exactly those t-subsets that contain x and exactly t-1 elements from B.
    
    # Precompute for each subset of points up to size t-1, the t-subsets that contain that subset.
    # Actually, let's precompute: for each (t-1)-subset S, the list of t-subsets that contain S.
    # Then when we have partial block B and add x, the newly covered t-subsets are the union over
    # all (t-1)-subsets S of B of {t-subsets containing S ∪ {x}}.
    # But that's expensive. Let me think of a better way.
    
    # Alternative: for each partial block B (as a frozenset), precompute which t-subsets are subsets of B.
    # Then when adding x, the newly covered t-subsets are those that are subsets of B ∪ {x} but not B.
    # A t-subset is a subset of B ∪ {x} iff it's a subset of B or it contains x and the rest is in B.
    # So newly covered = {ts : x in ts and ts \ {x} ⊆ B}
    
    # Let's precompute: for each (t-1)-subset S, the list of t-subsets containing S.
    # t-1 = 4 for our case. Number of 4-subsets of 19 is C(19,4) = 3876.
    # For each 4-subset S, the t-subsets containing S are S ∪ {x} for x not in S, so 19-4=15 each.
    
    t_minus_1 = t - 1
    # Precompute: for each (t-1)-subset, the array of t-subset indices that contain it
    tsub_containing = {}
    for s in itertools.combinations(range(v), t_minus_1):
        idxs = []
        for x in range(v):
            if x not in s:
                full = tuple(sorted(s + (x,)))
                idxs.append(tsub_to_idx[full])
        tsub_containing[s] = np.array(idxs, dtype=np.int32)
    
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []
    
    while np.any(uncovered):
        if time.time() > deadline - 2.0:
            break
        
        # Pick a random uncovered t-subset to start
        uncovered_indices = np.where(uncovered)[0]
        start_idx = rng.choice(uncovered_indices)
        start = all_tsubs[start_idx]
        block = list(start)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline - 2.0:
                break
            
            # For each candidate x not in block, compute how many uncovered t-subsets it would cover
            # A t-subset ts is covered by adding x iff x in ts and ts\{x} ⊆ block
            # So for each (t-1)-subset S of block, the t-subsets containing S ∪ {x} are candidates.
            # But we only count those that are currently uncovered.
            
            best_x = -1
            best_gain = -1
            
            # Get all (t-1)-subsets of block
            block_tminus1_subs = list(itertools.combinations(block, t_minus_1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for S in block_tminus1_subs:
                    idxs = tsub_containing.get(S)
                    if idxs is not None:
                        # Check how many of these are uncovered and contain x
                        # Actually, tsub_containing[S] gives t-subsets containing S, which are S ∪ {y} for y not in S.
                        # We want those where y = x, so we need to check if x is in each.
                        # Better: precompute differently.
                        pass
                
                # Let me redo this: for each (t-1)-subset S of block, the t-subsets that contain S ∪ {x}
                # are exactly {S ∪ {x}} if x not in S. So there's exactly one such t-subset.
                # Wait no: a t-subset containing S ∪ {x} must have all t elements, and S ∪ {x} has t elements.
                # So the only t-subset containing S ∪ {x} is S ∪ {x} itself.
                # So gain for x = number of (t-1)-subsets S of block such that S ∪ {x} is uncovered.
                
                gain = 0
                for S in block_tminus1_subs:
                    if x not in S:
                        full = tuple(sorted(S + (x,)))
                        if full in tsub_to_idx:
                            idx = tsub_to_idx[full]
                            if uncovered[idx]:
                                gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    if best_gain == 0:
                        # No more gain possible, break early
                        # But we still need to pick a point, so continue to find any with gain 0
                        pass
            
            if best_x == -1 or best_gain <= 0:
                # Pick a random point not in block
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_x = rng.choice(candidates)
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            uncovered[idx] = False
    
    return blocks


def _local_search(v, k, t, blocks, deadline, rng):
    """
    Local search: try to remove blocks that are redundant, or replace them with better ones.
    """
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Build coverage count: for each t-subset, how many blocks cover it
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            cover_count[tsub_to_idx[ts]] += 1
    
    removed = set()
    improved = True
    
    while improved and time.time() < deadline - 5.0:
        improved = False
        active = [i for i in range(len(blocks)) if i not in removed]
        rng.shuffle(active)
        
        for bi in active:
            if bi in removed:
                continue
            if time.time() > deadline - 5.0:
                break
            
            # Check if block bi is redundant: all its t-subsets are covered by other active blocks
            is_redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                idx = tsub_to_idx[ts]
                if cover_count[idx] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block bi
                removed.add(bi)
                for ts in itertools.combinations(blocks[bi], t):
                    cover_count[tsub_to_idx[ts]] -= 1
                improved = True
            else:
                # Try to find a replacement block that covers all uniquely covered t-subsets
                unique_subs = []
                for ts in itertools.combinations(blocks[bi], t):
                    idx = tsub_to_idx[ts]
                    if cover_count[idx] == 1:
                        unique_subs.append(ts)
                
                if not unique_subs:
                    continue
                
                # The replacement must contain all points in the union of unique_subs
                required = set()
                for ts in unique_subs:
                    required.update(ts)
                
                if len(required) > k:
                    continue
                
                need_more = k - len(required)
                remaining_points = [x for x in range(v) if x not in required]
                
                # Try a few random completions
                found = False
                for _ in range(3):
                    if need_more == 0:
                        new_block = sorted(required)
                    else:
                        extra = rng.sample(remaining_points, need_more)
                        new_block = sorted(required | set(extra))
                    
                    # Check if new_block covers all unique_subs (it should by construction)
                    # Add new_block, remove bi
                    new_idx = len(blocks)
                    blocks.append(new_block)
                    for ts in itertools.combinations(new_block, t):
                        cover_count[tsub_to_idx[ts]] += 1
                    
                    for ts in itertools.combinations(blocks[bi], t):
                        cover_count[tsub_to_idx[ts]] -= 1
                    
                    removed.add(bi)
                    improved = True
                    found = True
                    break
    
    result = [blocks[i] for i in range(len(blocks)) if i not in removed]
    return result


def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s kill
    rng = random.Random(42)
    
    # Phase 1: Fast greedy baseline
    blocks = _fast_greedy(v, k, t, rng, deadline)
    best = blocks
    best_len = len(blocks)
    
    # Phase 2: Local search
    if time.time() < deadline - 10.0:
        blocks = _local_search(v, k, t, blocks, deadline, rng)
        if len(blocks) < best_len:
            best = blocks
            best_len = len(blocks)
    
    # Phase 3: Multiple greedy restarts with different seeds
    seed = 43
    while time.time() < deadline - 15.0:
        rng2 = random.Random(seed)
        candidate = _fast_greedy(v, k, t, rng2, deadline - 5.0)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
            # Try local search on the better one
            if time.time() < deadline - 10.0:
                candidate = _local_search(v, k, t, candidate, deadline, rng2)
                if len(candidate) < best_len:
                    best = candidate
                    best_len = len(candidate)
        seed += 1
    
    return best
# EVOLVE-BLOCK-END
