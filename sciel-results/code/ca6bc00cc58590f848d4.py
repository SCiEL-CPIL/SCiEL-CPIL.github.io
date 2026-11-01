import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) with as few k-subsets of v points as possible
    such that every t-subset is contained in at least one block.
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    # Use a deadline with safety margin
    deadline = time.time() + 145.0
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    sub_to_id = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # For each (t-1)-subset, precompute which t-subsets contain it (for t>=3)
    # For t=3: pair -> list of tsub ids containing that pair
    if t >= 3:
        # Precompute (t-1)-subset -> list of t-subset ids
        t_minus_1_to_tsubs = {}
        for i, sub in enumerate(all_tsubs):
            for j in range(t):
                # (t-1)-subset = sub with element j removed
                rest = tuple(sub[:j] + sub[j+1:])
                if rest not in t_minus_1_to_tsubs:
                    t_minus_1_to_tsubs[rest] = []
                t_minus_1_to_tsubs[rest].append(i)
    else:
        t_minus_1_to_tsubs = None
    
    def greedy_cover(rng, time_limit):
        """
        Greedy covering: repeatedly pick a block that covers the most uncovered t-subsets.
        Uses randomized tie-breaking for diversity.
        Optimized inner loop.
        """
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        blocks = []
        
        while True:
            if time.time() > time_limit:
                return None
            
            # Find uncovered t-subsets
            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Pick a random uncovered t-subset as seed
            seed_idx = int(rng.choice(uncovered))
            seed_sub = all_tsubs[seed_idx]
            block = list(seed_sub)
            block_set = set(block)
            
            # Extend block to size k using greedy heuristic
            while len(block) < k:
                # Compute (t-1)-subsets of current block (fixed for this step)
                if t >= 3:
                    block_t_minus_1 = list(itertools.combinations(block, t - 1))
                else:
                    block_t_minus_1 = []
                
                best_gain = -1
                best_points = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    if t >= 3:
                        for rest in block_t_minus_1:
                            # rest is a (t-1)-tuple; the t-subset is rest + x
                            ts = tuple(sorted(rest + (x,)))
                            ts_id = sub_to_id.get(ts, -1)
                            if ts_id >= 0 and cover_count[ts_id] == 0:
                                gain += 1
                    else:
                        # t=2: adding x covers pair (block[0], x) and (block[1], x) etc.
                        for rest in block_t_minus_1:
                            ts = tuple(sorted(rest + (x,)))
                            ts_id = sub_to_id.get(ts, -1)
                            if ts_id >= 0 and cover_count[ts_id] == 0:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if not best_points:
                    candidates = [x for x in range(v) if x not in block_set]
                    if not candidates:
                        break
                    best_point = int(rng.choice(candidates))
                else:
                    best_point = int(rng.choice(best_points))
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update cover counts
            for ts in itertools.combinations(block, t):
                ts_id = sub_to_id[ts]
                cover_count[ts_id] += 1
        
        return blocks
    
    def compute_cover_count(blocks):
        cc = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cc[sub_to_id[ts]] += 1
        return cc
    
    def remove_redundant(blocks, cc, time_limit):
        """
        Iteratively remove blocks that are redundant (all their t-subsets are covered by other blocks).
        """
        current_blocks = [list(b) for b in blocks]
        current_cc = cc.copy()
        
        changed = True
        while changed:
            if time.time() > time_limit:
                break
            changed = False
            to_remove = []
            
            for i, b in enumerate(current_blocks):
                redundant = True
                for ts in itertools.combinations(b, t):
                    ts_id = sub_to_id[ts]
                    if current_cc[ts_id] <= 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                for i in to_remove:
                    b = current_blocks[i]
                    for ts in itertools.combinations(b, t):
                        current_cc[sub_to_id[ts]] -= 1
                current_blocks = [b for i, b in enumerate(current_blocks) if i not in remove_set]
                changed = True
        
        return current_blocks, current_cc
    
    def local_search(blocks, cc, rng, time_limit):
        """
        Local search: try removing each block and re-covering the newly uncovered t-subsets
        with a small greedy patch. Accept if patch size < 1 (i.e., no new blocks needed)
        or if total block count decreases.
        """
        current_blocks = [list(b) for b in blocks]
        current_cc = cc.copy()
        best_blocks = [list(b) for b in current_blocks]
        best_count = len(current_blocks)
        
        n_blocks = len(current_blocks)
        
        max_rounds = 80
        for r in range(max_rounds):
            if time.time() > time_limit:
                break
            
            # Try removing each block in random order
            order = list(range(len(current_blocks)))
            rng.shuffle(order)
            
            improved = False
            for i in order:
                if time.time() > time_limit:
                    break
                if len(current_blocks) <= best_count:
                    # Only try to improve
                    pass
                
                # Temporarily remove block i
                removed_block = current_blocks[i]
                # Identify t-subsets that become uncovered
                newly_uncovered = []
                for ts in itertools.combinations(removed_block, t):
                    ts_id = sub_to_id[ts]
                    if current_cc[ts_id] == 1:
                        newly_uncovered.append(ts_id)
                    current_cc[ts_id] -= 1
                
                if not newly_uncovered:
                    # Block was fully redundant
                    current_blocks.pop(i)
                    current_cc = current_cc  # already updated
                    improved = True
                    # Re-index order
                    order = [o for o in order if o != i]
                    for o in order:
                        if o > i:
                            order[order.index(o)] = o - 1
                    continue
                
                # Try to cover newly_uncovered with a single new block
                # Greedy: find a k-subset that covers the most of newly_uncovered
                # Use the first uncovered t-subset as seed
                uncovered_set = set(newly_uncovered)
                
                # Try to build a block that covers many of these
                seed_ts_id = newly_uncovered[0]
                seed_sub = all_tsubs[seed_ts_id]
                block = list(seed_sub)
                block_set = set(block)
                
                # Count how many of newly_uncovered this seed block covers
                def count_covered(block_list):
                    cnt = 0
                    for ts in itertools.combinations(block_list, t):
                        ts_id = sub_to_id[ts]
                        if ts_id in uncovered_set:
                            cnt += 1
                    return cnt
                
                seed_cov = count_covered(block)
                
                # Extend to size k greedily maximizing coverage of newly_uncovered
                while len(block) < k:
                    block_t_minus_1 = list(itertools.combinations(block, t - 1))
                    best_gain = -1
                    best_pts = []
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for rest in block_t_minus_1:
                            ts = tuple(sorted(rest + (x,)))
                            ts_id = sub_to_id.get(ts, -1)
                            if ts_id >= 0 and ts_id in uncovered_set:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_pts = [x]
                        elif gain == best_gain:
                            best_pts.append(x)
                    if not best_pts:
                        candidates = [x for x in range(v) if x not in block_set]
                        if not candidates:
                            break
                        bp = int(rng.choice(candidates))
                    else:
                        bp = int(rng.choice(best_pts))
                    block.append(bp)
                    block_set.add(bp)
                
                block.sort()
                block_cov = count_covered(block)
                
                # Accept if this single block covers all newly_uncovered
                if block_cov >= len(newly_uncovered):
                    # Restore cc for the removed block's tsubs, then add new block
                    for ts in itertools.combinations(removed_block, t):
                        current_cc[sub_to_id[ts]] += 1
                    # Remove block i
                    current_blocks.pop(i)
                    # Add new block
                    current_blocks.append(block)
                    for ts in itertools.combinations(block, t):
                        current_cc[sub_to_id[ts]] += 1
                    improved = True
                    # Update order: removed index i, added at end
                    order = [o for o in order if o != i]
                    for o in order:
                        if o > i:
                            order[order.index(o)] = o - 1
                    order.append(len(current_blocks) - 1)
                    break  # restart the pass
                
                # Revert cc
                for ts in itertools.combinations(removed_block, t):
                    current_cc[sub_to_id[ts]] += 1
        
        if len(current_blocks) < best_count:
            best_blocks = [list(b) for b in current_blocks]
            best_count = len(current_blocks)
        
        return best_blocks
    
    # Main search: multiple randomized greedy restarts with redundancy removal
    best_blocks = None
    best_count = float('inf')
    
    # Use a variety of random seeds
    max_restarts = 60
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        r_rng = random.Random(42 + restart * 7)
        
        # Time budget per restart
        time_per_restart = max(0.5, (deadline - 5 - time.time()) / max(1, max_restarts - restart))
        
        blocks = greedy_cover(r_rng, time.time() + time_per_restart * 0.5)
        if blocks is None:
            continue
        
        # Compute cover count
        cc = compute_cover_count(blocks)
        
        # Remove redundant blocks
        blocks, cc = remove_redundant(blocks, cc, deadline - 5)
        
        # Local search: try to reduce block count further
        if time.time() < deadline - 10:
            blocks = local_search(blocks, cc, r_rng, min(deadline - 5, time.time() + time_per_restart * 0.3))
        
        # Recompute final cover count to ensure validity
        cc_final = compute_cover_count(blocks)
        if np.any(cc_final == 0):
            # Invalid, skip
            continue
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        # Early termination if we've found a very good solution
        if best_count <= 74:
            break
    
    if best_blocks is None:
        # Fallback: run one more greedy without time pressure
        r_rng = random.Random(12345)
        blocks = greedy_cover(r_rng, deadline - 1)
        if blocks is not None:
            cc = compute_cover_count(blocks)
            blocks, cc = remove_redundant(blocks, cc, deadline - 1)
            best_blocks = blocks
        else:
            best_blocks = []
    
    return best_blocks
