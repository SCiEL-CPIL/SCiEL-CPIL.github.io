import itertools
import time
import numpy as np
from math import comb

# Precompute all t-subsets of v points and map them to indices for fast lookup
# Since v, k, t are fixed for the task (19, 7, 4), we can precompute at module level
# or inside construct if we want to be safe, but module level is faster.
# However, the contract says construct(v, k, t) is the interface.
# We'll compute inside but cache if possible, or just compute.
# C(19, 4) = 3876. This is small enough to compute quickly.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe against 150s limit
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Bitmask representation of covered t-subsets
    # Use a Python int as a bitmask, or a numpy array of bools.
    # For 3876 subsets, a numpy array of bools is efficient for vectorized ops.
    covered = np.zeros(n_t_subsets, dtype=bool)
    
    blocks = []
    remaining = n_t_subsets
    
    # Greedy phase
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Find the best block to add
        # A block is a k-subset. It covers all t-subsets contained in it.
        # We want to maximize the number of newly covered t-subsets.
        
        # Optimization: Instead of trying all C(v,k) blocks, we can use a greedy heuristic.
        # Start with a point, then greedily add points that cover the most uncovered t-subsets.
        
        # To speed this up, we can precompute for each point x, the list of t-subsets containing x.
        # But simpler: just iterate.
        
        # Heuristic: Pick the first uncovered t-subset, and build a block around it.
        # Find first uncovered index
        if not np.any(~covered):
            break
        first_uncovered_idx = np.argmax(~covered)
        seed_t_subset = all_t_subsets[first_uncovered_idx]
        
        # Start block with one point from seed_t_subset
        # Actually, a better heuristic:
        # Try to find a k-subset that covers the most uncovered t-subsets.
        # Since C(19,7) is 50388, and for each we need to check 35 t-subsets,
        # 50388 * 35 ~ 1.7M operations. This might be slow in Python loops.
        
        # Let's use a faster greedy:
        # 1. Pick a point x that is in many uncovered t-subsets.
        # 2. Pick a point y that is in many uncovered t-subsets containing x.
        # ...
        
        # Even faster:
        # Precompute for each point x, a bitmask (or boolean array) of t-subsets containing x.
        # point_masks = []
        # for x in range(v):
        #     mask = np.zeros(n_t_subsets, dtype=bool)
        #     for i, s in enumerate(all_t_subsets):
        #         if x in s:
        #             mask[i] = True
        #     point_masks.append(mask)
            
        # This is O(v * n_t_subsets) which is 19 * 3876 ~ 73k. Very fast.
        
        if not hasattr(construct, 'point_masks'):
            construct.point_masks = []
            for x in range(v):
                mask = np.zeros(n_t_subsets, dtype=bool)
                # Optimization: precompute which indices contain x
                # s is a tuple of 4 ints.
                # We can use list comprehension
                for i, s in enumerate(all_t_subsets):
                    if x in s:
                        mask[i] = True
                construct.point_masks.append(mask)
        
        point_masks = construct.point_masks
        
        # Greedy block construction
        block = []
        # Pick first point: the one that covers the most uncovered t-subsets
        # gain_x = np.sum(point_masks[x] & ~covered)
        # best_x = np.argmax([np.sum(point_masks[x] & ~covered) for x in range(v)])
        
        # To avoid recomputing ~covered every time, we can just use it.
        # But ~covered is a numpy array.
        
        # Let's try a different approach for speed:
        # Iterate through points, calculate gain.
        
        current_covered = covered
        uncovered_mask = ~current_covered
        
        # Step 1: Pick first point
        gains = np.array([np.sum(point_masks[x] & uncovered_mask) for x in range(v)])
        best_x = np.argmax(gains)
        block.append(best_x)
        
        # Update uncovered_mask? No, we are building a block.
        # We want to add points that cover the most *new* t-subsets given the current block.
        
        # If block has size b, a new point y covers t-subsets that contain y and (b-1) points from block?
        # No, a t-subset is covered if ALL its t points are in the block.
        # So if we have block B, adding y covers t-subsets S such that S subset of B U {y} and y in S.
        # This means S must contain y and t-1 points from B.
        
        # Let's precompute for each pair (x, y) or just calculate on the fly.
        
        # For efficiency, let's just iterate.
        # We need to find y that maximizes:
        # | { S in uncovered : S subset of B U {y} and y in S } |
        
        # Let's use a different strategy:
        # Randomized greedy with restarts?
        # Or just simple greedy:
        
        # Let's stick to the simple greedy but optimized.
        # We have B. We want to add y.
        # The t-subsets covered by adding y are those S in uncovered such that y in S and S \ {y} subset of B.
        
        # Let's precompute for each point y, and each subset of size t-1 from B?
        # That's expensive.
        
        # Alternative:
        # Just check all y in range(v) \ B.
        # For each y, count how many uncovered t-subsets are "completed" by y.
        # A t-subset S is completed by y if y in S and S \ {y} is a subset of B.
        
        # We can precompute for each B (as a bitmask of points), which t-subsets are "almost covered"?
        
        # Let's try a simpler heuristic that is known to work well:
        # Start with a random point, then greedily add the point that covers the most uncovered t-subsets.
        
        # To make it faster, we can sample a few starting points.
        
        # Actually, the most effective improvement is usually just a better greedy criterion.
        # Let's use the "max gain" criterion properly.
        
        # Precompute for each point y, the list of t-subsets containing y.
        # t_subsets_by_point = []
        # for y in range(v):
        #     lst = [i for i, s in enumerate(all_t_subsets) if y in s]
        #     t_subsets_by_point.append(lst)
        
        if not hasattr(construct, 't_subsets_by_point'):
            construct.t_subsets_by_point = []
            for y in range(v):
                lst = [i for i, s in enumerate(all_t_subsets) if y in s]
                construct.t_subsets_by_point.append(lst)
        
        t_subsets_by_point = construct.t_subsets_by_point
        
        # Now, for a given block B, and candidate y:
        # Gain(y) = number of i in t_subsets_by_point[y] such that:
        # 1. not covered[i]
        # 2. all points in all_t_subsets[i] except y are in B
        
        # Let's represent B as a set for fast lookup.
        B_set = set(block)
        
        best_y = -1
        best_gain = -1
        
        for y in range(v):
            if y in B_set:
                continue
            gain = 0
            for i in t_subsets_by_point[y]:
                if not covered[i]:
                    # Check if S \ {y} subset of B
                    s = all_t_subsets[i]
                    # s is a tuple of 4 ints. One is y.
                    # We need to check if the other 3 are in B.
                    # Since t=4, we can just check the 3 others.
                    # This is fast.
                    ok = True
                    for p in s:
                        if p != y and p not in B_set:
                            ok = False
                            break
                    if ok:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_y = y
        
        if best_y == -1 or best_gain == 0:
            # If no gain, break or pick random?
            # If we are stuck, break the inner loop?
            # Actually, if best_gain is 0, adding any point covers nothing new?
            # That shouldn't happen if there are uncovered t-subsets.
            # Unless the uncovered t-subsets don't intersect with any y outside B?
            # Impossible, since uncovered t-subset S has points. If S is not covered,
            # and we are building B, eventually we must cover it.
            # If best_gain is 0, it means no point outside B completes any uncovered t-subset.
            # This implies all uncovered t-subsets are disjoint from (V \ B)?
            # No, it means for every uncovered S, and every y not in B, S \ {y} is not subset of B.
            # This is possible if B is "bad".
            # Let's just pick a random point if gain is 0 to escape.
            import random
            candidates = [x for x in range(v) if x not in B_set]
            if candidates:
                best_y = random.choice(candidates)
            else:
                break # Should not happen if len(block) < k
        
        block.append(best_y)
        
        # Check if we should stop early if we have k points
        if len(block) == k:
            # Add this block
            blocks.append(sorted(block))
            
            # Update covered
            for s in itertools.combinations(block, t):
                idx = t_subset_to_idx[s]
                if not covered[idx]:
                    covered[idx] = True
                    remaining -= 1
            # Continue to next block
            
        # If we ran out of time, break
        if time.time() > deadline:
            break

    # Local Search Phase (if time remains)
    # Try to replace a block with a better one, or remove a block if possible.
    # This is complex. Let's just return the greedy result for now.
    # A simple improvement: try to find if any block can be removed.
    # If a block B is removed, are all its t-subsets covered by other blocks?
    
    # This is O(N_blocks * C(k,t) * N_blocks) which is too slow.
    # Let's just return the greedy result.
    
    return blocks
