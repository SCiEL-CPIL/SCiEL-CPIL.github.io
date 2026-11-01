import itertools
import random
import time

def construct(v, k, t):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    total_subsets = len(all_t_subsets)
    
    # Pre-compute indices for faster access? Not really needed for v=19.
    
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    time_limit = 115  # Conservative limit
    
    # For efficient random selection of uncovered subsets
    # We keep the list of all subsets and a set of covered ones
    
    while time.time() - start_time < time_limit:
        covered_set = set()
        blocks = []
        
        # To speed up the "pick random uncovered" step, we can just
        # iterate or use a counter. 
        # Since total is ~11k, and we cover ~56 per block, 
        # the set of covered grows.
        
        while len(covered_set) < total_subsets:
            # Pick a random uncovered subset
            # Optimization: If covered_set is small, random pick is fast.
            # If covered_set is large, random pick might fail many times.
            # Better: Maintain a list of uncovered? 
            # Or just iterate from a random start index?
            
            # Let's try a simple approach:
            # Pick a random index. If covered, pick another.
            # To avoid infinite loops if almost all are covered, 
            # we can break if we fail too many times, 
            # but the loop condition handles termination.
            
            start_subset = None
            attempts = 0
            max_attempts = 1000
            while start_subset is None:
                idx = random.randint(0, total_subsets - 1)
                if all_t_subsets[idx] not in covered_set:
                    start_subset = all_t_subsets[idx]
                attempts += 1
                if attempts > max_attempts:
                    # Fallback: find first uncovered
                    for s in all_t_subsets:
                        if s not in covered_set:
                            start_subset = s
                            break
                    if start_subset is None:
                        break
                    break
            
            if start_subset is None:
                break
                
            block = list(start_subset)
            
            # Greedily add points
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_points = []
                
                # Calculate gain for each candidate
                for x in points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # Only possible if len(block) >= t-1
                    if len(block) >= t - 1:
                        # Iterate over all (t-1) subsets of block
                        for y_tuple in itertools.combinations(block, t - 1):
                            # Construct the t-subset
                            # y_tuple has t-1 elements, x is the new one
                            # We need to sort them to match the tuple in covered_set
                            # Since y_tuple is from combinations, it is sorted.
                            # We just need to insert x in the correct position
                            # or just sort the result.
                            # Sorting a 5-element tuple is very fast.
                            s = tuple(sorted(y_tuple + (x,)))
                            if s not in covered_set:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        if gain > 0: # Only keep ties if they are useful
                            best_points.append(x)
                        # If gain is 0, we might want to keep all 0s?
                        # If best_gain is 0, we just pick a random valid point.
                        # So we should include 0s in best_points if best_gain is 0.
                        if gain == 0 and best_gain == 0:
                            best_points.append(x)
                
                if not best_points:
                    # Should not happen
                    break
                    
                # Select a point
                # If best_gain > 0, pick from best_points (which only has max gain points)
                # If best_gain == 0, pick from best_points (which has all points with 0 gain)
                block.append(random.choice(best_points))
            
            # Finalize block
            block.sort()
            blocks.append(block)
            
            # Update covered_set
            for s in itertools.combinations(block, t):
                covered_set.add(s)
                
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    return best_blocks if best_blocks else []
