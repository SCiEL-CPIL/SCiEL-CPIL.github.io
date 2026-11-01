import itertools
import random
import time

def construct(v, k, t):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    total_subsets = len(all_t_subsets)
    
    def greedy_once():
        covered_set = set()
        blocks = []
        
        while len(covered_set) < total_subsets:
            # Pick a random uncovered subset
            start_subset = None
            attempts = 0
            max_attempts = 1000
            while start_subset is None:
                idx = random.randint(0, total_subsets - 1)
                if all_t_subsets[idx] not in covered_set:
                    start_subset = all_t_subsets[idx]
                attempts += 1
                if attempts > max_attempts:
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
                
                for x in points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    if len(block) >= t - 1:
                        for y_tuple in itertools.combinations(block, t - 1):
                            s = tuple(sorted(y_tuple + (x,)))
                            if s not in covered_set:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        if gain > 0:
                            best_points.append(x)
                        if gain == 0 and best_gain == 0:
                            best_points.append(x)
                
                if not best_points:
                    break
                    
                block.append(random.choice(best_points))
            
            block.sort()
            blocks.append(block)
            
            for s in itertools.combinations(block, t):
                covered_set.add(s)
                
        return blocks

    def local_search(blocks):
        # Convert to set of frozensets for fast lookup? 
        # We need to check validity: every t-subset is in at least one block.
        # To check if removing a block is valid, we need to know if any t-subset 
        # covered *only* by that block exists.
        
        # Precompute coverage counts?
        # Let's try a simpler approach:
        # For each block, calculate the set of t-subsets it covers.
        # If any of those subsets are not covered by any *other* block, it's not redundant.
        
        # This is O(N_blocks * C(k,t) * N_blocks) which might be slow if N_blocks is large (300+).
        # 300 * 56 * 300 = 5M ops per pass. 5M is fast in Python? Maybe 0.5s.
        # We can do this in a loop until no improvement.
        
        changed = True
        while changed:
            changed = False
            # Randomize order to avoid bias
            random.shuffle(blocks)
            
            # We need to know coverage by others.
            # Let's maintain a list of blocks.
            # For each block i:
            #   unique_covered = 0
            #   for s in combinations(blocks[i], t):
            #       count = 0
            #       for j in range(len(blocks)):
            #           if i == j: continue
            #           if s in set(combinations(blocks[j], t)): # Too slow
            #               count += 1
            #       if count == 0:
            #           unique_covered += 1
            #   if unique_covered == 0:
            #       remove blocks[i]
            #       changed = True
            #       break # Restart loop
            
            # Optimization: Precompute set of t-subsets for each block?
            # blocks_sets = [set(itertools.combinations(b, t)) for b in blocks]
            
            # Even better: 
            # Count how many times each t-subset is covered.
            # coverage_count = {}
            # for b in blocks:
            #   for s in combinations(b, t):
            #     coverage_count[s] = coverage_count.get(s, 0) + 1
            
            # Then a block is redundant if for all s in combinations(block, t), coverage_count[s] > 1.
            
            # Let's implement this.
            
            # 1. Calculate coverage counts
            coverage_count = {}
            block_subsets = []
            for b in blocks:
                subs = list(itertools.combinations(b, t))
                block_subsets.append(subs)
                for s in subs:
                    coverage_count[s] = coverage_count.get(s, 0) + 1
            
            # 2. Find a redundant block
            for i, b in enumerate(blocks):
                redundant = True
                for s in block_subsets[i]:
                    if coverage_count[s] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    del blocks[i]
                    del block_subsets[i]
                    # We don't need to update coverage_count immediately if we restart the while loop
                    changed = True
                    break
            
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    time_limit = 115
    
    while time.time() - start_time < time_limit:
        blocks = greedy_once()
        # Apply local search
        blocks = local_search(blocks)
        
        # Verify validity (just in case, though local search should maintain it)
        # And count
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    return best_blocks if best_blocks else []
