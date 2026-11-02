"""Improved program for C(19,7,4) covering design.
Uses dynamic heap-based greedy with local improvement via 1-for-1 replacements."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin under 120s
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to its index for O(1) lookup
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # We'll track which t-subsets are covered
    covered = [False] * n_t
    covered_count = [0] * n_t  # how many blocks cover each t-subset
    blocks = []
    
    def time_ok():
        return time.time() < deadline
    
    def add_block(block):
        """Cover all t-subsets within block."""
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            if not covered[idx]:
                covered[idx] = True
            covered_count[idx] += 1
        blocks.append(tuple(block))
    
    def remove_block(block):
        """Remove coverage from a block."""
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            covered_count[idx] -= 1
            if covered_count[idx] == 0:
                covered[idx] = False
        blocks.remove(block)
    
    def uncovered_count_for_point_in_block(block_set, x):
        """Count how many uncovered t-subsets would be covered by adding x to block."""
        if x in block_set:
            return 0
        count = 0
        for rest in itertools.combinations(block_set, t - 1):
            sub = tuple(sorted(rest + (x,)))
            idx = t_sub_to_idx[sub]
            if not covered[idx]:
                count += 1
        return count
    
    # Greedy construction with dynamic scoring
    while True:
        if not time_ok():
            break
        
        # Find an uncovered t-subset to seed the next block
        uncovered_idx = None
        for i in range(n_t):
            if not covered[i]:
                uncovered_idx = i
                break
        if uncovered_idx is None:
            break
        
        # Start block with the t-subset
        seed_subset = all_t_subsets[uncovered_idx]
        block_set = set(seed_subset)
        
        # Extend block to size k using greedy with dynamic gain
        while len(block_set) < k:
            if not time_ok():
                break
            
            best_x = None
            best_gain = -1
            for x in all_points:
                if x in block_set:
                    continue
                gain = uncovered_count_for_point_in_block(block_set, x)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block_set.add(best_x)
        
        if not time_ok():
            break
        
        # Add the block
        block = tuple(sorted(block_set))
        add_block(block)
    
    # Local improvement: try 1-for-1 block replacements
    if time_ok():
        # Precompute for each block, the t-subsets it covers
        block_tsubs = []
        for b in blocks:
            tsubs = [t_sub_to_idx[sub] for sub in itertools.combinations(b, t)]
            block_tsubs.append(tsubs)
        
        # Try to improve by replacing blocks
        improved = True
        while improved and time_ok():
            improved = False
            for i in range(len(blocks)):
                if not time_ok():
                    break
                # Try to find a better block to replace blocks[i]
                # The new block must cover all t-subsets that blocks[i] uniquely covers
                # and should cover additional uncovered t-subsets
                
                # Find t-subsets uniquely covered by block i
                unique_tsubs = []
                for idx in block_tsubs[i]:
                    if covered_count[idx] == 1:
                        unique_tsubs.append(idx)
                
                if not unique_tsubs:
                    # Block i covers no unique t-subsets, try to remove it
                    remove_block(blocks[i])
                    improved = True
                    break
                
                # Try to find a replacement block that covers all unique_tsubs
                # and maximizes additional coverage
                best_new_block = None
                best_additional = -1
                
                # Generate candidate blocks containing all unique t-subsets
                # A block must contain the union of all unique t-subsets
                must_contain = set()
                for idx in unique_tsubs:
                    for pt in all_t_subsets[idx]:
                        must_contain.add(pt)
                
                if len(must_contain) > k:
                    continue
                
                # Try different extensions
                remaining_slots = k - len(must_contain)
                candidates_pool = [p for p in all_points if p not in must_contain]
                
                # Try a few random extensions
                for trial in range(20):
                    if not time_ok():
                        break
                    if remaining_slots == 0:
                        candidate = tuple(sorted(must_contain))
                    else:
                        extra = random.sample(candidates_pool, remaining_slots)
                        candidate = tuple(sorted(must_contain | set(extra)))
                    
                    # Check if this candidate is already in blocks
                    if candidate in blocks:
                        continue
                    
                    # Calculate additional coverage
                    additional = 0
                    for sub in itertools.combinations(candidate, t):
                        idx = t_sub_to_idx[sub]
                        if not covered[idx]:
                            additional += 1
                        elif covered_count[idx] > 1:
                            # Still counts as "additional" in the sense of redundancy
                            pass
                    
                    if additional > best_additional:
                        best_additional = additional
                        best_new_block = candidate
                
                if best_new_block is not None and best_additional > 0:
                    # Replace block i with best_new_block
                    remove_block(blocks[i])
                    add_block(best_new_block)
                    # Recompute block_tsubs
                    block_tsubs = []
                    for b in blocks:
                        tsubs = [t_sub_to_idx[sub] for sub in itertools.combinations(b, t)]
                        block_tsubs.append(tsubs)
                    improved = True
                    break
        
        # After improvement, try to remove redundant blocks
        if time_ok():
            for i in range(len(blocks) - 1, -1, -1):
                if not time_ok():
                    break
                # Check if block i is redundant
                redundant = True
                for idx in block_tsubs[i]:
                    if covered_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(blocks[i])
                    block_tsubs.pop(i)
    
    return [list(b) for b in blocks]

# EVOLVE-BLOCK-END
