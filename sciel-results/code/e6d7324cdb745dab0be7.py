import itertools
import random
import time
from collections import Counter

def construct(v, k, t):
    # Parameters for the specific task, but function should be general
    # v=19, k=7, t=4 expected
    
    # Precompute all t-subsets for efficiency
    all_t_subsets = list(itertools.combinations(range(v), t))
    all_t_set = set(all_t_subsets)
    
    # 1. Greedy Initialization - improved version
    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(all_t_set)
        blocks = []
        
        # For each point, precompute which t-subsets it participates in
        # This helps with fast gain computation
        
        while uncovered:
            # Pick the uncovered subset with the fewest possible completions
            # to a k-block (i.e., most constrained)
            best_start = None
            best_min_completions = float('inf')
            
            # Sample a few uncovered subsets to find a good starting point
            sample_size = min(50, len(uncovered))
            sample = random.sample(list(uncovered), sample_size) if len(uncovered) > sample_size else list(uncovered)
            
            for sub in sample:
                # Count how many ways we can extend this to a k-block
                # We need k-t more points from v - t available points
                # This is C(v-t, k-t) but we want to find the most constrained
                # Actually, let's just pick randomly for speed, but with bias
                pass
            
            # For simplicity and speed, pick a random uncovered subset
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            
            while len(block) < k:
                # Find the point that covers the most uncovered subsets
                best_point = None
                best_gain = -1
                # Precompute (t-1) subsets of current block
                current_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in current_t_minus_1:
                        new_subset = tuple(sorted(rest + (x,)))
                        if new_subset in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    # Fallback: pick any random point not in block
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        best_point = random.choice(candidates)
                    else:
                        break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        return blocks

    # 2. Fast Local Search Optimizer - improved version
    def optimize(blocks, time_limit=110):
        # State: list of blocks, coverage_count dict
        current_blocks = [tuple(b) for b in blocks]
        
        def calc_coverage(blist):
            cov = Counter()
            for b in blist:
                for sub in itertools.combinations(b, t):
                    cov[sub] += 1
            return cov

        coverage_count = calc_coverage(current_blocks)
        
        # Helper to check if a block is redundant (all its t-subsets have count > 1)
        def is_redundant(b, cov):
            for sub in itertools.combinations(b, t):
                if cov[sub] <= 1:
                    return False
            return True

        start_time = time.time()
        no_improve_streak = 0
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Phase 1: Remove redundant blocks
            i = 0
            while i < len(current_blocks):
                b = current_blocks[i]
                if is_redundant(b, coverage_count):
                    # Remove b
                    for sub in itertools.combinations(b, t):
                        coverage_count[sub] -= 1
                        if coverage_count[sub] == 0:
                            del coverage_count[sub]
                    current_blocks.pop(i)
                    improved = True
                    # Do not increment i, check the next block at the same index
                else:
                    i += 1
            
            if improved:
                no_improve_streak = 0
                continue # Restart loop as indices changed
            else:
                no_improve_streak += 1
                # If stuck, try swaps or break if time is low
                if no_improve_streak > 200 and time.time() - start_time > time_limit - 10:
                    break
                if time.time() - start_time > time_limit - 5:
                    break

            # Phase 2: Try to replace a block with a better one
            if len(current_blocks) > 1:
                # Pick a random index to potentially replace
                idx = random.randint(0, len(current_blocks) - 1)
                old_block = current_blocks[idx]
                
                # Count unique subsets (covered only by this block)
                unique_subsets = []
                for sub in itertools.combinations(old_block, t):
                    if coverage_count[sub] == 1:
                        unique_subsets.append(sub)
                
                # Generate several random candidate blocks and pick the best
                best_new_block = None
                best_score = len(unique_subsets)  # Must cover all unique subsets
                
                for _ in range(10):
                    new_block = tuple(sorted(random.sample(range(v), k)))
                    new_set = set(itertools.combinations(new_block, t))
                    
                    # Check if all unique subsets are covered
                    if all(sub in new_set for sub in unique_subsets):
                        # Score: number of subsets covered by new_block that are currently covered by old_block
                        # We want to maximize overlap to enable removals
                        overlap = sum(1 for sub in itertools.combinations(old_block, t) if sub in new_set)
                        # Also consider how many new subsets this covers
                        new_coverage = sum(1 for sub in new_set if coverage_count.get(sub, 0) == 0)
                        score = overlap + new_coverage * 0.1  # Prefer overlap
                        if score > best_score:
                            best_score = score
                            best_new_block = new_block
                
                if best_new_block is not None and best_new_block != old_block:
                    # Perform replacement
                    # Remove old_block
                    for sub in itertools.combinations(old_block, t):
                        coverage_count[sub] -= 1
                        if coverage_count[sub] == 0:
                            del coverage_count[sub]
                    # Add new_block
                    for sub in itertools.combinations(best_new_block, t):
                        coverage_count[sub] += 1
                    current_blocks[idx] = best_new_block
                    improved = True
                    no_improve_streak = 0
                    # Go back to Phase 1 to see if this enabled removals
                    continue
        
        return [list(b) for b in current_blocks]

    # 3. Cleanup: Remove redundant blocks (just in case)
    def clean_up(blocks):
        block_set = [tuple(b) for b in blocks]
        coverage_count = Counter()
        for b in block_set:
            for sub in itertools.combinations(b, t):
                coverage_count[sub] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in block_set:
                redundant = True
                for sub in itertools.combinations(b, t):
                    if coverage_count[sub] <= 1:
                        redundant = False
                        break
                if redundant:
                    changed = True
                    for sub in itertools.combinations(b, t):
                        coverage_count[sub] -= 1
                        if coverage_count[sub] == 0:
                            del coverage_count[sub]
                else:
                    new_blocks.append(b)
            block_set = new_blocks
        
        return [list(b) for b in block_set]

    # Main Execution
    # Run greedy multiple times and pick the best
    best_initial = None
    best_len = float('inf')
    
    # Time budget for greedy initialization
    greedy_start = time.time()
    greedy_time_limit = 30  # 30 seconds for greedy
    
    for i in range(50):
        if time.time() - greedy_start > greedy_time_limit:
            break
        initial_blocks = greedy_cover(seed=i)
        if len(initial_blocks) < best_len:
            best_len = len(initial_blocks)
            best_initial = initial_blocks
    
    # Run Optimizer
    remaining_time = 120 - (time.time() - greedy_start) - 5  # Leave 5s for cleanup
    optimized_blocks = optimize(best_initial, time_limit=max(10, remaining_time))
    
    # Final cleanup
    final_blocks = clean_up(optimized_blocks)
    
    return final_blocks
