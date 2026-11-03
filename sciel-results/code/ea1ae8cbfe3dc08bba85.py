import itertools
import time
import random
import math
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using greedy initialization + local search.
    """
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each k-subset, which t-subsets it covers
    # This is expensive for large v,k,t but manageable for 19,7,4
    # C(19,7) = 50388, C(7,4) = 35 per block
    # Instead, we'll compute coverage on-the-fly but cache per block
    
    # Greedy initialization
    uncovered = [0] * n_t  # count of uncovered t-subsets (1 = uncovered)
    for i in range(n_t):
        uncovered[i] = 1
    
    blocks = []
    covered_count = 0
    
    while covered_count < n_t:
        # Find most constrained uncovered t-subset (one that appears in fewest potential blocks)
        # For simplicity, pick the first uncovered one
        # Better: pick one with fewest completions
        best_block = None
        best_gain = -1
        
        # Try to find a good block
        # Find an uncovered t-subset
        uncovered_indices = [i for i in range(n_t) if uncovered[i] > 0]
        if not uncovered_indices:
            break
        
        # Pick the most "constrained" uncovered t-subset
        # A t-subset S is constrained if few (k-t)-subsets can extend it
        # For now, pick one randomly from uncovered for variety
        random.shuffle(uncovered_indices)
        
        # Try several candidate starting t-subsets
        for _ in range(3):
            if covered_count >= n_t:
                break
            start_idx = uncovered_indices[_] if _ < len(uncovered_indices) else uncovered_indices[0]
            start = all_t_subsets[start_idx]
            
            # Build block greedily from this start
            block = list(start)
            block_set = set(block)
            
            while len(block) < k:
                best_point = -1
                best_g = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets this point would cover
                    # when added to current block
                    g = 0
                    # For each (t-1)-subset of current block, check if adding x covers an uncovered t-subset
                    for rest in itertools.combinations(block, t-1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = t_subset_index.get(ts, -1)
                        if idx >= 0 and uncovered[idx] > 0:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_point = x
                if best_point >= 0:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    break
            
            if len(block) == k:
                # Count gain of this block
                g = 0
                for ts in itertools.combinations(block, t):
                    idx = t_subset_index.get(ts, -1)
                    if idx >= 0 and uncovered[idx] > 0:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_block = sorted(block)
        
        if best_block is None or best_gain <= 0:
            # Fallback: just pick any uncovered t-subset and extend
            start_idx = uncovered_indices[0]
            start = all_t_subsets[start_idx]
            block = list(start)
            block_set = set(block)
            while len(block) < k:
                best_point = -1
                best_g = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    g = 0
                    for rest in itertools.combinations(block, t-1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = t_subset_index.get(ts, -1)
                        if idx >= 0 and uncovered[idx] > 0:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_point = x
                if best_point >= 0:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    break
            best_block = sorted(block)
        
        blocks.append(best_block)
        
        # Update coverage
        for ts in itertools.combinations(best_block, t):
            idx = t_subset_index.get(ts, -1)
            if idx >= 0 and uncovered[idx] > 0:
                uncovered[idx] += 1
                covered_count += 1
        
        if time.time() > deadline - 30:
            break
    
    # Local search: try to remove redundant blocks and swap to improve
    # A block is redundant if all its t-subsets are covered by other blocks
    def compute_coverage(blocks_list):
        """Compute coverage count for each t-subset."""
        cov = [0] * n_t
        for b in blocks_list:
            for ts in itertools.combinations(b, t):
                idx = t_subset_index.get(ts, -1)
                if idx >= 0:
                    cov[idx] += 1
        return cov
    
    def is_complete(cov):
        return all(c >= 1 for c in cov)
    
    # Local search with simulated annealing
    current_blocks = blocks[:]
    current_cov = compute_coverage(current_blocks)
    current_uncovered = sum(1 for c in current_cov if c == 0)
    best_blocks = current_blocks[:]
    best_uncovered = current_uncovered
    best_size = len(current_blocks)
    
    if best_uncovered == 0:
        best_size = len(current_blocks)
    
    T = 100.0
    cooling = 0.999
    no_improve = 0
    
    while time.time() < deadline - 5:
        # Try different moves
        move_type = random.random()
        
        if move_type < 0.4 and len(current_blocks) > 1:
            # Try removing a block
            i = random.randint(0, len(current_blocks) - 1)
            # Check if removing block i keeps coverage
            new_cov = current_cov[:]
            b = current_blocks[i]
            for ts in itertools.combinations(b, t):
                idx = t_subset_index.get(ts, -1)
                if idx >= 0:
                    new_cov[idx] -= 1
            new_uncovered = sum(1 for c in new_cov if c == 0)
            
            if new_uncovered <= current_uncovered:
                # Accept
                del current_blocks[i]
                current_cov = new_cov
                current_uncovered = new_uncovered
                if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and len(current_blocks) < best_size):
                    best_uncovered = current_uncovered
                    best_size = len(current_blocks)
                    best_blocks = current_blocks[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Simulated annealing acceptance
                delta = new_uncovered - current_uncovered
                if random.random() < math.exp(-delta / max(T, 1e-10)):
                    del current_blocks[i]
                    current_cov = new_cov
                    current_uncovered = new_uncovered
                    no_improve += 1
                else:
                    no_improve += 1
                    
        elif move_type < 0.7:
            # Try swapping a block with a new one
            if len(current_blocks) == 0:
                continue
            i = random.randint(0, len(current_blocks) - 1)
            
            # Generate a random new block biased toward uncovered areas
            # Find uncovered t-subsets
            uncov_indices = [j for j in range(n_t) if current_cov[j] == 0]
            if uncov_indices:
                start_idx = random.choice(uncov_indices)
                start = all_t_subsets[start_idx]
            else:
                # All covered, try to improve by finding a block that helps more
                start = tuple(random.sample(range(v), t))
            
            block = list(start)
            block_set = set(block)
            while len(block) < k:
                best_point = -1
                best_g = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    g = 0
                    for rest in itertools.combinations(block, t-1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = t_subset_index.get(ts, -1)
                        if idx >= 0 and current_cov[idx] == 0:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_point = x
                if best_point >= 0:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    break
            
            if len(block) == k:
                new_block = sorted(block)
                
                # Check if this is a duplicate
                if new_block in current_blocks:
                    no_improve += 1
                    continue
                
                # Try replacing block i with new_block
                new_cov = current_cov[:]
                # Remove old block coverage
                old_b = current_blocks[i]
                for ts in itertools.combinations(old_b, t):
                    idx = t_subset_index.get(ts, -1)
                    if idx >= 0:
                        new_cov[idx] -= 1
                # Add new block coverage
                for ts in itertools.combinations(new_block, t):
                    idx = t_subset_index.get(ts, -1)
                    if idx >= 0:
                        new_cov[idx] += 1
                
                new_uncovered = sum(1 for c in new_cov if c == 0)
                
                if new_uncovered < current_uncovered:
                    current_blocks[i] = new_block
                    current_cov = new_cov
                    current_uncovered = new_uncovered
                    if current_uncovered < best_uncovered or (current_uncovered == best_uncovered and len(current_blocks) < best_size):
                        best_uncovered = current_uncovered
                        best_size = len(current_blocks)
                        best_blocks = current_blocks[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                elif new_uncovered == current_uncovered:
                    # Accept with some probability to explore
                    if random.random() < 0.3:
                        current_blocks[i] = new_block
                        current_cov = new_cov
                        no_improve += 1
                    else:
                        no_improve += 1
                else:
                    delta = new_uncovered - current_uncovered
                    if random.random() < math.exp(-delta / max(T, 1e-10)):
                        current_blocks[i] = new_block
                        current_cov = new_cov
                        current_uncovered = new_uncovered
                        no_improve += 1
                    else:
                        no_improve += 1
            else:
                no_improve += 1
        else:
            # Try adding a block (if we have uncovered)
            if current_uncovered > 0:
                uncov_indices = [j for j in range(n_t) if current_cov[j] == 0]
                if uncov_indices:
                    start_idx = random.choice(uncov_indices)
                    start = all_t_subsets[start_idx]
                    block = list(start)
                    block_set = set(block)
                    while len(block) < k:
                        best_point = -1
                        best_g = -1
                        for x in range(v):
                            if x in block_set:
                                continue
                            g = 0
                            for rest in itertools.combinations(block, t-1):
                                ts = tuple(sorted(rest + (x,)))
                                idx = t_subset_index.get(ts, -1)
                                if idx >= 0 and current_cov[idx] == 0:
                                    g += 1
                            if g > best_g:
                                best_g = g
                                best_point = x
                        if best_point >= 0:
                            block.append(best_point)
                            block_set.add(best_point)
                        else:
                            break
                    if len(block) == k:
                        new_block = sorted(block)
                        if new_block not in current_blocks:
                            current_blocks.append(new_block)
                            # Update coverage
                            for ts in itertools.combinations(new_block, t):
                                idx = t_subset_index.get(ts, -1)
                                if idx >= 0:
                                    current_cov[idx] += 1
                            current_uncovered = sum(1 for c in current_cov if c == 0)
                            if current_uncovered < best_uncovered:
                                best_uncovered = current_uncovered
                                best_size = len(current_blocks)
                                best_blocks = current_blocks[:]
                                no_improve = 0
                            else:
                                no_improve += 1
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        T *= cooling
        if T < 0.01:
            T = 0.01
        
        # Restart if stuck
        if no_improve > 200:
            no_improve = 0
            T = max(T, 50.0)
            # Perturb: randomly swap a few blocks
            for _ in range(min(3, len(current_blocks))):
                if len(current_blocks) < 2:
                    break
                i = random.randint(0, len(current_blocks) - 1)
                j = random.randint(0, len(current_blocks) - 1)
                if i != j:
                    current_blocks[i], current_blocks[j] = current_blocks[j], current_blocks[i]
                    # Recompute coverage
                    current_cov = compute_coverage(current_blocks)
                    current_uncovered = sum(1 for c in current_cov if c == 0)
    
    # Use best found
    result = best_blocks if best_uncovered == 0 else current_blocks
    
    # Final verification and cleanup
    # Remove any duplicate blocks
    seen = set()
    unique_blocks = []
    for b in result:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    # Verify completeness
    final_cov = [0] * n_t
    for b in unique_blocks:
        for ts in itertools.combinations(b, t):
            idx = t_subset_index.get(ts, -1)
            if idx >= 0:
                final_cov[idx] += 1
    
    if all(c >= 1 for c in final_cov):
        return unique_blocks
    else:
        # Fallback: use greedy result which should be complete
        return blocks if all(c >= 1 for c in compute_coverage(blocks)) else unique_blocks
