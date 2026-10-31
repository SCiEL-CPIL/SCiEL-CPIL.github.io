import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # For each possible block (k-subset), precompute which t-subsets it covers
    # This is expensive for large v,k but for v=19,k=7,t=4: C(19,7)=50388 blocks, each covers C(7,4)=35 subsets
    # We'll compute coverage on the fly using bitsets
    
    def block_to_tsubs(block):
        """Return list of t-subset indices covered by this block."""
        return [sub_to_idx[sub] for sub in itertools.combinations(block, t)]
    
    def greedy_cover(seed=42):
        """Fast greedy using numpy for coverage tracking."""
        rng = np.random.default_rng(seed)
        covered = np.zeros(num_t_subs, dtype=np.int8)
        blocks = []
        
        # Precompute for each point, the t-subsets containing it (as indices)
        # point_tsubs[x] = list of t-subset indices that contain point x
        point_tsubs = [[] for _ in range(v)]
        for idx, sub in enumerate(all_t_subs):
            for x in sub:
                point_tsubs[x].append(idx)
        
        while True:
            # Check if all covered
            if covered.min() > 0:
                break
            
            # Find uncovered subsets
            uncovered_mask = covered == 0
            if not uncovered_mask.any():
                break
            
            # Pick a random uncovered subset to start
            uncovered_indices = np.where(uncovered_mask)[0]
            start_idx = rng.choice(uncovered_indices)
            start_sub = all_t_subs[start_idx]
            block = list(start_sub)
            
            # Greedily add points to maximize coverage of uncovered subsets
            while len(block) < k:
                best_point = None
                best_gain = -1
                
                # For each candidate point, count how many uncovered t-subsets it would cover
                # A t-subset is covered by adding point x to block if:
                # - x is in the t-subset
                # - the other t-1 elements are in block
                # - the t-subset is currently uncovered
                
                # Precompute: for current block, which (t-1)-subsets are in it
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # Check each t-subset containing x
                    for tsub_idx in point_tsubs[x]:
                        if covered[tsub_idx] > 0:
                            continue
                        tsub = all_t_subs[tsub_idx]
                        # Check if all other elements (t-1 of them) are in block
                        others = [y for y in tsub if y != x]
                        if all(y in block_set for y in others):
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None or best_gain == 0:
                    # Pick random point
                    candidates = [x for x in range(v) if x not in block_set]
                    if not candidates:
                        break
                    best_point = rng.choice(candidates)
                
                block.append(best_point)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update coverage
            for tsub_idx in block_to_tsubs(block):
                covered[tsub_idx] += 1
        
        return blocks
    
    def optimize(blocks, time_limit=105):
        """Simulated annealing with block deletion and replacement."""
        rng = np.random.default_rng()
        current_blocks = [tuple(b) for b in blocks]
        
        # Build coverage count
        covered = np.zeros(num_t_subs, dtype=np.int32)
        for b in current_blocks:
            for tsub_idx in block_to_tsubs(b):
                covered[tsub_idx] += 1
        
        def is_redundant(b, cov):
            for tsub_idx in block_to_tsubs(b):
                if cov[tsub_idx] <= 1:
                    return False
            return True
        
        start_time = time.time()
        temp = 1.0
        cooling = 0.999
        no_improve = 0
        best_len = len(current_blocks)
        
        while time.time() - start_time < time_limit:
            if temp < 0.01:
                temp = 1.0
                no_improve = 0
            
            improved = False
            
            # Try to delete a random block
            if len(current_blocks) > 1:
                idx = rng.integers(0, len(current_blocks))
                b = current_blocks[idx]
                if is_redundant(b, covered):
                    for tsub_idx in block_to_tsubs(b):
                        covered[tsub_idx] -= 1
                    current_blocks.pop(idx)
                    improved = True
                    no_improve = 0
                    continue
            
            # Try to replace a random block with a better one
            if len(current_blocks) > 0:
                idx = rng.integers(0, len(current_blocks))
                old_block = current_blocks[idx]
                
                # Generate a new random block
                new_block = tuple(sorted(rng.choice(v, size=k, replace=False)))
                
                # Check validity: all t-subsets of old_block with count==1 must be in new_block
                valid = True
                for tsub_idx in block_to_tsubs(old_block):
                    if covered[tsub_idx] == 1:
                        # Check if this t-subset is in new_block
                        tsub = all_t_subs[tsub_idx]
                        new_block_set = set(new_block)
                        if not all(x in new_block_set for x in tsub):
                            valid = False
                            break
                
                if valid:
                    # Compute delta: how many new t-subsets does new_block cover that aren't already covered?
                    # and how many t-subsets of old_block become uncovered?
                    old_tsubs = block_to_tsubs(old_block)
                    new_tsubs = block_to_tsubs(new_block)
                    
                    # Would any t-subset become uncovered?
                    would_uncover = 0
                    for tsub_idx in old_tsubs:
                        if covered[tsub_idx] == 1 and tsub_idx not in new_tsubs:
                            would_uncover += 1
                    
                    if would_uncover == 0:
                        # Safe replacement
                        for tsub_idx in old_tsubs:
                            covered[tsub_idx] -= 1
                        for tsub_idx in new_tsubs:
                            covered[tsub_idx] += 1
                        current_blocks[idx] = new_block
                        improved = True
                        no_improve = 0
                        continue
            
            # Try to find a block that can be deleted after a small perturbation
            # Pick a random block, try to modify it to make it redundant
            if len(current_blocks) > 2:
                idx = rng.integers(0, len(current_blocks))
                b = current_blocks[idx]
                b_list = list(b)
                
                # Try removing one point and adding another
                if k > 1:
                    remove_idx = rng.integers(0, k)
                    x_remove = b_list[remove_idx]
                    b_list.pop(remove_idx)
                    
                    # Find best point to add
                    best_point = None
                    best_gain = -1
                    b_set = set(b_list)
                    
                    for x in range(v):
                        if x in b_set:
                            continue
                        gain = 0
                        # How many t-subsets of (b_list + x) are uncovered by other blocks?
                        for tsub in itertools.combinations(b_list + [x], t):
                            tsub_idx = sub_to_idx[tsub]
                            # Count coverage excluding this block
                            if covered[tsub_idx] == 1:
                                # Only this block covers it, so if we remove x, it might be uncovered
                                # But we're adding x, so we need to check if this t-subset is in old_block
                                pass
                        # Simpler: just check if this helps make the block redundant
                        # A block is redundant if all its t-subsets have count > 1
                        # So we want to add a point such that the new block's t-subsets
                        # are all covered by other blocks
                        new_block_tuple = tuple(sorted(b_list + [x]))
                        new_tsubs = block_to_tsubs(new_block_tuple)
                        all_covered_by_others = True
                        for tsub_idx in new_tsubs:
                            # This t-subset would be covered by the new block
                            # But we're replacing old_block, so we need to check
                            # if it's covered by other blocks (excluding old_block)
                            # Since we haven't removed old_block yet, covered includes it
                            # We need: covered[tsub_idx] - 1 >= 1, i.e., covered[tsub_idx] >= 2
                            # OR the t-subset is not in old_block
                            old_tsubs_set = set(block_to_tsubs(b))
                            if tsub_idx in old_tsubs_set:
                                if covered[tsub_idx] < 2:
                                    all_covered_by_others = False
                                    break
                            else:
                                if covered[tsub_idx] < 1:
                                    all_covered_by_others = False
                                    break
                        
                        if all_covered_by_others:
                            # This replacement would allow deletion
                            # Actually, we're replacing, not deleting
                            # Let's just do the replacement
                            best_gain = 100
                            best_point = x
                            break
                        
                        # Fallback: just count how many new t-subsets are covered
                        for tsub in itertools.combinations(b_list + [x], t):
                            tsub_idx = sub_to_idx[tsub]
                            if covered[tsub_idx] == 0:
                                gain += 1
                    
                    if best_point is not None:
                        b_list.append(best_point)
                        b_list.sort()
                        new_block = tuple(b_list)
                        
                        # Validate
                        old_tsubs = block_to_tsubs(b)
                        new_tsubs = block_to_tsubs(new_block)
                        valid = True
                        for tsub_idx in old_tsubs:
                            if covered[tsub_idx] == 1 and tsub_idx not in new_tsubs:
                                valid = False
                                break
                        
                        if valid:
                            for tsub_idx in old_tsubs:
                                covered[tsub_idx] -= 1
                            for tsub_idx in new_tsubs:
                                covered[tsub_idx] += 1
                            current_blocks[idx] = new_block
                            improved = True
                            no_improve = 0
                            continue
            
            if not improved:
                no_improve += 1
                temp *= cooling
            
            # Check time
            if time.time() - start_time > time_limit - 2:
                break
        
        return [list(b) for b in current_blocks]
    
    def final_cleanup(blocks):
        """Remove all redundant blocks."""
        current_blocks = [tuple(b) for b in blocks]
        covered = np.zeros(num_t_subs, dtype=np.int32)
        for b in current_blocks:
            for tsub_idx in block_to_tsubs(b):
                covered[tsub_idx] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in current_blocks:
                if is_redundant(b, covered):
                    for tsub_idx in block_to_tsubs(b):
                        covered[tsub_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            current_blocks = new_blocks
        
        return [list(b) for b in current_blocks]
    
    # Main execution
    best_blocks = None
    best_len = float('inf')
    
    # Run greedy with different seeds
    for seed in range(30):
        blocks = greedy_cover(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    # Optimize the best initial solution
    if best_blocks is not None:
        optimized = optimize(best_blocks, time_limit=105)
        final = final_cleanup(optimized)
        return final
    else:
        return best_blocks
