"""Different strategy: simulated annealing on block-replacement moves, seeded by a fast greedy covering."""
import time
import random
import itertools
import math

def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin below 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Helper: count how many times each t-subset is covered
    def build_cover_count(blocks):
        cc = {}
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cc[sub] = cc.get(sub, 0) + 1
        return cc
    
    # Helper: check if a block is fully covered (all its t-subsets have count > 1)
    def is_redundant(block, cc):
        for sub in itertools.combinations(block, t):
            if cc.get(sub, 0) <= 1:
                return False
        return True
    
    # Helper: remove redundant blocks iteratively
    def remove_redundant(blocks):
        cc = build_cover_count(blocks)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                if time.time() >= deadline:
                    break
                if is_redundant(blocks[i], cc):
                    for sub in itertools.combinations(blocks[i], t):
                        cc[sub] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks
    
    # Helper: check full coverage
    def is_valid(blocks):
        cc = build_cover_count(blocks)
        for sub in all_t_subsets:
            if cc.get(sub, 0) == 0:
                return False
        return True
    
    # Fast greedy to get initial covering
    def greedy_initial(seed):
        rng = random.Random(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered and time.time() < deadline:
            # Pick the uncovered subset with highest "frequency" in random samples
            # For speed, just pick a random uncovered one
            start_sub = rng.choice(list(uncovered))
            block = list(start_sub)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                # Use a small random sample of candidates for speed
                candidates = [x for x in range(v) if x not in block_set]
                if len(candidates) > 10:
                    candidates = rng.sample(candidates, 10)
                
                block_t1 = list(itertools.combinations(block, t - 1))
                for x in candidates:
                    count = 0
                    for sub in block_t1:
                        new_t = tuple(sorted(sub + (x,)))
                        if new_t in uncovered:
                            count += 1
                    if count > best_gain:
                        best_gain = count
                        best_point = x
                
                if best_point == -1:
                    best_point = rng.choice([x for x in range(v) if x not in block_set])
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        return remove_redundant(blocks)
    
    # Generate initial covering
    blocks = greedy_initial(42)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Simulated annealing: try to replace a block with a different one that covers at least one uncovered t-subset
    # Actually, since we have a valid covering, we try to remove blocks by finding replacements
    # Strategy: pick a block, try to replace it with a new block that covers all t-subsets that were uniquely covered by the old block
    
    # Precompute for each block which t-subsets it uniquely covers
    def get_unique_coverage(blocks, cc):
        unique = {}
        for i, block in enumerate(blocks):
            u = []
            for sub in itertools.combinations(block, t):
                if cc.get(sub, 0) == 1:
                    u.append(sub)
            unique[i] = u
        return unique
    
    # Try to find a new block that covers a given set of t-subsets
    def find_block_for_subsets(subsets, seed):
        rng = random.Random(seed)
        if not subsets:
            return None
        
        # The new block must contain all elements that appear in the subsets
        # Actually, each subset must be contained in the new block
        # So the new block must be a superset of each subset
        # Find the union of all elements in the subsets
        required = set()
        for sub in subsets:
            required.update(sub)
        
        if len(required) > k:
            return None
        
        # We need to add (k - len(required)) more points
        available = [x for x in range(v) if x not in required]
        
        # Try random completions
        for _ in range(50):
            if not available:
                break
            extra = rng.sample(available, k - len(required))
            block = sorted(required | set(extra))
            # Verify all subsets are covered
            block_set = set(block)
            valid = True
            for sub in subsets:
                if not all(x in block_set for x in sub):
                    valid = False
                    break
            if valid:
                return block
        return None
    
    # Main SA loop
    cc = build_cover_count(blocks)
    current_count = len(blocks)
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() < deadline:
        # Check if we can remove any redundant blocks
        blocks = remove_redundant(blocks)
        cc = build_cover_count(blocks)
        
        if len(blocks) < best_count:
            best_blocks = list(blocks)
            best_count = len(blocks)
        
        if len(blocks) == 0:
            break
        
        # Pick a random block to try to replace
        i = random.randint(0, len(blocks) - 1)
        old_block = blocks[i]
        
        # Find t-subsets uniquely covered by this block
        unique_subs = []
        for sub in itertools.combinations(old_block, t):
            if cc.get(sub, 0) == 1:
                unique_subs.append(sub)
        
        if not unique_subs:
            # This block is redundant, remove it
            for sub in itertools.combinations(old_block, t):
                cc[sub] -= 1
            blocks.pop(i)
            continue
        
        # Try to find a new block that covers all unique_subs
        new_block = find_block_for_subsets(unique_subs, random.randint(0, 10**6))
        
        if new_block is not None:
            # Check if new_block is the same as old_block
            if set(new_block) == set(old_block):
                continue
            
            # Compute the change in coverage
            # Remove old block's contributions
            for sub in itertools.combinations(old_block, t):
                cc[sub] -= 1
            
            # Add new block's contributions
            for sub in itertools.combinations(new_block, t):
                cc[sub] = cc.get(sub, 0) + 1
            
            # Check if all unique_subs are still covered
            valid = True
            for sub in unique_subs:
                if cc.get(sub, 0) == 0:
                    valid = False
                    break
            
            if valid:
                # Accept the move
                blocks[i] = new_block
            else:
                # Reject: undo
                for sub in itertools.combinations(new_block, t):
                    cc[sub] -= 1
                for sub in itertools.combinations(old_block, t):
                    cc[sub] = cc.get(sub, 0) + 1
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final cleanup
    blocks = remove_redundant(blocks)
    if len(blocks) < best_count:
        best_blocks = list(blocks)
        best_count = len(blocks)
    
    # Verify validity
    if not is_valid(best_blocks):
        # Fallback: regenerate
        best_blocks = greedy_initial(123)
    
    return best_blocks
