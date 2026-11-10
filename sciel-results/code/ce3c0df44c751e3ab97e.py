import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    
    # Precompute auxiliary sets for gain calculation
    # aux[x] contains all (t-1)-subsets U such that U U {x} is a t-subset
    # We will update these as subsets get covered.
    aux = [set() for _ in range(v)]
    for ts in all_t_subsets:
        for i in range(t):
            x = ts[i]
            u = ts[:i] + ts[i+1:]
            aux[x].add(u)
            
    def greedy_run(seed):
        random.seed(seed)
        unc = set(all_t_subsets)
        # Copy aux sets
        aux_c = [set(s) for s in aux]
        blocks = []
        
        while unc:
            # Pick a random anchor from uncovered to diversify
            anchor = random.choice(list(unc))
            block = list(anchor)
            
            # Extend block to size k
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                
                # Generate all (t-1)-subsets of current block
                # If t-1 > len(block), gain is 0 for all
                if len(block) < t - 1:
                    # No (t-1) subsets can be formed, so no gain from adding x 
                    # unless we consider that x completes a subset with elements NOT in block?
                    # Wait. Gain(x) = count of U in aux[x] such that U subset of block.
                    # If len(block) < t-1, there are no (t-1) subsets in block.
                    # So gain is 0 for all x.
                    # We should just pick a random x or the one with most potential?
                    # Actually, if len(block) < t-1, we can't cover any NEW t-subsets 
                    # by adding x, because a t-subset needs t-1 other elements.
                    # So we just pick a random x to break symmetry? Or min index?
                    # Let's pick random to allow different paths.
                    candidates = [x for x in range(v) if x not in block_set]
                    if candidates:
                        x = random.choice(candidates)
                    else:
                        break
                    block.append(x)
                    block_set.add(x)
                    continue
                
                # Generate (t-1) subsets of block
                block_t_minus_1 = set(itertools.combinations(block, t-1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many U in aux_c[x] are in block_t_minus_1
                    # Intersection size
                    gain = len(aux_c[x] & block_t_minus_1)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    # Tie breaking: random
                    elif gain == best_gain and best_x != -1:
                        if random.random() < 0.5:
                            best_x = x
                            
                if best_x == -1:
                    # Should not happen if v > k
                    break
                block.append(best_x)
                block_set.add(best_x)
                
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered and aux
            # Find all t-subsets of the new block
            for ts in itertools.combinations(block, t):
                if ts in unc:
                    unc.remove(ts)
                    for i in range(t):
                        x = ts[i]
                        u = ts[:i] + ts[i+1:]
                        if u in aux_c[x]:
                            aux_c[x].remove(u)
                            
        return blocks

    def remove_redundant(blocks):
        # Convert blocks to sets for faster lookup
        block_sets = [set(b) for b in blocks]
        
        # Count coverage for each t-subset
        # Map t-subset to list of block indices covering it
        cover_map = {ts: [] for ts in all_t_subsets}
        for i, b in enumerate(block_sets):
            for ts in itertools.combinations(b, t):
                cover_map[ts].append(i)
                
        # Try to remove blocks
        # If a block has any t-subset with count 1, it's essential.
        # If all t-subsets in a block have count > 1, it might be redundant.
        # We iterate and try to remove.
        
        essential = [True] * len(blocks)
        
        # First pass: mark blocks that are the *only* cover for some subset
        for ts, covers in cover_map.items():
            if len(covers) == 1:
                essential[covers[0]] = True
            else:
                # If a block is the only one for ANY subset, it's essential.
                # We need to check if block i is the only one for ts.
                pass
        
        # Better approach:
        # A block is redundant if removing it doesn't uncover any t-subset.
        # i.e., for all ts in block, len(cover_map[ts]) > 1.
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not essential[i]:
                    continue
                # Check if block i is essential
                b = blocks[i]
                is_essential = False
                for ts in itertools.combinations(b, t):
                    if len(cover_map[ts]) == 1:
                        is_essential = True
                        break
                if not is_essential:
                    essential[i] = False
                    changed = True
                    # Remove block i from cover_map
                    for ts in itertools.combinations(b, t):
                        cover_map[ts].remove(i)
                        
        return [blocks[i] for i in range(len(blocks)) if essential[i]]

    best_blocks = None
    seed = 42
    
    while time.time() < deadline:
        blocks = greedy_run(seed)
        # Post-process
        cleaned = remove_redundant(blocks)
        
        if best_blocks is None or len(cleaned) < len(best_blocks):
            best_blocks = cleaned
            # If we get a very good result, we might want to keep going
            # to see if we can do better, or break if we hit a known bound?
            # Best known is 151.
            if len(cleaned) <= 151:
                # We found a very good one. Try a few more to see if we can beat it?
                # Or just return?
                # Let's try a few more restarts just in case.
                pass
        seed += 1
        
    if best_blocks is None:
        # Fallback: simple greedy
        blocks = greedy_run(0)
        best_blocks = remove_redundant(blocks)
        
    return [list(b) for b in best_blocks]
