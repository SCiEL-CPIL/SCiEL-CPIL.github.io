import itertools
import random
import time

# Cache for precomputed data to avoid regeneration if called multiple times
# (Though construct is usually called once per evaluation)
_cache = {}

def _get_data(v, k, t):
    if (v, k, t) in _cache:
        return _cache[(v, k, t)]
    
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Map each block to its t-subsets
    block_subsets = []
    for b in all_blocks:
        subs = list(itertools.combinations(b, t))
        block_subsets.append(subs)
        
    # Map each t-subset to list of block indices containing it
    subset_to_blocks = {}
    for i, subs in enumerate(block_subsets):
        for s in subs:
            if s not in subset_to_blocks:
                subset_to_blocks[s] = []
            subset_to_blocks[s].append(i)
            
    all_t_subsets = list(subset_to_blocks.keys())
    
    data = {
        'blocks': all_blocks,
        'block_subsets': block_subsets,
        'subset_to_blocks': subset_to_blocks,
        'all_t_subsets': all_t_subsets
    }
    _cache[(v, k, t)] = data
    return data

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    data = _get_data(v, k, t)
    blocks = data['blocks']
    block_subsets = data['block_subsets']
    subset_to_blocks = data['subset_to_blocks']
    all_t_subsets = data['all_t_subsets']
    
    # If v, k, t are small enough, we could do exact, but 19,7,4 is large.
    # Greedy with restarts.
    
    best_solution = None
    best_len = float('inf')
    
    # We will run restarts until time runs out
    # Each restart is a greedy pass
    
    while time.time() < deadline:
        uncovered = set(all_t_subsets)
        selected_indices = []
        
        while uncovered:
            # Pick a random uncovered t-subset to focus on
            # Using random.choice on a set is O(1) in Python 3.7+? No, it's O(n) to pick?
            # Actually, random.choice requires a sequence. 
            # We can pick a random key from the dict? 
            # Better: maintain a list of uncovered? 
            # Removing from a set is O(1). 
            # To pick a random element from a set, we can do:
            #   random.choice(list(uncovered)) -> O(n)
            # This might be slow if uncovered is large.
            # Alternative: Pick a random index from the original list and check if it's uncovered?
            # Or just pick the first one? 
            # Let's try picking a random one from the list of all subsets and checking if it's uncovered.
            # But if most are covered, this is inefficient.
            
            # Optimization: Keep a list of uncovered subsets?
            # Removing from a list is O(n).
            # Let's stick to Set and use a trick:
            # We can iterate through a random permutation of all subsets? No.
            
            # Let's use a simpler approach for selection:
            # Pick a random element from the set.
            # In CPython, iterating a set is fast.
            # We can do:
            #   s = next(iter(uncovered)) # O(1) but always same order?
            #   Actually, hash randomization makes order random-ish.
            #   But to be safe and truly random:
            
            # Let's just pick a random subset from the full list and check if it's in uncovered.
            # If not, retry. Since we start with all uncovered, it's fast.
            # As coverage increases, this gets slower.
            
            # Better heuristic: Pick the subset with the fewest available blocks?
            # That's "minimum degree" heuristic.
            # Let's try: Pick S from uncovered that minimizes len(subset_to_blocks[S] intersect available)?
            # All blocks are available. So len(subset_to_blocks[S]) is constant for a given S.
            # So just pick any S.
            
            # To avoid O(n) list(uncovered), let's maintain a list of uncovered indices?
            # No, let's just use random.choice on a list of keys of the subset_to_blocks dict?
            # No, that includes covered ones.
            
            # Let's try this: 
            # random_s = random.choice(all_t_subsets)
            # while random_s not in uncovered: random_s = random.choice(all_t_subsets)
            # This is expected O(1) when sparse, but bad when dense? No, dense means many uncovered, so high chance of hit.
            # Sparse means few uncovered, low chance of hit.
            
            # Let's use a different strategy:
            # Pick a random block from the blocks containing a random uncovered subset.
            # To get a random uncovered subset, we can sample from the set.
            # Python sets support O(1) deletion but not O(1) random access.
            
            # Workaround: Keep a list `uncovered_list` and a set `uncovered_set`.
            # When removing, remove from set. 
            # To pick random, pick random index from uncovered_list, check if in set.
            # This is efficient.
            
            # Re-initialize for this restart
            uncovered_set = set(all_t_subsets)
            uncovered_list = list(all_t_subsets)
            selected_indices = []
            
            while uncovered_set:
                # Pick a random uncovered subset
                # Try a few times to find one in the list
                s = None
                for _ in range(10):
                    idx = random.randint(0, len(uncovered_list) - 1)
                    candidate = uncovered_list[idx]
                    if candidate in uncovered_set:
                        s = candidate
                        break
                if s is None:
                    # Fallback: find first one
                    for c in uncovered_list:
                        if c in uncovered_set:
                            s = c
                            break
                
                # Get candidates for this subset
                candidates = subset_to_blocks[s]
                
                # Find best candidate block
                best_block_idx = -1
                max_gain = -1
                
                for b_idx in candidates:
                    # Count how many subsets of this block are uncovered
                    gain = 0
                    for sub in block_subsets[b_idx]:
                        if sub in uncovered_set:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_block_idx = b_idx
                        
                # Add best block
                selected_indices.append(best_block_idx)
                
                # Update uncovered
                for sub in block_subsets[best_block_idx]:
                    if sub in uncovered_set:
                        uncovered_set.remove(sub)
                        # We don't remove from list to keep it simple, 
                        # the membership check handles it.
                        
        # End while uncovered
        
        # End restart
        current_len = len(selected_indices)
        if current_len < best_len:
            best_len = current_len
            best_solution = [blocks[i] for i in selected_indices]
            
        # Check time
        if time.time() > deadline:
            break
            
    if best_solution is None:
        # Fallback: return a trivial valid cover if we failed? 
        # Or just the last one?
        # If we didn't finish even one restart, we need a valid cover.
        # A greedy without restarts is safer.
        # But the loop above runs at least once if time permits.
        # If time was 0, we might have best_solution None.
        # Let's do a single deterministic greedy if best_solution is None.
        uncovered_set = set(all_t_subsets)
        selected_indices = []
        while uncovered_set:
            # Pick first uncovered
            s = next(iter(uncovered_set))
            candidates = subset_to_blocks[s]
            best_block_idx = -1
            max_gain = -1
            for b_idx in candidates:
                gain = 0
                for sub in block_subsets[b_idx]:
                    if sub in uncovered_set:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = b_idx
            selected_indices.append(best_block_idx)
            for sub in block_subsets[best_block_idx]:
                if sub in uncovered_set:
                    uncovered_set.remove(sub)
        best_solution = [blocks[i] for i in selected_indices]
        
    return best_solution

