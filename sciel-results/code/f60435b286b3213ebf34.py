import itertools
import random
import time
import math

def construct(v, k, t):
    # Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # block_triples[i] = list of triple indices covered by block i
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(ts)
        
    # triple_to_blocks[i] = list of block indices covering triple i
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_to_blocks[t_idx].append(i)

    # Local Search function
    def local_search(K, max_iters=100000, timeout=10):
        start = time.time()
        # Initialize random set of K blocks
        selected_indices = set(random.sample(range(num_blocks), K))
        coverage_count = [0] * num_triples
        
        # Calculate initial coverage
        for idx in selected_indices:
            for t_idx in block_triples[idx]:
                coverage_count[t_idx] += 1
                
        uncovered_count = 0
        for c in coverage_count:
            if c == 0:
                uncovered_count += 1
                
        if uncovered_count == 0:
            return list(selected_indices)

        current_temp = 1.0
        cooling_rate = 0.99
        min_temp = 0.01
        
        best_uncovered = uncovered_count
        best_set = set(selected_indices)
        
        for i in range(max_iters):
            if time.time() - start > timeout:
                break
                
            # Pick a random block to remove
            remove_idx = random.choice(list(selected_indices))
            
            # Pick a random block to add (not currently selected)
            # Efficient way: pick random index, if in selected, retry or pick another
            # For small K vs num_blocks, random sampling is okay
            add_idx = random.randint(0, num_blocks - 1)
            if add_idx in selected_indices:
                # Find a valid one quickly
                while add_idx in selected_indices:
                    add_idx = random.randint(0, num_blocks - 1)
            
            # Calculate Delta
            # Remove remove_idx
            delta = 0
            affected_triples = set()
            
            # To avoid double counting if triples are shared, we just simulate
            # But simulating full state is expensive.
            # Let's calculate delta based on counts.
            
            # Triples in remove_idx
            for t_idx in block_triples[remove_idx]:
                if coverage_count[t_idx] == 1:
                    delta += 1 # Becomes uncovered
                affected_triples.add(t_idx)
                
            # Triples in add_idx
            for t_idx in block_triples[add_idx]:
                if t_idx in affected_triples:
                    # This triple is in both. 
                    # Remove: count goes from C to C-1
                    # Add: count goes from C-1 to C
                    # Net change: 0
                    # But we need to be careful if C was 1.
                    # If C==1: Remove -> 0 (uncovered +1). Add -> 1 (covered -1). Net 0.
                    # If C>1: Remove -> C-1 (no change in unc). Add -> C (no change). Net 0.
                    # So shared triples have net delta 0?
                    # Wait.
                    # State: count[t] = C.
                    # Remove: count[t] becomes C-1.
                    # Add: count[t] becomes C.
                    # Final count is C.
                    # Initial count was C.
                    # So yes, shared triples have 0 delta on uncovered count.
                    pass
                else:
                    if coverage_count[t_idx] == 0:
                        delta -= 1 # Becomes covered
                    affected_triples.add(t_idx)

            # Acceptance
            if delta < 0 or (delta == 0 and random.random() < 0.1) or (delta > 0 and random.random() < math.exp(-delta / current_temp)):
                # Apply move
                selected_indices.remove(remove_idx)
                selected_indices.add(add_idx)
                
                for t_idx in block_triples[remove_idx]:
                    coverage_count[t_idx] -= 1
                for t_idx in block_triples[add_idx]:
                    coverage_count[t_idx] += 1
                    
                uncovered_count += delta
                
                if uncovered_count < best_uncovered:
                    best_uncovered = uncovered_count
                    best_set = set(selected_indices)
                    
                if uncovered_count == 0:
                    return list(selected_indices)
                    
            current_temp *= cooling_rate
            if current_temp < min_temp:
                current_temp = min_temp

        return None

    # Search for minimal K
    # Lower bound is 69. Best known 76.
    # We try from 69 upwards.
    start_time = time.time()
    max_time = 110
    
    for K in range(69, 80):
        if time.time() - start_time > max_time - 5:
            break
            
        # Try a few restarts for this K
        found = None
        restarts = 5 if K < 72 else 2
        for r in range(restarts):
            res = local_search(K, max_iters=50000, timeout=5)
            if res is not None:
                found = res
                break
                
        if found:
            return [list(all_blocks[i]) for i in found]
            
    # Fallback: Return a valid solution found by simple greedy if local search failed
    # Or just return the best we can get with a larger K
    # Simple Greedy Fallback
    uncovered = set(range(num_triples))
    gains = [len(ts) for ts in block_triples]
    blocks_used = []
    while uncovered:
        max_gain = -1
        candidates = []
        for i in range(num_blocks):
            if gains[i] > max_gain:
                max_gain = gains[i]
                candidates = [i]
            elif gains[i] == max_gain:
                candidates.append(i)
        if max_gain == 0: break
        b = random.choice(candidates)
        blocks_used.append(b)
        for t_idx in block_triples[b]:
            if t_idx in uncovered:
                uncovered.remove(t_idx)
                for b_prime in triple_to_blocks[t_idx]:
                    gains[b_prime] -= 1
    return [list(all_blocks[i]) for i in blocks_used]
