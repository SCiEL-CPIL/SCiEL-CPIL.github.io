import itertools
import time
import random
import numpy as np
from collections import defaultdict

def _precompute_triple_indices(v, t):
    """
    Maps each t-subset to a unique integer index and precomputes the indices 
    for every k-subset.
    Returns:
        total_triples: total number of t-subsets
        block_triple_indices: dict mapping block tuple -> list of triple indices
    """
    # We need a mapping from t-subset to index.
    # Since v=24, t=3, total is 2024.
    # We can use a dictionary for the mapping.
    triple_to_idx = {}
    idx = 0
    for tri in itertools.combinations(range(v), t):
        triple_to_idx[tri] = idx
        idx += 1
    total_triples = idx
    
    # Precompute for all k-subsets? C(24,7) = 346104.
    # Storing a list of 35 ints for each is ~12MB. Acceptable.
    # But we might not need all. We only need it for blocks we generate.
    # However, for SA, we generate random blocks.
    # It's better to compute on the fly or cache.
    # Let's just return the mapper.
    return triple_to_idx, total_triples

def _greedy_initial_cover(v, k, t, rng, time_limit):
    """
    Fast greedy to get a starting point for SA.
    """
    start_time = time.time()
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() > time_limit:
            break
        # Pick random uncovered triple
        pivot = tuple(rng.choice(list(uncovered)))
        pivot_set = set(pivot)
        other_points = [p for p in range(v) if p not in pivot_set]
        
        best_gain = -1
        best_block = None
        
        # Sample candidates instead of exhaustive if time is tight?
        # For initial cover, exhaustive C(21,4)=5985 is fast enough for ~70 steps.
        for combo in itertools.combinations(other_points, k - t):
            block = tuple(sorted(pivot_set | set(combo)))
            gain = 0
            for tri in itertools.combinations(block, t):
                if tri in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block:
            blocks.append(list(best_block))
            for tri in itertools.combinations(best_block, t):
                uncovered.discard(tri)
            
    return blocks

def _reduce_covering(blocks, v, k, t):
    """
    Remove redundant blocks.
    """
    if not blocks:
        return blocks
    
    triple_count = defaultdict(int)
    block_triples = []
    for block in blocks:
        tris = list(itertools.combinations(block, t))
        block_triples.append(tris)
        for tri in tris:
            triple_count[tri] += 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_triples = []
        for i, block in enumerate(blocks):
            essential = False
            for tri in block_triples[i]:
                if triple_count[tri] == 1:
                    essential = True
                    break
            if essential:
                new_blocks.append(block)
                new_block_triples.append(block_triples[i])
            else:
                changed = True
                for tri in block_triples[i]:
                    triple_count[tri] -= 1
        blocks = new_blocks
        block_triples = new_block_triples
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # 1. Generate Initial Cover
    rng = random.Random(42)
    blocks = _greedy_initial_cover(v, k, t, rng, deadline - 10)
    blocks = _reduce_covering(blocks, v, k, t)
    
    best_blocks = blocks
    best_size = len(blocks)
    
    # 2. Simulated Annealing
    # State: list of blocks
    # Cost: number of blocks + penalty * (number of uncovered triples)
    # We want to minimize cost.
    
    # Precompute triple indices for vectorization
    triple_to_idx, total_triples = _precompute_triple_indices(v, t)
    
    # Helper to get triple indices for a block
    def get_block_indices(block):
        return [triple_to_idx[tri] for tri in itertools.combinations(block, t)]
    
    # Initialize SA state
    current_blocks = [list(b) for b in blocks]
    current_indices = [get_block_indices(b) for b in current_blocks]
    
    # Coverage count array
    coverage = np.zeros(total_triples, dtype=np.int32)
    for inds in current_indices:
        coverage[inds] += 1
        
    current_uncovered = np.sum(coverage == 0)
    current_cost = len(current_blocks) + 100 * current_uncovered
    
    best_cost = current_cost
    best_state_blocks = [list(b) for b in current_blocks]
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    rng_sa = random.Random(123)
    
    while T > T_min and time.time() < deadline - 5:
        # Perturbation
        move_type = rng_sa.choice(['swap', 'remove', 'add'])
        
        old_cost = current_cost
        old_uncovered = current_uncovered
        old_n_blocks = len(current_blocks)
        
        # Save state for revert
        revert_info = None
        
        if move_type == 'swap':
            if not current_blocks: continue
            i = rng_sa.randrange(len(current_blocks))
            old_block = current_blocks[i]
            old_inds = current_indices[i]
            
            new_block = sorted(rng_sa.sample(range(v), k))
            new_inds = get_block_indices(new_block)
            
            # Check validity of swap:
            # Uncovered triples must not increase.
            # Ideally, we want to reduce uncovered or keep same and maybe reduce blocks later.
            # For SA, we accept based on cost.
            
            # Calculate change in coverage
            # Decrement old, increment new
            coverage[old_inds] -= 1
            coverage[new_inds] += 1
            
            new_uncovered = np.sum(coverage == 0)
            new_cost = old_n_blocks + 100 * new_uncovered
            
            if new_cost < old_cost or rng_sa.expovariate(1.0) < np.exp(-(new_cost - old_cost) / T):
                # Accept
                current_blocks[i] = new_block
                current_indices[i] = new_inds
                current_cost = new_cost
                current_uncovered = new_uncovered
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state_blocks = [list(b) for b in current_blocks]
                    
            else:
                # Revert
                coverage[old_inds] += 1
                coverage[new_inds] -= 1
                
        elif move_type == 'remove':
            if len(current_blocks) <= 1: continue
            i = rng_sa.randrange(len(current_blocks))
            old_block = current_blocks[i]
            old_inds = current_indices[i]
            
            # Try to remove
            coverage[old_inds] -= 1
            new_uncovered = np.sum(coverage == 0)
            new_n_blocks = old_n_blocks - 1
            new_cost = new_n_blocks + 100 * new_uncovered
            
            if new_cost < old_cost or rng_sa.expovariate(1.0) < np.exp(-(new_cost - old_cost) / T):
                # Accept
                current_blocks.pop(i)
                current_indices.pop(i)
                current_cost = new_cost
                current_uncovered = new_uncurrent_uncovered if 'new_uncovered' in locals() else 0 # fix var
                current_uncovered = new_uncovered
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state_blocks = [list(b) for b in current_blocks]
            else:
                # Revert
                coverage[old_inds] += 1
                
        elif move_type == 'add':
            # Add a random block
            new_block = sorted(rng_sa.sample(range(v), k))
            new_inds = get_block_indices(new_block)
            
            # Check if this block is already present? 
            # For simplicity, we allow duplicates in SA state temporarily, 
            # but reduction step later will clean it.
            # Actually, duplicate blocks don't help coverage but increase cost.
            # So adding a duplicate is always bad unless it allows a swap later?
            # No, just bad.
            # Let's check if it exists.
            # Simple check:
            block_tuple = tuple(new_block)
            exists = False
            for b in current_blocks:
                if tuple(b) == block_tuple:
                    exists = True
                    break
            if exists:
                continue
                
            coverage[new_inds] += 1
            new_uncovered = np.sum(coverage == 0)
            new_n_blocks = old_n_blocks + 1
            new_cost = new_n_blocks + 100 * new_uncovered
            
            if new_cost < old_cost or rng_sa.expovariate(1.0) < np.exp(-(new_cost - old_cost) / T):
                # Accept
                current_blocks.append(new_block)
                current_indices.append(new_inds)
                current_cost = new_cost
                current_uncovered = new_uncovered
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state_blocks = [list(b) for b in current_blocks]
            else:
                # Revert
                coverage[new_inds] -= 1
                
        T *= cooling_rate
        
        # Periodic reduction to keep state clean and potentially find better local minima
        # Every 1000 steps or so? Or when we hit a new best?
        # Let's do it when we hit a new best.
        if best_state_blocks != current_blocks and len(best_state_blocks) < len(current_blocks):
             # Reset current to best to continue from there?
             # No, SA should continue. But we can clean up current state if it's bloated.
             pass
             
    # Final cleanup
    # The SA state might have duplicates or redundant blocks.
    # We need to return a valid covering.
    # If current_uncovered > 0, we must repair it.
    
    # Use best_state_blocks found during SA
    final_blocks = best_state_blocks
    
    # Verify coverage of final_blocks
    # If invalid, repair using greedy on missing triples
    covered_triples = set()
    for b in final_blocks:
        for tri in itertools.combinations(b, t):
            covered_triples.add(tri)
            
    all_triples = set(itertools.combinations(range(v), t))
    missing = all_triples - covered_triples
    
    if missing:
        # Repair
        blocks_to_add = _greedy_initial_cover_repair(v, k, t, list(missing), rng_sa, deadline)
        final_blocks.extend(blocks_to_add)
        
    # Final reduction
    final_blocks = _reduce_covering(final_blocks, v, k, t)
    
    return final_blocks

def _greedy_initial_cover_repair(v, k, t, missing_triples, rng, time_limit):
    """
    Greedy cover for specific missing triples.
    """
    uncovered = set(missing_triples)
    blocks = []
    start_time = time.time()
    
    while uncovered:
        if time.time() > time_limit:
            break
        pivot = tuple(rng.choice(list(uncovered)))
        pivot_set = set(pivot)
        other_points = [p for p in range(v) if p not in pivot_set]
        
        best_gain = -1
        best_block = None
        
        for combo in itertools.combinations(other_points, k - t):
            block = tuple(sorted(pivot_set | set(combo)))
            gain = 0
            for tri in itertools.combinations(block, t):
                if tri in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block:
            blocks.append(list(best_block))
            for tri in itertools.combinations(best_block, t):
                uncovered.discard(tri)
                
    return blocks

# Fix the variable scope issue in the 'remove' branch of construct
# I will rewrite the construct function to be cleaner and safer.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Generate Initial Cover
    rng = random.Random(42)
    blocks = _greedy_initial_cover(v, k, t, rng, deadline - 10)
    blocks = _reduce_covering(blocks, v, k, t)
    
    best_blocks = blocks
    best_size = len(blocks)
    
    # 2. Simulated Annealing
    triple_to_idx, total_triples = _precompute_triple_indices(v, t)
    
    def get_block_indices(block):
        return [triple_to_idx[tri] for tri in itertools.combinations(block, t)]
    
    current_blocks = [list(b) for b in blocks]
    current_indices = [get_block_indices(b) for b in current_blocks]
    
    coverage = np.zeros(total_triples, dtype=np.int32)
    for inds in current_indices:
        coverage[inds] += 1
        
    current_uncovered = int(np.sum(coverage == 0))
    current_cost = len(current_blocks) + 100 * current_uncovered
    
    best_cost = current_cost
    best_state_blocks = [list(b) for b in current_blocks]
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    rng_sa = random.Random(123)
    
    while T > T_min and time.time() < deadline - 5:
        move_type = rng_sa.choice(['swap', 'remove', 'add'])
        
        old_cost = current_cost
        old_uncovered = current_uncovered
        old_n_blocks = len(current_blocks)
        
        if move_type == 'swap':
            if not current_blocks: continue
            i = rng_sa.randrange(len(current_blocks))
            old_block = current_blocks[i]
            old_inds = current_indices[i]
            
            new_block = sorted(rng_sa.sample(range(v), k))
            new_inds = get_block_indices(new_block)
            
            coverage[old_inds] -= 1
            coverage[new_inds] += 1
            
            new_uncovered = int(np.sum(coverage == 0))
            new_cost = old_n_blocks + 100 * new_uncovered
            
            if new_cost < old_cost or rng_sa.expovariate(1.0) < np.exp(-(new_cost - old_cost) / T):
                current_blocks[i] = new_block
                current_indices[i] = new_inds
                current_cost = new_cost
                current_uncovered = new_uncovered
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state_blocks = [list(b) for b in current_blocks]
            else:
                coverage[old_inds] += 1
                coverage[new_inds] -= 1
                
        elif move_type == 'remove':
            if len(current_blocks) <= 1: continue
            i = rng_sa.randrange(len(current_blocks))
            old_block = current_blocks[i]
            old_inds = current_indices[i]
            
            coverage[old_inds] -= 1
            new_uncovered = int(np.sum(coverage == 0))
            new_n_blocks = old_n_blocks - 1
            new_cost = new_n_blocks + 100 * new_uncovered
            
            if new_cost < old_cost or rng_sa.expovariate(1.0) < np.exp(-(new_cost - old_cost) / T):
                current_blocks.pop(i)
                current_indices.pop(i)
                current_cost = new_cost
                current_uncovered = new_uncovered
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state_blocks = [list(b) for b in current_blocks]
            else:
                coverage[old_inds] += 1
                
        elif move_type == 'add':
            new_block = sorted(rng_sa.sample(range(v), k))
            block_tuple = tuple(new_block)
            
            # Skip if exists
            exists = any(tuple(b) == block_tuple for b in current_blocks)
            if exists:
                continue
                
            new_inds = get_block_indices(new_block)
            
            coverage[new_inds] += 1
            new_uncovered = int(np.sum(coverage == 0))
            new_n_blocks = old_n_blocks + 1
            new_cost = new_n_blocks + 100 * new_uncovered
            
            if new_cost < old_cost or rng_sa.expovariate(1.0) < np.exp(-(new_cost - old_cost) / T):
                current_blocks.append(new_block)
                current_indices.append(new_inds)
                current_cost = new_cost
                current_uncovered = new_uncovered
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state_blocks = [list(b) for b in current_blocks]
            else:
                coverage[new_inds] -= 1
                
        T *= cooling_rate
        
    # Final cleanup
    final_blocks = best_state_blocks
    
    # Verify and repair
    covered_triples = set()
    for b in final_blocks:
        for tri in itertools.combinations(b, t):
            covered_triples.add(tri)
            
    all_triples = set(itertools.combinations(range(v), t))
    missing = all_triples - covered_triples
    
    if missing:
        blocks_to_add = _greedy_initial_cover_repair(v, k, t, list(missing), rng_sa, deadline)
        final_blocks.extend(blocks_to_add)
        
    final_blocks = _reduce_covering(final_blocks, v, k, t)
    
    return final_blocks
