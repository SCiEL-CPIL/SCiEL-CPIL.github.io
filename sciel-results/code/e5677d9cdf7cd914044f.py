import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    # Precompute triple indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_id = {triple: i for i, triple in enumerate(all_triples)}
    id_to_triple = {i: triple for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute triple ID lookup table: idx_table[u, w, x] is ID of {u, w, x}
    idx_table = np.full((v, v, v), -1, dtype=np.int32)
    for a in range(v):
        for b in range(v):
            if a == b: continue
            for c in range(v):
                if a == c or b == c: continue
                p, q, r = sorted((a, b, c))
                idx_table[a, b, c] = triple_to_id[(p, q, r)]
                
    # Precompute pair-triple vectors for fast gain calculation
    # pair_triples[u, w] is an array of length v where element x is idx_table[u, w, x]
    pair_triples = np.full((v, v, v), -1, dtype=np.int32)
    for u in range(v):
        for w in range(v):
            if u == w: continue
            pair_triples[u, w] = idx_table[u, w, :]

    def run_greedy(seed_val):
        random.seed(seed_val)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        blocks = []
        
        while True:
            uncovered_ids = np.where(cover_count == 0)[0]
            if len(uncovered_ids) == 0:
                break
            
            # Pick a random uncovered triple to start the block
            start_tid = random.choice(uncovered_ids)
            a, b, c = id_to_triple[start_tid]
            block = [a, b, c]
            
            # Add k - t more points
            while len(block) < k:
                pairs = list(itertools.combinations(block, 2))
                
                if len(pairs) == 0:
                    break
                    
                # Build a matrix of triple IDs for all candidates x
                # Shape: (num_pairs, v)
                triple_ids_for_candidates = np.vstack([pair_triples[u, w] for u, w in pairs])
                
                # Check which are new (cover_count == 0)
                valid_mask = triple_ids_for_candidates != -1
                safe_ids = np.where(valid_mask, triple_ids_for_candidates, 0)
                is_new = (cover_count[safe_ids] == 0) & valid_mask
                gains = is_new.sum(axis=0) # Shape (v,)
                
                # Mask out points already in block
                for p in block:
                    gains[p] = -1
                    
                best_gain = np.max(gains)
                if best_gain <= 0:
                    break
                    
                max_gain_points = np.where(gains == best_gain)[0]
                if len(max_gain_points) > 1:
                    best_point = random.choice(max_gain_points.tolist())
                else:
                    best_point = max_gain_points[0]
                    
                block.append(best_point)
            
            # Sort block for consistency
            block.sort()
            blocks.append(block)
            
            # Update cover counts
            for u, w, z in itertools.combinations(block, t):
                tid = triple_to_id[(u, w, z)]
                cover_count[tid] += 1
                
        return blocks

    def local_search_remove(blocks, seed_val):
        # Try to remove redundant blocks
        cover_count = np.zeros(num_triples, dtype=np.int32)
        
        # Initial count
        for block in blocks:
            for u, w, z in itertools.combinations(block, t):
                tid = triple_to_id[(u, w, z)]
                cover_count[tid] += 1
                
        indices = list(range(len(blocks)))
        random.seed(seed_val)
        random.shuffle(indices)
        
        removed = [False] * len(blocks)
        
        # Multiple passes to ensure all possible removals are found
        # Limit passes to avoid excessive time if many blocks
        max_passes = 5
        for _ in range(max_passes):
            changed = False
            for i in indices:
                if removed[i]:
                    continue
                    
                block = blocks[i]
                redundant = True
                for u, w, z in itertools.combinations(block, t):
                    tid = triple_to_id[(u, w, z)]
                    if cover_count[tid] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    removed[i] = True
                    changed = True
                    for u, w, z in itertools.combinations(block, t):
                        tid = triple_to_id[(u, w, z)]
                        cover_count[tid] -= 1
            if not changed:
                break
                
        new_blocks = [b for i, b in enumerate(blocks) if not removed[i]]
        return new_blocks

    # Main driver
    deadline = time.time() + 110.0
    best_blocks = None
    best_len = float('inf')
    
    # First run
    current_blocks = run_greedy(0)
    improved_blocks = local_search_remove(current_blocks, 0)
    if len(improved_blocks) < best_len:
        best_len = len(improved_blocks)
        best_blocks = improved_blocks
        
    seed_val = 1
    while time.time() < deadline:
        current_blocks = run_greedy(seed_val)
        improved_blocks = local_search_remove(current_blocks, seed_val)
        if len(improved_blocks) < best_len:
            best_len = len(improved_blocks)
            best_blocks = improved_blocks
        seed_val += 1
        
    return best_blocks
