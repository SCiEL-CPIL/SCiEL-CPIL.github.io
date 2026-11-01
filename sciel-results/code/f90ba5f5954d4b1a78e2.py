import itertools
import random
import time

_CACHE = {}

def _get_data(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    triples = list(itertools.combinations(points, t))
    tri_to_id = {tri: i for i, tri in enumerate(triples)}
    num_triples = len(triples)
    
    blocks = list(itertools.combinations(points, k))
    num_blocks = len(blocks)
    
    block_triples = []
    triple_blocks = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(blocks):
        b_tris = []
        for tri in itertools.combinations(block, t):
            tid = tri_to_id[tri]
            b_tris.append(tid)
            triple_blocks[tid].append(b_idx)
        block_triples.append(b_tris)
        
    _CACHE[key] = (blocks, block_triples, triple_blocks, num_triples)
    return _CACHE[key]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    blocks, block_triples, triple_blocks, num_triples = _get_data(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    def idx_to_blocks(indices):
        return [list(blocks[i]) for i in indices]
    
    def minimize(indices):
        if not indices:
            return []
        
        cover_count = [0] * num_triples
        for b_idx in indices:
            for tid in block_triples[b_idx]:
                cover_count[tid] += 1
        
        changed = True
        current_indices = list(indices)
        
        while changed:
            changed = False
            new_indices = []
            for b_idx in current_indices:
                essential = False
                for tid in block_triples[b_idx]:
                    if cover_count[tid] == 1:
                        essential = True
                        break
                if essential:
                    new_indices.append(b_idx)
                else:
                    for tid in block_triples[b_idx]:
                        cover_count[tid] -= 1
                    changed = True
            current_indices = new_indices
            
        return current_indices

    while time.time() < deadline:
        uncovered = set(range(num_triples))
        selected = []
        
        while uncovered:
            # Pick a random uncovered triple
            # Converting to list is O(N), N is small
            target_tid = random.choice(list(uncovered))
            
            candidates = triple_blocks[target_tid]
            
            best_gain = -1
            best_candidates = []
            
            for b_idx in candidates:
                gain = 0
                for tid in block_triples[b_idx]:
                    if tid in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain:
                    best_candidates.append(b_idx)
            
            chosen_idx = random.choice(best_candidates)
            selected.append(chosen_idx)
            
            for tid in block_triples[chosen_idx]:
                uncovered.discard(tid)
                
        minimized = minimize(selected)
        
        if len(minimized) < best_len:
            best_len = len(minimized)
            best_blocks = idx_to_blocks(minimized)
            
            if best_len <= 69:
                break
                
    if best_blocks is None:
        return []
        
    return best_blocks
