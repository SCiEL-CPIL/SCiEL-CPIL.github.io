import itertools
import random
import time

_PRECOMP_CACHE = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _PRECOMP_CACHE:
        return _PRECOMP_CACHE[key]
        
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    block_triples = []
    for block in all_blocks:
        block_set = set(block)
        covered_triples = []
        for triple in all_triples:
            if set(triple).issubset(block_set):
                covered_triples.append(triple_to_idx[triple])
        block_triples.append(covered_triples)
        
    blocks_by_triple = [[] for _ in range(num_triples)]
    for b_idx, triples_in_block in enumerate(block_triples):
        for t_idx in triples_in_block:
            blocks_by_triple[t_idx].append(b_idx)
            
    _PRECOMP_CACHE[key] = (all_blocks, block_triples, blocks_by_triple, num_triples)
    return _PRECOMP_CACHE[key]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_blocks, block_triples, blocks_by_triple, num_triples = _get_precomputed(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        coverage = [0] * num_triples
        uncovered_set = set(range(num_triples))
        current_blocks = []
        
        rng = random.Random()
        
        while uncovered_set:
            anchor_triple_idx = rng.choice(list(uncovered_set))
            candidates = blocks_by_triple[anchor_triple_idx]
            
            best_block_idx = -1
            best_score_val = -1.0
            
            for b_idx in candidates:
                score = 0.0
                for t_idx in block_triples[b_idx]:
                    # Weighted score: prefer covering triples with low coverage
                    score += 1.0 / (1.0 + coverage[t_idx])
                
                if score > best_score_val:
                    best_score_val = score
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            current_blocks.append(best_block_idx)
            
            for t_idx in block_triples[best_block_idx]:
                if t_idx in uncovered_set:
                    uncovered_set.remove(t_idx)
                coverage[t_idx] += 1
            
        # Polish
        while True:
            coverage = [0] * num_triples
            for b_idx in current_blocks:
                for t_idx in block_triples[b_idx]:
                    coverage[t_idx] += 1
            
            removed_any = False
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                b_idx = current_blocks[i]
                is_redundant = True
                for t_idx in block_triples[b_idx]:
                    if coverage[t_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for t_idx in block_triples[b_idx]:
                        coverage[t_idx] -= 1
                    current_blocks.pop(i)
                    removed_any = True
                    break
            
            if not removed_any:
                break
        
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = [list(all_blocks[b]) for b in current_blocks]
            
    if best_blocks is None:
        return []
        
    return best_blocks
