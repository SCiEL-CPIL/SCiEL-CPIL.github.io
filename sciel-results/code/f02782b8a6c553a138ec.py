import itertools
import random
import time

def construct(v, k, t):
    if v != 24 or k != 7 or t != 3:
        return _simple_greedy(v, k, t)
        
    start_time = time.time()
    deadline = start_time + 140.0
    
    all_triples = list(itertools.combinations(range(v), t))
    triple_set = set(all_triples)
    
    best_blocks = None
    best_count = float('inf')
    
    while time.time() < deadline:
        blocks = _randomized_greedy_fast(v, k, t, triple_set, random_seed=random.randint(0, 10**6))
        blocks = _reduce_blocks(blocks, v, t)
        blocks = _local_search_swap(blocks, v, k, t, time_limit=3.0)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = _simple_greedy(v, k, t)
        
    best_blocks = _reduce_blocks(best_blocks, v, t)
    return [list(b) for b in best_blocks]

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        target = min(uncovered)
        best_block = None
        best_gain = -1
        remaining_points = [p for p in range(v) if p not in target]
        need = k - t
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block
        blocks.append(best_block)
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    return blocks

def _randomized_greedy_fast(v, k, t, triple_set, random_seed):
    rng = random.Random(random_seed)
    uncovered = set(triple_set)
    blocks = []
    
    while uncovered:
        unc_list = list(uncovered)
        target = unc_list[rng.randrange(len(unc_list))]
        
        best_block = None
        best_gain = -1
        target_set = set(target)
        remaining_points = [p for p in range(v) if p not in target_set]
        need = k - t
        
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block
            elif gain == best_gain and rng.random() < 0.5:
                best_block = block
                    
        blocks.append(best_block)
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
            
    return blocks

def _reduce_blocks(blocks, v, t):
    if not blocks:
        return []
        
    # Precompute triple sets for each block
    block_triples = [set(itertools.combinations(b, t)) for b in blocks]
    
    # For each triple, find which blocks cover it
    triple_to_blocks = {}
    for i, triple_set_i in enumerate(block_triples):
        for sub in triple_set_i:
            if sub not in triple_to_blocks:
                triple_to_blocks[sub] = []
            triple_to_blocks[sub].append(i)
            
    active = [True] * len(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if not active[i]:
                continue
            redundant = True
            for sub in block_triples[i]:
                count = sum(1 for j in triple_to_blocks[sub] if active[j])
                if count == 1:
                    redundant = False
                    break
            if redundant:
                active[i] = False
                changed = True
                
    return [blocks[i] for i in range(len(blocks)) if active[i]]

def _local_search_swap(blocks, v, k, t, time_limit=3.0):
    if not blocks:
        return blocks
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n = len(blocks)
        
        # Precompute triple sets
        block_triples = [set(itertools.combinations(b, t)) for b in blocks]
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            # Find unique triples of block i
            unique = []
            for sub in block_triples[i]:
                # Check if any other block covers sub
                covered = False
                for j in range(n):
                    if j != i and sub in block_triples[j]:
                        covered = True
                        break
                if not covered:
                    unique.append(sub)
            
            if not unique:
                blocks = blocks[:i] + blocks[i+1:]
                improved = True
                break
            
            # Try to find a replacement block covering all unique triples
            required = set()
            for ut in unique:
                required.update(ut)
            
            if len(required) > k:
                continue
            
            remaining = [p for p in range(v) if p not in required]
            need_more = k - len(required)
            
            best_new = None
            best_score = -1
            
            for combo in itertools.combinations(remaining, need_more):
                new_block = tuple(sorted(tuple(required) + combo))
                new_triples = set(itertools.combinations(new_block, t))
                # Score: number of triples in new_block covered by other blocks
                score = 0
                for sub in new_triples:
                    for j in range(n):
                        if j != i and sub in block_triples[j]:
                            score += 1
                            break
                if score > best_score:
                    best_score = score
                    best_new = new_block
            
            if best_new is not None:
                blocks[i] = best_new
                improved = True
                break
    
    return _reduce_blocks(blocks, v, t)
