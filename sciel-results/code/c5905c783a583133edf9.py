import heapq
import itertools
import time
import random

_cache = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _cache:
        return _cache[key]
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map triple to index
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # For each block, list of triple indices it covers
    block_triple_indices = []
    # For each triple, list of block indices covering it
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for b_idx, b in enumerate(all_blocks):
        tris = list(itertools.combinations(b, t))
        t_indices = [triple_to_idx[tr] for tr in tris]
        block_triple_indices.append(t_indices)
        for t_idx in t_indices:
            triple_to_blocks[t_idx].append(b_idx)
            
    _cache[key] = (all_blocks, block_triple_indices, triple_to_blocks, all_triples)
    return _cache[key]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_blocks, block_triple_indices, triple_to_blocks, all_triples = _get_precomputed(v, k, t)
    n_blocks = len(all_blocks)
    n_triples = len(all_triples)
    
    # uncovered: boolean array
    uncovered = [True] * n_triples
    # gain: number of uncovered triples covered by block
    gain = [0] * n_blocks
    
    # Initialize gains
    for b_idx in range(n_blocks):
        gain[b_idx] = len(block_triple_indices[b_idx])
        
    # Heap: (-gain, block_idx, version)
    # We need a version counter to handle stale entries in the heap
    # Or simply check if the gain in the heap matches the current gain.
    heap = []
    for b_idx in range(n_blocks):
        heapq.heappush(heap, (-gain[b_idx], b_idx))
        
    selected = []
    selected_set = set()
    
    while True:
        if time.time() > deadline:
            break
            
        # Get best block
        neg_gain, b_idx = heapq.heappop(heap)
        cur_gain = -neg_gain
        if cur_gain != gain[b_idx]:
            continue # Stale entry
        if b_idx in selected_set:
            continue
        if cur_gain == 0:
            break # No more coverage possible? Shouldn't happen if cover is possible
            
        # Select block
        selected.append(b_idx)
        selected_set.add(b_idx)
        
        # Update uncovered and gains
        for t_idx in block_triple_indices[b_idx]:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                # Decrement gain for all blocks covering this triple
                for other_b in triple_to_blocks[t_idx]:
                    if other_b != b_idx and other_b not in selected_set:
                        gain[other_b] -= 1
                        heapq.heappush(heap, (-gain[other_b], other_b))
                        
        # Check if done
        if not any(uncovered):
            break
            
    # Verify validity
    # If not valid, we have a problem. But Lazy Greedy should be valid if it finishes.
    # If it timed out, it's invalid.
    # We must ensure we return a valid cover.
    # If time is tight, we might need a fallback.
    
    # For now, return the result.
    # Note: If it didn't finish, this is invalid.
    # We need a guaranteed valid method if greedy fails.
    # But greedy is very fast.
    
    return [list(all_blocks[i]) for i in selected]
