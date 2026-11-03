import numpy as np
import itertools
import time
import random
import multiprocessing as mp
from functools import partial

# Global variables for worker processes
G_v = 0
G_k = 0
G_t = 0
G_blocks = []
G_block_triples = None
G_triple_to_blocks = []
G_n_triples = 0
G_n_blocks = 0
G_deadline = 0

def _init_worker(v, k, t, blocks, block_triples, triple_to_blocks, n_triples, n_blocks, deadline):
    global G_v, G_k, G_t, G_blocks, G_block_triples, G_triple_to_blocks, G_n_triples, G_n_blocks, G_deadline
    G_v = v
    G_k = k
    G_t = t
    G_blocks = blocks
    G_block_triples = block_triples
    G_triple_to_blocks = triple_to_blocks
    G_n_triples = n_triples
    G_n_blocks = n_blocks
    G_deadline = deadline

def _greedy_and_local(seed):
    global G_block_triples, G_triple_to_blocks, G_n_triples, G_blocks, G_deadline, G_k
    
    if time.time() > G_deadline:
        return None, float('inf')
        
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    
    # Greedy Set Cover
    uncovered = np.ones(G_n_triples, dtype=bool)
    selected = []
    
    # Precompute weights? For simplicity, use unweighted first, then weighted.
    # Let's use a dynamic weight: 1 / (coverage_count + 1)
    # But coverage counts change.
    # Standard greedy: max |B \cap Uncovered|
    
    # To speed up, we can precompute the list of triples for each block.
    # G_block_triples is (n_blocks, C(k,t)) int32.
    
    while np.any(uncovered):
        if time.time() > G_deadline:
            return None, float('inf')
            
        # Calculate gains
        # gains[i] = number of uncovered triples in block i
        # We can compute this as sum(uncovered[block_triples[i]])
        # Vectorized: np.sum(uncovered[G_block_triples], axis=1)
        # This is a bit slow if done every iteration for 347k blocks.
        # Optimization: Only consider blocks that might have high gain?
        # Or just do it. 347k * 35 operations is ~12M ops. NumPy handles this in ~10-20ms.
        
        gains = np.sum(uncovered[G_block_triples], axis=1)
        
        # Random tie breaking
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        candidates = np.where(gains == max_gain)[0]
        if len(candidates) > 0:
            # Pick random candidate to diversify
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            selected.append(best_idx)
            uncovered[G_block_triples[best_idx]] = False
            
    if not selected:
        return None, float('inf')
        
    # Local Search: Remove and Swap
    # Build coverage counts for the current solution
    coverage_count = np.zeros(G_n_triples, dtype=np.int32)
    for b_idx in selected:
        coverage_count[G_block_triples[b_idx]] += 1
        
    selected_set = set(selected)
    
    improved = True
    while improved and time.time() < G_deadline:
        improved = False
        # Shuffle to avoid bias
        random.shuffle(selected)
        
        for b_idx in selected:
            if time.time() > G_deadline:
                return None, float('inf')
                
            # Try to remove b_idx
            # Decrement coverage counts for triples in b_idx
            b_triples = G_block_triples[b_idx]
            for t_idx in b_triples:
                coverage_count[t_idx] -= 1
                
            # Check if any triple is now uncovered (count == 0)
            uncovered_in_b = [t_idx for t_idx in b_triples if coverage_count[t_idx] == 0]
            
            if not uncovered_in_b:
                # Safe to remove
                selected_set.remove(b_idx)
                improved = True
            else:
                # Need to swap. Pick one uncovered triple t0
                t0 = uncovered_in_b[0]
                
                # Find candidate blocks covering t0
                candidates = G_triple_to_blocks[t0]
                
                best_gain = 0
                best_swap = None
                best_new_selected = None
                
                for c_idx in candidates:
                    if c_idx in selected_set:
                        continue
                        
                    c_triples = G_block_triples[c_idx]
                    
                    # Gain: how many triples in c_idx are currently uncovered (count==0)
                    # OR how many new triples does c_idx cover that b_idx did not?
                    # Actually, we are swapping b_idx for c_idx.
                    # Triples in b_idx that are unique to b_idx (count was 1) become uncovered.
                    # Triples in c_idx that are currently uncovered (count==0) become covered.
                    
                    # Let's calculate the net change in number of covered triples.
                    # Current state: b_idx is "removed" (counts decremented).
                    # If we add c_idx:
                    # 1. Triples in c_idx with count==0 will become covered.
                    # 2. Triples in c_idx with count>0 stay covered.
                    # 3. Triples in b_idx that were unique (count==0 now) remain uncovered unless c_idx covers them.
                    
                    # Simpler metric: 
                    # Gain = (number of triples in c_idx with count==0) - (number of triples in b_idx with count==0 that are NOT in c_idx)
                    
                    c_triples_set = set(c_triples.tolist())
                    
                    new_covered = 0
                    for t_idx in c_triples:
                        if coverage_count[t_idx] == 0:
                            new_covered += 1
                            
                    lost = 0
                    for t_idx in b_triples:
                        if coverage_count[t_idx] == 0 and t_idx not in c_triples_set:
                            lost += 1
                            
                    gain = new_covered - lost
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = c_idx
                        best_new_selected = c_triples
                        
                if best_swap is not None and best_gain > 0:
                    # Perform swap
                    selected_set.remove(b_idx)
                    selected_set.add(best_swap)
                    # Update coverage counts
                    # We already decremented b_idx. Now increment best_swap.
                    coverage_count[best_new_selected] += 1
                    improved = True
                else:
                    # Restore counts for b_idx
                    coverage_count[b_triples] += 1
                    
    # Final check: is it a valid cover?
    if coverage_count.min() < 0: # Should not happen
        return None, float('inf')
        
    # Ensure all covered
    if np.any(coverage_count == 0):
        return None, float('inf')
        
    return list(selected_set), len(selected_set)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # c(k, t)
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # Precompute block_triples matrix
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    # Precompute triple_to_blocks list
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[i, idx] = t_idx
            triple_to_blocks[t_idx].append(i)
            idx += 1
            
    # Set global variables for workers
    _init_worker(v, k, t, blocks, block_triples, triple_to_blocks, n_triples, n_blocks, deadline)
    
    best_solution = None
    best_size = float('inf')
    
    # Use multiprocessing for parallel restarts
    # Number of workers depends on CPU count, but let's cap it to avoid overhead
    num_workers = min(4, mp.cpu_count())
    
    # Create seeds for different trials
    seeds = [int(time.time()) + i for i in range(num_workers * 5)]
    
    with mp.Pool(processes=num_workers, initializer=_init_worker, 
                 initargs=(v, k, t, blocks, block_triples, triple_to_blocks, n_triples, n_blocks, deadline)) as pool:
        
        # Submit multiple tasks
        results = []
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            results.append(pool.apply_async(_greedy_and_local, (seed,)))
            
        # Collect results
        for r in results:
            try:
                # Wait with timeout? apply_async doesn't support timeout directly in wait.
                # We can just wait. The workers check deadline.
                selected_indices, size = r.get(timeout=deadline - time.time())
                if selected_indices is not None and size < best_size:
                    best_size = size
                    best_solution = [list(blocks[i]) for i in selected_indices]
            except Exception as e:
                pass
        
    # If no parallel result or if we have time, do a sequential one with more care
    if time.time() < deadline - 20:
        # Sequential high-quality run
        # Use a better greedy: Weighted by 1/freq
        # But for simplicity, let's just run one more random restart sequentially
        selected_indices, size = _greedy_and_local(int(time.time()))
        if selected_indices is not None and size < best_size:
            best_size = size
            best_solution = [list(blocks[i]) for i in selected_indices]
            
    if best_solution is None:
        # Fallback: greedy without local search, single run
        # This should not happen if the above works
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        while np.any(uncovered):
            gains = np.sum(uncovered[block_triples], axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
        best_solution = [list(blocks[i]) for i in selected]
        
    return best_solution

if __name__ == '__main__':
    # Test locally
    v, k, t = 24, 7, 3
    start = time.time()
    sol = construct(v, k, t)
    end = time.time()
    print(f"Time: {end-start:.2f}s, Blocks: {len(sol)}")
    
    # Verify
    covered = set()
    for b in sol:
        for c in itertools.combinations(b, t):
            covered.add(c)
    all_triples = set(itertools.combinations(range(v), t))
    print(f"Covered: {len(covered)}/{len(all_triples)}")
