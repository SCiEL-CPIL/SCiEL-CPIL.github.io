import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy + local search."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Precompute t-subsets and k-blocks
    tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubs)
    tsub_idx = {ts: i for i, ts in enumerate(tsubs)}
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Map block tuple to list of t-subset indices it covers
    block_tsubs = {}
    block_tsub_sets = {}
    for b in blocks:
        ts_idxs = [tsub_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs[b] = ts_idxs
        block_tsub_sets[b] = set(ts_idxs)
        
    # Map t-subset tuple to list of blocks containing it
    tsub_blocks = {ts: [] for ts in tsubs}
    for b in blocks:
        for ts in itertools.combinations(b, t):
            tsub_blocks[ts].append(b)

    def _greedy_cover(rng):
        """Randomized greedy set cover."""
        uncovered = set(range(n_tsubs))
        selected_blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            target_idx = list(uncovered)[rng.randrange(len(uncovered))]
            target_tsub = tsubs[target_idx]
            
            # Get candidate blocks covering this t-subset
            candidates = tsub_blocks[target_tsub]
            
            best_block = None
            best_gain = -1
            
            for b in candidates:
                gain = len(block_tsub_sets[b] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is None:
                break
                
            selected_blocks.append(best_block)
            uncovered -= block_tsub_sets[best_block]
            
        return selected_blocks

    def _local_search(blocks, rng, time_limit):
        """Local search: delete redundant blocks, swap to escape local minima."""
        start_time_ls = time.time()
        blocks = list(blocks)
        
        # Build initial coverage counts
        cov = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            for idx in block_tsubs[b]:
                cov[idx] += 1
        
        no_improve_count = 0
        
        while time.time() - start_time_ls < time_limit:
            improved = False
            
            # 1. Try to delete redundant blocks
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() - start_time_ls > time_limit:
                    break
                b = blocks[i]
                idxs = block_tsubs[b]
                if np.min(cov[idxs]) > 1:
                    cov[idxs] -= 1
                    blocks.pop(i)
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue
            
            # 2. Try to swap blocks
            if not blocks:
                break
                
            found_swap = False
            num_tries = 50
            for _ in range(num_tries):
                if time.time() - start_time_ls > time_limit:
                    break
                    
                i = rng.randrange(len(blocks))
                b = blocks[i]
                idxs_b = block_tsubs[b]
                
                crit_mask = (cov[idxs_b] == 1)
                critical_idxs = idxs_b[crit_mask]
                
                if len(critical_idxs) == 0:
                    cov[idxs_b] -= 1
                    blocks.pop(i)
                    improved = True
                    found_swap = True
                    no_improve_count = 0
                    break
                
                first_crit = critical_idxs[rng.randrange(len(critical_idxs))]
                candidates = tsub_blocks[tsubs[first_crit]]
                
                candidates = list(candidates)
                rng.shuffle(candidates)
                
                for b_prime in candidates:
                    if b_prime == b:
                        continue
                    
                    if len(set(critical_idxs).intersection(block_tsub_sets[b_prime])) == len(critical_idxs):
                        cov[idxs_b] -= 1
                        for idx in block_tsubs[b_prime]:
                            cov[idx] += 1
                        blocks[i] = b_prime
                        improved = True
                        found_swap = True
                        no_improve_count = 0
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                if no_improve_count > 20 and len(blocks) > 10:
                    # Kick
                    kick_size = rng.randint(1, 3)
                    if len(blocks) > kick_size:
                        kick_indices = rng.sample(range(len(blocks)), kick_size)
                        kick_indices.sort(reverse=True)
                        
                        for idx in kick_indices:
                            b = blocks[idx]
                            cov[block_tsubs[b]] -= 1
                            blocks.pop(idx)
                        
                        uncovered_idxs = np.where(cov == 0)[0]
                        
                        if len(uncovered_idxs) > 0:
                            uncovered_set = set(uncovered_idxs)
                            while uncovered_set:
                                target_idx = list(uncovered_set)[rng.randrange(len(uncovered_set))]
                                target_tsub = tsubs[target_idx]
                                candidates = tsub_blocks[target_tsub]
                                
                                best_block = None
                                best_gain = -1
                                for b in candidates:
                                    gain = len(block_tsub_sets[b] & uncovered_set)
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_block = b
                                
                                if best_block is None:
                                    break
                                
                                blocks.append(best_block)
                                uncovered_set -= block_tsub_sets[best_block]
                                for idx in block_tsubs[best_block]:
                                    cov[idx] += 1
                        no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > 100:
                        break
                
        return blocks

    # Main loop
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Greedy restarts
    greedy_deadline = start_time + 15.0
    candidates_solutions = []
    num_restarts = 0
    while time.time() < greedy_deadline:
        rng = random.Random(num_restarts * 101 + 7)
        blocks = _greedy_cover(rng)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        candidates_solutions.append(blocks)
        num_restarts += 1
        
    # Keep top 3 distinct solutions
    # Sort by length
    candidates_solutions.sort(key=lambda x: len(x))
    top_solutions = candidates_solutions[:3]
    
    # Phase 2: Local search on top solutions
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 0 and top_solutions:
        time_per_solution = remaining_time / len(top_solutions)
        
        for sol in top_solutions:
            if time.time() > deadline - 5.0:
                break
            rng = random.Random(42)
            refined_sol = _local_search(sol, rng, time_per_solution)
            if len(refined_sol) < best_count:
                best_count = len(refined_sol)
                best_blocks = refined_sol

    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = _greedy_cover(rng)
        
    # Final validation
    seen = set()
    final_blocks = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen and len(set(b)) == k and all(0 <= x < v for x in b):
            seen.add(key)
            final_blocks.append(list(b))
            
    return final_blocks
