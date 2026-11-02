"""Covering design C(v,k,t) using Simulated Annealing on Set Covering."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute all possible k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the indices of t-subsets it covers
    block_to_tsubs = []
    for block in all_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        block_to_tsubs.append(indices)
    
    # Reverse index: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for bi, indices in enumerate(block_to_tsubs):
        for idx in indices:
            tsub_to_blocks[idx].append(bi)
    
    def get_block_tsub_indices(block_tuple):
        """Return list of t-subset indices covered by a block (as tuple)."""
        indices = []
        for ts in itertools.combinations(block_tuple, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        return indices

    def greedy_build(seed):
        """Fast greedy construction to get an initial valid covering."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int32)
        blocks_used = []
        used_block_indices = set()
        
        while True:
            if uncovered.sum() == 0:
                break
            
            uncovered_indices = np.where(uncovered == 1)[0]
            if len(uncovered_indices) == 0:
                break
            
            best_block_idx = -1
            best_gain = -1
            best_block_tsubs = []
            
            # Pick a random uncovered t-subset to guide selection (faster than full scan)
            # Or pick the most constrained one if time permits, but random is faster
            # Let's use a heuristic: pick the uncovered t-subset with fewest remaining covering blocks
            # For speed, we just sample a few
            sample_size = min(10, len(uncovered_indices))
            sample_idx = rng.sample(uncovered_indices.tolist(), sample_size)
            
            for ts_idx in sample_idx:
                covering_blocks = [bi for bi in tsub_to_blocks[ts_idx] if bi not in used_block_indices]
                if not covering_blocks:
                    continue
                rng.shuffle(covering_blocks)
                for bi in covering_blocks[:5]:
                    block_tsubs = block_to_tsubs[bi]
                    gain = np.sum(uncovered[np.array(block_tsubs, dtype=np.int32)] == 1)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = bi
                        best_block_tsubs = block_tsubs
            
            if best_block_idx == -1:
                # Fallback
                ts_idx = uncovered_indices[0]
                for bi in tsub_to_blocks[ts_idx]:
                    if bi in used_block_indices:
                        continue
                    block_tsubs = block_to_tsubs[bi]
                    gain = np.sum(uncovered[np.array(block_tsubs, dtype=np.int32)] == 1)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = bi
                        best_block_tsubs = block_tsubs
            
            if best_block_idx == -1:
                break
            
            used_block_indices.add(best_block_idx)
            blocks_used.append(all_blocks[best_block_idx])
            
            for idx in best_block_tsubs:
                uncovered[idx] = 0
            
            if time.time() > deadline - 30:
                break
        
        return blocks_used

    def reduce_blocks(blocks, max_passes=5):
        """Aggressively remove redundant blocks."""
        if not blocks:
            return blocks
        
        blocks_list = list(blocks)
        
        for _ in range(max_passes):
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            block_tsubs_list = []
            for block in blocks_list:
                indices = get_block_tsub_indices(tuple(block))
                block_tsubs_list.append(indices)
                for idx in indices:
                    cov_count[idx] += 1
            
            improved = True
            while improved:
                improved = False
                order = list(range(len(blocks_list)))
                random.shuffle(order)
                
                for i in order:
                    if i >= len(blocks_list):
                        continue
                    redundant = True
                    for idx in block_tsubs_list[i]:
                        if cov_count[idx] <= 1:
                            redundant = False
                            break
                    
                    if redundant:
                        for idx in block_tsubs_list[i]:
                            cov_count[idx] -= 1
                        blocks_list.pop(i)
                        block_tsubs_list.pop(i)
                        improved = True
            
            if not improved:
                break
        
        return blocks_list

    def simulated_annealing(initial_blocks, seed, time_limit=10):
        """
        Simulated Annealing on the set of blocks.
        State: set of block indices.
        Moves: Add a block, Remove a block, Swap a block.
        Energy: lambda * uncovered_count + block_count.
        We want to minimize this. Uncovered must be 0 for validity.
        """
        rng = random.Random(seed)
        blocks_set = set()
        for b in initial_blocks:
            blocks_set.add(all_blocks.index(b)) # This is slow, use dict
        
        # Better representation: list of block indices
        blocks_list = [all_blocks.index(b) for b in initial_blocks]
        block_set = set(blocks_list)
        
        # Initial coverage
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for bi in blocks_list:
            for idx in block_to_tsubs[bi]:
                cov_count[idx] += 1
        
        uncovered_count = np.sum(cov_count == 0)
        current_energy = uncovered_count * 1000 + len(blocks_list)
        
        best_blocks_list = list(blocks_list)
        best_energy = current_energy
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time and T > T_min:
            # Generate a move
            move_type = rng.random()
            
            if move_type < 0.3:
                # Add a block
                if len(blocks_list) < n_blocks:
                    # Pick a random block not in set
                    # Optimization: pick a block that covers many uncovered
                    if uncovered_count > 0:
                        uncovered_indices = np.where(cov_count == 0)[0]
                        if len(uncovered_indices) > 0:
                            target_ts = uncovered_indices[rng.randrange(len(uncovered_indices))]
                            candidates = tsub_to_blocks[target_ts]
                            if candidates:
                                new_bi = candidates[rng.randrange(len(candidates))]
                                if new_bi not in block_set:
                                    # Apply move
                                    block_set.add(new_bi)
                                    blocks_list.append(new_bi)
                                    for idx in block_to_tsubs[new_bi]:
                                        cov_count[idx] += 1
                                    new_uncovered = np.sum(cov_count == 0)
                                    new_energy = new_uncovered * 1000 + len(blocks_list)
                                    delta = new_energy - current_energy
                                    
                                    if delta < 0 or rng.random() < np.exp(-delta / T):
                                        current_energy = new_energy
                                        uncovered_count = new_uncovered
                                        if current_energy < best_energy:
                                            best_energy = current_energy
                                            best_blocks_list = list(blocks_list)
                                    else:
                                        # Revert
                                        blocks_list.pop()
                                        block_set.remove(new_bi)
                                        for idx in block_to_tsubs[new_bi]:
                                            cov_count[idx] -= 1
                                continue
                    else:
                        # No uncovered, just add random
                        new_bi = rng.randrange(n_blocks)
                        if new_bi not in block_set:
                            block_set.add(new_bi)
                            blocks_list.append(new_bi)
                            for idx in block_to_tsubs[new_bi]:
                                cov_count[idx] += 1
                            new_uncovered = np.sum(cov_count == 0)
                            new_energy = new_uncovered * 1000 + len(blocks_list)
                            delta = new_energy - current_energy
                            
                            if delta < 0 or rng.random() < np.exp(-delta / T):
                                current_energy = new_energy
                                uncovered_count = new_uncovered
                                if current_energy < best_energy:
                                    best_energy = current_energy
                                    best_blocks_list = list(blocks_list)
                            else:
                                blocks_list.pop()
                                block_set.remove(new_bi)
                                for idx in block_to_tsubs[new_bi]:
                                    cov_count[idx] -= 1
                        continue
                else:
                    continue
                    
            elif move_type < 0.6:
                # Remove a block
                if len(blocks_list) > 0:
                    idx_to_remove = rng.randrange(len(blocks_list))
                    bi = blocks_list[idx_to_remove]
                    
                    # Apply move
                    blocks_list.pop(idx_to_remove)
                    block_set.remove(bi)
                    for idx in block_to_tsubs[bi]:
                        cov_count[idx] -= 1
                    new_uncovered = np.sum(cov_count == 0)
                    new_energy = new_uncovered * 1000 + len(blocks_list)
                    delta = new_energy - current_energy
                    
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_energy = new_energy
                        uncovered_count = new_uncovered
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_blocks_list = list(blocks_list)
                    else:
                        # Revert
                        blocks_list.insert(idx_to_remove, bi)
                        block_set.add(bi)
                        for idx in block_to_tsubs[bi]:
                            cov_count[idx] += 1
                else:
                    continue
                    
            else:
                # Swap a block
                if len(blocks_list) > 0:
                    idx_to_swap = rng.randrange(len(blocks_list))
                    old_bi = blocks_list[idx_to_swap]
                    
                    # Pick new block
                    # Prefer blocks covering uncovered
                    if uncovered_count > 0:
                        uncovered_indices = np.where(cov_count == 0)[0]
                        if len(uncovered_indices) > 0:
                            target_ts = uncovered_indices[rng.randrange(len(uncovered_indices))]
                            candidates = [c for c in tsub_to_blocks[target_ts] if c not in block_set]
                            if candidates:
                                new_bi = candidates[rng.randrange(len(candidates))]
                            else:
                                new_bi = rng.randrange(n_blocks)
                                if new_bi in block_set:
                                    continue
                    else:
                        new_bi = rng.randrange(n_blocks)
                        if new_bi in block_set:
                            continue
                    
                    # Apply move
                    blocks_list[idx_to_swap] = new_bi
                    block_set.remove(old_bi)
                    block_set.add(new_bi)
                    for idx in block_to_tsubs[old_bi]:
                        cov_count[idx] -= 1
                    for idx in block_to_tsubs[new_bi]:
                        cov_count[idx] += 1
                    new_uncovered = np.sum(cov_count == 0)
                    new_energy = new_uncovered * 1000 + len(blocks_list)
                    delta = new_energy - current_energy
                    
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_energy = new_energy
                        uncovered_count = new_uncovered
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_blocks_list = list(blocks_list)
                    else:
                        # Revert
                        blocks_list[idx_to_swap] = old_bi
                        block_set.remove(new_bi)
                        block_set.add(old_bi)
                        for idx in block_to_tsubs[new_bi]:
                            cov_count[idx] -= 1
                        for idx in block_to_tsubs[old_bi]:
                            cov_count[idx] += 1
                else:
                    continue
            
            T *= cooling
            
        # Return best found, ensure it's valid (uncovered == 0)
        # If best_energy has uncovered > 0, we need a valid one.
        # The initial state was valid. If we never accepted a state with uncovered > 0 as "best" 
        # because we track best_energy which includes penalty, the best_energy state might be invalid 
        # if it has lower penalty than a valid state with more blocks? 
        # No, penalty is 1000 per uncovered. A valid state with 200 blocks has energy 200. 
        # An invalid state with 100 blocks and 1 uncovered has energy 1100. 
        # So valid states always have lower energy if block count < 1000. 
        # But if we have 1200 blocks valid, energy 1200. Invalid with 100 blocks, 1 uncovered, energy 1100. 
        # The invalid one would be "best". 
        # We must return a VALID covering.
        
        # So we track best_valid_energy separately.
        # Let's re-run SA logic or just take the initial if best is invalid.
        # Better: track best_valid_blocks separately.
        
        return best_blocks_list, best_energy

    # Main Search Loop
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Generate initial solutions with Greedy
    # We run a few greedy starts
    initial_solutions = []
    for seed in range(5):
        if time.time() > deadline - 40:
            break
        blocks = greedy_build(seed)
        reduced = reduce_blocks(blocks, max_passes=2)
        initial_solutions.append(reduced)
        if len(reduced) < best_count:
            best_blocks = reduced
            best_count = len(reduced)
            
    # Phase 2: Simulated Annealing on the best initial solutions
    # We have ~40-70 seconds left.
    # Run SA on the top 2-3 initial solutions.
    rng_main = random.Random(42)
    
    num_sa_runs = 3
    sa_time_each = 15  # seconds
    
    for i in range(num_sa_runs):
        if time.time() > deadline - 10:
            break
        if i < len(initial_solutions):
            start_blocks = initial_solutions[i]
        else:
            start_blocks = best_blocks
            
        sa_blocks, sa_energy = simulated_annealing(start_blocks, seed=100+i, time_limit=sa_time_each)
        
        # Check validity of sa_blocks
        # sa_energy includes penalty. If sa_energy < best_count (which implies 0 uncovered and fewer blocks), it's good.
        # But we need to be sure it's valid.
        # Let's verify validity explicitly for the SA result.
        cov = np.zeros(n_tsubs, dtype=np.int32)
        for bi in sa_blocks:
            for idx in block_to_tsubs[bi]:
                cov[idx] += 1
        if np.min(cov) >= 1:
            # It is valid
            # Reduce it further
            reduced_sa = reduce_blocks([all_blocks[bi] for bi in sa_blocks], max_passes=3)
            if len(reduced_sa) < best_count:
                best_blocks = reduced_sa
                best_count = len(reduced_sa)
                
    # Phase 3: Final Polish on the best solution found
    if best_blocks is not None and time.time() < deadline - 5:
        # Try a quick local search on the best
        rng_polish = random.Random(999)
        current = list(best_blocks)
        
        # Simple swap-based local search
        # Try to swap one block for another to reduce count? No, swap keeps count.
        # Try to remove one block and add one block that covers more?
        # Actually, just run reduce_blocks again with more passes
        current = reduce_blocks(current, max_passes=5)
        if len(current) < best_count:
            best_blocks = current
            best_count = len(current)
            
    return best_blocks if best_blocks else greedy_build(0)
