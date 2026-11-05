import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, store the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        idxs = [t_sub_to_idx[t_sub] for t_sub in itertools.combinations(k_sub, t)]
        k_sub_t_idxs.append(idxs)

    # For each t-subset, store the list of k-subset indices that cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    # Convert to numpy arrays for faster operations
    # Store as list of arrays
    k_sub_t_idxs_np = [np.array(idxs, dtype=np.int32) for idxs in k_sub_t_idxs]
    t_sub_to_k_idxs_np = [np.array(idxs, dtype=np.int32) for idxs in t_sub_to_k_idxs]

    def time_ok():
        return time.time() < deadline

    def reduce_blocks(block_indices, coverage):
        """Remove redundant blocks. Returns new list of block indices and updated coverage."""
        if not block_indices:
            return [], coverage
            
        current_blocks = list(block_indices)
        changed = True
        while changed and time_ok():
            changed = False
            new_blocks = []
            for b_idx in current_blocks:
                idxs = k_sub_t_idxs_np[b_idx]
                # Check if any t-subset in this block has cover_count == 1
                if np.any(coverage[idxs] == 1):
                    new_blocks.append(b_idx)
                else:
                    # Redundant
                    changed = True
                    coverage[idxs] -= 1
            
            current_blocks = new_blocks
        
        return current_blocks, coverage

    def greedy_cover_random(uncovered_set, used_blocks_set, rng, coverage):
        """Vectorized greedy set cover. Returns list of block indices."""
        blocks = []
        uncovered = set(uncovered_set)
        
        while uncovered and time_ok():
            # Pick a random uncovered t-subset
            s_idx = rng.choice(list(uncovered))
            candidates = t_sub_to_k_idxs_np[s_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            # Shuffle candidates to break ties randomly
            cand_list = candidates.copy()
            rng.shuffle(cand_list)
            
            for b_idx in cand_list:
                if b_idx in used_blocks_set:
                    continue
                idxs = k_sub_t_idxs_np[b_idx]
                gain = np.sum(coverage[idxs] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break

            blocks.append(best_block_idx)
            used_blocks_set.add(best_block_idx)
            
            idxs = k_sub_t_idxs_np[best_block_idx]
            coverage[idxs] += 1
            
            for t_idx in idxs:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
        
        return blocks

    def simulated_annealing(block_indices, coverage, rng, max_iters=200):
        """Simulated annealing to improve the covering."""
        if not block_indices:
            return block_indices, coverage
            
        blocks = list(block_indices)
        cov = coverage.copy()
        best_blocks = list(blocks)
        best_cov = cov.copy()
        best_len = len(blocks)
        
        T = 5.0  # Initial temperature
        cooling_rate = 0.95
        
        for i in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # Pick a random move: swap, add, or remove
            move_type = rng.random()
            
            old_len = len(blocks)
            
            if move_type < 0.5 and len(blocks) > 0:
                # Swap: remove one, add one
                b_remove = rng.choice(blocks)
                # Pick a random block to add
                b_add = rng.randint(0, n_k - 1)
                if b_add in blocks:
                    continue
                    
                # Apply move
                blocks.remove(b_remove)
                blocks.append(b_add)
                
                # Update coverage
                idxs_remove = k_sub_t_idxs_np[b_remove]
                idxs_add = k_sub_t_idxs_np[b_add]
                cov[idxs_remove] -= 1
                cov[idxs_add] += 1
                
                # Check validity
                if np.any(cov < 0):
                    # Invalid, revert
                    cov[idxs_remove] += 1
                    cov[idxs_add] -= 1
                    blocks.remove(b_add)
                    blocks.append(b_remove)
                    continue
                
                # Reduce
                blocks, cov = reduce_blocks(blocks, cov)
                
            elif move_type < 0.75:
                # Add
                b_add = rng.randint(0, n_k - 1)
                if b_add in blocks:
                    continue
                
                blocks.append(b_add)
                idxs_add = k_sub_t_idxs_np[b_add]
                cov[idxs_add] += 1
                
                # Reduce
                blocks, cov = reduce_blocks(blocks, cov)
                
            else:
                # Remove (only if it doesn't break coverage)
                if len(blocks) <= 1:
                    continue
                b_remove = rng.choice(blocks)
                idxs_remove = k_sub_t_idxs_np[b_remove]
                
                # Check if removal is safe
                if np.any(cov[idxs_remove] == 1):
                    continue  # Can't remove
                
                blocks.remove(b_remove)
                cov[idxs_remove] -= 1
                
            new_len = len(blocks)
            
            # Acceptance criterion
            if new_len <= old_len:
                # Always accept improvements or equal
                best_blocks = list(blocks)
                best_cov = cov.copy()
                best_len = new_len
            else:
                # Accept with probability exp(-delta / T)
                delta = new_len - old_len
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    best_blocks = list(blocks)
                    best_cov = cov.copy()
                    best_len = new_len
                else:
                    # Revert to best
                    blocks = list(best_blocks)
                    cov = best_cov.copy()
            
            T *= cooling_rate
        
        return best_blocks, best_cov

    # Main strategy:
    # 1. Run greedy multiple times with different randomizations to find the best covering.
    # 2. Apply reduction to each.
    # 3. Apply simulated annealing to the best ones.
    # 4. Keep the best (smallest) one.
    
    best_blocks = None
    best_size = float('inf')
    best_coverage = None
    
    num_restarts = 30  # Adjust based on time
    
    for i in range(num_restarts):
        if not time_ok():
            break
            
        rng = random.Random(i)
        used = set()
        coverage = np.zeros(n_t, dtype=np.int32)
        blocks = greedy_cover_random(set(range(n_t)), used, rng, coverage)
        
        # Reduce
        reduced, coverage = reduce_blocks(blocks, coverage)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_blocks = reduced
            best_coverage = coverage.copy()
            
            # Apply simulated annealing
            if time.time() < deadline - 20.0:
                sa_blocks, sa_cov = simulated_annealing(best_blocks, best_coverage, rng, max_iters=100)
                if len(sa_blocks) < best_size:
                    best_size = len(sa_blocks)
                    best_blocks = sa_blocks
                    best_coverage = sa_cov
    
    if best_blocks is None:
        # Fallback
        rng = random.Random(0)
        used = set()
        coverage = np.zeros(n_t, dtype=np.int32)
        blocks = greedy_cover_random(set(range(n_t)), used, rng, coverage)
        best_blocks, best_coverage = reduce_blocks(blocks, coverage)
        
    # Convert indices back to lists of points
    return [list(all_k_subs[idx]) for idx in best_blocks]
