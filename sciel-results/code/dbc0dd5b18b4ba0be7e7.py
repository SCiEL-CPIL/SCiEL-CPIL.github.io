import itertools
import random
import time

def construct(v, k, t):
    points = range(v)
    
    # Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Precompute all k-subsets
    k_subsets = list(itertools.combinations(points, k))
    num_k_subsets = len(k_subsets)
    
    # For each t-subset, precompute the list of indices of k-subsets that contain it
    t_to_k_containing = [[] for _ in range(num_t_subsets)]
    # For each k-subset, precompute the list of t-subset indices it covers
    k_to_t_indices = [[] for _ in range(num_k_subsets)]
    
    for k_idx, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            ts_idx = t_subset_to_idx[ts]
            t_to_k_containing[ts_idx].append(k_idx)
            k_to_t_indices[k_idx].append(ts_idx)
            
    def run_greedy(seed):
        random.seed(seed)
        uncovered = set(range(num_t_subsets))
        blocks = []
        coverage_count = [0] * num_t_subsets
        
        # Greedy construction
        while uncovered:
            # Pick a random uncovered t-subset
            # Converting to list is O(N) but N is small (~4000)
            target_idx = random.choice(list(uncovered))
            
            # Get all k-subsets that contain this t-subset
            candidates = t_to_k_containing[target_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            for k_idx in candidates:
                # Calculate gain: number of uncovered t-subsets covered by this k-subset
                gain = 0
                for ts_idx in k_to_t_indices[k_idx]:
                    if ts_idx in uncovered:
                        gain += 1
                        # Early exit if we found all 35
                        if gain == 35: 
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = k_idx
            
            if best_block_idx == -1:
                # Should not happen if the problem is solvable
                break
                
            # Add the best block
            blocks.append(best_block_idx)
            
            # Update coverage and uncovered
            for ts_idx in k_to_t_indices[best_block_idx]:
                if coverage_count[ts_idx] == 0:
                    uncovered.discard(ts_idx)
                coverage_count[ts_idx] += 1
        
        # Redundancy Removal
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                b_idx = blocks[i]
                can_remove = True
                for ts_idx in k_to_t_indices[b_idx]:
                    if coverage_count[ts_idx] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    blocks.pop(i)
                    for ts_idx in k_to_t_indices[b_idx]:
                        coverage_count[ts_idx] -= 1
                    changed = True
                    break
            
        # Convert indices back to blocks
        return [k_subsets[i] for i in blocks]
    
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    time_limit = 110 # Leave margin for 120s limit
    
    seed = 0
    while time.time() - start_time < time_limit:
        blocks = run_greedy(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        seed += 1
        
    if best_blocks is None:
        return run_greedy(0)
        
    return best_blocks
