import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Pre-compute mappings
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    k_to_idx = {ks: i for i, ks in enumerate(k_subsets)}
    num_k_subsets = len(k_subsets)
    
    # For each t-subset, list of k-subsets containing it
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    # For each k-subset, list of t-subsets contained in it
    k_to_ts = [[] for _ in range(num_k_subsets)]
    
    for i, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            j = t_to_idx[ts]
            t_to_blocks[j].append(i)
            k_to_ts[i].append(j)
            
    # Greedy with anchor
    def greedy_once(seed):
        random.seed(seed)
        # Use a list of booleans for uncovered
        uncovered = [True] * num_t_subsets
        num_uncovered = num_t_subsets
        blocks_idx = []
        
        while num_uncovered > 0:
            # Find an uncovered t-subset
            # Instead of scanning, we can pick a random index and check
            # Or keep a list of uncovered indices
            # For simplicity, scan or use a set
            # Let's use a set of uncovered indices for efficiency in picking
            # But updating the set is O(1) per removal.
            # Initial set creation is O(N).
            # Let's just pick a random index until we find an uncovered one.
            # With 3876 items, this is fast.
            anchor_idx = -1
            while True:
                cand = random.randint(0, num_t_subsets - 1)
                if uncovered[cand]:
                    anchor_idx = cand
                    break
            
            # Get candidates: blocks containing anchor_idx
            candidates = t_to_blocks[anchor_idx]
            
            # Score each candidate by number of newly covered uncovered t-subsets
            best_gain = -1
            best_block = -1
            best_cand_list = []
            
            for b_idx in candidates:
                gain = 0
                for ts_idx in k_to_ts[b_idx]:
                    if uncovered[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_cand_list = [b_idx]
                elif gain == best_gain:
                    best_cand_list.append(b_idx)
            
            # Pick one
            chosen = random.choice(best_cand_list)
            blocks_idx.append(chosen)
            
            # Update uncovered
            for ts_idx in k_to_ts[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    num_uncovered -= 1
                    
        return blocks_idx

    best_idx = None
    best_count = float('inf')
    
    restarts = 0
    while time.time() < deadline - 10.0:
        idxs = greedy_once(restarts)
        if len(idxs) < best_count:
            best_count = len(idxs)
            best_idx = idxs
        restarts += 1
        
    if not best_idx:
        # Fallback
        return [list(k_subsets[0])] * 100 # Dummy
        
    # Local Search
    current_idx = list(best_idx)
    # cover_count
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for idx in current_idx:
        for j in k_to_ts[idx]:
            cover_count[j] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_idx) - 1, -1, -1):
            idx_to_remove = current_idx[i]
            removable = True
            for j in k_to_ts[idx_to_remove]:
                if cover_count[j] == 1:
                    removable = False
                    break
            
            if removable:
                for j in k_to_ts[idx_to_remove]:
                    cover_count[j] -= 1
                current_idx.pop(i)
                improved = True
                break # Restart loop
                
    return [list(k_subsets[i]) for i in current_idx]
