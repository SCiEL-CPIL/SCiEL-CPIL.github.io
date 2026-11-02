import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115
    
    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all blocks and their t-subset indices
    all_blocks = []
    block_tsub_indices = []
    block_to_idx = {}
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    
    for block in itertools.combinations(range(v), k):
        b = list(block)
        b_idx = len(all_blocks)
        all_blocks.append(b)
        block_to_idx[tuple(b)] = b_idx
        
        t_indices = []
        for ts in itertools.combinations(b, t):
            t_idx = tsub_to_idx[ts]
            t_indices.append(t_idx)
            tsub_to_blocks[t_idx].append(b_idx)
            
        block_tsub_indices.append(t_indices)
        
    # Convert to NumPy array for fast slicing
    block_tsub_indices_np = np.array(block_tsub_indices, dtype=np.int32)
    num_blocks = len(all_blocks)
    
    def greedy_cover(seed):
        r = random.Random(seed)
        counts = np.zeros(num_tsubs, dtype=np.uint16)
        blocks = []
        uncovered_mask = np.ones(num_tsubs, dtype=bool)
        
        while np.any(uncovered_mask):
            # Pick random uncovered t-subset
            while True:
                idx = r.randrange(num_tsubs)
                if uncovered_mask[idx]:
                    break
            
            target_t_idx = idx
            candidates = tsub_to_blocks[target_t_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                t_indices = block_tsub_indices_np[b_idx]
                gain = np.sum(counts[t_indices] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    
            if best_block_idx != -1:
                blocks.append(best_block_idx)
                t_indices = block_tsub_indices_np[best_block_idx]
                counts[t_indices] += 1
                uncovered_mask[t_indices] = (counts[t_indices] == 0)
            else:
                break
                
        return blocks

    def local_search(blocks, seed):
        r = random.Random(seed)
        blocks = list(blocks)
        counts = np.zeros(num_tsubs, dtype=np.uint16)
        
        for b_idx in blocks:
            counts[block_tsub_indices_np[b_idx]] += 1
            
        improved = True
        no_improve_count = 0
        
        while improved and (time.time() - start_time < time_limit * 0.9):
            improved = False
            
            # 1. Remove redundant blocks
            indices = list(range(len(blocks)))
            r.shuffle(indices)
            
            for i in indices:
                b_idx = blocks[i]
                t_indices = block_tsub_indices_np[b_idx]
                if np.all(counts[t_indices] > 1):
                    counts[t_indices] -= 1
                    blocks.pop(i)
                    improved = True
                    break
                    
            if improved:
                continue
                
            # 2. Perturbation
            if not blocks:
                break
                
            perturbed = False
            for _ in range(10):
                if time.time() - start_time > time_limit * 0.95:
                    break
                
                old_len = len(blocks)
                old_counts = counts.copy()
                old_blocks = blocks.copy()
                
                i = r.randrange(len(blocks))
                b_idx = blocks[i]
                
                counts[block_tsub_indices_np[b_idx]] -= 1
                blocks.pop(i)
                
                uncovered_indices = np.where(counts == 0)[0]
                
                while len(uncovered_indices) > 0:
                    if time.time() - start_time > time_limit * 0.95:
                        break
                    u_idx = r.choice(uncovered_indices.tolist())
                    candidates = tsub_to_blocks[u_idx]
                    best_block_idx = -1
                    best_gain = -1
                    
                    for cb_idx in candidates:
                        if cb_idx == b_idx:
                            continue
                        t_indices = block_tsub_indices_np[cb_idx]
                        gain = np.sum(counts[t_indices] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_block_idx = cb_idx
                            
                    if best_block_idx == -1:
                        best_block_idx = b_idx
                    else:
                        blocks.append(best_block_idx)
                        counts[block_tsub_indices_np[best_block_idx]] += 1
                        uncovered_indices = np.where(counts == 0)[0]
                        
                if len(blocks) < old_len:
                    perturbed = True
                    improved = True
                    break
                else:
                    blocks = old_blocks
                    counts = old_counts
                    
            if not perturbed:
                break

        return blocks

    best_blocks = None
    best_len = float('inf')
    
    num_iterations = 0
    while time.time() - start_time < time_limit * 0.8:
        num_iterations += 1
        seed = num_iterations
        blocks = greedy_cover(seed)
        blocks = local_search(blocks, seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [all_blocks[i] for i in blocks]
            
    if best_blocks:
        best_indices = [block_to_idx[tuple(b)] for b in best_blocks]
        best_indices = local_search(best_indices, 999)
        best_blocks = [all_blocks[i] for i in best_indices]
        
    if best_blocks is None:
        best_blocks = []
        
    cov = defaultdict(int)
    for b in best_blocks:
        for ts in itertools.combinations(b, t):
            cov[ts] += 1
            
    for ts in all_tsubs:
        if cov.get(ts, 0) == 0:
            b = list(ts)
            r = random.Random(999)
            while len(b) < k:
                x = r.choice([p for p in all_points if p not in b])
                b.append(x)
            b.sort()
            best_blocks.append(b)
            
    return best_blocks
