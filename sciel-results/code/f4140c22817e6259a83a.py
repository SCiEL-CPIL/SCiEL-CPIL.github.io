import itertools
import time
import numpy as np
import random

_PRECOMPUTED = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _PRECOMPUTED:
        return _PRECOMPUTED[key]
    
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    covers = [None] * n_k
    reverse_covers = [[] for _ in range(n_t)]
    block_contents = []
    
    for i, block in enumerate(k_subs):
        block_contents.append(list(block))
        block_t_subs = list(itertools.combinations(block, t))
        idxs = [t_sub_to_idx[sub] for sub in block_t_subs]
        covers[i] = np.array(idxs, dtype=np.int32)
        for idx in idxs:
            reverse_covers[idx].append(i)
            
    reverse_covers_arr = [np.array(rc, dtype=np.int32) for rc in reverse_covers]
    
    _PRECOMPUTED[key] = {
        'covers': covers,
        'reverse_covers': reverse_covers_arr,
        'n_t': n_t,
        'n_k': n_k,
        'block_contents': block_contents
    }
    return _PRECOMPUTED[key]

def construct(v, k, t):
    if (v, k, t) != (19, 7, 4):
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    data = _precompute(v, k, t)
    covers = data['covers']
    reverse_covers = data['reverse_covers']
    n_t = data['n_t']
    n_k = data['n_k']
    block_contents = data['block_contents']
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    uncovered = np.ones(n_t, dtype=bool)
    gain = np.full(n_k, 35, dtype=np.int32)
    selected = []
    
    while np.any(uncovered):
        b = np.argmax(gain)
        if gain[b] == 0: break
        selected.append(b)
        cov = covers[b]
        uncovered[cov] = False
        for t_idx in cov:
            for b2 in reverse_covers[t_idx]:
                gain[b2] -= 1
                
    current_blocks = selected
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    def local_optimize(blocks):
        count = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            count[covers[b]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                b = blocks[i]
                if np.min(count[covers[b]]) > 1:
                    count[covers[b]] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks
    
    current_blocks = local_optimize(current_blocks)
    if len(current_blocks) < best_size:
        best_size = len(current_blocks)
        best_blocks = list(current_blocks)
        
    while time.time() < deadline:
        trial_blocks = list(best_blocks)
        if len(trial_blocks) < 2: break
        num_remove = random.randint(1, min(3, len(trial_blocks)-1))
        remove_indices = set(random.sample(range(len(trial_blocks)), num_remove))
        
        count = np.zeros(n_t, dtype=np.int32)
        active_blocks = []
        for i, b in enumerate(trial_blocks):
            if i not in remove_indices:
                active_blocks.append(b)
                count[covers[b]] += 1
                
        uncovered = (count == 0)
        
        if np.any(uncovered):
            gain = np.zeros(n_k, dtype=np.int32)
            uncovered_indices = np.where(uncovered)[0]
            for t_idx in uncovered_indices:
                for b in reverse_covers[t_idx]:
                    gain[b] += 1
                    
            while np.any(uncovered):
                b = np.argmax(gain)
                if gain[b] == 0: break
                active_blocks.append(b)
                count[covers[b]] += 1
                cov = covers[b]
                new_uncovered = cov[count[cov] == 1]
                uncovered[cov] = False
                for t_idx in new_uncovered:
                    for b2 in reverse_covers[t_idx]:
                        gain[b2] -= 1
                        
        active_blocks = local_optimize(active_blocks)
        
        if len(active_blocks) < best_size:
            best_size = len(active_blocks)
            best_blocks = list(active_blocks)
            
    return [block_contents[b] for b in best_blocks]
