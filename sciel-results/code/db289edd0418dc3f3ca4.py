import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 55.0
    n = 12
    m = 40
    total = 1 << n
    
    # Precompute comparability matrix
    # comp[i, j] is 1 if i and j are comparable and i != j
    masks = np.arange(total, dtype=np.int32)
    
    comp = np.zeros((total, total), dtype=np.uint8)
    for i in range(total):
        # j is comparable to i if i subset j or j subset i
        # i subset j: (i & j) == i
        # j subset i: (i & j) == j
        inter = i & masks
        row = (inter == i) | (inter == masks)
        row[i] = False
        comp[i, :] = row
        
    # Initial solution: All subsets of {0,1,2,3,4} (masks 0-31)
    # Plus 8 supersets involving element 5
    initial = list(range(32))
    for i in range(8):
        initial.append(31 | (1 << (5 + i)))
        
    # Ensure distinct and exactly 40
    seen = set()
    selected = []
    for mask in initial:
        if mask not in seen:
            seen.add(mask)
            selected.append(mask)
    while len(selected) < m:
        mask = np.random.randint(0, total)
        if mask not in seen:
            seen.add(mask)
            selected.append(mask)
    selected = selected[:m]
    
    sel_idx = np.array(selected, dtype=np.int32)
    
    def get_score(sel_idx):
        sel_vec = np.zeros(total, dtype=np.uint8)
        sel_vec[sel_idx] = 1
        counts = comp @ sel_vec
        return int(np.sum(counts[sel_idx]) // 2)
        
    current_score = get_score(sel_idx)
    best_score = current_score
    best_sel = sel_idx.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = np.arange(m)
        
        for idx in indices:
            if time.time() >= deadline:
                break
            
            old = sel_idx[idx]
            
            # Create selection vector without 'old'
            sel_vec = np.zeros(total, dtype=np.uint8)
            sel_vec[sel_idx] = 1
            sel_vec[old] = 0
            
            # Compute neighbors count for all candidates
            neighbors_count = comp @ sel_vec
            
            # Loss is the number of comparable pairs 'old' had with other selected items
            loss = neighbors_count[old]
            
            # Gains for adding j
            scores = neighbors_count.copy()
            # Mask out selected items (including old, which is already 0 in sel_vec, 
            # but we need to ensure we don't pick any currently selected item)
            # Actually, sel_vec has 0 for old and other selected items have 1.
            # We want to pick an item with sel_vec[j] == 0.
            scores[sel_vec.astype(bool)] = -1
            
            best_candidate = np.argmax(scores)
            best_gain = scores[best_candidate]
            
            if best_gain > loss:
                sel_idx[idx] = best_candidate
                current_score += best_gain - loss
                if current_score > best_score:
                    best_score = current_score
                    best_sel = sel_idx.copy()
                improved = True
                
    result = []
    for mask in best_sel:
        result.append([int(b) for b in range(n) if (mask >> b) & 1])
    return result
