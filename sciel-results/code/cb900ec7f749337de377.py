import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    indices = list(range(D))
    
    # Pre-generate W3 vectors
    w3_list = []
    for combo in itertools.combinations(indices, 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            w3_list.append(v)
    w3_vectors = np.array(w3_list)
    n_w3 = len(w3_vectors)
    
    # Pre-extract pairs for W3 to avoid repeated zero-checking
    w3_pairs = []
    for v in w3_vectors:
        pairs = set()
        for k in range(D):
            if v[k] != 0:
                pairs.add((k, v[k]))
        w3_pairs.append(pairs)
        
    # Pre-generate W2 vectors
    w2_list = []
    for i, j in itertools.combinations(indices, 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                w2_list.append(v)
    w2_vectors = np.array(w2_list)
    n_w2 = len(w2_vectors)
    
    # Pre-extract pairs for W2
    w2_pairs = []
    for v in w2_vectors:
        pairs = set()
        for k in range(D):
            if v[k] != 0:
                pairs.add((k, v[k]))
        w2_pairs.append(pairs)
        
    best_w3_sel = []
    best_w2_sel = []
    best_score = 0
    
    # Try a few random orderings
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        perm = np.random.permutation(n_w3)
        w3_idx_shuffled = perm
        
        selected_w3_indices = []
        selected_pairs = []
        
        for idx in w3_idx_shuffled:
            pairs = w3_pairs[idx]
            conflict = False
            for sp in selected_pairs:
                if len(pairs & sp) >= 2:
                    conflict = True
                    break
            
            if not conflict:
                selected_w3_indices.append(idx)
                selected_pairs.append(pairs)
                
        # Build forbidden W2 pairs
        forbidden_w2_pairs = set()
        for sp in selected_pairs:
            sp_list = list(sp)
            for a in range(3):
                for b in range(a+1, 3):
                    p1 = sp_list[a]
                    p2 = sp_list[b]
                    if p1 > p2:
                        p1, p2 = p2, p1
                    forbidden_w2_pairs.add((p1, p2))
                    
        selected_w2_indices = []
        for i in range(n_w2):
            pairs = w2_pairs[i]
            p_list = list(pairs)
            p1, p2 = p_list
            if p1 > p2:
                p1, p2 = p2, p1
            if (p1, p2) not in forbidden_w2_pairs:
                selected_w2_indices.append(i)
                
        score = len(selected_w3_indices) + len(selected_w2_indices)
        if score > best_score:
            best_score = score
            best_w3_sel = [w3_vectors[i] for i in selected_w3_indices]
            best_w2_sel = [w2_vectors[i] for i in selected_w2_indices]
            
    if best_w3_sel:
        if best_w2_sel:
            final_vectors = np.vstack([np.array(best_w3_sel), np.array(best_w2_sel)])
        else:
            final_vectors = np.array(best_w3_sel)
    else:
        final_vectors = np.array(best_w2_sel)
        
    return final_vectors
