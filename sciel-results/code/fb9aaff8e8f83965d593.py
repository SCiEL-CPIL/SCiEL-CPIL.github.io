import time
import random

UNIVERSE = 1 << 12
comp_mask = [0] * UNIVERSE

def _precompute():
    global comp_mask
    for i in range(UNIVERSE):
        m = 0
        for j in range(UNIVERSE):
            if i != j:
                if (i & j) == i or (i & j) == j:
                    m |= (1 << j)
        comp_mask[i] = m

_precompute()

def _count_score(S_mask, S_list):
    total = 0
    for i in S_list:
        total += (comp_mask[i] & S_mask).bit_count()
    return total // 2

def _local_search(S_list, S_mask, score):
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_out = -1
        best_in = -1
        
        for i_out in S_list:
            S_mask_wo = S_mask ^ (1 << i_out)
            loss = (comp_mask[i_out] & S_mask_wo).bit_count()
            
            for i_in in range(UNIVERSE):
                if (S_mask >> i_in) & 1:
                    continue
                gain = (comp_mask[i_in] & S_mask_wo).bit_count()
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_out = i_out
                    best_in = i_in
        
        if best_delta > 0:
            S_list.remove(best_out)
            S_list.append(best_in)
            S_mask = (S_mask ^ (1 << best_out)) | (1 << best_in)
            score += best_delta
            improved = True
            
    return S_list, S_mask, score

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best_score = -1
    best_S_list = []
    
    # 1. Heuristic Start
    chain_masks = [(1 << k) - 1 for k in range(13)]
    S_list = list(chain_masks)
    S_mask = 0
    for m in S_list:
        S_mask |= (1 << m)
    
    while len(S_list) < 40:
        best_gain = -1
        best_m = -1
        for m in range(UNIVERSE):
            if (S_mask >> m) & 1:
                continue
            gain = (comp_mask[m] & S_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_m = m
        S_list.append(best_m)
        S_mask |= (1 << best_m)
        
    score = _count_score(S_mask, S_list)
    S_list, S_mask, score = _local_search(S_list, S_mask, score)
    
    if score > best_score:
        best_score = score
        best_S_list = list(S_list)
        
    # 2. Random Restarts
    while time.time() < deadline:
        S_list = random.sample(range(UNIVERSE), 40)
        S_mask = 0
        for m in S_list:
            S_mask |= (1 << m)
        
        score = _count_score(S_mask, S_list)
        S_list, S_mask, score = _local_search(S_list, S_mask, score)
        
        if score > best_score:
            best_score = score
            best_S_list = list(S_list)
            
    result = []
    for m in best_S_list:
        result.append([b for b in range(12) if (m >> b) & 1])
        
    return result
