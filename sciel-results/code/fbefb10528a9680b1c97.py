import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    # 1. Initial Construction: Core + Tail
    core = []
    for mask in range(32):
        s = []
        for b in range(5):
            if (mask >> b) & 1:
                s.append(b)
        core.append(s)
    
    base = [0, 1, 2, 3, 4]
    tail = []
    for i in range(8):
        s = base.copy()
        for j in range(5, 5 + i + 1):
            s.append(j)
        tail.append(s)
        
    result = core + tail
    
    # Convert to bitmasks for efficient subset checks
    # Universe is 0..11
    def to_mask(lst):
        m = 0
        for v in lst:
            m |= (1 << v)
        return m
        
    masks = [to_mask(s) for s in result]
    
    # Helper to calculate score of a list of masks
    def calc_score(m_list):
        score = 0
        n = len(m_list)
        for i in range(n):
            mi = m_list[i]
            for j in range(i + 1, n):
                mj = m_list[j]
                # Check if mi is strict subset of mj
                if (mi & mj) == mi and mi != mj:
                    score += 1
                # Check if mj is strict subset of mi
                elif (mi & mj) == mj and mi != mj:
                    score += 1
        return score

    # Local Search
    current_score = calc_score(masks)
    mask_set = set(masks)
    
    # Precompute neighbors? No, generate on the fly.
    # Neighbor generation: flip one bit in 0..11
    # Also, to escape local optima, occasionally try a random mask?
    # Let's stick to 1-bit flips first.
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for i in range(40):
            if time.time() > deadline:
                break
            m_i = masks[i]
            # Generate neighbors
            neighbors = []
            for b in range(12):
                # Flip bit b
                new_m = m_i ^ (1 << b)
                # Check if new_m is already in the set (excluding i)
                # Optimization: if new_m is in mask_set, and it's not m_i, skip
                # But we need to know if it's m_i.
                if new_m != m_i:
                    # Check if new_m is in the current set of masks
                    # We can't just check mask_set because m_i is in mask_set.
                    # If new_m is in mask_set, it might be m_i (no, we excluded) or another mask.
                    # If it's another mask, it's a duplicate -> invalid.
                    # So if new_m in mask_set, skip.
                    if new_m in mask_set:
                        continue
                    neighbors.append(new_m)
            
            # Evaluate best neighbor
            best_neighbor = -1
            best_score = current_score
            for nm in neighbors:
                # Calculate delta score
                # Remove contributions of i, add contributions of nm
                # Old contributions: pairs (i, j)
                old_contrib = 0
                for j in range(40):
                    if j == i: continue
                    mj = masks[j]
                    if (m_i & mj) == m_i and m_i != mj: old_contrib += 1
                    elif (m_i & mj) == mj and m_i != mj: old_contrib += 1
                
                new_contrib = 0
                for j in range(40):
                    if j == i: continue
                    mj = masks[j]
                    if (nm & mj) == nm and nm != mj: new_contrib += 1
                    elif (nm & mj) == mj and nm != mj: new_contrib += 1
                
                new_score = current_score - old_contrib + new_contrib
                if new_score > best_score:
                    best_score = new_score
                    best_neighbor = nm
            
            if best_neighbor != -1:
                # Update
                mask_set.remove(m_i)
                mask_set.add(best_neighbor)
                masks[i] = best_neighbor
                current_score = best_score
                improved = True
                # Continue to next i or restart? 
                # Usually, restart or continue. Continuing is fine.
                
    # Convert back to lists
    final_result = []
    for m in masks:
        s = []
        for b in range(12):
            if (m >> b) & 1:
                s.append(b)
        final_result.append(s)
        
    return final_result
