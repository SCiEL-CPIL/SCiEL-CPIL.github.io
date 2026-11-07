import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 12
    m = 40
    total = 1 << n
    
    # Precompute comparability as bitmasks (symmetric)
    comp = [0] * total
    for i in range(total):
        for j in range(i + 1, total):
            if (i & j) == i or (i & j) == j:
                comp[i] |= (1 << j)
                comp[j] |= (1 << i)
    
    def score_of(sel, sel_mask):
        s = 0
        for x in sel:
            s += (comp[x] & sel_mask).bit_count()
        return s // 2
    
    def make_initial(seed):
        rng = random.Random(seed)
        initial = []
        initial.append(0)
        for i in range(n):
            initial.append(1 << i)
        for i in range(1, n):
            initial.append((1 << 0) | (1 << i))
        for i in range(2, n):
            initial.append((1 << 0) | (1 << 1) | (1 << i))
        for i in range(3, n):
            initial.append((1 << 0) | (1 << 1) | (1 << 2) | (1 << i))
        seen = set()
        selected = []
        for mask in initial:
            if mask not in seen:
                seen.add(mask)
                selected.append(mask)
        if len(selected) >= m:
            rng.shuffle(selected)
            selected = selected[:m]
        else:
            all_masks = list(range(total))
            rng.shuffle(all_masks)
            for mask in all_masks:
                if mask not in seen:
                    seen.add(mask)
                    selected.append(mask)
                    if len(selected) == m:
                        break
        return selected
    
    best_score = -1
    best_sel = None
    
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        selected = make_initial(r)
        sel_mask = 0
        for x in selected:
            sel_mask |= (1 << x)
        current = score_of(selected, sel_mask)
        best_local = current
        best_local_sel = list(selected)
        
        # Simulated annealing with single swaps
        temperature = 10.0
        cooling = 0.9995
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(m):
                if time.time() >= deadline - 2.0:
                    break
                old = selected[idx]
                sel_wo = sel_mask & ~(1 << old)
                loss = (comp[old] & sel_wo).bit_count()
                
                new = random.randrange(total)
                if (sel_mask >> new) & 1:
                    continue
                gain = (comp[new] & sel_wo).bit_count()
                delta = gain - loss
                
                accept = False
                if delta >= 0:
                    accept = True
                else:
                    ratio = delta / max(temperature, 0.1)
                    if ratio > -50:
                        accept = random.random() < pow(2.718281828, ratio)
                
                if accept:
                    selected[idx] = new
                    sel_mask = (sel_mask & ~(1 << old)) | (1 << new)
                    current += delta
                    if current > best_local:
                        best_local = current
                        best_local_sel = list(selected)
                    improved = True
                    break
            temperature *= cooling
        
        if best_local > best_score:
            best_score = best_local
            best_sel = best_local_sel
    
    # Final greedy polish: try all candidates for each position
    if best_sel is not None and time.time() < deadline - 1.0:
        selected = list(best_sel)
        sel_mask = 0
        for x in selected:
            sel_mask |= (1 << x)
        current = best_score
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for idx in range(m):
                if time.time() >= deadline - 0.5:
                    break
                old = selected[idx]
                sel_wo = sel_mask & ~(1 << old)
                loss = (comp[old] & sel_wo).bit_count()
                best_gain = -1
                best_new = None
                for new in range(total):
                    if (sel_mask >> new) & 1:
                        continue
                    gain = (comp[new] & sel_wo).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new
                if best_new is not None and best_gain > loss:
                    selected[idx] = best_new
                    sel_mask = (sel_mask & ~(1 << old)) | (1 << best_new)
                    current += best_gain - loss
                    improved = True
                    break
        
        if current > best_score:
            best_score = current
            best_sel = selected
    
    result = []
    for mask in best_sel:
        result.append([b for b in range(n) if (mask >> b) & 1])
    return [[int(x) for x in sl] for sl in result]
