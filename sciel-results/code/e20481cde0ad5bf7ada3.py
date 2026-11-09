import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability matrix
    comp = [bytearray(total_masks) for _ in range(total_masks)]
    for a in range(total_masks):
        for b in range(a + 1, total_masks):
            ab = a & b
            if ab == a:  # a ⊆ b
                if a != b:
                    comp[a][b] = 1
                    comp[b][a] = 1
            elif ab == b:  # b ⊆ a
                if a != b:
                    comp[a][b] = 1
                    comp[b][a] = 1
    
    def compute_score(masks):
        count = 0
        for i in range(len(masks)):
            mi = masks[i]
            cmi = comp[mi]
            for j in range(i + 1, len(masks)):
                mj = masks[j]
                if cmi[mj]:
                    count += 1
        return count
    
    def local_search(initial_masks, time_limit):
        current = list(initial_masks)
        current_score = compute_score(current)
        current_set = set(current)
        
        score_with = [0] * total_masks
        for x in current:
            cx = comp[x]
            for k in range(total_masks):
                score_with[k] += cx[k]
        
        best_score = current_score
        best_solution = list(current)
        
        while time.time() < time_limit:
            improved = False
            for out_idx in range(m):
                out_mask = current[out_idx]
                lost = score_with[out_mask]
                cout = comp[out_mask]
                for in_mask in range(total_masks):
                    if in_mask in current_set:
                        continue
                    cin = comp[in_mask]
                    gained = score_with[in_mask] - cin[out_mask]
                    delta = gained - lost
                    if delta > 0:
                        current[out_idx] = in_mask
                        current_set.remove(out_mask)
                        current_set.add(in_mask)
                        current_score += delta
                        for k in range(total_masks):
                            score_with[k] += cin[k] - cout[k]
                        if current_score > best_score:
                            best_score = current_score
                            best_solution = list(current)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return best_solution, best_score
    
    def structured_initial():
        masks = []
        for size in [3, 4, 5, 6, 7, 8]:
            for mask in range(total_masks):
                if bin(mask).count('1') == size:
                    masks.append(mask)
                    if len(masks) >= m:
                        return masks
        return masks[:m]
    
    def random_initial():
        return random.sample(range(total_masks), m)
    
    best_overall = None
    best_overall_score = 0
    
    # Structured initial
    if time.time() < deadline - 15:
        init = structured_initial()
        tl = min(deadline - 10, time.time() + 25)
        sol, sc = local_search(init, tl)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = sol
    
    # Random restarts
    while time.time() < deadline - 5:
        init = random_initial()
        tl = min(deadline - 2, time.time() + 15)
        if tl <= time.time():
            break
        sol, sc = local_search(init, tl)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = sol
    
    if best_overall is None:
        best_overall = list(range(m))
    
    result = []
    for mask in best_overall:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    return result
