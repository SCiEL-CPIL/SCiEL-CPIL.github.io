import time
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability matrix
    # M[i][j] = 1 if i subset j or j subset i
    M = np.zeros((total_masks, total_masks), dtype=np.uint8)
    
    for i in range(total_masks):
        for j in range(total_masks):
            if i != j:
                if (i & j) == i or (i & j) == j:
                    M[i][j] = 1
    
    # Helper to calculate score
    def calc_score(fam):
        count = 0
        for i in range(m):
            for j in range(i+1, m):
                if M[fam[i]][fam[j]]:
                    count += 1
        return count

    # Helper to calculate degrees
    def calc_degrees(fam):
        deg = np.zeros(total_masks, dtype=np.int32)
        for x in fam:
            for y in fam:
                if x != y and M[x][y]:
                    deg[x] += 1
        return deg

    def run_sa(init_fam, time_limit):
        fam = init_fam[:]
        deg = calc_degrees(fam)
        in_fam = [False] * total_masks
        for x in fam:
            in_fam[x] = True
            
        current_score = calc_score(fam)
        best_score = current_score
        best_fam = fam[:]
        
        start_time = time.time()
        T = 5.0
        cooling = 0.999
        T_min = 0.01
        
        # Precompute list of all masks for random selection
        all_masks = list(range(total_masks))
        
        while time.time() - start_time < time_limit:
            # Pick random index
            i = random.randint(0, m - 1)
            old_val = fam[i]
            
            # Pick random new value not in fam
            # Optimization: just pick random, if in fam, retry
            while True:
                new_val = random.choice(all_masks)
                if not in_fam[new_val]:
                    break
            
            # Calculate delta
            # Loss: deg[old_val] (pairs old_val had with others)
            # Gain: deg[new_val] - M[new_val][old_val] (pairs new_val has with others, excluding old_val which is being removed)
            # Note: deg[new_val] is computed w.r.t current fam which includes old_val.
            # So deg[new_val] includes M[new_val][old_val].
            # We remove old_val, so the pair (new_val, old_val) is not formed in the new set.
            # So gain is deg[new_val] - M[new_val][old_val].
            
            loss = deg[old_val]
            gain = deg[new_val] - M[new_val][old_val]
            delta = gain - loss
            
            accept = False
            if delta >= 0:
                accept = True
            else:
                # Boltzmann factor
                # exp(delta / T)
                # delta is negative
                if random.random() < pow(2.718281828, delta / T):
                    accept = True
            
            if accept:
                # Perform swap
                fam[i] = new_val
                in_fam[old_val] = False
                in_fam[new_val] = True
                
                # Update degrees
                # For all y in fam (except old_val, now new_val is there)
                # Actually, for all y in current fam (which includes new_val, excludes old_val)
                # deg[y] changes by M[new_val][y] - M[old_val][y]
                
                # We need to iterate over all y in fam
                # Note: deg[new_val] itself needs to be updated?
                # deg[new_val] in the new context is the number of comparable sets in the new fam.
                # Let's just recalculate deg for all affected nodes?
                # Affected nodes: all y in fam.
                
                for y in fam:
                    # y is in the new fam
                    # deg[y] was based on old fam.
                    # Change: remove old_val, add new_val.
                    # delta_deg = M[y][new_val] - M[y][old_val]
                    # Note M is symmetric.
                    deg[y] += M[y][new_val] - M[y][old_val]
                
                # Update current score
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_fam = fam[:]
                
                T *= cooling
                if T < T_min:
                    T = T_min
            else:
                T *= cooling
                if T < T_min:
                    T = T_min
            
            # Optional: Annealing schedule restart?
            # If T is very low, we might get stuck.
            # Simple restart: if no improvement for long, reset T?
            # Or just let it cool. With short time limits, cooling is fast.
            
        return best_fam, best_score

    # Initialization 1: Chain + Greedy
    def init_chain_greedy():
        fam = []
        cur = 0
        fam.append(cur)
        for i in range(n):
            cur |= (1 << i)
            fam.append(cur)
        
        used = set(fam)
        # Greedy add
        for _ in range(m - len(fam)):
            best_gain = -1
            best_mask = -1
            # Sample candidates to speed up? Or iterate all?
            # Iterating all 4096 is fast enough for 27 steps.
            for s in range(total_masks):
                if s in used:
                    continue
                gain = 0
                for f in fam:
                    if M[s][f]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            fam.append(best_mask)
            used.add(best_mask)
        return fam

    # Initialization 2: Random
    def init_random():
        fam = random.sample(range(total_masks), m)
        return fam

    best_overall_fam = []
    best_overall_score = -1
    
    # Time budget management
    total_time_limit = 55.0 # Leave 5s for safety
    start_total = time.time()
    
    # Run multiple SA instances
    # 1. Chain Greedy
    fam1 = init_chain_greedy()
    fam1, score1 = run_sa(fam1, 10.0)
    if score1 > best_overall_score:
        best_overall_score = score1
        best_overall_fam = fam1
    
    # 2. Random Start (a few times)
    for _ in range(3):
        if time.time() - start_total > total_time_limit:
            break
        fam_r = init_random()
        # Quick local search to improve random start
        fam_r, score_r = run_sa(fam_r, 5.0)
        if score_r > best_overall_score:
            best_overall_score = score_r
            best_overall_fam = fam_r
            
    # 3. Perturb best found
    if time.time() - start_total < total_time_limit:
        # Take best, perturb 5 elements
        fam_p = best_overall_fam[:]
        used = set(fam_p)
        for _ in range(5):
            i = random.randint(0, m-1)
            old = fam_p[i]
            used.remove(old)
            # Pick random new
            while True:
                new = random.choice(range(total_masks))
                if new not in used:
                    break
            fam_p[i] = new
            used.add(new)
            
        fam_p, score_p = run_sa(fam_p, 10.0)
        if score_p > best_overall_score:
            best_overall_score = score_p
            best_overall_fam = fam_p

    # Final Polish: Run SA on best found for remaining time
    remaining = total_time_limit - (time.time() - start_total)
    if remaining > 1.0:
        best_overall_fam, _ = run_sa(best_overall_fam, remaining)

    # Convert masks to lists
    result = []
    for mask in best_overall_fam:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result
