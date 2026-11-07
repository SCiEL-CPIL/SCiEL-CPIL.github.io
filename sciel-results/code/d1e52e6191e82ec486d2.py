import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    def is_subset(a, b):
        return (a & b) == a

    def calc_score(family):
        score = 0
        for i in range(len(family)):
            a = family[i]
            for j in range(i + 1, len(family)):
                b = family[j]
                if is_subset(a, b) or is_subset(b, a):
                    score += 1
        return score

    def greedy_init(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        family = []
        in_family = [False] * total_masks
        
        # Start with empty set
        family.append(0)
        in_family[0] = True
        
        while len(family) < m:
            best_mask = -1
            best_gain = -1
            # Shuffle masks to break symmetry randomly if needed?
            # No, just pick best. If tie, pick first.
            # To add randomness, we can perturb gains slightly or pick randomly from top K.
            # Let's stick to deterministic greedy first, then rely on restarts.
            
            for mask in range(total_masks):
                if in_family[mask]:
                    continue
                gain = 0
                for a in family:
                    if is_subset(mask, a) or is_subset(a, mask):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            if best_mask == -1:
                break
            family.append(best_mask)
            in_family[best_mask] = True
            
            if time.time() > deadline - 5.0:
                return family
        return family

    def local_search(family, in_family):
        current_score = calc_score(family)
        
        while time.time() < deadline - 2.0:
            contribs = [0] * len(family)
            for i in range(len(family)):
                a = family[i]
                for j in range(len(family)):
                    if i == j: continue
                    b = family[j]
                    if is_subset(a, b) or is_subset(b, a):
                        contribs[i] += 1
            
            best_delta = 0
            best_i = -1
            best_new_mask = -1
            
            for i in range(len(family)):
                old_mask = family[i]
                old_contrib = contribs[i]
                
                for new_mask in range(total_masks):
                    if in_family[new_mask]:
                        continue
                    
                    c_new_f = 0
                    for j in range(len(family)):
                        if j == i: continue
                        a = family[j]
                        if is_subset(new_mask, a) or is_subset(a, new_mask):
                            c_new_f += 1
                    
                    comp_with_old = 0
                    if is_subset(new_mask, old_mask) or is_subset(old_mask, new_mask):
                        comp_with_old = 1
                    
                    c_new_f_minus_old = c_new_f - comp_with_old
                    delta = c_new_f_minus_old - old_contrib
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_new_mask = new_mask
            
            if best_delta > 0:
                old_mask = family[best_i]
                in_family[old_mask] = False
                family[best_i] = best_new_mask
                in_family[best_new_mask] = True
                current_score += best_delta
            else:
                break
        
        return current_score, family, in_family

    best_score = -1
    best_family = None
    
    # Run multiple trials
    # Estimate time per trial. Greedy is fast. Local search is fast.
    # Let's do as many as we can.
    
    trial_count = 0
    while time.time() < deadline - 5.0:
        trial_count += 1
        seed = trial_count
        fam = greedy_init(seed)
        in_fam = [False] * total_masks
        for mask in fam:
            in_fam[mask] = True
        
        score, fam, in_fam = local_search(fam, in_fam)
        
        if score > best_score:
            best_score = score
            best_family = fam[:]
            
        # If we have very high score, maybe stop? 
        # Max possible score is binom(40, 2) = 780.
        # If we get close, we can stop.
        if best_score > 700:
            break

    if best_family is None:
        # Fallback
        best_family = list(range(m))

    result = []
    for mask in best_family:
        subset = []
        for b in range(n):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
        
    return result
