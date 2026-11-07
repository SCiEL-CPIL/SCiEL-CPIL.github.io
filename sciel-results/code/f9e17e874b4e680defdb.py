"""Improved program: Greedy Initialization with Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = P['n']
    m = P['m']
    N = 1 << n
    
    # Precompute comparability bitmasks
    # Comp[i] is an integer where bit j is set if i and j are comparable (including i itself)
    Comp = [0] * N
    
    for i in range(N):
        mask = 0
        # i is comparable with all subsets of i
        sub = i
        while True:
            mask |= (1 << sub)
            if sub == 0:
                break
            sub = (sub - 1) & i
        # i is comparable with all supersets of i
        super_comp = (N - 1) ^ i
        sup = 0
        while True:
            s = i | sup
            mask |= (1 << s)
            if sup == 0:
                break
            sup = (sup - 1) & super_comp
        Comp[i] = mask
        
    # Greedy Initialization
    def greedy_init():
        fam = []
        fam_m = 0
        all_masks = list(range(N))
        # Randomize the order of all_masks to break ties and allow diversity in restarts
        random.shuffle(all_masks)
        
        for _ in range(m):
            best_cand = -1
            best_score = -1
            # To avoid bias from the fixed order, we could shuffle again, 
            # but iterating through all is deterministic given the mask.
            # The tie-breaking is the order in all_masks.
            for c in all_masks:
                if (fam_m >> c) & 1:
                    continue
                score = (Comp[c] & fam_m).bit_count()
                if score > best_score:
                    best_score = score
                    best_cand = c
            
            if best_cand != -1:
                fam.append(best_cand)
                fam_m |= (1 << best_cand)
            else:
                break
        return fam, fam_m

    # Simulated Annealing
    def run_sa(initial_fam, initial_fam_m, max_time):
        fam = initial_fam[:]
        fam_m = initial_fam_m
        current_score = 0
        for mask in fam:
            current_score += (Comp[mask] & fam_m).bit_count()
        current_score //= 2
        
        best_score = current_score
        best_fam = fam[:]
        
        T = 500.0
        cooling_rate = 0.9995
        min_T = 0.5
        
        # Estimate steps per second to adjust if needed, but we just loop until time runs out
        while T > min_T:
            if time.time() > max_time:
                break
                
            idx = random.randint(0, m - 1)
            old_mask = fam[idx]
            
            new_mask = random.randint(0, N - 1)
            if (fam_m >> new_mask) & 1:
                continue
            
            S_mask = fam_m & ~(1 << old_mask)
            comp_old = (Comp[old_mask] & S_mask).bit_count()
            comp_new = (Comp[new_mask] & S_mask).bit_count()
            
            delta = comp_new - comp_old
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                fam[idx] = new_mask
                fam_m = (fam_m & ~(1 << old_mask)) | (1 << new_mask)
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_fam = fam[:]
            
            T *= cooling_rate
            
        return best_fam, best_score

    best_overall_fam = []
    best_overall_score = -1
    
    # Run multiple restarts to ensure we find a good solution
    # Greedy is fast, so we can afford a few restarts
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        init_fam, init_fam_m = greedy_init()
        # Run SA for a portion of the remaining time
        # Or just run SA until T cools down, but check time in loop.
        # The loop inside run_sa checks time.
        # We want to distribute time.
        # Let's just run SA for the full cooling schedule, but it will break early if time runs out.
        # To make it fair, we can limit the number of steps per restart.
        
        # Actually, just running the standard SA loop is fine.
        # But if we do 5 restarts, each SA should have a time limit.
        time_per_restart = (deadline - time.time()) / (num_restarts - r)
        
        res_fam, res_score = run_sa(init_fam, init_fam_m, time.time() + time_per_restart)
        
        if res_score > best_overall_score:
            best_overall_score = res_score
            best_overall_fam = res_fam
            
    family = best_overall_fam
    
    # Convert to required format
    result = []
    for mask in family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
