import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    base = 51975
    # Generate odd divisors > 1
    moduli = [d for d in range(3, base + 1, 2) if base % d == 0]
    moduli_set = set(moduli)
    
    # Identify "anchor" moduli: the largest prime-power factors
    anchors = [27, 25, 7, 11]
    non_anchors = [m for m in moduli if m not in anchors]
    
    best_uncovered = float('inf')
    best_residues = None
    
    def compute_uncovered(count_cover):
        return int(np.sum(count_cover == 0))
    
    def greedy_initialize(order, seed):
        """Greedy initialization processing moduli in given order."""
        random.seed(seed)
        count_cover = np.zeros(base, dtype=np.int8)
        residues = {}
        
        for m in order:
            best_r = 0
            best_score = -1
            for r in range(m):
                s = int(np.sum(count_cover[r::m] == 0))
                if s > best_score:
                    best_score = s
                    best_r = r
            residues[m] = best_r
            count_cover[best_r::m] += 1
        
        uncovered = compute_uncovered(count_cover)
        return residues, count_cover, uncovered
    
    def local_search(residues, count_cover, current_uncovered, time_budget):
        """Simulated annealing local search."""
        random.seed(int(time.time() * 1000) % 100000)
        temperature = 30.0
        cooling = 0.997
        min_temperature = 0.05
        best_local = current_uncovered
        best_local_residues = residues.copy()
        
        while temperature > min_temperature and time.time() < deadline - time_budget:
            # Choose move type: single (70%) or pairwise swap (30%)
            if random.random() < 0.7:
                m = random.choice(moduli)
                r_old = residues[m]
                r_new = random.randint(0, m - 1)
                if r_new == r_old:
                    continue
                
                count_cover[r_old::m] -= 1
                old_uncovered = compute_uncovered(count_cover)
                
                count_cover[r_new::m] += 1
                new_uncovered = compute_uncovered(count_cover)
                
                delta = new_uncovered - old_uncovered
                
                if delta <= 0 or (delta > 0 and random.random() < np.exp(-delta / temperature)):
                    residues[m] = r_new
                    current_uncovered = new_uncovered
                    if current_uncovered < best_local:
                        best_local = current_uncovered
                        best_local_residues = residues.copy()
                else:
                    count_cover[r_new::m] -= 1
                    count_cover[r_old::m] += 1
            else:
                m1, m2 = random.sample(moduli, 2)
                r1_old = residues[m1]
                r2_old = residues[m2]
                
                count_cover[r1_old::m1] -= 1
                count_cover[r2_old::m2] -= 1
                
                count_cover[r2_old::m1] += 1
                count_cover[r1_old::m2] += 1
                
                new_uncovered = compute_uncovered(count_cover)
                delta = new_uncovered - current_uncovered
                
                if delta <= 0 or (delta > 0 and random.random() < np.exp(-delta / temperature)):
                    residues[m1] = r2_old
                    residues[m2] = r1_old
                    current_uncovered = new_uncovered
                    if current_uncovered < best_local:
                        best_local = current_uncovered
                        best_local_residues = residues.copy()
                else:
                    count_cover[r2_old::m1] -= 1
                    count_cover[r1_old::m2] -= 1
                    count_cover[r1_old::m1] += 1
                    count_cover[r2_old::m2] += 1
            
            temperature *= cooling
        
        return best_local_residues, best_local
    
    def deterministic_polish(residues, count_cover, current_uncovered):
        """Deterministic coordinate descent pass."""
        for m in moduli:
            if time.time() > deadline - 5:
                break
            
            r_old = residues[m]
            count_others = count_cover.copy()
            count_others[r_old::m] -= 1
            
            best_r = r_old
            best_score = int(np.sum(count_others[r_old::m] == 0))
            
            if m <= 1000:
                candidates = range(m)
            else:
                candidates = [r_old] + random.sample(range(m), min(100, m - 1))
            
            for r in candidates:
                if r == r_old:
                    continue
                s = int(np.sum(count_others[r::m] == 0))
                if s > best_score:
                    best_score = s
                    best_r = r
            
            if best_r != r_old:
                count_cover[r_old::m] -= 1
                count_cover[best_r::m] += 1
                residues[m] = best_r
        
        final_uncovered = compute_uncovered(count_cover)
        return residues, final_uncovered
    
    # Try dual greedy: ascending and descending order
    # Ascending: small moduli first
    asc_order = sorted(non_anchors) + anchors
    # Descending: large moduli first  
    desc_order = sorted(non_anchors, reverse=True) + anchors
    
    # Try multiple seeds with each order
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        
        # Ascending greedy
        res_asc, cnt_asc, unc_asc = greedy_initialize(asc_order, seed * 1000 + 1)
        if unc_asc < best_uncovered:
            best_uncovered = unc_asc
            best_residues = res_asc.copy()
        
        # Descending greedy
        res_desc, cnt_desc, unc_desc = greedy_initialize(desc_order, seed * 1000 + 2)
        if unc_desc < best_uncovered:
            best_uncovered = unc_desc
            best_residues = res_desc.copy()
    
    # Run local search from the best greedy solution
    if best_residues is not None:
        # Reconstruct count_cover from best_residues
        count_cover = np.zeros(base, dtype=np.int8)
        for m, r in best_residues.items():
            count_cover[r::m] += 1
        current_uncovered = compute_uncovered(count_cover)
        
        # Simulated annealing
        residues_ls, best_local = local_search(
            best_residues.copy(), count_cover, current_uncovered, time_budget=15
        )
        if best_local < best_uncovered:
            best_uncovered = best_local
            best_residues = residues_ls.copy()
        
        # Deterministic polish
        if time.time() < deadline - 10:
            count_cover2 = np.zeros(base, dtype=np.int8)
            for m, r in best_residues.items():
                count_cover2[r::m] += 1
            residues_polished, unc_polished = deterministic_polish(
                best_residues.copy(), count_cover2, compute_uncovered(count_cover2)
            )
            if unc_polished < best_uncovered:
                best_uncovered = unc_polished
                best_residues = residues_polished.copy()
    
    # Final fallback if nothing worked
    if best_residues is None:
        best_residues = {}
        count_cover = np.zeros(base, dtype=np.int8)
        for m in sorted(moduli):
            best_r = 0
            best_score = -1
            for r in range(m):
                s = int(np.sum(count_cover[r::m] == 0))
                if s > best_score:
                    best_score = s
                    best_r = r
            best_residues[m] = best_r
            count_cover[best_r::m] += 1
    
    classes = [[m, best_residues[m]] for m in moduli]
    return classes
