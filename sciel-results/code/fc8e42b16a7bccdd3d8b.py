import time
import random
import math
import numpy as np

N = 51975

def get_divisors(n):
    divs = []
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.append(n)
    return sorted(divs)

MODULI = get_divisors(N)
K = len(MODULI)
MODULI_DESC = sorted(MODULI, reverse=True)

def compute_counts(residues):
    counts = np.zeros(N, dtype=np.int32)
    for m, r in zip(MODULI, residues):
        counts[r::m] += 1
    return counts

def greedy_build_descending():
    """Greedy: process moduli in descending order, pick best residue for each."""
    covered = np.zeros(N, dtype=np.uint8)
    residues = [0] * K
    # Map from modulus to index in MODULI
    mod_to_idx = {m: i for i, m in enumerate(MODULI)}
    
    for m in MODULI_DESC:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = int(np.sum(1 - covered[r::m]))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[mod_to_idx[m]] = best_r
        covered[best_r::m] = 1
    
    counts = compute_counts(residues)
    score = int(np.sum(counts == 0))
    return residues, counts, score

def greedy_build_ascending():
    """Greedy: process moduli in ascending order, pick best residue for each."""
    covered = np.zeros(N, dtype=np.uint8)
    residues = [0] * K
    mod_to_idx = {m: i for i, m in enumerate(MODULI)}
    
    for m in MODULI:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = int(np.sum(1 - covered[r::m]))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[mod_to_idx[m]] = best_r
        covered[best_r::m] = 1
    
    counts = compute_counts(residues)
    score = int(np.sum(counts == 0))
    return residues, counts, score

def sa_optimize(residues, counts, score, start_time, time_limit):
    """Simulated annealing with delta computation."""
    current_residues = residues[:]
    current_counts = counts.copy()
    current_score = score
    
    best_residues = current_residues[:]
    best_score = current_score
    
    T = 1.0
    cooling_rate = 0.9997
    min_T = 1e-5
    
    while time.time() - start_time < time_limit:
        i = random.randint(0, K - 1)
        m = MODULI[i]
        
        old_r = current_residues[i]
        new_r = random.randint(0, m - 1)
        
        if new_r == old_r:
            continue
        
        delta = 0
        
        # Process S_old (removing coverage)
        for x in range(old_r, N, m):
            c = current_counts[x]
            if c == 1:
                delta += 1
            current_counts[x] = c - 1
        
        # Process S_new (adding coverage)
        for x in range(new_r, N, m):
            c = current_counts[x]
            if c == 0:
                delta -= 1
            current_counts[x] = c + 1
        
        new_score = current_score + delta
        
        # Acceptance
        if delta < 0 or (delta == 0) or random.random() < math.exp(-delta / T):
            current_residues[i] = new_r
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_residues = current_residues[:]
        else:
            # Revert
            for x in range(old_r, N, m):
                current_counts[x] += 1
            for x in range(new_r, N, m):
                current_counts[x] -= 1
            current_residues[i] = old_r
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
    
    return best_residues, best_score

def local_search_polish(residues, counts, score, start_time, time_limit):
    """Local search: try all residue changes for each modulus, accept if improves."""
    current_residues = residues[:]
    current_counts = counts.copy()
    current_score = score
    best_residues = current_residues[:]
    best_score = current_score
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Shuffle order to avoid bias
        indices = list(range(K))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time >= time_limit:
                break
            m = MODULI[i]
            old_r = current_residues[i]
            best_r = old_r
            best_delta = 0
            
            for new_r in range(m):
                if new_r == old_r:
                    continue
                
                delta = 0
                # Process S_old (removing coverage)
                for x in range(old_r, N, m):
                    c = current_counts[x]
                    if c == 1:
                        delta += 1
                    current_counts[x] = c - 1
                
                # Process S_new (adding coverage)
                for x in range(new_r, N, m):
                    c = current_counts[x]
                    if c == 0:
                        delta -= 1
                    current_counts[x] = c + 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_r = new_r
                
                # Revert
                for x in range(old_r, N, m):
                    current_counts[x] += 1
                for x in range(new_r, N, m):
                    current_counts[x] -= 1
            
            if best_delta < 0:
                # Apply best change
                new_r = best_r
                for x in range(old_r, N, m):
                    current_counts[x] -= 1
                for x in range(new_r, N, m):
                    current_counts[x] += 1
                current_residues[i] = new_r
                current_score += best_delta
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_residues = current_residues[:]
    
    return best_residues, best_score

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s limit
    
    # Phase 1: Try multiple greedy constructions
    best_residues = None
    best_score = N + 1
    best_counts = None
    
    # Descending greedy
    residues_d, counts_d, score_d = greedy_build_descending()
    if score_d < best_score:
        best_score = score_d
        best_residues = residues_d[:]
        best_counts = counts_d.copy()
    
    # Ascending greedy
    residues_a, counts_a, score_a = greedy_build_ascending()
    if score_a < best_score:
        best_score = score_a
        best_residues = residues_a[:]
        best_counts = counts_a.copy()
    
    # Random init
    residues_r = [random.randint(0, m - 1) for m in MODULI]
    counts_r = compute_counts(residues_r)
    score_r = int(np.sum(counts_r == 0))
    if score_r < best_score:
        best_score = score_r
        best_residues = residues_r[:]
        best_counts = counts_r.copy()
    
    # Phase 2: SA optimization from best initial
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10:
        sa_time = min(remaining - 15, 80)
        best_residues, best_score = sa_optimize(
            best_residues, best_counts, best_score, start_time, start_time + sa_time
        )
        best_counts = compute_counts(best_residues)
    
    # Phase 3: Local search polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        ls_time = remaining - 3
        best_residues, best_score = local_search_polish(
            best_residues, best_counts, best_score, start_time, start_time + ls_time
        )
    
    classes = [[m, r] for m, r in zip(MODULI, best_residues)]
    return classes
