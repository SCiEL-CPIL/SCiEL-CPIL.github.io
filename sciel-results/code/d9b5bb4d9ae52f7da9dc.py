import random
import time
import numpy as np

def construct():
    base = 51975
    # Generate divisors > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    # We must use distinct moduli. The problem implies we can choose a subset or use all?
    # "With every divisor of 51975 available once... construct() -> classes [[m, r], ...] with distinct moduli m > 1 dividing 51975"
    # Usually, using all available moduli gives the best chance to cover everything.
    # Let's assume we use ALL divisors.
    
    moduli = divisors
    k = len(moduli)
    
    def get_score(residues):
        """
        residues: list of ints, length k
        returns: number of uncovered residues modulo base
        """
        covered = np.zeros(base, dtype=bool)
        for i in range(k):
            m = moduli[i]
            r = residues[i]
            if r < base:
                # Mark covered
                covered[r::m] = True
        return base - np.sum(covered)

    def run_sa(initial_residues, time_limit=100):
        start_time = time.time()
        current_residues = initial_residues[:]
        current_score = get_score(current_residues)
        
        best_residues = current_residues[:]
        best_score = current_score
        
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() - start_time < time_limit:
            if T < min_T:
                break
            
            # Pick a random index to modify
            idx = random.randint(0, k - 1)
            old_r = current_residues[idx]
            
            # Generate new random residue
            new_r = random.randint(0, moduli[idx] - 1)
            
            if new_r == old_r:
                continue
                
            current_residues[idx] = new_r
            new_score = get_score(current_residues)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
            else:
                # Revert
                current_residues[idx] = old_r
                
            T *= cooling
            
        return best_score, best_residues

    # Initial states to try
    # 1. All zeros
    init1 = [0] * k
    # 2. Random
    init2 = [random.randint(0, moduli[i] - 1) for i in range(k)]
    # 3. Greedy (simple heuristic)
    def greedy_init():
        covered = bytearray(base)
        res = []
        # Sort by modulus size? Smaller moduli cover more? No, larger moduli cover fewer points but are more specific.
        # Usually, covering systems use small moduli to cover large chunks.
        # Let's just do a simple greedy: for each modulus, pick r that maximizes newly covered.
        for m in moduli:
            best_r = 0
            max_cov = -1
            for r in range(m):
                # Count how many are currently uncovered
                # This is slow in pure python, but we only do it once per init.
                # Optimize: just pick r=0 for simplicity in init, or use numpy.
                # Let's stick to r=0 for simplicity in this init, or random.
                # Actually, let's use a quick numpy check for the init
                pass
            # For simplicity, let's just use random for init2 and init1.
            res.append(0)
        return res
    
    # Better init: Use the greedy logic from the seed but vectorized?
    # Let's just run SA from a few good starting points.
    
    candidates = []
    
    # Candidate 1: All zeros
    candidates.append([0] * k)
    
    # Candidate 2: Random
    candidates.append([random.randint(0, moduli[i] - 1) for i in range(k)])
    
    # Candidate 3: "Greedy" - for each m, pick r that covers most *uncovered* spots.
    # We can do this efficiently with numpy.
    covered = np.zeros(base, dtype=bool)
    greedy_res = []
    # Process moduli in increasing order? Or decreasing?
    # Small moduli cover many spots. Large moduli cover few.
    # If we use small moduli first, we cover a lot. Then large moduli fill gaps.
    sorted_moduli = sorted(moduli)
    index_map = {m: i for i, m in enumerate(moduli)}
    
    temp_covered = np.zeros(base, dtype=bool)
    temp_res = [0] * k
    for m in sorted_moduli:
        best_r = 0
        max_new = -1
        # Try a subset of residues if m is large? No, m is at most 51975.
        # But iterating all r for all m is O(sum(m)) which is large.
        # Let's just pick r=0 for this init to be fast.
        # Or, pick r that is currently uncovered?
        # Let's stick to simple inits.
        temp_res[index_map[m]] = 0
        temp_covered[0::m] = True
        
    candidates.append(temp_res)
    
    best_overall_score = float('inf')
    best_overall_res = None
    
    # Run SA on each candidate, allocating time
    # We have ~110s. Let's give 30s to each of 3 candidates.
    time_per_candidate = 30
    
    for init in candidates:
        score, res = run_sa(init, time_limit=time_per_candidate)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_res = res
            
    # Construct output
    classes = []
    for i in range(k):
        classes.append([moduli[i], best_overall_res[i]])
        
    return classes
