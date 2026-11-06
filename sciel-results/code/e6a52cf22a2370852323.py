import math
import random
import time

def construct():
    start_time = time.time()
    deadline = 45.0  # Leave 15s margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcds for speed
    gcd_cache = {}
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            gcd_cache[(i, j)] = math.gcd(i, j)
            
    def get_gcd(m1, m2):
        if m1 == m2: return m1
        if m1 > m2: m1, m2 = m2, m1
        return gcd_cache[(m1, m2)]

    def is_valid_pair(m1, r1, m2, r2):
        g = get_gcd(m1, m2)
        return (r1 % g) != (r2 % g)

    def try_greedy_assignment(moduli_set, random_seed=42):
        """
        Attempts to assign residues to all moduli in moduli_set.
        Returns a list of [residue, modulus] if successful, else None.
        """
        random.seed(random_seed)
        # Order moduli by size? Or random?
        # Smaller moduli have fewer residue options, so they are more constrained.
        # Let's sort by modulus ascending.
        sorted_mods = sorted(moduli_set)
        
        assigned = [] # list of (modulus, residue)
        
        for m in sorted_mods:
            found = False
            # Try residues in a random order to break symmetry
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                valid = True
                for m_prev, r_prev in assigned:
                    if not is_valid_pair(m, r, m_prev, r_prev):
                        valid = False
                        break
                if valid:
                    assigned.append((m, r))
                    found = True
                    break
            
            if not found:
                return None
                
        return [[r, m] for m, r in assigned]

    best_solution = []
    best_count = 0
    
    # Strategy 1: Try all even moduli (largest potential set)
    even_moduli = [m for m in moduli if m % 2 == 0]
    for attempt in range(1000):
        if time.time() > deadline:
            break
        sol = try_greedy_assignment(even_moduli, random_seed=attempt)
        if sol:
            if len(sol) > best_count:
                best_count = len(sol)
                best_solution = sol
            # If we found a full solution, it's likely the max for this set
            if len(sol) == len(even_moduli):
                return sol

    # Strategy 2: Try multiples of 3
    mod3 = [m for m in moduli if m % 3 == 0]
    for attempt in range(500):
        if time.time() > deadline:
            break
        sol = try_greedy_assignment(mod3, random_seed=1000 + attempt)
        if sol:
            if len(sol) > best_count:
                best_count = len(sol)
                best_solution = sol
            if len(sol) == len(mod3):
                return sol

    # Strategy 3: Try multiples of 5
    mod5 = [m for m in moduli if m % 5 == 0]
    for attempt in range(200):
        if time.time() > deadline:
            break
        sol = try_greedy_assignment(mod5, random_seed=2000 + attempt)
        if sol:
            if len(sol) > best_count:
                best_count = len(sol)
                best_solution = sol
            if len(sol) == len(mod5):
                return sol

    # Strategy 4: Random subsets of evens (in case full evens is impossible)
    # Pick random subsets of even_moduli of decreasing size
    # But we already tried full evens. If it failed, maybe a subset works?
    # Greedy on a subset might succeed if the full set is over-constrained.
    # However, if we can't solve full evens, a subset is likely solvable.
    # Let's try to construct a large subset greedily.
    
    def greedy_subset_search():
        selected = []
        residues = {}
        # Start with random order of evens
        candidates = even_moduli[:]
        random.shuffle(candidates)
        
        for m in candidates:
            if time.time() > deadline:
                break
            # Try to find a residue for m
            residues_list = list(range(m))
            random.shuffle(residues_list)
            found = False
            for r in residues_list:
                valid = True
                for m_prev in selected:
                    r_prev = residues[m_prev]
                    if not is_valid_pair(m, r, m_prev, r_prev):
                        valid = False
                        break
                if valid:
                    selected.append(m)
                    residues[m] = r
                    found = True
                    break
            # If not found, skip m (greedy choice)
        
        return [[residues[m], m] for m in selected]

    for attempt in range(500):
        if time.time() > deadline:
            break
        sol = greedy_subset_search()
        if sol and len(sol) > best_count:
            best_count = len(sol)
            best_solution = sol

    if not best_solution:
        # Fallback: A known small solution
        # {6, 10, 15, 30} - Check if valid
        # gcd(6,10)=2. r6=0, r10=1. 0%2=0, 1%2=1. OK.
        # gcd(6,15)=3. r6=0, r15=2. 0%3=0, 2%3=2. OK.
        # gcd(10,15)=5. r10=1, r15=2. 1%5=1, 2%5=2. OK.
        # gcd(6,30)=6. r6=0, r30=3. 0%6=0, 3%6=3. OK.
        # gcd(10,30)=10. r10=1, r30=3. 1%10=1, 3%10=3. OK.
        # gcd(15,30)=15. r15=2, r30=3. 2%15=2, 3%15=3. OK.
        return [[0, 6], [1, 10], [2, 15], [3, 30]]
        
    return best_solution
