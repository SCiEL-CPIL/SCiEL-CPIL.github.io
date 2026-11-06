import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    # Hard deadline: 110 seconds to be safe against the 120s/150s limits
    deadline = start_time + 110
    
    all_moduli = list(range(20, 201))
    
    # Pre-compute gcds for speed
    # gcd_cache[m1][m2] = gcd(m1, m2)
    # But m1, m2 up to 200, so 200x200 = 40000 entries. Fast enough.
    gcd_cache = {}
    for m1 in all_moduli:
        for m2 in all_moduli:
            if m1 not in gcd_cache:
                gcd_cache[m1] = {}
            gcd_cache[m1][m2] = gcd(m1, m2)

    def check_validity(moduli, node_limit=50000):
        """
        Returns True if there exist residues for the given moduli.
        Uses backtracking with forward checking and a node limit.
        """
        n = len(moduli)
        if n == 0:
            return True, []
        
        # Order indices by modulus size? 
        # Smaller moduli have fewer residues, so they are more constrained.
        # Let's sort indices by modulus ascending to assign most constrained first.
        indices = list(range(n))
        indices.sort(key=lambda i: moduli[i])
        
        residues = [None] * n
        nodes_visited = 0
        
        def backtrack(idx):
            nonlocal nodes_visited
            if nodes_visited > node_limit:
                return False
            
            if idx == n:
                return True
            
            original_idx = indices[idx]
            m = moduli[original_idx]
            
            # Try residues 0 to m-1
            # Optimization: We can skip residues that are congruent to previous ones mod gcd
            for r in range(m):
                nodes_visited += 1
                if nodes_visited > node_limit:
                    return False
                
                ok = True
                # Check against all previously assigned indices (0 to idx-1 in sorted order)
                for j in range(idx):
                    prev_idx = indices[j]
                    m_j = moduli[prev_idx]
                    g = gcd_cache[m][m_j]
                    if (r - residues[prev_idx]) % g == 0:
                        ok = False
                        break
                if ok:
                    residues[original_idx] = r
                    if backtrack(idx + 1):
                        return True
                    residues[original_idx] = None
            return False
        
        if backtrack(0):
            return True, residues
        else:
            return False, None

    def get_score(moduli):
        return sum(Fraction(1, m) for m in moduli)

    best_score = Fraction(0, 1)
    best_classes = []
    
    # Strategy 1: Greedy construction
    # We want to maximize sum(1/m), so we prefer small moduli.
    # However, small moduli are often "incompatible" with each other if they share factors.
    # Example: 20 and 22 share gcd 2. If r1=0, r2 must be odd.
    
    # Let's try to build a set starting from the smallest moduli.
    # We will do multiple restarts with different randomizations of the order of *attempting* to add.
    
    for attempt in range(50): # Reduced from 100
        if time.time() > deadline - 20:
            break
        
        # Create a list of candidates
        # Bias towards smaller moduli
        candidates = all_moduli[:]
        
        if attempt == 0:
            # Pure ascending
            candidates.sort()
        else:
            # Shuffle, but keep small ones somewhat prioritized?
            # Or just shuffle completely.
            random.shuffle(candidates)
            # Sort by a key that is m + random_noise?
            # Let's just shuffle. The greedy "add if fits" will naturally pick up small ones if they fit.
            # Actually, if we shuffle, we might pick a large m first, then a small m might not fit later?
            # No, if small m fits with large m, it fits. The constraint is symmetric.
            # But adding small m first restricts the space more for subsequent additions?
            # Actually, smaller m has fewer residues, so it's more restrictive.
            # It is generally better to add restrictive (small) moduli first.
            # So let's sort by m ascending, but add some randomness to the order of *equal* or *near* moduli?
            # Or just stick to ascending for the main attempts, and use random for variations.
            pass 

        # For variations, let's perturb the order
        if attempt > 0:
            # Take the sorted list and do some random swaps
            candidates.sort()
            for _ in range(10):
                i = random.randint(0, len(candidates)-1)
                j = random.randint(0, len(candidates)-1)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        
        current_set = []
        current_residues = []
        
        for m in candidates:
            if time.time() > deadline - 20:
                break
            
            # Try to add m
            # Optimization: Only check validity if it's likely to be worth it?
            # 1/m is always positive, so any valid addition improves score.
            trial_set = current_set + [m]
            
            # Fast check: If m is coprime to all existing, we can always find a residue?
            # No, we need to check the specific residues.
            # But check_validity is the bottleneck.
            
            # Heuristic: If len(current_set) is large, check_validity is slow.
            # We can limit the size of the set we try to build in one go?
            # No, we want the max size.
            
            # To speed up, we can skip check_validity if we know it will fail?
            # Hard to know.
            
            valid, res = check_validity(trial_set)
            if valid:
                current_set.append(m)
                current_residues = res
        
        if current_set:
            s = get_score(current_set)
            if s > best_score:
                best_score = s
                best_classes = [[current_set[i], current_residues[i]] for i in range(len(current_set))]
                
    # Strategy 2: Local Search / Polish
    # Try to add any missing modulus to the best set found.
    if best_classes and time.time() < deadline - 5:
        current_moduli = [c[0] for c in best_classes]
        current_residues = [c[1] for c in best_classes]
        current_score = best_score
        current_set_list = [c[0] for c in best_classes]
        
        # Iterate through all moduli not in the set, try to add them
        # Sort by m ascending (highest value first)
        missing = [m for m in all_moduli if m not in current_moduli]
        missing.sort()
        
        added = True
        while added and time.time() < deadline - 2:
            added = False
            for m in missing:
                if time.time() > deadline - 2:
                    break
                
                trial_set = current_moduli + [m]
                valid, res = check_validity(trial_set)
                if valid:
                    current_moduli.append(m)
                    current_residues = res
                    missing.remove(m)
                    s = get_score(current_moduli)
                    if s > best_score:
                        best_score = s
                        best_classes = [[current_moduli[i], current_residues[i]] for i in range(len(current_moduli))]
                    added = True
                    # Restart loop to re-evaluate with new set
                    break
                    
    # Fallback
    if not best_classes:
        best_classes = [[20, 0], [22, 1]]
        best_score = Fraction(1, 20) + Fraction(1, 22)

    return best_classes
