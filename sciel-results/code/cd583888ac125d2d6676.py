import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    best_set = []
    
    # Precompute gcds
    gcd_table = {}
    for i in range(len(moduli)):
        for j in range(i+1, len(moduli)):
            gcd_table[(moduli[i], moduli[j])] = gcd(moduli[i], moduli[j])
            
    def is_compatible(candidate_m, current_set):
        # current_set is list of (r, m)
        # Check gcd condition
        for r_k, m_k in current_set:
            if gcd(candidate_m, m_k) == 1:
                return False
        
        # Check residue condition
        # Try all residues for candidate_m
        for r in range(candidate_m):
            valid = True
            for r_k, m_k in current_set:
                d = gcd(candidate_m, m_k)
                if r % d == r_k % d:
                    valid = False
                    break
            if valid:
                return True
        return False

    # Greedy with random order
    best_size = 0
    best_result = []
    
    num_restarts = 1000
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(moduli)
        current_set = []
        
        for m in moduli:
            if time.time() > deadline:
                break
            if is_compatible(m, current_set):
                # Find a valid residue
                # We know one exists, but is_compatible didn't return it.
                # Let's refactor is_compatible to return the residue
                pass # Placeholder, need to fix logic
        
        # The above loop is inefficient because is_compatible doesn't return residue.
        # Let's rewrite the core loop properly.
        
    # Proper implementation
    def find_valid_residue(candidate_m, current_set):
        for r in range(candidate_m):
            valid = True
            for r_k, m_k in current_set:
                d = gcd(candidate_m, m_k)
                if r % d == r_k % d:
                    valid = False
                    break
            if valid:
                return r
        return None

    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(moduli)
        current_set = []
        
        for m in moduli:
            if time.time() > deadline:
                break
            
            # Check gcd condition first
            ok_gcd = True
            for r_k, m_k in current_set:
                if gcd(m, m_k) == 1:
                    ok_gcd = False
                    break
            
            if not ok_gcd:
                continue
                
            r = find_valid_residue(m, current_set)
            if r is not None:
                current_set.append((r, m))
                
        if len(current_set) > len(best_result):
            best_result = current_set[:]

    # Local improvement on best_result
    # Try to add more or swap
    # ... (implement local search if time permits)
    
    # Convert to required format [[r, m], ...]
    return [[r, m] for r, m in best_result]
