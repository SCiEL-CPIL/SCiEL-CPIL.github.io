import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    max_modulus = 128
    all_moduli = list(range(2, max_modulus + 1))
    
    def is_disjoint(r1, m1, r2, m2):
        g = gcd(m1, m2)
        return (r1 - r2) % g != 0

    def is_valid_in_set(r, m, current_set):
        for r_i, m_i in current_set:
            if not is_disjoint(r, m, r_i, m_i):
                return False
        return True

    def find_valid_residue(m, current_set):
        # Try residues in a randomized order or specific order
        # To break symmetries, we can try 0, m-1, 1, m-2 etc
        candidates = list(range(m))
        random.shuffle(candidates)
        for r in candidates:
            if is_valid_in_set(r, m, current_set):
                return r
        return None

    def greedy_build(order):
        current_set = []
        for m in order:
            if m in [x[1] for x in current_set]:
                continue
            r = find_valid_residue(m, current_set)
            if r is not None:
                current_set.append((r, m))
        return current_set

    def local_search(initial_set):
        current_set = list(initial_set)
        best_set = list(current_set)
        
        # Convert to list of tuples for easy manipulation
        # We want to maximize len(current_set)
        
        used_moduli = set(m for _, m in current_set)
        
        # Try to add any missing modulus
        while True:
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break
            
            improved = False
            # Try adding a new modulus
            missing = [m for m in all_moduli if m not in used_moduli]
            random.shuffle(missing)
            
            for m in missing:
                r = find_valid_residue(m, current_set)
                if r is not None:
                    current_set.append((r, m))
                    used_moduli.add(m)
                    improved = True
                    break
            
            if improved:
                if len(current_set) > len(best_set):
                    best_set = list(current_set)
                continue
            
            # If no improvement, try a swap (remove one, add one)
            # This might help escape local minima
            if not current_set:
                break
                
            # Pick a random element to remove
            idx = random.randint(0, len(current_set) - 1)
            r_out, m_out = current_set[idx]
            current_set.pop(idx)
            used_moduli.remove(m_out)
            
            # Try to add a different modulus
            missing = [m for m in all_moduli if m not in used_moduli]
            random.shuffle(missing)
            
            added = False
            for m_in in missing:
                r_in = find_valid_residue(m_in, current_set)
                if r_in is not None:
                    current_set.append((r_in, m_in))
                    used_moduli.add(m_in)
                    added = True
                    break
            
            # If we didn't add anything, put the old one back
            if not added:
                current_set.append((r_out, m_out))
                used_moduli.add(m_out)
                
            if len(current_set) > len(best_set):
                best_set = list(current_set)

        return best_set

    best_result = []
    
    # Run multiple greedy starts with different orders
    orders = [
        all_moduli,
        sorted(all_moduli, reverse=True),
        sorted(all_moduli, key=lambda x: -x.bit_count()), # More factors first?
    ]
    
    for _ in range(20):
        orders.append(random.sample(all_moduli, len(all_moduli)))
        
    for order in orders:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
            
        g_set = greedy_build(order)
        if len(g_set) > len(best_result):
            best_result = g_set
            
        # Local search on the greedy result
        ls_set = local_search(g_set)
        if len(ls_set) > len(best_result):
            best_result = ls_set

    # Convert to required format
    # List of [residue, modulus]
    return [[r, m] for r, m in best_result]
