import random
import time

def construct():
    deadline = time.time() + 80  # Safety margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    best_size = 0
    best_solution = []
    
    # Try many random orders
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        random.shuffle(moduli)
        current_set = []
        current_residues = {}
        
        for m in moduli:
            # Try to find a residue for m
            valid_res = None
            # We can restrict search space if needed, but m <= 128 is small
            for r in range(m):
                ok = True
                for m_i, r_i in current_residues.items():
                    g = __import__('math').gcd(m, m_i)
                    if (r - r_i) % g == 0:
                        ok = False
                        break
                if ok:
                    valid_res = r
                    break
            
            if valid_res is not None:
                current_set.append(m)
                current_residues[m] = valid_res
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_solution = [[current_residues[m], m] for m in current_set]
            
    # Return the best solution found
    if not best_solution:
        # Fallback to powers of 2
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_solution
