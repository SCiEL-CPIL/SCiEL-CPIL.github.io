import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin
    
    # Candidate moduli
    moduli = list(range(2, 129))
    
    # Heuristic: sort moduli. 
    # Moduli with higher "potential" first?
    # Let's try sorting by number of divisors descending, or just descending value.
    # Powers of 2 are strong candidates.
    # Let's try a few orderings and pick the best result.
    
    best_result = []
    
    orderings = [
        sorted(moduli, reverse=True),
        sorted(moduli, key=lambda x: x.bit_count(), reverse=True),
        sorted(moduli, key=lambda x: x.bit_count()),
        sorted(moduli),
        # Powers of 2 first
        [2**i for i in range(7)] + [m for m in moduli if m not in [2**i for i in range(7)]]
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
            
        current_set = [] # List of (r, m)
        current_moduli = set()
        
        for m in order:
            if time.time() > deadline:
                break
                
            # Find a valid residue r for m
            # r must be in [0, m-1]
            # For each existing (r_j, m_j), we need gcd(m, m_j) not divide (r - r_j)
            
            valid = True
            candidate_r = -1
            
            # Try residues 0 to m-1
            # Optimization: we can check constraints incrementally
            for r in range(m):
                ok = True
                for r_j, m_j in current_set:
                    g = math.gcd(m, m_j)
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                if ok:
                    candidate_r = r
                    break
            
            if candidate_r != -1:
                current_set.append((candidate_r, m))
                current_moduli.add(m)
                
        if len(current_set) > len(best_result):
            best_result = current_set
            if time.time() > deadline:
                break
                
    # Convert to list of lists [residue, modulus]
    return [[r, m] for r, m in best_result]
