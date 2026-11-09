import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    moduli = list(range(2, 129))
    # We want to maximize the count.
    # A valid set is a list of [r, m]
    
    best_count = 0
    best_solution = []
    
    # Precompute gcds for speed? Or compute on fly.
    
    def is_valid_candidate(m, current_pairs):
        # Returns list of valid residues for m
        valid_residues = []
        for r in range(m):
            ok = True
            for r_c, m_c in current_pairs:
                g = gcd(m, m_c)
                if (r - r_c) % g == 0:
                    ok = False
                    break
            if ok:
                valid_residues.append(r)
        return valid_residues

    def backtrack(index, current_pairs, current_count):
        nonlocal best_count, best_solution
        if time.time() > deadline:
            return
        
        if current_count > best_count:
            best_count = current_count
            best_solution = current_pairs.copy()
            
        if index == len(moduli):
            return
            
        m = moduli[index]
        
        # Option 1: Skip m
        backtrack(index + 1, current_pairs, current_count)
        
        # Option 2: Include m
        valid_residues = is_valid_candidate(m, current_pairs)
        if valid_residues:
            # Pick a residue. To maximize future options, maybe pick the one that
            # conflicts least with others? Or just pick the first one.
            # For a better search, we might want to try all valid residues,
            # but that explodes. 
            # Heuristic: Pick residue 0? Or random?
            # Let's try the first valid residue.
            r = valid_residues[0]
            current_pairs.append([r, m])
            backtrack(index + 1, current_pairs, current_count + 1)
            current_pairs.pop()

    # Order moduli to help search.
    # Larger moduli have more residues, so they are "easier" to place?
    # Or smaller moduli have fewer residues, so they are "harder" and should be placed first?
    # Usually, placing constrained variables first is better.
    # Small moduli have fewer residues, so they are more constrained.
    # Let's try ascending order.
    
    backtrack(0, [], 0)
    
    if not best_solution:
        # Fallback to seed
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_solution
