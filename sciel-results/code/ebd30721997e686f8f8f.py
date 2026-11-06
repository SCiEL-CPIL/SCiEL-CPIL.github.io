import random
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    max_time = 55.0  # Safety margin before 60s
    
    # Candidate moduli: all even numbers from 2 to 128
    # Why even? Because gcd(m1, m2) >= 2, allowing us to choose residues that are not congruent mod gcd.
    # If we mix odd and even, gcd might be 1, which forces intersection.
    # So, sticking to one prime factor (2) is a strong heuristic.
    moduli = list(range(2, 129, 2))
    
    best_count = 0
    best_solution = []
    
    # Greedy randomized search
    # We will try to assign a residue to each modulus in moduli.
    # If we can assign to all, we get 64 classes.
    # If not, we get fewer.
    
    # Precompute gcds for speed? Not necessary for 64 items.
    
    def is_compatible(r_new, m_new, existing):
        for r_old, m_old in existing:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True

    # Run multiple random restarts
    attempts = 0
    while time.time() - start_time < max_time:
        attempts += 1
        random.shuffle(moduli)
        current = []
        for m in moduli:
            # Try all residues for this modulus
            residues = list(range(m))
            random.shuffle(residues)
            found = False
            for r in residues:
                if is_compatible(r, m, current):
                    current.append((r, m))
                    found = True
                    break
            if not found:
                # If no residue works for this modulus, we skip it
                # But to maximize, maybe we should try to see if we can swap?
                # For simplicity, we just skip.
                pass
        
        if len(current) > best_count:
            best_count = len(current)
            best_solution = current
            
        # Early stop if we found a perfect solution (all 64 moduli)
        if best_count == len(moduli):
            break
            
    # Convert to list of lists as required
    return [[r, m] for r, m in best_solution]
# EVOLVE-BLOCK-END
