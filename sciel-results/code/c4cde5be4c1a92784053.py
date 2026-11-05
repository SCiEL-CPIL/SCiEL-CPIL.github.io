import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    best_result = []
    
    def is_valid_assignment(residues_moduli, r, m):
        """
        Check if residue r for modulus m is valid with respect to 
        the list of (residue, modulus) pairs in residues_moduli.
        """
        for r_j, m_j in residues_moduli:
            g = math.gcd(m, m_j)
            if g > 1:
                if r % g == r_j % g:
                    return False
            # if g == 1, they are coprime, so intersection is guaranteed? 
            # Wait. If gcd=1, solution always exists. 
            # So they are NEVER disjoint. 
            # So we can never add a modulus coprime to an existing one.
            # The check above: if g==1, r%1 == r_j%1 is 0==0, so it returns False.
            # This is correct. We cannot have coprime moduli.
        return True

    def find_residue(m, current_classes):
        """
        Find a valid residue for m given current_classes.
        Returns residue or None.
        """
        # Try residues in a deterministic order first, then random?
        # Deterministic 0..m-1 is fast.
        for r in range(m):
            if is_valid_assignment(current_classes, r, m):
                return r
        return None

    # Run multiple random restarts
    # We can also try sorting moduli by some heuristic
    # e.g., descending number of divisors?
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.shuffle(moduli)
        
        current_classes = []
        
        for m in moduli:
            r = find_residue(m, current_classes)
            if r is not None:
                current_classes.append((r, m))
                
        if len(current_classes) > len(best_result):
            best_result = current_classes.copy()
            # If we hit a very high number, we might break early?
            # Max possible is 127, but likely much lower.
            # If we get > 60, it's good.
            
        # Optional: Local search / Swap
        # If we get stuck, try to swap a modulus?
        # For now, simple restarts.

    # Convert to list of lists [residue, modulus]
    # Sort by modulus for canonical output? Not required but good.
    result = [[r, m] for r, m in best_result]
    return result

