import math
import random
import time

def construct():
    max_mod = 128
    
    def is_compatible(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0

    def solve_csp(moduli):
        """
        Solves the residue assignment for a fixed set of moduli using backtracking.
        Returns list of [residue, modulus] or None.
        """
        if not moduli:
            return []
        
        # Sort moduli to optimize backtracking.
        # Heuristic: Moduli with many non-coprime neighbors first?
        # Or small moduli first (fewer residues)?
        # Let's try sorting by value ascending, as smaller moduli have fewer residues to try.
        sorted_moduli = sorted(moduli)
        
        assignment = {}
        
        # Precompute gcds with all other moduli in the set
        # gcd_with[m][idx]
        
        def backtrack(idx):
            if idx == len(sorted_moduli):
                return True
            
            m = sorted_moduli[idx]
            
            # Check residues 0..m-1
            # Optimization: Shuffle residues? No, deterministic is better for reproducibility of the specific set.
            # But we can try residues that are "unconstrained" first?
            # For now, iterate 0..m-1.
            
            for r in range(m):
                valid = True
                # Check against all already assigned
                for m2, r2 in assignment.items():
                    if not is_compatible(r, m, r2, m2):
                        valid = False
                        break
                
                if valid:
                    assignment[m] = r
                    if backtrack(idx + 1):
                        return True
                    del assignment[m]
            
            return False

        if backtrack(0):
            return [[assignment[m], m] for m in sorted_moduli]
        return None

    # Strategy: Try sets of multiples of small primes.
    # Multiples of 2 failed due to parity constraints on triangles.
    # Multiples of 3 might work.
    # Multiples of 5 might work.
    
    best_result = []
    
    # Candidate sets
    candidate_sets = []
    
    # 1. All multiples of 3
    multiples_3 = [m for m in range(3, max_mod + 1, 3)]
    candidate_sets.append(multiples_3)
    
    # 2. All multiples of 5
    multiples_5 = [m for m in range(5, max_mod + 1, 5)]
    candidate_sets.append(multiples_5)
    
    # 3. All multiples of 7
    multiples_7 = [m for m in range(7, max_mod + 1, 7)]
    candidate_sets.append(multiples_7)
    
    # 4. Mix of multiples of 3 and 5?
    # If we mix, we need $\gcd(3k, 5l) > 1$.
    # $\gcd(3,5)=1$. So 3 and 5 cannot be in the same set unless we remove one or ensure they don't conflict?
    # They ALWAYS conflict if coprime.
    # So we cannot mix primes.
    
    # 5. What about a set of evens that is "triangle-free" in the coprime-half graph?
    # This is complex. Let's stick to prime multiples first.
    
    # Try to solve each candidate set
    for s in candidate_sets:
        result = solve_csp(s)
        if result and len(result) > len(best_result):
            best_result = result

    # If all failed, or best is small, try a greedy fallback on all moduli
    # But the problem asks to maximize.
    
    # If best_result is empty, return a valid small set (e.g. just [0,2])
    if not best_result:
        return [[0, 2]]
        
    return best_result
