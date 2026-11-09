"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    def is_compatible(r_new, m_new, chosen):
        for r, m in chosen:
            g = gcd(m_new, m)
            if r_new % g == r % g:
                return False
        return True

    def solve_residues(moduli, time_limit=5):
        """
        Attempts to assign residues to the given moduli.
        Returns a list of [r, m] if successful, else None.
        """
        # Sort moduli by number of divisors or magnitude?
        # Smaller moduli have fewer residues, so they are more constrained.
        # Let's sort by modulus ascending.
        sorted_mods = sorted(moduli)
        assignment = {}
        
        # Backtracking search
        def backtrack(idx):
            if time.time() > start_time + time_limit:
                return False
            if idx == len(sorted_mods):
                return True
            
            m = sorted_mods[idx]
            # Try residues in a random or specific order
            # Residues 0..m-1
            # Heuristic: try residues that are "different" from existing ones
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                # Check compatibility with all previously assigned
                compatible = True
                for j in range(idx):
                    m_j = sorted_mods[j]
                    r_j = assignment[m_j]
                    g = gcd(m, m_j)
                    if r % g == r_j % g:
                        compatible = False
                        break
                
                if compatible:
                    assignment[m] = r
                    if backtrack(idx + 1):
                        return True
                    del assignment[m]
            
            return False

        if backtrack(0):
            return [[assignment[m], m] for m in sorted_mods]
        return None

    best_result = []
    best_count = 0
    
    random.seed(42)
    
    # Candidate sets of moduli
    # 1. All even numbers
    evens = [m for m in all_moduli if m % 2 == 0]
    # 2. All multiples of 4
    mult4 = [m for m in all_moduli if m % 4 == 0]
    # 3. All multiples of 3
    mult3 = [m for m in all_moduli if m % 3 == 0]
    # 4. All multiples of 5
    mult5 = [m for m in all_moduli if m % 5 == 0]
    # 5. All numbers divisible by 2 or 3? (Careful: 2 and 3 are coprime)
    # We need pairwise gcd > 1.
    # So we can't mix 2 and 3 directly unless we drop one or ensure they share a factor (impossible).
    # So sets must be "homogeneous" in prime factors.
    
    candidates = [evens, mult4, mult3, mult5]
    
    # Also try subsets of evens that might work better?
    # The full set of evens likely fails residue assignment due to parity conflicts.
    # Let's try smaller subsets of evens, e.g., powers of 2.
    powers2 = [2**i for i in range(1, 8) if 2**i <= 128]
    candidates.append(powers2)
    
    # Try to solve for each candidate
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        
        # Shuffle to break symmetry if needed?
        # solve_residues handles the order.
        res = solve_residues(cand, time_limit=10)
        
        if res and len(res) > best_count:
            best_count = len(res)
            best_result = res[:]

    # If no candidate worked well, fall back to the old greedy
    if best_count == 0:
        # Old greedy logic
        orderings = [sorted(all_moduli)]
        for order in orderings:
            if time.time() > deadline - 5:
                break
            chosen = []
            for m in order:
                if time.time() > deadline - 5:
                    break
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if is_compatible(r, m, chosen):
                        chosen.append((r, m))
                        break
            if len(chosen) > best_count:
                best_count = len(chosen)
                best_result = chosen[:]

    if best_count > 0:
        return [[r, m] for r, m in best_result]
    
    return [[1, 2]]
# EVOLVE-BLOCK-END
