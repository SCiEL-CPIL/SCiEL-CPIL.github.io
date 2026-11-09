"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Moduli in [20, 200]
    all_moduli = list(range(20, 201))
    n = len(all_moduli)
    
    # Precompute gcd table
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = gcd(all_moduli[i], all_moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute 1/m as Fraction
    recip = [Fraction(1, m) for m in all_moduli]
    
    def can_add(classes, m_idx, r):
        """Check if (m_idx, r) is disjoint from all existing classes."""
        m = all_moduli[m_idx]
        for (m2_idx, r2) in classes:
            g = gcd_table[m_idx][m2_idx]
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_residue(classes, m_idx):
        """Find a valid residue for m_idx given current classes. Return -1 if none."""
        m = all_moduli[m_idx]
        # For each existing class, the forbidden residues form an arithmetic progression
        # r ≡ r2 (mod gcd(m, m2))
        # We need r not in any of these sets.
        
        # Collect constraints
        constraints = []
        for (m2_idx, r2) in classes:
            g = gcd_table[m_idx][m2_idx]
            if g == 1:
                # No constraint (any residue works w.r.t. this class)
                continue
            # r ≡ r2 (mod g) is forbidden
            constraints.append((g, r2 % g))
        
        if not constraints:
            return 0
        
        # Try residues 0, 1, 2, ..., m-1
        for r in range(m):
            ok = True
            for (g, r2_mod) in constraints:
                if r % g == r2_mod:
                    ok = False
                    break
            if ok:
                return r
        return -1
    
    # Strategy: Greedy with backtracking
    # Sort moduli by 1/m descending (i.e., m ascending) to prioritize small moduli
    order = sorted(range(n), key=lambda i: all_moduli[i])
    
    best_classes = []
    best_score = Fraction(0, 1)
    
    def backtrack(pos, classes, current_score):
        nonlocal best_classes, best_score
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_classes = list(classes)
        
        # Try adding more moduli from position pos onwards
        for i in range(pos, n):
            if time.time() > deadline:
                return
            
            m_idx = order[i]
            r = find_residue(classes, m_idx)
            if r >= 0:
                classes.append((m_idx, r))
                backtrack(i + 1, classes, current_score + recip[m_idx])
                classes.pop()
    
    # Run backtracking
    backtrack(0, [], Fraction(0, 1))
    
    # Convert best_classes to the required format
    result = [[all_moduli[m_idx], r] for (m_idx, r) in best_classes]
    
    # Verify validity
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            m1, r1 = result[i]
            m2, r2 = result[j]
            g = gcd(m1, m2)
            if (r1 - r2) % g == 0:
                # Invalid - should not happen
                pass
    
    return result
# EVOLVE-BLOCK-END
