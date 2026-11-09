"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    """
    Find a set of pairwise disjoint residue classes with distinct moduli in [20, 200]
    maximizing the sum of 1/m.
    
    Uses a greedy approach with local search (swap improvements).
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    def is_compatible(assign, m, r):
        """Check if residue r for modulus m is compatible with all assigned moduli."""
        for m2, r2 in assign.items():
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def try_place(assign, m, r=None):
        """Place modulus m with residue r (or find smallest valid r). Returns True on success."""
        if r is not None:
            if 0 <= r < m and is_compatible(assign, m, r):
                assign[m] = r
                return True
            return False
        for r in range(m):
            if is_compatible(assign, m, r):
                assign[m] = r
                return True
        return False
    
    def score(assign):
        """Compute the score as a float."""
        return sum(1.0 / m for m in assign)
    
    def exact_score(assign):
        """Compute the score as an exact Fraction."""
        total = Fraction(0, 1)
        for m in assign:
            total += Fraction(1, m)
        return total
    
    # Strategy: Greedy construction + local search with swaps
    
    # Phase 1: Greedy construction
    # Try multiple orderings and keep the best
    
    moduli = list(range(20, 201))
    best_assign = None
    best_score = 0.0
    
    # Ordering 1: Ascending moduli (small first, high 1/m)
    for order in [
        sorted(moduli, key=lambda m: m),
        sorted(moduli, key=lambda m: -m),
        sorted(moduli, key=lambda m: m % 2),  # even first
        sorted(moduli, key=lambda m: (m % 2, m)),  # even first, then ascending
        sorted(moduli, key=lambda m: (-1 if m % 2 == 0 else 1, m)),
        # Try multiples of small numbers first
        sorted(moduli, key=lambda m: min(m // p for p in [2, 3, 4, 5, 6, 7, 8, 9, 10] if m % p == 0) if any(m % p == 0 for p in [2, 3, 4, 5, 6, 7, 8, 9, 10]) else m),
    ]:
        if elapsed() > deadline:
            break
        
        assign = {}
        for m in order:
            try_place(assign, m)
        
        s = score(assign)
        if s > best_score:
            best_score = s
            best_assign = dict(assign)
    
    # Phase 2: Local search - try to swap out a modulus for a better one
    improved = True
    while improved and elapsed() < deadline:
        improved = False
        assign = dict(best_assign)
        s = score(assign)
        
        # Try adding each unplaced modulus
        for m_new in moduli:
            if m_new in assign:
                continue
            # Try all residues for m_new
            placed = False
            for r in range(m_new):
                if is_compatible(assign, m_new, r):
                    # Check if this improves the score
                    new_s = s + 1.0 / m_new
                    if new_s > s + 1e-12:
                        assign[m_new] = r
                        s = new_s
                        improved = True
                        placed = True
                        break
            if placed:
                # Continue trying to add more
                continue
        
        # Try removing a modulus and replacing with two others
        # This is more expensive, so only do it if we have time
        if elapsed() < deadline - 30:
            for m_remove in list(assign.keys()):
                if elapsed() > deadline:
                    break
                # Try removing m_remove and see if we can add two better ones
                assign2 = dict(assign)
                del assign2[m_remove]
                s2 = score(assign2)
                
                # Try to add moduli not in assign2
                added = []
                for m_add in moduli:
                    if m_add in assign2:
                        continue
                    if elapsed() > deadline:
                        break
                    for r in range(m_add):
                        if is_compatible(assign2, m_add, r):
                            assign2[m_add] = r
                            added.append(m_add)
                            s2 += 1.0 / m_add
                            break
                    if len(added) >= 2:
                        break
                
                if s2 > s + 1e-12:
                    assign = assign2
                    s = s2
                    improved = True
        
        if s > best_score:
            best_score = s
            best_assign = dict(assign)
    
    # Phase 3: Final polish - try all residues for each modulus to optimize
    if elapsed() < deadline - 10:
        assign = dict(best_assign)
        s = score(assign)
        
        for m in list(assign.keys()):
            if elapsed() > deadline:
                break
            current_r = assign[m]
            # Try other residues
            for r in range(m):
                if r == current_r:
                    continue
                if is_compatible(assign, m, r):
                    # Score doesn't change with residue, but this might allow other improvements
                    # For now, just keep the first valid residue
                    break
        
        if score(assign) > best_score:
            best_assign = assign
            best_score = score(assign)
    
    # Convert to list of [m, r] pairs
    result = [[m, r] for m, r in sorted(best_assign.items())]
    
    return result

# EVOLVE-BLOCK-END
