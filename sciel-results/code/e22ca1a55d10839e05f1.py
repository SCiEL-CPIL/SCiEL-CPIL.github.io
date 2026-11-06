"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction
import time
import sys

def construct():
    """
    Find a maximum weight independent set on the conflict graph of candidate (m, r) pairs.
    Nodes: (m, r) where 20 <= m <= 200, 0 <= r < m.
    Edges: two nodes (m1, r1) and (m2, r2) conflict if r1 ≡ r2 (mod gcd(m1, m2)).
    Weight: 1/m for each node.
    
    We use branch-and-bound with a fractional knapsack upper bound.
    """
    
    # Generate all candidate (m, r) pairs
    candidates = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r))
    
    n = len(candidates)
    
    # Precompute weights as Fractions for exact arithmetic
    weights = [Fraction(1, m) for (m, r) in candidates]
    
    # Precompute conflict masks
    # conflict_mask[i] is a bitset of all j != i that conflict with i
    # For efficiency, we'll use Python integers as bitsets
    
    # First, let's precompute gcd for all pairs of moduli
    # moduli_values = [m for (m, r) in candidates]
    # But we need gcd(m_i, m_j) for each pair
    
    # Let's precompute a matrix of gcd values
    # Actually, for the conflict check: (m1, r1) conflicts with (m2, r2) iff r1 % gcd(m1, m2) == r2 % gcd(m1, m2)
    
    # Let's build conflict masks
    # To make this feasible, we'll compute conflicts on the fly but cache gcd values
    
    gcd_cache = {}
    
    def get_gcd(m1, m2):
        if m1 > m2:
            m1, m2 = m2, m1
        key = (m1, m2)
        if key not in gcd_cache:
            gcd_cache[key] = math.gcd(m1, m2)
        return gcd_cache[key]
    
    def conflicts(i, j):
        m1, r1 = candidates[i]
        m2, r2 = candidates[j]
        g = get_gcd(m1, m2)
        return (r1 % g) == (r2 % g)
    
    # Build conflict masks as integers (bitsets)
    # This is O(n^2) which is ~20000^2 = 400 million, too slow
    # We need a smarter approach
    
    # Actually, let's reconsider. n = sum_{m=20}^{200} m = (20+200)*181/2 = 19295
    # n^2 = 372 million, which is too slow in Python
    
    # Alternative: use a different search strategy
    # Since weights are 1/m, and we want to maximize sum, we prefer small m.
    # Let's try a greedy + local search approach, but with a better initial solution
    
    # Key insight: for each modulus m, we can only pick one residue.
    # So the problem is: choose a subset of moduli S, and for each m in S, choose r_m,
    # such that for all m, m' in S: r_m != r_m' (mod gcd(m, m')).
    
    # Let's try a different approach:
    # 1. Start with a good initial solution (e.g., all moduli 20, 40, ..., 200 with residues 0, 1, ..., 9)
    # 2. Use simulated annealing or hill climbing to improve
    
    import random
    
    def check_disjoint(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def is_valid_solution(classes):
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                if m1 == m2:
                    return False
                if not check_disjoint(m1, r1, m2, r2):
                    return False
        return True
    
    def score(classes):
        return sum(Fraction(1, m) for m, r in classes)
    
    # Start with current solution
    current = [[20 * j, j - 1] for j in range(1, 11)]
    current_score = score(current)
    
    # Try to improve by adding moduli
    used_moduli = set(m for m, r in current)
    
    for m in range(20, 201):
        if m in used_moduli:
            continue
        for r in range(m):
            valid = True
            for m2, r2 in current:
                if not check_disjoint(m, r, m2, r2):
                    valid = False
                    break
            if valid:
                current.append([m, r])
                used_moduli.add(m)
                break
    
    # Now try to swap: replace a modulus with a smaller one if possible
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            m_i, r_i = current[i]
            for m_new in range(20, m_i):
                if m_new in used_moduli - {m_i}:
                    continue
                for r_new in range(m_new):
                    valid = True
                    for j in range(len(current)):
                        if j == i:
                            continue
                        m_j, r_j = current[j]
                        if not check_disjoint(m_new, r_new, m_j, r_j):
                            valid = False
                            break
                    if valid:
                        old_m = current[i][0]
                        current[i] = [m_new, r_new]
                        used_moduli.discard(old_m)
                        used_moduli.add(m_new)
                        improved = True
                        break
                if improved:
                    break
    
    # Convert to float for return
    return [[m, r] for m, r in current]

# EVOLVE-BLOCK-END
