"""Optimized odd covering system search using bitset greedy maximum coverage."""
import numpy as np

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11

def _get_divisors(n):
    """Get all divisors of n greater than 1."""
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

def construct():
    base = BASE
    # Precompute all divisors > 1 of 51975
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size (larger moduli cover fewer residues per class, 
    # but smaller moduli cover more residues per class)
    # We want to cover as many residues as possible, so smaller moduli 
    # cover more residues per class. But we need distinct moduli.
    # Strategy: process moduli in order of decreasing coverage potential.
    # A class (m, r) covers floor(base/m) or ceil(base/m) residues.
    # Smaller m => more residues covered per class.
    
    # Precompute masks for each divisor
    # For each divisor m, the number of residues covered by any class is 
    # either base//m or base//m + 1 (if base % m != 0)
    # Since m divides base, every class covers exactly base//m residues.
    
    # Greedy approach: at each step, pick the (m, r) pair that covers 
    # the most currently uncovered residues.
    
    # Use numpy for efficient bitset operations
    # Represent covered set as a boolean array of length base
    covered = np.zeros(base, dtype=bool)
    uncovered_count = base
    classes = []
    
    # Precompute for each divisor m, the residue class indices
    # For efficiency, we'll iterate through moduli in a smart order
    
    # Key insight: smaller moduli cover more residues, so they should be 
    # considered first in terms of potential coverage. But we need to 
    # check which specific residue class covers the most uncovered.
    
    # Let's sort divisors by base//m descending (i.e., m ascending)
    # since smaller m means more residues per class
    divisors_sorted = sorted(divisors, key=lambda m: -base // m)
    
    used_moduli = set()
    
    # For efficiency, we'll use a priority-based approach
    # At each step, for each unused modulus, find the best residue class
    # and pick the one with maximum uncovered coverage
    
    # To speed up, we can precompute the residue class patterns
    # For modulus m, residue r covers indices r, r+m, r+2m, ...
    
    # Let's use a different approach: maintain uncovered as a set or array
    # and for each candidate (m,r), count how many uncovered residues it covers
    
    # For performance with base=51975, we can use numpy slicing
    # covered[r::m] gives us the elements at positions r, r+m, r+2m, ...
    
    # Greedy: repeatedly pick the (m, r) that covers the most uncovered residues
    # But checking all (m, r) pairs each iteration is expensive.
    
    # Alternative: process moduli in order of size (smallest first), 
    # and for each modulus, pick the best residue class.
    # This is the original approach but let's verify it's optimal enough.
    
    # Actually, let's try a smarter greedy: 
    # 1. Start with all residues uncovered
    # 2. For each step, evaluate all unused (m, r) pairs
    # 3. Pick the one covering the most uncovered residues
    # 4. Mark those as covered
    # 5. Repeat until no more improvement
    
    # But this is O(num_divisors * avg_class_size * num_steps) which could be slow.
    
    # Let's estimate: num_divisors of 51975 = 3^3 * 5^2 * 7 * 11
    # Number of divisors = (3+1)*(2+1)*(1+1)*(1+1) = 4*3*2*2 = 48
    # So 47 divisors > 1.
    
    # For each modulus m, there are m possible residue classes.
    # Total candidate pairs = sum of m for all divisors m > 1 of 51975.
    # This is manageable.
    
    # Let's implement the full greedy:
    # At each step, find the (m, r) that maximizes uncovered coverage.
    
    # Precompute all candidate (m, r) pairs
    candidates = []
    for m in divisors:
        for r in range(m):
            candidates.append((m, r))
    
    # Sort candidates by potential coverage (base // m) descending
    # This helps us check high-coverage candidates first
    candidates.sort(key=lambda x: -base // x[0])
    
    used_m = set()
    remaining = np.ones(base, dtype=bool)  # True = uncovered
    
    for _ in range(len(divisors)):  # at most len(divisors) steps
        best_gain = -1
        best_pair = None
        
        for m, r in candidates:
            if m in used_m:
                continue
            # Count uncovered residues in class (m, r)
            # Indices: r, r+m, r+2m, ...
            indices = np.arange(r, base, m)
            gain = remaining[indices].sum()
            if gain > best_gain:
                best_gain = gain
                best_pair = (m, r)
        
        if best_pair is None or best_gain <= 0:
            break
        
        m, r = best_pair
        used_m.add(m)
        classes.append([m, r])
        indices = np.arange(r, base, m)
        remaining[indices] = False
    
    return classes
