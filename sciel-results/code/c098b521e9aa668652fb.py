import math
import time
import random

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    
    Key Insight:
    Two residue classes (r1, m1) and (r2, m2) are disjoint iff r1 % gcd(m1, m2) != r2 % gcd(m1, m2).
    
    Strategy:
    1. Use a randomized greedy approach to select a subset of moduli and assign residues.
    2. Start with a deterministic baseline (multiples of 11) and then try to improve via randomized search.
    3. The search tries to add more moduli by finding valid residues.
    """
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    def is_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def can_add(r, m, existing):
        for r_e, m_e in existing:
            if not is_disjoint(r, m, r_e, m_e):
                return False
        return True
    
    def find_best_residue(m, existing):
        # Find a residue r in [0, m) that is disjoint with all existing classes
        # For efficiency, we can check residues in a smart order
        # First try 0, then 1, ... but also consider residues that are "far" from existing ones mod gcd
        for r in range(m):
            if can_add(r, m, existing):
                return r
        return None
    
    def greedy_construct(moduli_order, existing):
        """Try to add moduli in the given order, assigning residues greedily."""
        result = list(existing)
        for m in moduli_order:
            if m in [m_e for _, m_e in result]:
                continue
            r = find_best_residue(m, result)
            if r is not None:
                result.append((r, m))
        return result
    
    # Baseline: multiples of 11
    baseline_moduli = [11 * k for k in range(1, 12)]
    baseline = []
    for idx, m in enumerate(baseline_moduli):
        baseline.append((idx % m, m))
    
    best_solution = baseline
    best_size = len(baseline)
    
    # All possible moduli 2..128
    all_moduli = list(range(2, 129))
    
    # Try to improve by adding more moduli
    # Strategy: Randomly shuffle moduli and try greedy addition
    # Also try different starting points
    
    num_attempts = 0
    max_attempts = 5000
    
    while time.time() < deadline and num_attempts < max_attempts:
        num_attempts += 1
        
        # Random strategy: shuffle moduli and try greedy
        random.shuffle(all_moduli)
        
        # Start with empty set or a small random subset
        if num_attempts % 3 == 0:
            # Start empty
            current = []
        elif num_attempts % 3 == 1:
            # Start with a random subset of baseline
            current = []
            random.shuffle(baseline)
            for item in baseline[:random.randint(3, 6)]:
                current.append(item)
        else:
            # Start with best solution so far
            current = list(best_solution)
        
        # Try to add moduli in random order
        # Skip moduli already in current
        current_moduli_set = set(m for _, m in current)
        remaining = [m for m in all_moduli if m not in current_moduli_set]
        random.shuffle(remaining)
        
        for m in remaining:
            if time.time() > deadline:
                break
            r = find_best_residue(m, current)
            if r is not None:
                current.append((r, m))
        
        if len(current) > best_size:
            best_size = len(current)
            best_solution = current
    
    # Convert to required format
    result = []
    for r, m in best_solution:
        result.append([r, m])
    
    return result
