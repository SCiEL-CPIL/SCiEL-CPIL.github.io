import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Candidate moduli: all integers from 20 to 200
    # We want to maximize sum of 1/m, so prefer smaller m
    # Greedy: try to include moduli in order of increasing m
    # For each, find a compatible residue
    
    def is_compatible(placed, m, r):
        """Check if residue r for modulus m is compatible with all placed classes."""
        for m2, r2 in placed:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def find_residue(placed, m):
        """Find any residue r in [0, m) compatible with placed classes."""
        for r in range(m):
            if is_compatible(placed, m, r):
                return r
        return None
    
    # Greedy: process moduli from smallest to largest
    # This maximizes the chance of including small-m moduli
    best_result = None
    best_score = Fraction(0, 1)
    
    # Try greedy in order of increasing m
    placed = []
    score = Fraction(0, 1)
    for m in range(20, 201):
        r = find_residue(placed, m)
        if r is not None:
            placed.append((m, r))
            score += Fraction(1, m)
    
    if placed:
        best_result = [[m, r] for m, r in placed]
        best_score = score
    
    # Check time
    if time.time() > deadline:
        return best_result
    
    # Try to improve: remove some moduli to add others with better 1/m ratios
    # Actually, greedy in increasing m should be near-optimal for this structure
    # But let's try a few different orderings
    
    # Try: prioritize moduli with many small prime factors (more gcd conflicts)
    # vs. moduli that are prime (fewer conflicts)
    
    # Alternative: use only moduli that are multiples of a small number
    # e.g., multiples of 2: can have at most 2 distinct residues mod 2
    # multiples of 3: at most 3 residues mod 3
    # etc.
    
    # Let's try a smarter approach: use moduli that are pairwise "independent"
    # Key insight: if gcd(m1, m2) = 1, then any residues work (always disjoint)
    # So we want moduli with small pairwise gcds
    
    # Generate candidates: primes in [20,200], and products of small primes
    # But let's stick with the greedy for now and verify it's good
    
    # Actually, let's try a local search: swap out some moduli for better ones
    # This is complex, so let's just return the greedy result
    
    return best_result
