"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    # Greedy algorithm: add elements from largest to smallest.
    # A set S is valid if no reciprocal of an element in S is a sum of reciprocals of a subset of S \ {element}.
    # If we add elements in decreasing order, when adding k, all existing elements j > k.
    # 1. Can 1/k be formed by a subset of existing S?
    #    If yes, then 1/k = sum(1/j for j in T). This is a violation (k is target).
    # 2. Can 1/j (existing) be formed by a subset of S U {k} \ {j}?
    #    This requires 1/j = 1/k + sum(...).
    #    Since j > k, 1/j < 1/k, so 1/j - 1/k < 0.
    #    Sum of positive reciprocals cannot be negative.
    #    So this is impossible.
    # Thus, we only need to check if 1/k is a subset sum of the current S.
    
    S = []
    reachable = {Fraction(0)}
    
    for k in range(24, 0, -1):
        target = Fraction(1, k)
        if target in reachable:
            continue
        # Add k to S
        S.append(k)
        # Update reachable sums
        new_reachable = set(reachable)
        for s in reachable:
            new_reachable.add(s + target)
        reachable = new_reachable
        
    return S
# EVOLVE-BLOCK-END
