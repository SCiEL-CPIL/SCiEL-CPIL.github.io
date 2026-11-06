"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from itertools import combinations
import time

def construct():
    deadline = time.time() + 50  # safety margin
    
    # Precompute all reciprocal identities within [1,24]
    # For each i, find all subsets T of [1,24]\{i} such that sum(1/j for j in T) == 1/i
    # We only need subsets of size 2 to 4 (larger subsets are less likely and we can bound)
    
    n = 24
    elements = list(range(1, n+1))
    
    # Find all conflict constraints: (target, frozenset(other_elements))
    # where 1/target == sum(1/j for j in other_elements)
    conflicts = []  # list of (target, frozenset)
    
    # For efficiency, only consider subsets up to size 4
    # 1/i = 1/a + 1/b: a*b = i*(a+b), so a and b > i
    # 1/i = 1/a + 1/b + 1/c: more complex
    # Let's enumerate subsets of size 2, 3, 4
    
    for i in elements:
        target = Fraction(1, i)
        others = [j for j in elements if j != i]
        
        # Size 2
        for a, b in combinations(others, 2):
            if Fraction(1, a) + Fraction(1, b) == target:
                conflicts.append((i, frozenset([a, b])))
        
        # Size 3
        for a, b, c in combinations(others, 3):
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                conflicts.append((i, frozenset([a, b, c])))
        
        # Size 4
        for a, b, c, d in combinations(others, 4):
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                conflicts.append((i, frozenset([a, b, c, d])))
        
        if time.time() > deadline:
            break
    
    # Now find the largest subset S of [1,24] such that for no (i, T) in conflicts
    # do we have i in S and T ⊆ S
    
    # Use backtracking with greedy ordering
    # Sort elements by degree of involvement in conflicts (most constrained first)
    
    # Build conflict map: for each element, which conflicts involve it
    elem_conflicts = {e: [] for e in elements}
    for idx, (target, T) in enumerate(conflicts):
        elem_conflicts[target].append(idx)
        for j in T:
            elem_conflicts[j].append(idx)
    
    # Order elements by number of conflicts they're involved in (descending)
    order = sorted(elements, key=lambda e: -len(elem_conflicts[e]))
    
    best_set = set()
    best_size = 0
    
    # Backtracking
    def backtrack(idx, current_set):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        if idx == len(order):
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Pruning: even if we add all remaining, can't beat best
        remaining = len(order) - idx
        if len(current_set) + remaining <= best_size:
            return
        
        e = order[idx]
        
        # Option 1: exclude e
        backtrack(idx + 1, current_set)
        
        # Option 2: include e, if no conflict is violated
        valid = True
        for idx_c in elem_conflicts[e]:
            target, T = conflicts[idx_c]
            if target == e:
                # Check if T ⊆ current_set
                if T.issubset(current_set):
                    valid = False
                    break
            else:
                # Check if e in T and target in current_set
                if e in T and target in current_set:
                    valid = False
                    break
        
        if valid:
            backtrack(idx + 1, current_set | {e})
    
    backtrack(0, set())
    
    if best_size > 0:
        return sorted(best_set)
    else:
        return [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# EVOLVE-BLOCK-END
