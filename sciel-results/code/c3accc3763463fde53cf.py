"""Rewritten program: maximize exact reciprocal sum of a subset of [2,256]
with lcm(a,b) > 256 for all distinct pairs, using local search with swaps."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm, gcd
from fractions import Fraction
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Precompute lcm > 256 compatibility
    # compatible[i][j] = True iff lcm(i+2, j+2) > 256 for i != j
    # Actually let's use 0-indexed: element index i corresponds to value i+2
    N = 255  # values 2..256
    vals = list(range(2, 257))
    
    # Precompute which pairs are compatible (lcm > 256)
    # For speed, precompute a bitmask or just use a set of incompatible pairs
    # incompatible[i] = set of j such that lcm(vals[i], vals[j]) <= 256
    incompatible = [set() for _ in range(N)]
    for i in range(N):
        vi = vals[i]
        for j in range(i+1, N):
            vj = vals[j]
            if lcm(vi, vj) <= 256:
                incompatible[i].add(j)
                incompatible[j].add(i)
    
    def is_valid_with_set(i, selected_indices):
        """Check if index i is compatible with all indices in selected_indices."""
        for j in selected_indices:
            if j in incompatible[i]:
                return False
        return True
    
    def reciprocal_sum(selected_indices):
        """Compute exact reciprocal sum as Fraction."""
        s = Fraction(0, 1)
        for i in selected_indices:
            s += Fraction(1, vals[i])
        return s
    
    def reciprocal_sum_float(selected_indices):
        s = 0.0
        for i in selected_indices:
            s += 1.0 / vals[i]
        return s
    
    # Greedy initialization: add elements in order of largest reciprocal (smallest value)
    # but only if compatible with all already selected
    def greedy_init(order):
        selected = []
        selected_set = set()
        for i in order:
            if is_valid_with_set(i, selected_set):
                selected.append(i)
                selected_set.add(i)
        return selected
    
    # Generate multiple greedy inits with different orderings
    best_selected = None
    best_score = Fraction(0, 1)
    
    # Strategy 1: greedy by smallest value first (largest reciprocal)
    order1 = list(range(N))  # index 0 = value 2, index 1 = value 3, etc.
    sel1 = greedy_init(order1)
    score1 = reciprocal_sum(sel1)
    best_selected = sel1
    best_score = score1
    
    # Strategy 2: greedy by largest value first
    order2 = list(range(N-1, -1, -1))
    sel2 = greedy_init(order2)
    score2 = reciprocal_sum(sel2)
    if score2 > best_score:
        best_selected = sel2
        best_score = score2
    
    # Strategy 3: random orderings
    random.seed(42)
    for trial in range(20):
        if time.time() - start_time > DEADLINE - 30:
            break
        order = list(range(N))
        random.shuffle(order)
        sel = greedy_init(order)
        score = reciprocal_sum(sel)
        if score > best_score:
            best_selected = sel
            best_score = score
    
    # Local search: 1-swap moves (remove one, add one)
    # Also try 1-add (just add if possible)
    def local_search(selected):
        selected_set = set(selected)
        score = reciprocal_sum(selected)
        improved = True
        while improved:
            improved = False
            # Try adding each unselected element
            for i in range(N):
                if i not in selected_set:
                    if is_valid_with_set(i, selected_set):
                        new_score = score + Fraction(1, vals[i])
                        if new_score > score:
                            selected.append(i)
                            selected_set.add(i)
                            score = new_score
                            improved = True
            # Try 1-swap: remove j from selected, add i not in selected
            # Only consider swaps where i has larger reciprocal than j (i < j in value)
            # or where the net effect is positive
            for j in list(selected):
                vj = vals[j]
                # For each candidate i to add (not in selected, and compatible with selected minus j)
                # We need i compatible with all in selected_set - {j}
                for i in range(N):
                    if i in selected_set or i == j:
                        continue
                    # Check i compatible with selected_set - {j}
                    ok = True
                    for k in selected_set:
                        if k == j:
                            continue
                        if k in incompatible[i]:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Net change: +1/vals[i] - 1/vj
                    delta = Fraction(1, vals[i]) - Fraction(1, vj)
                    if delta > 0:
                        selected.remove(j)
                        selected_set.discard(j)
                        selected.append(i)
                        selected_set.add(i)
                        score += delta
                        improved = True
                        break  # restart scan after a swap
                if improved:
                    break
            if time.time() - start_time > DEADLINE - 5:
                break
        return selected, score
    
    # Run local search on best solution
    best_selected, best_score = local_search(best_selected)
    
    # Try local search on other initializations too
    for trial in range(10):
        if time.time() - start_time > DEADLINE - 10:
            break
        order = list(range(N))
        random.shuffle(order)
        sel = greedy_init(order)
        sel, score = local_search(sel)
        if score > best_score:
            best_selected = sel
            best_score = score
    
    # Convert back to values
    result = sorted([vals[i] for i in best_selected])
    return result
# EVOLVE-BLOCK-END
