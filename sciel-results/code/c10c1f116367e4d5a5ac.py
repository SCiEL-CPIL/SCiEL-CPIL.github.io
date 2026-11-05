"""Rewritten program: maximize reciprocal sum of a subset S ⊆ [2,256]
such that lcm(a,b) > 256 for all distinct a,b in S."""

import time
import math
from fractions import Fraction
from itertools import combinations

# Precompute gcd table for speed
def _build_gcd_table(n):
    """Build a 2D list of gcds for 1..n."""
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    return gcd_table

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    MAX_VAL = 256
    # We want lcm(a,b) > 256, i.e., a*b/gcd(a,b) > 256
    # Equivalently: a*b > 256 * gcd(a,b)
    
    # Build compatibility: two numbers x,y (x != y) are compatible if lcm(x,y) > 256
    # Precompute which pairs are INCOMPATIBLE (lcm <= 256)
    
    # For each number, compute its list of incompatible partners
    incompatible = [set() for _ in range(MAX_VAL + 1)]
    
    for a in range(2, MAX_VAL + 1):
        for b in range(a + 1, MAX_VAL + 1):
            # lcm(a,b) = a * b // gcd(a,b)
            # Check if a*b//gcd(a,b) <= 256
            # To avoid large numbers, check: a * b <= 256 * gcd(a,b)
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                # Incompatible
                incompatible[a].add(b)
                incompatible[b].add(a)
    
    # Now we want to find a subset S ⊆ {2,...,256} maximizing sum(1/x)
    # subject to: for all distinct a,b in S, a and b are NOT in incompatible[a]
    
    # This is a maximum weight independent set in a graph where:
    # - Vertices: 2..256 (255 vertices)
    # - Edge between a,b if lcm(a,b) <= 256 (incompatible)
    # - Weight of vertex x: 1/x
    # We want max weight independent set.
    
    # Strategy: Greedy + local search with swaps
    # 1. Greedy: sort by weight descending (i.e., value ascending), add if compatible
    # 2. Local search: try single swaps (remove one, add one), 2-for-1, 1-for-2
    
    # Greedy initialization
    # Sort candidates by 1/x descending means x ascending
    candidates = list(range(2, MAX_VAL + 1))
    
    # Greedy: add in order of decreasing weight (increasing x), check compatibility
    selected = []
    selected_set = set()
    
    for x in sorted(candidates, key=lambda v: -Fraction(1, v)):  # decreasing weight
        if all(x not in incompatible[y] for y in selected_set):
            selected.append(x)
            selected_set.add(x)
    
    def compute_score(s):
        """Compute exact reciprocal sum as Fraction."""
        return sum(Fraction(1, x) for x in s)
    
    best_set = set(selected)
    best_score = compute_score(best_set)
    
    # Local search
    def is_compatible_with_set(x, s):
        """Check if x can be added to set s."""
        for y in s:
            if x in incompatible[y]:
                return False
        return True
    
    def local_search(current_set, current_score, time_limit):
        """Perform local search with single-swap, 2-for-1, 1-for-2 moves."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current_list = list(current_set)
            
            # Single swap: remove one, add one
            for i, rem in enumerate(current_list):
                rem_weight = Fraction(1, rem)
                for add in range(2, MAX_VAL + 1):
                    if add in current_set:
                        continue
                    # Check if add is compatible with all except rem
                    if is_compatible_with_set(add, current_set - {rem}):
                        new_score = current_score - rem_weight + Fraction(1, add)
                        if new_score > current_score:
                            current_set.discard(rem)
                            current_set.add(add)
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # 2-for-1: remove two, add one
            for rem1_idx in range(len(current_list)):
                rem1 = current_list[rem1_idx]
                for rem2_idx in range(rem1_idx + 1, len(current_list)):
                    rem2 = current_list[rem2_idx]
                    rem_weight = Fraction(1, rem1) + Fraction(1, rem2)
                    for add in range(2, MAX_VAL + 1):
                        if add in current_set:
                            continue
                        if is_compatible_with_set(add, current_set - {rem1, rem2}):
                            new_score = current_score - rem_weight + Fraction(1, add)
                            if new_score > current_score:
                                current_set.discard(rem1)
                                current_set.discard(rem2)
                                current_set.add(add)
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # 1-for-2: remove one, add two
            for rem in current_list:
                rem_weight = Fraction(1, rem)
                remaining = current_set - {rem}
                # Find two numbers that can be added together and with remaining
                found = False
                for add1 in range(2, MAX_VAL + 1):
                    if add1 in current_set:
                        continue
                    if not is_compatible_with_set(add1, remaining):
                        continue
                    for add2 in range(add1 + 1, MAX_VAL + 1):
                        if add2 in current_set:
                            continue
                        if add1 in incompatible[add2]:
                            continue
                        if not is_compatible_with_set(add2, remaining | {add1}):
                            continue
                        new_score = current_score - rem_weight + Fraction(1, add1) + Fraction(1, add2)
                        if new_score > current_score:
                            current_set.discard(rem)
                            current_set.add(add1)
                            current_set.add(add2)
                            current_score = new_score
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            if not improved and time.time() > time_limit - 5:
                break
        
        return current_set, current_score
    
    # Run local search
    best_set, best_score = local_search(set(best_set), best_score, deadline - 5)
    
    # Additional rounds with perturbation if time remains
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() >= deadline - 5:
                break
            # Perturb: remove a random low-weight element
            perturbed = set(best_set)
            if len(perturbed) > 1:
                # Remove the element with smallest weight (largest value)
                worst = max(perturbed, key=lambda x: x)
                perturbed.discard(worst)
                # Try to add back something different
                best_set, best_score = local_search(perturbed, compute_score(perturbed), deadline - 5)
    
    return list(best_set)
