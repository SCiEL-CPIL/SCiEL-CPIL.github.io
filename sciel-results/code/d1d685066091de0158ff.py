"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import sys
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    def can_form_target(target_frac, selected_set):
        """Check if target_frac can be formed by a non-empty subset of reciprocals of selected_set."""
        if not selected_set:
            return False
        
        elems = sorted(selected_set)  # Smallest elements first (largest reciprocals)
        recips = [Fraction(1, y) for y in elems]
        n = len(recips)
        
        # Quick check: sum of all < target
        total = sum(recips)
        if total < target_frac:
            return False
        
        # Check single element
        for r in recips:
            if r == target_frac:
                return True
        
        # DFS with pruning
        suffix_sums = [0] * (n + 1)
        for idx in range(n - 1, -1, -1):
            suffix_sums[idx] = suffix_sums[idx + 1] + recips[idx]
        
        found = [False]
        
        def dfs(idx, acc):
            if found[0]:
                return
            if acc == target_frac:
                found[0] = True
                return
            if idx >= n:
                return
            if acc > target_frac:
                return
            if acc + suffix_sums[idx] < target_frac:
                return
            
            dfs(idx + 1, acc + recips[idx])
            dfs(idx + 1, acc)
        
        dfs(0, Fraction(0))
        return found[0]
    
    def is_valid_set(s):
        """Check if set s is valid (no reciprocal subsums)."""
        for elem in s:
            target = Fraction(1, elem)
            others = s - {elem}
            if can_form_target(target, others):
                return False
        return True
    
    def greedy_decreasing(start_set=None, order=None):
        """Greedy construction from largest to smallest, optionally starting from a set."""
        if start_set is None:
            start_set = set()
        selected = set(start_set)
        
        if order is None:
            order = range(24, 0, -1)
        
        for i in order:
            if time.time() > deadline:
                break
            if i in selected:
                continue
            target = Fraction(1, i)
            if not can_form_target(target, selected):
                selected.add(i)
        
        return selected
    
    # Phase 1: Standard greedy
    best = greedy_decreasing()
    best_size = len(best)
    
    # Phase 2: Try different starting elements
    for first in range(24, 0, -1):
        if time.time() > deadline:
            break
        selected2 = greedy_decreasing(start_set={first})
        if len(selected2) > best_size:
            best = selected2
            best_size = len(selected2)
    
    # Phase 3: Local search with single swaps (accept equal-size valid sets too for exploration)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = set(best)
        for remove in list(current):
            if time.time() > deadline:
                break
            for add in range(24, 0, -1):
                if add in current:
                    continue
                candidate = set(current)
                candidate.remove(remove)
                candidate.add(add)
                if is_valid_set(candidate):
                    if len(candidate) > len(current):
                        best = candidate
                        best_size = len(best)
                        improved = True
                        break
            if improved:
                break
    
    # Phase 4: Try adding elements one by one
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        if i in best:
            continue
        candidate = set(best)
        candidate.add(i)
        if is_valid_set(candidate):
            best = candidate
            best_size = len(best)
    
    # Phase 5: 2-for-2 swap local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = list(best)
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(current)):
                if time.time() > deadline:
                    break
                rem1, rem2 = current[i], current[j]
                for a in range(24, 0, -1):
                    if time.time() > deadline:
                        break
                    if a == rem1 or a == rem2 or a in best:
                        continue
                    for b in range(a - 1, 0, -1):
                        if time.time() > deadline:
                            break
                        if b == rem1 or b == rem2 or b in best:
                            continue
                        candidate = (best - {rem1, rem2}) | {a, b}
                        if len(candidate) > len(best):
                            continue
                        if is_valid_set(candidate) and len(candidate) > len(best):
                            best = candidate
                            best_size = len(best)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Phase 6: Randomized restarts
    rng = random.Random(42)
    for restart in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, 25))
        rng.shuffle(order)
        order.sort(reverse=True)
        selected2 = greedy_decreasing(order=order)
        if len(selected2) > best_size:
            best = selected2
            best_size = len(best)
    
    # Phase 7: Final polish
    for i in range(1, 25):
        if time.time() > deadline:
            break
        if i in best:
            continue
        candidate = set(best)
        candidate.add(i)
        if is_valid_set(candidate):
            best = candidate
            best_size = len(best)
    
    result = sorted(best)
    
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
