import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Known solution as fallback
    KNOWN = [0, 1, 4, 10, 12, 28, 32, 45, 63]
    
    def is_valid(arr):
        if len(arr) > 32 or 0 not in arr or max(arr) > 10**6:
            return False
        diffs = set()
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))
    
    # Bitmask-based BFS/Dijkstra to find minimum diameter
    # State: (sorted_tuple_of_selected, used_diffs_bitmask, max_element)
    # We use a priority queue keyed by max_element (diameter)
    
    import heapq
    
    def solve_min_diameter(max_diameter, max_time):
        """
        Find a Sidon set with diameter <= max_diameter using BFS with bitmask.
        Returns the set if found, None otherwise.
        """
        # State: (diameter, selected_tuple, used_mask)
        # used_mask: bit i set means difference (i+1) is in used
        # We track all differences, not just 1..30, to ensure Sidon property
        
        # Actually, we need to track ALL differences to ensure Sidon property.
        # Let's use a different approach: track the set of used differences explicitly
        # but use bitmask for coverage of 1..30
        
        # Better approach: use Dijkstra where state is (max_elem, frozenset_of_used_diffs, frozenset_of_selected)
        # This is too expensive. Let's use a smarter approach.
        
        # Alternative: iterative deepening with memoization
        # State: (last_selected, used_diffs_frozen, coverage_mask)
        # But used_diffs_frozen is too large
        
        # Let's use a different strategy: backtracking with strong pruning
        # and bitmask for coverage
        
        best = None
        
        def backtrack(selected, used_diffs, coverage_mask, next_min):
            nonlocal best
            if time.time() > max_time:
                return
            if coverage_mask == (1 << 30) - 1:
                # All differences 1..30 covered
                if best is None or max(selected) < max(best):
                    best = selected[:]
                return
            
            if len(selected) >= 32:
                return
            
            if next_min > max_diameter:
                return
            
            # Pruning: can we still cover all remaining differences?
            remaining = (1 << 30) - 1 - coverage_mask
            if remaining.bit_count() > 0:
                # Each new element can cover at most (len(selected)) new differences
                # (differences to each existing element)
                max_additional = (max_diameter - next_min + 1) * len(selected)
                if max_additional < remaining.bit_count():
                    return
            
            for x in range(next_min, max_diameter + 1):
                if x in selected:
                    continue
                
                # Check if adding x creates duplicate differences
                new_diffs = []
                valid = True
                for s in selected:
                    d = x - s
                    if d <= 0:
                        continue
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Compute new coverage
                new_coverage = coverage_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_coverage |= (1 << (d - 1))
                
                # Recurse
                new_selected = selected + [x]
                new_used = used_diffs | set(new_diffs)
                backtrack(new_selected, new_used, new_coverage, x + 1)
                
                if best is not None and time.time() > max_time:
                    return
        
        backtrack([0], set(), 0, 1)
        return best
    
    # Try increasing diameters
    for D in range(45, 151):
        if time.time() > deadline - 10:
            break
        r = solve_min_diameter(D, deadline - 5)
        if r is not None and is_valid(r):
            return r
    
    # Try larger diameters with more time
    for D in range(150, 301):
        if time.time() > deadline - 3:
            break
        r = solve_min_diameter(D, deadline - 2)
        if r is not None and is_valid(r):
            return r
    
    # Greedy fallback
    def greedy():
        selected = [0]
        used = set()
        for target in range(1, 31):
            if target in used:
                continue
            best_x = None
            best_cover = -1
            for x in range(1, 5000):
                if x in selected:
                    continue
                conflict = False
                new_diffs = []
                for s in selected:
                    d = x - s
                    if d <= 0:
                        continue
                    if d in used:
                        conflict = True
                        break
                    new_diffs.append(d)
                if conflict:
                    continue
                cover = sum(1 for d in new_diffs if 1 <= d <= 30 and d not in used)
                if target in new_diffs:
                    best_x = x
                    break
                elif cover > best_cover:
                    best_x = x
                    best_cover = cover
            if best_x is not None:
                selected.append(best_x)
                for s in selected[:-1]:
                    d = best_x - s
                    if d > 0:
                        used.add(d)
        if is_valid(selected):
            return sorted(selected)
        return None
    
    result = greedy()
    if result is not None:
        return result
    
    r = solve_min_diameter(500, deadline - 1)
    if r is not None and is_valid(r):
        return r
    
    return KNOWN

# EVOLVE-BLOCK-END
