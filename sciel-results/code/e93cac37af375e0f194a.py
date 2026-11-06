"""Sidon set construction with backtracking and heuristic ordering."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Prescribed elements
    base = [1, 2, 5, 11]
    
    # Precompute initial sums
    def compute_sums(elements):
        sums = set()
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                sums.add(elements[i] + elements[j])
        return sums
    
    initial_sums = compute_sums(base)
    
    # Try to build a large Sidon set using backtracking with heuristics
    def can_add(element, current_elements, current_sums):
        """Check if element can be added to maintain Sidon property."""
        new_sums = set()
        for x in current_elements:
            s = element + x
            if s in current_sums or s in new_sums:
                return False
            new_sums.add(s)
        # Check 2*element
        double_s = 2 * element
        if double_s in current_sums or double_s in new_sums:
            return False
        new_sums.add(double_s)
        return True
    
    def get_conflict_count(element, current_elements, current_sums):
        """Count how many sums would conflict if we added element."""
        conflicts = 0
        for x in current_elements:
            s = element + x
            if s in current_sums:
                conflicts += 1
        double_s = 2 * element
        if double_s in current_sums:
            conflicts += 1
        return conflicts
    
    # Heuristic: order candidates by how "spread out" they are
    def heuristic_order(candidates, current_elements):
        """Sort candidates to try most promising ones first."""
        if not candidates:
            return []
        
        # Prefer elements that are far from existing elements to reduce sum collisions
        def score(x):
            # Distance to nearest existing element
            min_dist = min(abs(x - e) for e in current_elements) if current_elements else x
            # Also consider distance to existing sums
            return min_dist
        
        return sorted(candidates, key=score, reverse=True)
    
    best_solution = list(base)
    best_size = len(base)
    
    def backtrack(current_elements, current_sums, candidates, depth):
        nonlocal best_solution, best_size, deadline
        
        if time.time() > deadline:
            return
        
        if len(current_elements) > best_size:
            best_size = len(current_elements)
            best_solution = list(current_elements)
        
        if not candidates:
            return
        
        # Try adding each candidate
        for idx, candidate in enumerate(candidates):
            if time.time() > deadline:
                return
            
            if can_add(candidate, current_elements, current_sums):
                new_elements = current_elements + [candidate]
                new_sums = current_sums | {candidate + x for x in current_elements} | {2 * candidate}
                
                # Remaining candidates
                remaining = [c for c in candidates[idx+1:] if c > candidate]
                
                backtrack(new_elements, new_sums, remaining, depth + 1)
                
                # If we've already found a solution of size 20 or more, we can stop
                if best_size >= 20:
                    return
    
    # Generate candidate elements in [1, 284] excluding base elements
    all_candidates = [x for x in range(1, 285) if x not in set(base)]
    
    # Use heuristic ordering for initial candidates
    ordered_candidates = heuristic_order(all_candidates, base)
    
    # Run backtracking
    backtrack(base, initial_sums, ordered_candidates, 0)
    
    # If backtracking didn't find good solution, try greedy approach as fallback
    if len(best_solution) < 15:
        greedy_solution = list(base)
        greedy_sums = initial_sums
        
        for x in range(12, 285):
            if time.time() > deadline:
                break
            if can_add(x, greedy_solution, greedy_sums):
                greedy_solution.append(x)
                greedy_sums |= {x + y for y in greedy_solution} | {2 * x}
        
        if len(greedy_solution) > len(best_solution):
            best_solution = greedy_solution
    
    return best_solution
# EVOLVE-BLOCK-END
