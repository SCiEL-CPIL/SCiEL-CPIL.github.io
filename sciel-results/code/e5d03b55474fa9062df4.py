# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses optimized greedy construction, randomized restarts, and efficient local search.
    """
    m = 205
    k = 2
    deadline = time.time() + 580.0  # 580s safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # For fast lookup, also create a boolean array
    forbidden_arr = np.zeros(m, dtype=bool)
    for f in forbidden:
        forbidden_arr[f] = True
    
    # Precompute conflict neighbors: for each residue, which other residues conflict with it
    # conflict_neighbors[i] = set of j such that (i-j) % m is in forbidden
    conflict_neighbors = [set() for _ in range(m)]
    for i in range(m):
        for f in forbidden:
            j = (i - f) % m
            conflict_neighbors[i].add(j)
            j2 = (i + f) % m
            conflict_neighbors[i].add(j2)
    
    def is_valid_addition(candidate, selected_set):
        """Check if candidate can be added to selected_set without creating a forbidden pair."""
        # Check against all selected elements
        for s in selected_set:
            if s in conflict_neighbors[candidate]:
                return False
        return True

    def greedy_construct(order):
        """Greedy construction using a specific order of candidates."""
        selected = []
        selected_set = set()
        for candidate in order:
            if candidate not in selected_set:
                # Fast check: see if candidate conflicts with any selected
                valid = True
                for s in selected_set:
                    if s in conflict_neighbors[candidate]:
                        valid = False
                        break
                if valid:
                    selected.append(candidate)
                    selected_set.add(candidate)
        return selected

    def local_search(initial_set, max_iterations=200):
        """Efficient local search: try to add elements, and if stuck, try targeted swaps."""
        selected = list(initial_set)
        selected_set = set(selected)
        best_size = len(selected)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding any element that doesn't conflict
            for candidate in range(m):
                if candidate not in selected_set:
                    valid = True
                    for s in selected_set:
                        if s in conflict_neighbors[candidate]:
                            valid = False
                            break
                    if valid:
                        selected.append(candidate)
                        selected_set.add(candidate)
                        best_size = len(selected)
                        improved = True
                        break
            
            if improved:
                continue
            
            # If we can't add, try a swap: remove one element, then add a different one
            # Only try swaps that might enable future additions
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                # Build new set without this element
                new_selected_set = set(selected_set)
                new_selected_set.discard(removed)
                
                # Find candidates that can be added to the reduced set
                added_any = False
                for candidate in range(m):
                    if candidate in new_selected_set or candidate == removed:
                        continue
                    valid = True
                    for s in new_selected_set:
                        if s in conflict_neighbors[candidate]:
                            valid = False
                            break
                    if valid:
                        # Check if this swap enables at least one more addition
                        test_set = new_selected_set | {candidate}
                        can_add_more = False
                        for c in range(m):
                            if c in test_set:
                                continue
                            v = True
                            for s in test_set:
                                if s in conflict_neighbors[c]:
                                    v = False
                                    break
                            if v:
                                can_add_more = True
                                break
                        if can_add_more:
                            selected_set = test_set
                            selected = list(selected_set)
                            best_size = len(selected)
                            improved = True
                            added_any = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        return selected

    def randomized_greedy(n_restarts=100):
        """Try multiple random orderings."""
        best = []
        for _ in range(n_restarts):
            if time.time() > deadline:
                break
            order = list(range(m))
            random.shuffle(order)
            result = greedy_construct(order)
            if len(result) > len(best):
                best = result
        return best

    # Phase 1: Deterministic greedy with natural order
    best_result = greedy_construct(list(range(m)))
    best_size = len(best_result)
    
    # Phase 2: Randomized greedy (more restarts)
    random.seed(42)
    result = randomized_greedy(n_restarts=300)
    if len(result) > best_size:
        best_result = result
        best_size = len(result)
    
    # Phase 3: Local search from best result
    if time.time() < deadline - 30:
        result = local_search(best_result, max_iterations=300)
        if len(result) > best_size:
            best_result = result
            best_size = len(result)
    
    # Phase 4: More randomized restarts with local search
    random.seed(123)
    for restart in range(80):
        if time.time() > deadline - 30:
            break
        order = list(range(m))
        random.shuffle(order)
        result = greedy_construct(order)
        if len(result) >= best_size - 2:  # Allow slightly smaller to try improving
            result = local_search(result, max_iterations=150)
            if len(result) > best_size:
                best_result = result
                best_size = len(result)
    
    # Phase 5: Additional local search passes from different starting points
    if time.time() < deadline - 15:
        # Try local search from a few more random greedy solutions
        random.seed(777)
        for _ in range(30):
            if time.time() > deadline - 15:
                break
            order = list(range(m))
            random.shuffle(order)
            result = greedy_construct(order)
            result = local_search(result, max_iterations=100)
            if len(result) > best_size:
                best_result = result
                best_size = len(result)
    
    # Final validation
    for i, a in enumerate(best_result):
        for b in best_result[i+1:]:
            d1 = (a - b) % m
            d2 = (b - a) % m
            if forbidden_arr[d1] or forbidden_arr[d2]:
                # Should not happen, but if it does, return empty to be safe
                return np.array([], dtype=np.int64)
    
    return np.array(best_result, dtype=np.int64)


# EVOLVE-BLOCK-END
