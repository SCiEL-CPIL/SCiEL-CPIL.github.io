# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/mZ of maximum
    size that does NOT contain any two elements differing by a nonzero perfect k-th power
    modulo m. Here m = 205 and k = 2 (squares), i.e. forbidden differences are the nonzero
    quadratic residues modulo 205.

    Uses greedy initialization followed by local search with 1-swap and 2-swap moves.

    Returns:
        subset: np.ndarray of shape (s,), dtype int, a list of distinct residues in
            {0, ..., 204} with no two differing by a nonzero square modulo 205.
    """
    m = 205
    k = 2
    
    # Deadline guard: 600s limit, use 580s for safety
    start_time = time.time()
    deadline = start_time + 580

    # Forbidden differences: the nonzero k-th power residues modulo m.
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Build conflict matrix: conflict[i][j] = True if (i-j)%m or (j-i)%m is in forbidden
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i+1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i][j] = True
                conflict[j][i] = True

    def is_valid_addition(candidate, selected_set):
        """Check if candidate can be added to selected_set without conflicts."""
        for s in selected_set:
            if conflict[candidate][s]:
                return False
        return True

    def greedy_init():
        """Greedy construction of initial independent set."""
        selected = []
        selected_set = set()
        for candidate in range(m):
            if is_valid_addition(candidate, selected_set):
                selected.append(candidate)
                selected_set.add(candidate)
        return selected, selected_set

    def local_search(selected, selected_set, max_iters=100000):
        """
        Local search with 1-swap (remove one, add one) and 2-swap (remove two, add two) moves.
        Returns improved selected list and set.
        """
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 30:
                break
            improved = False
            iters += 1
            
            # Try 1-swap: remove one element, try to add another
            for i in range(len(selected)):
                if time.time() > deadline - 30:
                    break
                removed = selected[i]
                new_selected = selected[:i] + selected[i+1:]
                new_selected_set = set(new_selected)
                
                # Try adding each non-selected element
                for candidate in range(m):
                    if candidate in new_selected_set:
                        continue
                    if is_valid_addition(candidate, new_selected_set):
                        # Accept the swap
                        new_selected.append(candidate)
                        new_selected_set.add(candidate)
                        selected = new_selected
                        selected_set = new_selected_set
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 2-swap: remove two elements, try to add two
            n = len(selected)
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline - 30:
                        break
                    removed1 = selected[i]
                    removed2 = selected[j]
                    new_selected = [x for x in selected if x != removed1 and x != removed2]
                    new_selected_set = set(new_selected)
                    
                    # Try adding pairs of non-selected elements
                    for c1 in range(m):
                        if c1 in new_selected_set:
                            continue
                        if is_valid_addition(c1, new_selected_set):
                            temp_set = new_selected_set | {c1}
                            for c2 in range(m):
                                if c2 in temp_set or c2 == c1:
                                    continue
                                if is_valid_addition(c2, temp_set):
                                    # Accept the 2-swap
                                    new_selected.append(c1)
                                    new_selected.append(c2)
                                    selected = new_selected
                                    selected_set = new_selected_set
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return selected, selected_set

    # Run multiple restarts with different random orderings
    best_selected = None
    best_size = 0

    # First, deterministic greedy
    selected, selected_set = greedy_init()
    selected, selected_set = local_search(selected, selected_set)
    if len(selected) > best_size:
        best_size = len(selected)
        best_selected = selected[:]

    # Randomized restarts
    random.seed(42)
    while time.time() < deadline - 60:
        # Random ordering for greedy
        candidates = list(range(m))
        random.shuffle(candidates)
        
        selected = []
        selected_set = set()
        for candidate in candidates:
            if is_valid_addition(candidate, selected_set):
                selected.append(candidate)
                selected_set.add(candidate)
        
        # Local search
        selected, selected_set = local_search(selected, selected_set)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
        
        # If we found size 12 or better, try more local search on that
        if best_size >= 12:
            # Additional local search passes
            for _ in range(3):
                if time.time() > deadline - 30:
                    break
                selected, selected_set = local_search(best_selected, set(best_selected), max_iters=50000)
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = selected[:]

    # Final check
    if best_selected is None:
        best_selected = [0]

    return np.array(sorted(best_selected), dtype=np.int64)


# EVOLVE-BLOCK-END
