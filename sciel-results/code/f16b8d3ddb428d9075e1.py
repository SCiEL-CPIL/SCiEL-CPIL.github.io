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

    Returns:
        subset: np.ndarray of shape (s,), dtype int, a list of distinct residues in
            {0, ..., 204} with no two differing by a nonzero square modulo 205.
    """
    m = 205
    k = 2
    deadline = time.time() + 580  # Safety margin before 600s limit

    # Forbidden differences: the nonzero k-th power residues modulo m.
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Precompute for each element, which other elements are forbidden with it
    # Using bitmasks for fast operations
    # forbidden_mask[i] is a bitmask where bit j is set if j is forbidden with i
    forbidden_mask = [0] * m
    for i in range(m):
        mask = 0
        for f in forbidden:
            j1 = (i + f) % m
            j2 = (i - f) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        forbidden_mask[i] = mask

    def greedy(order):
        """Greedy: process candidates in given order, add if compatible with all selected."""
        selected = []
        selected_mask = 0
        for candidate in order:
            # Check if candidate conflicts with any selected
            if (selected_mask & (1 << candidate)) == 0:
                # Check against all selected vertices
                ok = True
                for s in selected:
                    if (forbidden_mask[s] >> candidate) & 1:
                        ok = False
                        break
                if ok:
                    selected.append(candidate)
                    selected_mask |= (1 << candidate)
        return selected

    def greedy_min_degree(seed_val):
        """
        Greedy independent set construction using minimum-degree heuristic.
        """
        rng = random.Random(seed_val)
        candidates = set(range(m))
        selected = []
        
        while candidates:
            min_deg = m
            min_nodes = []
            for v in candidates:
                deg = 0
                for u in candidates:
                    if u != v and ((forbidden_mask[v] >> u) & 1):
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [v]
                elif deg == min_deg:
                    min_nodes.append(v)
            
            v = rng.choice(min_nodes)
            selected.append(v)
            candidates.discard(v)
            # Remove all neighbors of v from candidates
            for u in list(candidates):
                if (forbidden_mask[v] >> u) & 1:
                    candidates.discard(u)
                    
        return selected

    def local_search_improve(initial_set, seed_val):
        """
        Improve an initial independent set using local search with exchange moves.
        """
        rng = random.Random(seed_val)
        current = set(initial_set)
        best = set(current)
        best_size = len(current)
        
        def can_add(v, s):
            """Check if v can be added to set s."""
            for u in s:
                if (forbidden_mask[u] >> v) & 1:
                    return False
            return True

        def try_add_maximally(s):
            """Try to add as many vertices as possible to s."""
            improved = True
            while improved:
                improved = False
                for v in range(m):
                    if v not in s and can_add(v, s):
                        s.add(v)
                        improved = True
                        break
            return s

        # Phase 1: Greedy addition
        current = try_add_maximally(set(current))
        if len(current) > best_size:
            best = set(current)
            best_size = len(current)

        # Phase 2: Exchange-based local search
        # Try 1-k exchanges: remove one vertex, try to add multiple others
        max_exchange_rounds = 100
        for round_num in range(max_exchange_rounds):
            if time.time() > deadline:
                break
            
            improved_this_round = False
            
            # Try removing each vertex and see if we can add more
            for u in list(current):
                if time.time() > deadline:
                    break
                
                new_set = set(current)
                new_set.discard(u)
                
                # Try to add vertices greedily
                new_set = try_add_maximally(new_set)
                
                if len(new_set) > len(current):
                    current = new_set
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
                    improved_this_round = True
                    break  # Restart with new current
            else:
                # No improvement this round, try random 1-1 swaps to escape local minima
                if current:
                    u = rng.choice(list(current))
                    new_set = set(current)
                    new_set.discard(u)
                    
                    # Find candidates that can be added
                    candidates_to_add = [v for v in range(m) if v not in new_set and can_add(v, new_set)]
                    
                    if candidates_to_add:
                        v = rng.choice(candidates_to_add)
                        new_set.add(v)
                        new_set = try_add_maximally(new_set)
                        
                        # Accept with some probability even if same size (to explore)
                        if len(new_set) >= len(current):
                            if len(new_set) > len(current) or rng.random() < 0.2:
                                current = new_set
                                if len(current) > best_size:
                                    best = set(current)
                                    best_size = len(current)
                                improved_this_round = True
                if not improved_this_round and round_num > 20:
                    break  # Stagnation detected
            
        return best

    # Start with deterministic greedy in natural order
    best = greedy(list(range(m)))
    best_size = len(best)

    # Run multiple heuristics
    attempts = 0
    
    # First, try greedy_min_degree runs
    for i in range(30):
        if time.time() > deadline:
            break
        result = greedy_min_degree(i)
        if len(result) > best_size:
            best = list(result)
            best_size = len(result)

    # Now, run local search on the best set found so far
    if time.time() < deadline - 10:
        improved_best = local_search_improve(best, 42)
        if len(improved_best) > best_size:
            best = list(improved_best)
            best_size = len(improved_best)

    # Continue with random greedy and local search improvements
    while time.time() < deadline - 5:
        # Alternate between random greedy and min-degree greedy
        if attempts % 2 == 0:
            order = list(range(m))
            random.shuffle(order)
            result = greedy(order)
        else:
            result = greedy_min_degree(attempts)
        
        # Apply local search to the result if it's promising
        if len(result) >= best_size - 2:  # Only improve if close to best
            improved_result = local_search_improve(result, attempts)
            if len(improved_result) > best_size:
                best = list(improved_result)
                best_size = len(improved_result)
        
        attempts += 1

    return np.array(best, dtype=np.int64)


# EVOLVE-BLOCK-END
