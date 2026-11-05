"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin 10s
    
    def get_seed_graph():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    best_n, best_edges = get_seed_graph()
    
    # Targets to try, descending
    # Exoo found 42. Let's try 42, 41, 40.
    targets = [42, 41, 40, 39, 38]
    
    for target_n in targets:
        if time.time() > deadline:
            break
        
        time_budget = deadline - time.time()
        if time_budget < 5:
            break
            
        # Try to solve for target_n
        found = solve_for_n(target_n, time_budget)
        if found:
            n, edges = found
            if n > best_n:
                best_n, best_edges = n, edges
                # If we found a very good one (e.g. 42), we can stop or continue?
                # Since 42 is the known max lower bound, we can stop if we find 42.
                if n >= 42:
                    break
    return best_n, best_edges

def solve_for_n(n, time_budget):
    # Randomized local search
    # Use a set of edges for O(1) lookup
    # Init random graph
    random.seed(42 + n) # Reproducibility per N? Or just random.
    
    # We will try multiple restarts
    max_restarts = max(1, int(time_budget / 10)) # 10s per restart
    
    for r in range(max_restarts):
        if time.time() > time_budget:
            break
            
        edges_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    edges_set.add((i, j))
                    
        # Initial count
        # We don't strictly need the count, just find violations
        
        while time.time() < time_budget:
            # Find one violation
            combo, type_v = find_one_violation(n, edges_set)
            if not combo:
                # No violation found
                # Convert set to list of tuples
                edges_list = list(edges_set)
                return n, edges_list
            
            # Try to fix by flipping an edge in the combo
            fixed = False
            # Get all pairs in combo
            pairs = list(itertools.combinations(combo, 2))
            random.shuffle(pairs)
            
            for u, v in pairs:
                if u > v: u, v = v, u
                edge = (u, v)
                
                # Current state
                if edge in edges_set:
                    # Remove it
                    edges_set.remove(edge)
                else:
                    # Add it
                    edges_set.add(edge)
                    
                # Check if this flip resolved the specific violation and didn't create too many new ones?
                # Simple heuristic: Just check if the specific combo is fixed.
                # And maybe do a light check.
                # For robustness, let's just accept the flip if it fixes the combo.
                # The loop will catch new violations in the next iteration.
                # However, we might oscillate.
                # Let's check if the count of violations decreased? Too slow.
                # Let's just accept the flip.
                fixed = True
                break
            
            if not fixed:
                # Should not happen, but if so, revert and break?
                # Actually, if we removed/added, we changed the graph.
                # If we couldn't fix the *specific* combo, we still changed the graph.
                # Let's just continue.
                pass

    return None

def find_one_violation(n, edges_set):
    for combo in itertools.combinations(range(n), 5):
        is_k5 = True
        is_ik5 = True
        for i in range(5):
            for j in range(i+1, 5):
                u, v = combo[i], combo[j]
                if u > v: u, v = v, u
                if (u, v) in edges_set:
                    is_ik5 = False
                else:
                    is_k5 = False
                if not is_k5 and not is_ik5:
                    break
            if not is_k5 and not is_ik5:
                break
        if is_k5:
            return combo, 'K5'
        if is_ik5:
            return combo, 'IK5'
    return None, None
# EVOLVE-BLOCK-END
