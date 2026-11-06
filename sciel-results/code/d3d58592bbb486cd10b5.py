import time
import random
from itertools import combinations

def calc_alpha(triples):
    """
    Calculate the independence number of the 3-uniform hypergraph.
    Returns (alpha, max_independent_set)
    """
    n = 16
    triple_sets = [frozenset(t) for t in triples]
    
    # We want the largest k such that there exists a subset of size k with no triple.
    # We check from 16 down to 1.
    for k in range(n, 0, -1):
        for subset in combinations(range(n), k):
            s = frozenset(subset)
            is_independent = True
            for t_set in triple_sets:
                if t_set.issubset(s):
                    is_independent = False
                    break
            if is_independent:
                return k, subset
    return 1, (0,)

def is_linear(triple, existing_triples):
    """Check if a new triple is linear with all existing triples."""
    a, b, c = triple
    # Normalize pairs
    pairs = [tuple(sorted((a,b))), tuple(sorted((a,c))), tuple(sorted((b,c)))]
    
    for t in existing_triples:
        x, y, z = t
        old_set = {x, y, z}
        new_set = {a, b, c}
        inter = old_set.intersection(new_set)
        if len(inter) == 2:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_triples = None
    best_alpha = 16
    
    # Helper to add triple if linear
    def try_add(triples, new_triple):
        if is_linear(new_triple, triples):
            return triples + [new_triple], True
        return triples, False

    # Initial random linear 3-graph
    def random_init():
        triples = []
        used_pairs = set()
        points = list(range(16))
        random.shuffle(points)
        
        # Greedy random addition
        for _ in range(100):
            a, b, c = random.sample(points, 3)
            p1 = tuple(sorted((a,b)))
            p2 = tuple(sorted((a,c)))
            p3 = tuple(sorted((b,c)))
            
            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                triples.append([a,b,c])
        return triples

    # Strategy: Random restarts + Hole Punching
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        triples = random_init()
        if not triples:
            continue
            
        alpha, max_set = calc_alpha(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if best_alpha <= 5:
                break

        # Hole Punching Loop
        # We try to break the current largest independent set 'max_set'
        # by adding a triple contained within it.
        punched = False
        while not punched and time.time() < deadline:
            # Try to find a linear triple T subset of max_set
            found = False
            # Shuffle subsets to avoid bias
            subsets = list(combinations(max_set, 3))
            random.shuffle(subsets)
            
            for sub in subsets:
                # Check linearity
                if is_linear(sub, triples):
                    triples.append(list(sub))
                    punched = True
                    # Recompute alpha
                    alpha, max_set = calc_alpha(triples)
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_triples = triples
                        if best_alpha <= 5:
                            return best_triples
                    break
            
            if not punched:
                # If we couldn't punch the hole, we might be stuck in a local optimum
                # Try a "swap" or "remove" move to escape?
                # For now, break and restart. 
                # Or try to remove a triple that is NOT in max_set?
                # Let's just break and restart for simplicity and robustness.
                break
                
    return best_triples
