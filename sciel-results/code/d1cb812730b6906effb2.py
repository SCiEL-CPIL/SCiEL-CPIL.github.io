import time
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute all forbidden triples: a = bc/(b+c), a,b,c distinct, all in [1,180]
    # a = bc/(b+c) => a(b+c) = bc => ab + ac = bc
    # For given b,c: a = bc/(b+c). Must be integer.
    
    N = 180
    # Build adjacency: for each element, which triples does it participate in?
    # triple_map[x] = list of triples containing x
    
    triples = set()
    for b in range(1, N+1):
        for c in range(b+1, N+1):  # b < c to avoid duplicates
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    # Ensure distinct: a != b and a != c already checked
                    # Store as sorted tuple
                    t = tuple(sorted((a, b, c)))
                    triples.add(t)
    
    triples_list = list(triples)
    
    # Greedy: start with all elements, remove elements to break all triples
    # At each step, remove the element in the most remaining triples
    
    remaining = set(range(1, N+1))
    
    # Count how many triples each element is in (among those with all elements in remaining)
    # We'll maintain: for each element, count of active triples containing it
    
    # Initialize counts
    elem_triple_count = defaultdict(int)
    for t in triples_list:
        for x in t:
            elem_triple_count[x] += 1
    
    # Active triples: all start active
    active_triples = set(triples_list)
    
    removed = set()
    
    while active_triples and time.time() < deadline:
        # Find element with max count among elements in active triples
        # Only consider elements that are in at least one active triple
        max_count = -1
        best_elem = None
        for t in active_triples:
            for x in t:
                if x in remaining:
                    if elem_triple_count[x] > max_count:
                        max_count = elem_triple_count[x]
                        best_elem = x
        
        if best_elem is None:
            break
        
        # Remove best_elem
        remaining.discard(best_elem)
        removed.add(best_elem)
        
        # Update: remove all active triples containing best_elem
        to_remove = [t for t in active_triples if best_elem in t]
        for t in to_remove:
            active_triples.discard(t)
            for x in t:
                elem_triple_count[x] -= 1
    
    # If we still have time, try a local search / refinement
    # Try to add back removed elements if they don't create conflicts
    # Actually, let's try a different approach: local search on the remaining set
    
    # Try to improve: for each removed element, check if it can be added back
    # without creating a new forbidden triple
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in sorted(removed):
            # Check if adding x to remaining creates any forbidden triple
            can_add = True
            for t in triples_list:
                if x in t:
                    other_two = [y for y in t if y != x]
                    if all(y in remaining for y in other_two):
                        can_add = False
                        break
            if can_add:
                remaining.add(x)
                removed.discard(x)
                improved = True
    
    return sorted(remaining)

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
