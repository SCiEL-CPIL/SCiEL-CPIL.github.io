import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 180
    numbers = list(range(1, N + 1))
    
    # Precompute all forbidden triples
    # 1/a = 1/b + 1/c => a = bc/(b+c), a < b, a < c, all distinct
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if a < b and 1 <= a <= N:
                    triples.append((a, b, c))
    
    # Build conflict structure: for each element, which triples it participates in
    # We want to find max subset with no complete triple
    
    # Start with a good initial solution: all numbers > 90
    current = set(range(91, N + 1))
    
    # Greedy: try to add elements from 1 to 90
    # For each candidate, check if adding it creates any forbidden triple
    def has_conflict(s, candidate):
        """Check if adding candidate to set s creates a forbidden triple."""
        for (a, b, c) in triples:
            # Check if all three would be in s ∪ {candidate}
            elems = {a, b, c}
            if candidate in elems:
                others = elems - {candidate}
                if all(o in s for o in others):
                    return True
        return False
    
    # Try adding elements in order of "least conflict"
    # First, count how many triples each small number participates in
    conflict_count = {i: 0 for i in range(1, 91)}
    for (a, b, c) in triples:
        for elem in (a, b, c):
            if elem <= 90:
                conflict_count[elem] += 1
    
    # Try adding elements with fewest conflicts first
    candidates = sorted(range(1, 91), key=lambda x: conflict_count[x])
    
    for cand in candidates:
        if time.time() > deadline - 5.0:
            break
        if not has_conflict(current, cand):
            current.add(cand)
    
    # Local search: try to swap out one element for another to improve
    # Actually, let's try a different approach: simulated annealing or hill climbing
    # Try removing and re-adding
    
    # Let's try multiple restarts with different orderings
    best = set(current)
    
    # Try adding in different orders
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        random.seed(restart)
        cur = set(range(91, N + 1))
        candidates = list(range(1, 91))
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline - 5.0:
                break
            if not has_conflict(cur, cand):
                cur.add(cand)
        
        if len(cur) > len(best):
            best = set(cur)
    
    return list(best)
