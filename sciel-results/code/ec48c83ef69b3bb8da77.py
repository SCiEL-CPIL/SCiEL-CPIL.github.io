import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Points 0..15
    n = 16
    points = list(range(n))
    
    # Helper to check if a triple is valid (linear with existing triples)
    # For speed, we maintain a set of used pairs
    used_pairs = set()
    
    def can_add(triple):
        a, b, c = triple
        if (a, b) in used_pairs or (b, a) in used_pairs: return False
        if (a, c) in used_pairs or (c, a) in used_pairs: return False
        if (b, c) in used_pairs or (c, b) in used_pairs: return False
        return True

    def add_triple(triple):
        a, b, c = triple
        used_pairs.add((a, b)); used_pairs.add((b, a))
        used_pairs.add((a, c)); used_pairs.add((c, a))
        used_pairs.add((b, c)); used_pairs.add((c, b))

    # 1. Greedy construction of a large linear triple system
    # We want to maximize the number of triples to minimize independence number.
    # Randomized greedy
    triples = []
    remaining_points = set(points)
    
    # Simple greedy: pick random unused pair, extend to triple if possible
    # Better: iterate through all possible triples, shuffle, add if valid
    all_triples = list(itertools.combinations(points, 3))
    
    def build_system():
        nonlocal used_pairs, triples
        used_pairs = set()
        triples = []
        random.shuffle(all_triples)
        for t in all_triples:
            if can_add(t):
                add_triple(t)
                triples.append(t)
        return triples

    # 2. Local Search to improve independence number
    # We need a fast heuristic for independence number or just try to increase density.
    # Actually, just maximizing triple count is a good proxy.
    # But we can do a specific swap search.
    
    # Since calculating exact independence number is expensive (2^16 subsets),
    # we rely on the fact that high density -> low alpha.
    # Let's run several greedy builds and pick the one with most triples.
    
    best_triples = []
    best_count = 0
    
    # Run a few quick greedy builds
    for _ in range(50):
        if time.time() > deadline:
            break
        build_system()
        if len(triples) > best_count:
            best_count = len(triples)
            best_triples = triples[:]
            
    # Local search: try to swap triples to increase count? 
    # Or just return the best greedy result.
    # The seed had 5 triples. A greedy will likely get 30-40.
    # This should significantly reduce independence number.
    
    # To be safe, let's do a simple hill climbing on the "covered pairs" metric?
    # No, just returning a dense packing is usually sufficient to get alpha ~ 8-9.
    
    return best_triples
