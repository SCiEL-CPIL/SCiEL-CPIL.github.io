import time
import random
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Identify all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                lac = lcm(a, c)
                if lac != lab:
                    continue
                lbc = lcm(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))
    
    if not bad_triples:
        return numbers
        
    # 2. Multi-start Greedy Vertex Cover
    def greedy_remove(bad_triples, n, rng):
        counts = [0] * (n + 1)
        for t in bad_triples:
            for x in t:
                counts[x] += 1
                
        active_triples = [True] * len(bad_triples)
        active_count = len(bad_triples)
        
        removed = [False] * (n + 1)
        
        # Precompute element to triples mapping
        elem_to_triples = [[] for _ in range(n + 1)]
        for i, t in enumerate(bad_triples):
            for x in t:
                elem_to_triples[x].append(i)
                
        while active_count > 0:
            # Find candidates
            candidates = []
            for x in range(1, n + 1):
                if not removed[x]:
                    score = counts[x] + rng.random() * 0.5
                    candidates.append((score, x))
            
            if not candidates:
                break
                
            candidates.sort(key=lambda item: item[0], reverse=True)
            
            # Pick randomly from top 5
            k = min(5, len(candidates))
            _, target = rng.choice(candidates[:k])
            
            removed[target] = True
            
            # Update
            for idx in elem_to_triples[target]:
                if active_triples[idx]:
                    active_triples[idx] = False
                    active_count -= 1
                    t = bad_triples[idx]
                    for x in t:
                        if x != target:
                            counts[x] -= 1
                            
        result = [x for x in range(1, n + 1) if not removed[x]]
        return result

    best_result = []
    best_size = 0
    rng = random.Random(42)
    
    while time.time() < deadline:
        result = greedy_remove(bad_triples, n, rng)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
        if best_size == n:
            break
            
    if not best_result:
        return [1]
        
    return best_result
