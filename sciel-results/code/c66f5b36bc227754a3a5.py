import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Candidates: numbers in [1,255] divisible by 3, 5, or 17
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    n = len(candidates)
    
    # Build adjacency: edge if gcd > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy maximum clique heuristic
    def greedy_clique(order):
        clique = []
        for v in order:
            if all(u in adj[v] for u in clique):
                clique.append(v)
        return clique
    
    # Known good: multiples of 3
    best_clique = [i for i, x in enumerate(candidates) if x % 3 == 0]
    best_size = len(best_clique)
    
    # Try greedy by degree descending
    degrees = [len(adj[i]) for i in range(n)]
    order_desc = sorted(range(n), key=lambda i: -degrees[i])
    clique = greedy_clique(order_desc)
    if len(clique) > best_size:
        best_clique = clique
        best_size = len(clique)
    
    # Try greedy by degree ascending
    order_asc = sorted(range(n), key=lambda i: degrees[i])
    clique = greedy_clique(order_asc)
    if len(clique) > best_size:
        best_clique = clique
        best_size = len(clique)
    
    # Local search with add and swap moves
    if time.time() < deadline - 5:
        random.seed(42)
        current = set(best_clique)
        
        for _ in range(10000):
            if time.time() > deadline:
                break
            
            # Try to add a random vertex
            not_in = [i for i in range(n) if i not in current]
            if not_in:
                v = random.choice(not_in)
                if all(u in adj[v] for u in current):
                    new_set = current | {v}
                    if len(new_set) > best_size:
                        best_clique = list(new_set)
                        best_size = len(best_clique)
                        current = new_set
                    continue
            
            # Try swap: remove one, add another
            if len(current) > 1:
                to_remove = random.choice(list(current))
                candidate_new = current - {to_remove}
                not_in = [i for i in range(n) if i not in candidate_new]
                if not_in:
                    v = random.choice(not_in)
                    if all(u in adj[v] for u in candidate_new):
                        new_set = candidate_new | {v}
                        if len(new_set) > best_size:
                            best_clique = list(new_set)
                            best_size = len(best_clique)
                            current = new_set
    
    result = [candidates[i] for i in best_clique]
    
    # Ensure 255 is in the result
    if 255 not in result:
        idx_255 = candidates.index(255)
        # Check if 255 is compatible with all current elements
        if all(idx_255 in adj[candidates.index(x)] for x in result):
            result.append(255)
        else:
            # Replace an incompatible element with 255
            for i, x in enumerate(result):
                if x != 255 and idx_255 not in adj[candidates.index(x)]:
                    result[i] = 255
                    break
    
    return result
