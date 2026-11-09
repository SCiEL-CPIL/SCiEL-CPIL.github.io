import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Target n=21 first, then 20
    for n in [21, 20]:
        if time.time() > deadline:
            break
            
        # Simulated Annealing
        # Init
        graph = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    graph[i] |= (1 << j)
                    graph[j] |= (1 << i)
        
        current_score = fast_score(graph, n)
        best_graph = graph[:]
        best_score = current_score
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < deadline and T > T_min:
            # Pick move
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                T *= cooling
                continue
                
            # Flip
            if graph[i] & (1 << j):
                graph[i] ^= (1 << j)
                graph[j] ^= (1 << i)
            else:
                graph[i] |= (1 << j)
                graph[j] |= (1 << i)
            
            new_score = fast_score(graph, n)
            
            # Acceptance
            if new_score < current_score or random.random() < pow(2.71828, -(new_score - current_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_graph = graph[:]
                    # If best_score is 0, check exact
                    if best_score == 0:
                        if exact_check(best_graph, n):
                            edges = []
                            for u in range(n):
                                for v in range(u+1, n):
                                    if best_graph[u] & (1 << v):
                                        edges.append([u, v])
                            return {'n': n, 'edges': edges}
            else:
                # Undo
                if graph[i] & (1 << j):
                    graph[i] ^= (1 << j)
                    graph[j] ^= (1 << i)
                else:
                    graph[i] |= (1 << j)
                    graph[j] |= (1 << i)
            
            T *= cooling
            
        # If we exit loop, check best_graph
        if best_score == 0:
            if exact_check(best_graph, n):
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if best_graph[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
    # Fallback
    return {'n': 5, 'edges': []}

def fast_score(graph, n, sample_size=200):
    # Count regular size-6 subsets in sample
    count = 0
    for _ in range(sample_size):
        # Pick 6 distinct random vertices
        subset = random.sample(range(n), 6)
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        # Check regularity
        degs = set()
        for v in subset:
            d = bin(graph[v] & mask).count('1')
            degs.add(d)
            if len(degs) > 1:
                break
        
        if len(degs) == 1:
            count += 1
    return count

def exact_check(graph, n):
    # Check all subsets of size >= 6
    # Optimize: Check size 6 first.
    # If any regular, return False.
    
    # We can speed this up by checking only size 6 and 7.
    # If no size 6 or 7, it's very likely valid.
    # But we must be correct.
    
    # For n=20, 2^20 is 1M.
    # For n=21, 2^21 is 2M.
    # This is acceptable if called rarely.
    
    # To make it faster, we can use a list of masks? No.
    
    # Just loop.
    
    # We can break early if we find one.
    
    # Let's try to optimize the inner loop.
    
    # For each k:
    for k in range(6, n+1):
        for c in combinations(range(n), k):
            mask = 0
            for v in c:
                mask |= (1 << v)
            
            # Check degrees
            # We can do this faster?
            # Just compute degrees.
            
            d0 = bin(graph[c[0]] & mask).count('1')
            regular = True
            for v in c[1:]:
                if bin(graph[v] & mask).count('1') != d0:
                    regular = False
                    break
            
            if regular:
                return False
    return True
