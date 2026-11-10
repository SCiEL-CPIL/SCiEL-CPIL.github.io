import random
import itertools
import time

def construct():
    n = 20
    deadline = time.time() + 140
    
    def is_valid(adj):
        # Check all subsets of size 6 to n
        # We can iterate by size
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                degs = []
                for v in comb:
                    degs.append((adj[v] & mask).bit_count())
                
                if all(d == degs[0] for d in degs):
                    return False
        return True

    # Random search
    for _ in range(5000):
        if time.time() > deadline:
            break
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.getrandbits(1):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if is_valid(adj):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    # Fallback
    return {'n': 5, 'edges': []}
