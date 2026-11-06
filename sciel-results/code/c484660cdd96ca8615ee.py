import time
import random
from itertools import combinations

def construct():
    n = 20
    deadline = time.time() + 120.0

    def popcount(x):
        return bin(x).count("1")

    def is_regular(adj, subset):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        degs = [popcount(adj[v] & mask) for v in subset]
        return all(d == degs[0] for d in degs)

    def find_violation_full(adj):
        for k in range(6, n + 1):
            for comb in combinations(range(n), k):
                if is_regular(adj, comb):
                    return comb
        return None

    def find_violation_sampled(adj, num_samples=100):
        # Sample random subsets of size 6 to 10
        # If found, return it
        for _ in range(num_samples):
            k = random.randint(6, 10)
            comb = tuple(random.sample(range(n), k))
            if is_regular(adj, comb):
                return comb
        return None

    def random_graph(n):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def flip_edge(adj, u, v):
        if adj[u] & (1 << v):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[j] |= (1 << i) # Typo in thought, correct below

    # Corrected flip_edge
    def flip_edge(adj, u, v):
        if adj[u] & (1 << v):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    while time.time() < deadline - 10.0:
        adj = random_graph(n)
        
        # Local search loop
        steps = 0
        while steps < 2000 and time.time() < deadline - 5.0:
            steps += 1
            
            # Use sampled violation to guide search
            violation = find_violation_sampled(adj)
            
            if violation is None:
                # No violations in sample. Check fully.
                full_violation = find_violation_full(adj)
                if full_violation is None:
                    # Valid!
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                else:
                    # Found a missed violation. Use it.
                    violation = full_violation
            
            # Flip edge in violation
            u, v = random.sample(list(violation), 2)
            flip_edge(adj, u, v)
            
            # Random perturbation
            if random.random() < 0.05:
                u, v = random.sample(range(n), 2)
                flip_edge(adj, u, v)

    # Fallback to smaller n if 20 fails
    for target_n in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 2.0:
            break
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            adj = random_graph(target_n)
            if find_violation_full(adj) is None:
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': target_n, 'edges': edges}
                
    return {'n': 5, 'edges': []}
