import time
import itertools
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin
    
    n = 15
    # Partition: A = 0..6 (size 7), B = 7..14 (size 8)
    A = list(range(7))
    B = list(range(7, 15))
    
    # Start with complete bipartite K_{7,8}
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((min(a,b), max(a,b)))
    
    # Candidate edges: all pairs within A and within B
    candidates = []
    for i, j in itertools.combinations(A, 2):
        candidates.append((min(i, j), max(i, j)))
    for i, j in itertools.combinations(B, 2):
        candidates.append((min(i, j), max(i, j)))
    
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edges):
        deg = get_degrees(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Check every 3-edge path s-u-v-t
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def greedy_solve(seed):
        rng = random.Random(seed)
        cands = list(candidates)
        rng.shuffle(cands)
        edges = set(base_edges)
        for e in cands:
            if time.time() > DEADLINE:
                break
            edges.add(e)
            if is_valid(edges):
                pass
            else:
                edges.discard(e)
        return edges
    
    best_edges = set(base_edges)
    best_count = len(base_edges)
    
    # Try multiple random orderings
    num_trials = 200
    for trial in range(num_trials):
        if time.time() > DEADLINE - 5:
            break
        edges = greedy_solve(trial)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
    
    # Also try a few different base structures
    # Try K_{6,9}
    A2 = list(range(6))
    B2 = list(range(6, 15))
    base2 = set()
    for a in A2:
        for b in B2:
            base2.add((min(a,b), max(a,b)))
    candidates2 = []
    for i, j in itertools.combinations(A2, 2):
        candidates2.append((min(i, j), max(i, j)))
    for i, j in itertools.combinations(B2, 2):
        candidates2.append((min(i, j), max(i, j)))
    
    def greedy_solve2(seed):
        rng = random.Random(seed + 10000)
        cands = list(candidates2)
        rng.shuffle(cands)
        edges = set(base2)
        for e in cands:
            if time.time() > DEADLINE:
                break
            edges.add(e)
            if is_valid(edges):
                pass
            else:
                edges.discard(e)
        return edges
    
    for trial in range(100):
        if time.time() > DEADLINE - 3:
            break
        edges = greedy_solve2(trial)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
    
    # Convert to list format
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
