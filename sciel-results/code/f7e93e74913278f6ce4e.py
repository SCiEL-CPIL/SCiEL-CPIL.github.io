import itertools
import time

def is_triangle_free_diff(D, n):
    S = set()
    for d in D:
        S.add(d % n)
        S.add((-d) % n)
    
    for x in S:
        for y in S:
            if (x + y) % n in S:
                return False
    return True

def has_independent_set_10(n, D):
    adj = [set() for _ in range(n)]
    for d in D:
        for i in range(n):
            j = (i + d) % n
            adj[i].add(j)
            adj[j].add(i)
            
    # Simple backtracking
    def backtrack(current_size, available):
        if current_size == 10:
            return True
        if not available:
            return False
        if current_size + len(available) < 10:
            return False
        
        # Heuristic: pick vertex with max degree in available subgraph?
        # Or just pick one.
        # To improve pruning, we can try to pick a vertex that is connected to many others in available.
        # Let's just iterate.
        for v in available:
            new_available = available - {v} - adj[v]
            if backtrack(current_size + 1, new_available):
                return True
        return False

    return backtrack(0, set(range(n)))

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
        max_d = n // 2
        candidates = list(range(1, max_d + 1))
        
        # Try k from high to low
        for k in range(max_d, 4, -1):
            if time.time() > deadline:
                break
            for D in itertools.combinations(candidates, k):
                if time.time() > deadline:
                    break
                if is_triangle_free_diff(D, n):
                    if not has_independent_set_10(n, D):
                        edges = []
                        for i in range(n):
                            for d in D:
                                j = (i + d) % n
                                if i < j:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
    
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
