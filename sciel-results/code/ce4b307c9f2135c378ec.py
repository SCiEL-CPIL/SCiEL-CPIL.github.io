import time
import random

def instance_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def greedy_decompose(edges, rng):
    if not edges:
        return []
    
    max_v = max(max(e) for e in edges)
    adj = {i: set() for i in range(max_v + 1)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    unused = set(edges)
    paths = []
    
    while unused:
        # Pick a random edge from unused
        # Converting to list is O(E), E is small
        e = rng.choice(list(unused))
        u, v = e
        
        # Remove edge
        unused.remove(e)
        adj[u].remove(v)
        adj[v].remove(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Extend from v
        curr = v
        while True:
            # Find valid neighbors
            candidates = [x for x in adj[curr] if x not in path_set]
            if not candidates:
                break
            # Heuristic: max degree
            # Tie-break randomly if needed? 
            # max() with key is deterministic. 
            # To add randomness, we can pick from top-K or just use rng to break ties.
            # For simplicity and speed, max degree is a good heuristic.
            # Let's add slight randomness: pick from top 3 if available?
            # Or just max degree.
            best_n = max(candidates, key=lambda x: len(adj[x]))
            
            # Remove edge curr-best_n
            e_sorted = tuple(sorted((curr, best_n)))
            unused.remove(e_sorted)
            adj[curr].remove(best_n)
            adj[best_n].remove(curr)
            
            path.append(best_n)
            path_set.add(best_n)
            curr = best_n
            
        # Extend from u
        curr = u
        while True:
            candidates = [x for x in adj[curr] if x not in path_set]
            if not candidates:
                break
            best_n = max(candidates, key=lambda x: len(adj[x]))
            
            e_sorted = tuple(sorted((curr, best_n)))
            unused.remove(e_sorted)
            adj[curr].remove(best_n)
            adj[best_n].remove(curr)
            
            path.insert(0, best_n)
            path_set.add(best_n)
            curr = best_n
            
        paths.append(path)
        
    return paths

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        edges = instance_graph(n, p, seed)
        best_paths = None
        best_count = float('inf')
        
        # Run multiple trials
        # Estimate cost: 
        # Greedy is roughly O(E * N) per path, E paths worst case. O(E^2 N).
        # E ~ 50-80. N ~ 20-30.
        # 80^2 * 30 = 192,000 ops per run.
        # 1000 runs = 200M ops. Might be too slow in Python?
        # Let's try 500 runs first.
        
        for trial in range(1000):
            if time.time() > deadline:
                break
                
            # Use a different seed for each trial to vary the random choices
            # The rng inside greedy uses rng.choice and max (deterministic).
            # The only randomness is the initial edge pick.
            # To vary the extension, we need randomness there too.
            # Let's modify greedy to accept a rng and use it for tie-breaking or selection.
            
            rng = random.Random(seed * 10000 + trial)
            paths = greedy_decompose(edges, rng)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                
        if best_paths is None:
             # Fallback: one edge per path
             best_paths = [[list(e) for e in edges]]
             
        results.append(best_paths)
        
    return results
