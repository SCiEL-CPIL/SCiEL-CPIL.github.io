import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target n values in descending order
    for n in [47, 46]:
        min_deg_req = n - 39
        
        # Try multiple times to find a valid graph
        max_trials = 1000
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Initialize graph
            # Using sets for adjacency
            adj = [set() for _ in range(n)]
            edges = []
            
            # Precompute all potential edges
            # We will maintain a list of valid edges dynamically
            # Or just iterate through all pairs and check validity
            # Given n is small, iterating all pairs O(n^2) at each step is O(n^4) total?
            # n=47, n^2 ~ 2200. Steps ~ 1000. 2200 * 1000 = 2.2M checks.
            # Each check is O(deg^2) ~ 64. Total ~ 140M ops. Might be slow in Python.
            
            # Optimization:
            # Only check edges where u < v.
            # We can keep a list of candidate edges.
            
            # Let's use a simpler approach:
            # Randomly shuffle vertices? No, labels are fixed.
            # We'll just pick a random valid edge at each step to avoid bias,
            # but bias towards low degree helps min degree.
            
            # Strategy:
            # 1. List all pairs (u, v) with u < v.
            # 2. At each step, find valid ones.
            # 3. If valid set is not empty, pick one.
            
            # To speed up, we can pre-filter.
            # But for n=47, let's just do it directly.
            
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_pairs) # Randomize order to break symmetry
            
            # We will add edges one by one.
            # We need to check if adding (u, v) creates a C4.
            # Condition: N(u) and N(N(v)) are disjoint.
            
            def is_valid(u, v):
                # Check if N(u) intersects N(N(v))
                # N(v) is adj[v]
                # N(N(v)) is union of adj[x] for x in adj[v]
                # We can check: for w in adj[u], for x in adj[v], if x in adj[w]: return False
                # This is O(deg(u) * deg(v))
                for w in adj[u]:
                    for x in adj[v]:
                        if x in adj[w]:
                            return False
                return True

            # Greedy loop
            # We want to maximize min degree.
            # So we should prioritize adding edges to vertices with low degree.
            # Let's score valid edges by sum of degrees.
            
            # Instead of scanning all pairs every time, we can just scan all pairs, 
            # filter valid ones, and pick the best.
            # Since we only add ~ n^2/4 edges, and n is small, this is okay.
            
            added = 0
            # We can't just shuffle once, because validity changes.
            # We must re-evaluate.
            
            # To avoid getting stuck in a local minimum of "can't add any more",
            # we can try different strategies.
            # Strategy 1: Random valid edge.
            # Strategy 2: Min degree sum valid edge.
            
            # Let's alternate or randomize the strategy.
            strategy = random.choice(['random', 'min_deg'])
            
            while True:
                if time.time() > deadline:
                    break
                
                valid_edges = []
                for u, v in all_pairs:
                    if (u, v) in [(e[0], e[1]) for e in edges]: # This check is slow
                        continue
                    # Better: check if edge exists in adj
                    if v in adj[u]:
                        continue
                    
                    if is_valid(u, v):
                        deg_sum = len(adj[u]) + len(adj[v])
                        valid_edges.append((deg_sum, u, v))
                
                if not valid_edges:
                    break
                
                if strategy == 'min_deg':
                    # Pick edge with minimum degree sum
                    valid_edges.sort(key=lambda x: x[0])
                    # Add some randomness to tie-breaking
                    best_score = valid_edges[0][0]
                    candidates = [e for e in valid_edges if e[0] == best_score]
                    _, u, v = random.choice(candidates)
                else:
                    # Pick random valid edge
                    _, u, v = random.choice(valid_edges)
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                added += 1
            
            # Check min degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_req:
                return {'n': n, 'edges': edges}
            
            # If we finished a trial without success, continue to next trial
            # Note: 'added' might be 0 if no edges could be added initially?
            # No, empty graph is valid, so at least some edges can be added.
            
        # If n=47 failed, loop continues to n=46
        
    # Fallback: return a known valid small graph if all else fails
    # n=39, empty graph is valid?
    # Min degree 0. n-39 = 0. Yes.
    return {'n': 39, 'edges': []}
