import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s buffer for safety

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, time_limit=10.0):
        if not edges:
            return []
        
        # Initial paths: each edge is a path
        # Represent path as a tuple of vertices
        # We need to track used edges to ensure edge-disjointness in merges?
        # Actually, if we start with disjoint edges and only merge if the connecting edge is not part of either path,
        # and the resulting path is simple, edge-disjointness is maintained.
        
        paths = [tuple(e) for e in edges]
        # To speed up, we can use a set of paths? No, paths can be same? Unlikely.
        # Let's use a list.
        
        # We will iterate trying to merge.
        # To avoid O(N^2) checks every time, we can just re-scan.
        # N is small (~50-100 edges).
        
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_paths = []
            used = [False] * len(paths)
            
            for i in range(len(paths)):
                if used[i]:
                    continue
                p1 = paths[i]
                merged = False
                for j in range(i + 1, len(paths)):
                    if used[j]:
                        continue
                    p2 = paths[j]
                    
                    # Check 4 possible merges
                    # 1. p1 + p2
                    # 2. p1 + reversed(p2)
                    # 3. reversed(p1) + p2
                    # 4. reversed(p1) + reversed(p2)
                    
                    candidates = [
                        p1 + p2,
                        p1 + tuple(reversed(p2)),
                        tuple(reversed(p1)) + p2,
                        tuple(reversed(p1)) + tuple(reversed(p2))
                    ]
                    
                    valid_merge = None
                    for cand in candidates:
                        # Check simplicity
                        if len(set(cand)) == len(cand):
                            # Check if the connecting edge was already in p1 or p2
                            # The connecting edge is (cand[-2], cand[-1])? No.
                            # If p1 + p2, connection is (p1[-1], p2[0]).
                            # We need to ensure (p1[-1], p2[0]) is an edge in the graph
                            # and NOT in p1 and NOT in p2.
                            
                            # Since we started with edges as paths, and only merge if simple,
                            # do we need to check if the connecting edge is already used?
                            # Yes.
                            
                            # Let's identify the connecting edge for this candidate.
                            # For p1 + p2: edge (p1[-1], p2[0])
                            # For p1 + rev(p2): edge (p1[-1], p2[-1])
                            # For rev(p1) + p2: edge (p1[0], p2[0])
                            # For rev(p1) + rev(p2): edge (p1[0], p2[-1])
                            
                            # We can just check if the edge exists in the graph and is not in p1 or p2.
                            # But we don't have the graph edges set handy here?
                            # We can pass it in.
                            
                            pass # Logic below
                            
                    if valid_merge:
                        new_paths.append(valid_merge)
                        used[i] = True
                        used[j] = True
                        improved = True
                        merged = True
                        break
                if not merged:
                    new_paths.append(p1)
                    used[i] = True
            
            paths = new_paths
            
        return [list(p) for p in paths]

    # The above decompose function is a bit messy with the edge check.
    # Let's rewrite it more cleanly.

    def decompose_greedy(edges, time_limit=10.0):
        if not edges:
            return []
        
        # Build adjacency and edge set for O(1) lookup
        edge_set = set()
        adj = {i: set() for i in range(max(edges[0][0], edges[0][1]) + 1)}
        for u, v in edges:
            edge_set.add((u, v))
            edge_set.add((v, u))
            adj[u].add(v)
            adj[v].add(u)
            
        # Initialize paths
        paths = [tuple(e) for e in edges]
        
        # Helper to check if a path is valid (simple and uses valid edges)
        def is_valid_path(p):
            if len(set(p)) != len(p):
                return False
            for i in range(len(p) - 1):
                if (p[i], p[i+1]) not in edge_set:
                    return False
            return True

        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_paths = []
            used = [False] * len(paths)
            
            for i in range(len(paths)):
                if used[i]:
                    continue
                p1 = paths[i]
                merged = False
                for j in range(i + 1, len(paths)):
                    if used[j]:
                        continue
                    p2 = paths[j]
                    
                    # Try to merge p1 and p2
                    # Possible connections:
                    # 1. p1[-1] connected to p2[0] -> new_p = p1 + p2
                    # 2. p1[-1] connected to p2[-1] -> new_p = p1 + p2[::-1]
                    # 3. p1[0] connected to p2[0] -> new_p = p1[::-1] + p2
                    # 4. p1[0] connected to p2[-1] -> new_p = p1[::-1] + p2[::-1]
                    
                    candidates = []
                    if p1[-1] in adj and p2[0] in adj and (p1[-1], p2[0]) in edge_set:
                        # Check if edge (p1[-1], p2[0]) is not in p1 or p2
                        if (p1[-1], p2[0]) not in set(map(tuple, zip(p1, p1[1:]))) and \
                           (p1[-1], p2[0]) not in set(map(tuple, zip(p2, p2[1:]))):
                            candidates.append(p1 + p2)
                            
                    if p1[-1] in adj and p2[-1] in adj and (p1[-1], p2[-1]) in edge_set:
                        if (p1[-1], p2[-1]) not in set(map(tuple, zip(p1, p1[1:]))) and \
                           (p1[-1], p2[-1]) not in set(map(tuple, zip(p2, p2[1:]))):
                            candidates.append(p1 + tuple(reversed(p2)))
                            
                    if p1[0] in adj and p2[0] in adj and (p1[0], p2[0]) in edge_set:
                        if (p1[0], p2[0]) not in set(map(tuple, zip(p1, p1[1:]))) and \
                           (p1[0], p2[0]) not in set(map(tuple, zip(p2, p2[1:]))):
                            candidates.append(tuple(reversed(p1)) + p2)
                            
                    if p1[0] in adj and p2[-1] in adj and (p1[0], p2[-1]) in edge_set:
                        if (p1[0], p2[-1]) not in set(map(tuple, zip(p1, p1[1:]))) and \
                           (p1[0], p2[-1]) not in set(map(tuple, zip(p2, p2[1:]))):
                            candidates.append(tuple(reversed(p1)) + tuple(reversed(p2)))
                            
                    valid_merge = None
                    for cand in candidates:
                        if is_valid_path(cand):
                            valid_merge = cand
                            break
                            
                    if valid_merge:
                        new_paths.append(valid_merge)
                        used[i] = True
                        used[j] = True
                        improved = True
                        merged = True
                        break
                if not merged:
                    new_paths.append(p1)
                    used[i] = True
            
            paths = new_paths
            
        return [list(p) for p in paths]

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 5:
            # Not enough time, return trivial solution for remaining
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        # Run greedy decomposition
        # To improve, we can shuffle edges initially or use multiple random restarts
        best_paths = None
        best_count = len(edges)
        
        # Try a few random permutations of edges to see if it helps
        # But greedy merging is deterministic if we pick first valid merge.
        # Let's just run it once.
        
        paths = decompose_greedy(edges, time_limit=10.0)
        
        # Verify
        # (Verification omitted for brevity in thought process, but should be added in final code if unsure)
        
        results.append(paths)
        
    return results
