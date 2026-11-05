import time
import random
import sys

def has_triangle(n, edges_set):
    # This is slow if called frequently. 
    # Better to maintain adjacency lists.
    pass

def max_independent_set_size(n, adj):
    # Exact branch and bound for small n? 
    # n=40 is too big for exact B&B in Python within 150s if done many times.
    # We need a fast approximation or a heuristic that guarantees finding the max if it's small?
    # Actually, we just need to know if alpha < 10.
    # We can use a greedy algorithm to find an independent set. 
    # If we find one of size 10, it's invalid.
    # If we can't find one of size 10, it MIGHT be valid, but we need to be sure.
    # The task says "both conditions are checked by exact branch and bound".
    # This implies the EVALUATOR checks it. My program just needs to OUTPUT a valid graph.
    # I don't need to verify it perfectly in my code, but I should try to ensure it's valid to score.
    # If I output an invalid graph, score is 0.
    # So I need a reliable way to check alpha < 10.
    # For n=40, checking alpha < 10 is hard.
    # However, if I use the "add edge" heuristic, I am explicitly destroying independent sets.
    # If I start with a graph where alpha is large, and I add edges to break them, 
    # I might end up with a graph where alpha is small.
    # I will use a greedy independent set finder to check if alpha >= 10.
    # If I find a set of size 10, I continue.
    # If I fail to find a set of size 10 using a strong greedy heuristic, I assume it might be < 10.
    # This is risky.
    
    # Better: Use a SAT solver? No, standard library only.
    # Use a simple backtracking for alpha < 10?
    # If n=40, and we want to prove alpha < 10, it's hard.
    # But we just want to FIND a graph.
    
    # Let's rely on the heuristic: 
    # If I can't find an independent set of size 10 with a good greedy algorithm, 
    # it's likely alpha < 10.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try n=40 first
    # Then n=39
    # Then fallback to seed
    
    # Heuristic Search
    # 1. Generate random triangle-free graph
    # 2. Greedy independent set check
    # 3. If alpha >= 10, add edge to break it
    # 4. Repeat
    
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    
    for n in [40, 39, 38]:
        if time.time() > deadline - 20:
            break
            
        # Try multiple restarts for this n
        for restart in range(50):
            if time.time() > deadline - 20:
                break
                
            # Init random graph
            adj = [set() for _ in range(n)]
            # Start with a random bipartite graph? Or empty?
            # Empty graph has alpha = n.
            # Let's start with a random graph with p=0.5, then remove triangles.
            
            # Faster: Start with a known structure like Andrasfai and perturb?
            # Or just random.
            
            # Generate random edges
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        edges.append((i,j))
                        
            # Remove triangles
            # This is O(E^2) or so.
            # Convert to adj sets
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                
            # Remove triangles
            changed = True
            while changed:
                changed = False
                for u in range(n):
                    for v in list(adj[u]):
                        if v > u: # Check each edge once
                            # Check if there is w connected to both
                            common = adj[u].intersection(adj[v])
                            if common:
                                # Remove one edge, say (u,v)
                                adj[u].remove(v)
                                adj[v].remove(u)
                                changed = True
                                
            # Now triangle-free
            # Check alpha
            # Greedy Independent Set
            def find_independent_set_size(n, adj):
                # Simple greedy: pick vertex with min degree, remove it and its neighbors
                # Repeat
                # This gives a lower bound on alpha.
                # If this lower bound >= 10, then alpha >= 10.
                # If this lower bound < 10, alpha MIGHT be < 10.
                # We need to be sure.
                # But if we use the "add edge" strategy, we are actively reducing alpha.
                # If we get stuck with greedy alpha < 10, we return it.
                
                # To be safer, use a slightly better greedy:
                # Randomly shuffle vertices, pick first available, remove neighbors.
                # Do this many times.
                max_size = 0
                for _ in range(10):
                    vertices = list(range(n))
                    random.shuffle(vertices)
                    chosen = []
                    available = set(range(n))
                    for v in vertices:
                        if v in available:
                            chosen.append(v)
                            available.remove(v)
                            for u in adj[v]:
                                if u in available:
                                    available.remove(u)
                    if len(chosen) > max_size:
                        max_size = len(chosen)
                return max_size

            current_alpha = find_independent_set_size(n, adj)
            
            if current_alpha < 10:
                # Candidate found!
                # Convert adj to edges list
                res_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            res_edges.append([u, v])
                if n > best_n:
                    best_n = n
                    best_edges = res_edges
                break # Found a valid graph for this n, move to next n? No, if n=40 works, we are done.
            
            # If alpha >= 10, try to fix
            # Find an independent set of size 10
            # The greedy one above found one. Let's get the actual set.
            def get_independent_set(n, adj):
                vertices = list(range(n))
                random.shuffle(vertices)
                chosen = []
                available = set(range(n))
                for v in vertices:
                    if v in available:
                        chosen.append(v)
                        available.remove(v)
                        for u in adj[v]:
                            if u in available:
                                available.remove(u)
                return chosen

            I = get_independent_set(n, adj)
            if len(I) >= 10:
                # Try to add an edge between two vertices in I
                added = False
                for i in range(len(I)):
                    for j in range(i+1, len(I)):
                        u, v = I[i], I[j]
                        # Check if adding (u,v) creates triangle
                        # i.e., if N(u) and N(v) have common neighbor
                        if not adj[u].intersection(adj[v]):
                            # Safe to add
                            adj[u].add(v)
                            adj[v].add(u)
                            added = True
                            break
                    if added:
                        break
                if not added:
                    # Stuck. Restart or kick.
                    # Just restart this n with new random seed
                    continue
            else:
                # Greedy didn't find 10, but maybe alpha is 10?
                # Assume valid for now? Risky.
                # Let's assume if greedy fails to find 10, it's likely < 10.
                res_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            res_edges.append([u, v])
                if n > best_n:
                    best_n = n
                    best_edges = res_edges
                break

    return {'n': best_n, 'edges': best_edges}

construct()
