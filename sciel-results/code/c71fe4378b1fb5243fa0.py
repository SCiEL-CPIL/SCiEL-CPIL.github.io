"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    n = P['n']
    base_edges_set = set()
    for u, v in P['edges']:
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # Candidate edges: must connect opposite parity to maintain bipartiteness (triangle-free)
    # and must not be in base.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 == 1 and (u, v) not in base_edges_set:
                candidates.append((u, v))
                
    # Helper: Calculate all-pairs shortest paths (Floyd-Warshall or BFS from each node)
    # Since n=24, BFS from each node is efficient enough.
    def get_all_distances(adj):
        dist = [[-1] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[i][v] == -1:
                        dist[i][v] = dist[i][u] + 1
                        q.append(v)
        return dist

    def get_diameter(dist):
        max_d = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        return max_d

    # Initial graph: just the path
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    initial_dist = get_all_distances(adj)
    initial_diam = get_diameter(initial_dist)
    
    # Identify "critical pairs": pairs (u, v) such that dist(u, v) > 4
    # We need to select edges such that for all critical pairs, the new distance is <= 4.
    critical_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            if initial_dist[u][v] > 4:
                critical_pairs.append((u, v))
                
    # If no critical pairs, return empty
    if not critical_pairs:
        return []

    # Pre-compute which candidates "help" which critical pairs?
    # Actually, it's complex to pre-compute exactly which single edge reduces which pair to <= 4 
    # because edges work in combination. 
    # However, we can use a greedy set cover approach where we estimate impact.
    
    # Strategy:
    # 1. Start with a good initial solution (e.g., greedy selection of edges that reduce diameter most).
    # 2. Try to remove edges to see if validity holds.
    # 3. Try to swap edges to reduce count (harder, but maybe just removal is enough if initial is near-optimal).
    
    # Let's first find a valid solution using a greedy approach similar to the original but faster.
    current_edges = set()
    current_adj = [set(s) for s in adj]
    
    # Greedy: Pick edge that reduces the number of critical pairs most (or reduces diameter most).
    # To be efficient, we only check if an edge helps reduce the diameter or fixes specific pairs.
    
    # A stronger initial solution: 
    # We know that adding edges (0, 12), (6, 18) etc might work? 
    # Let's just run the greedy but with a better heuristic: 
    # For each candidate, calculate how many critical pairs it "shortcuts" significantly.
    
    # Actually, let's try a different angle: 
    # The minimum number of edges is likely small (3 or 4).
    # We can enumerate combinations of a small number of "high impact" edges.
    
    # High impact edges are those with large |u-v|.
    # Sort candidates by |u-v| descending.
    candidates_sorted = sorted(candidates, key=lambda x: abs(x[0]-x[1]), reverse=True)
    
    # Let's try to find a solution with k edges, starting from k=1, 2, 3...
    # Since n=24, diameter 4, the "radius" of the graph must be small.
    # A known result or heuristic: A path of length 23 needs to be compressed to diameter 4.
    # This is equivalent to finding a dominating set or similar in the line graph?
    
    # Let's try a randomized local search or simulated annealing? 
    # No, let's try a systematic search for small k.
    
    # Upper bound on k: The original greedy probably found ~5-7. 
    # Let's guess the optimum is 3 or 4.
    
    # We will test combinations of the top M candidates.
    # How many top candidates? 
    # If we take top 15, C(15, 3) = 455, C(15, 4) = 1365. Very fast.
    # If we take top 20, C(20, 4) = 4845. Still fast.
    
    # But which candidates are "top"? 
    # Maybe not just |u-v|. Maybe edges that cover the "middle" of the path.
    # Let's include a diverse set of candidates.
    # Select top 25 by |u-v|, plus some random ones? 
    # Let's stick to top 20 by |u-v| for determinism and speed.
    
    top_candidates = candidates_sorted[:20]
    
    best_solution = None
    best_k = 100
    
    # Try k = 1, 2, 3, 4, 5
    for k in range(1, 6):
        if time.time() > deadline:
            break
            
        # If we found a solution with k-1, we don't need to check k unless we want to verify minimality?
        # We want the MINIMUM. So if we find one with k, we can stop and return it?
        # Yes, because we are iterating k upwards.
        
        # Generate combinations of k edges from top_candidates
        # Optimization: Prune combinations that are obviously bad? 
        # For now, just check all.
        
        for combo in itertools.combinations(top_candidates, k):
            if time.time() > deadline:
                break
                
            # Build graph
            test_adj = [set(s) for s in adj]
            for u, v in combo:
                test_adj[u].add(v)
                test_adj[v].add(u)
                
            # Check validity
            dist = get_all_distances(test_adj)
            diam = get_diameter(dist)
            
            if diam <= 4:
                best_solution = combo
                best_k = k
                break
        
        if best_solution:
            break
            
    if best_solution is None:
        # Fallback: use the greedy result if exhaustive search failed or timed out
        # Or just return a known valid set if we have one?
        # Let's run the original greedy logic as a fallback to ensure we return *something* valid.
        
        current_edges = set()
        current_adj = [set(s) for s in adj]
        current_diam = initial_diam
        
        while current_diam > 4:
            if time.time() > deadline:
                break
            best_edge = None
            best_diam = current_diam + 1
            
            for u, v in candidates:
                if (u, v) in current_edges or (v, u) in current_edges:
                    continue
                current_adj[u].add(v)
                current_adj[v].add(u)
                new_diam = get_diameter(get_all_distances(current_adj))
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = (u, v)
                if new_diam <= 4:
                    break
                    
            if best_edge is None:
                break
                
            u, v = best_edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_edges.add((u, v))
            current_edges.add((v, u))
            current_diam = best_diam
            
        # Extract added edges
        added = []
        for u, v in current_edges:
            if (u, v) not in base_edges_set and (v, u) not in base_edges_set:
                 if u < v:
                     added.append([u, v])
                 elif v < u:
                     added.append([v, u])
                     
        final_added = []
        seen = set()
        for e in added:
            key = (e[0], e[1])
            if key not in seen:
                final_added.append([int(e[0]), int(e[1])])
                seen.add(key)
        return final_added
        
    # Convert best_solution to list of lists
    final_added = [[int(u), int(v)] for u, v in best_solution]
    return final_added

# EVOLVE-BLOCK-END
