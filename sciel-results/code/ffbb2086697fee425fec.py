import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    q = 5
    # Build PG(2, q) points: 31 points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # should be 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0 and i != j
    base_edges = []
    base_adj = [set() for _ in range(n)]
    base_edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                base_edges.append((i, j))
                base_adj[i].add(j)
                base_adj[j].add(i)
                base_edge_set.add((i, j))
    
    # Count C4s in a graph given edge set
    def count_c4s(adj, edge_set):
        # For each pair of vertices, count common neighbors
        # A C4 is formed by 4 vertices a,b,c,d with edges ab, bc, cd, da
        # Equivalent: for each unordered pair {u,v}, let c = |N(u) ∩ N(v)|
        # Then number of C4s = sum over all pairs {u,v} of C(c,2) / 2
        # Wait, that's not quite right. Let me think again.
        # A C4 is a cycle of length 4. For each pair of non-adjacent vertices {u,v},
        # if they have k common neighbors, then each pair of common neighbors forms a C4.
        # But we need to be careful about double counting.
        # Actually: number of C4s = (1/4) * sum over all pairs {u,v} of C(|N(u)∩N(v)|, 2)
        # No wait. Each C4 has 2 pairs of opposite vertices. For each C4, it contributes
        # to exactly 2 pairs (the two diagonals). For a pair {u,v} that are opposite in a C4,
        # they share exactly 2 common neighbors (the other two vertices of the C4).
        # So: number of C4s = (1/2) * sum over all pairs {u,v} of C(|N(u)∩N(v)|, 2)
        # Wait, that overcounts. Let me reconsider.
        # For a fixed pair {u,v}, if |N(u)∩N(v)| = k, then there are C(k,2) ways to choose
        # two common neighbors a,b, which gives a 4-cycle u-a-v-b-u (if all edges exist).
        # But this 4-cycle is counted once for the pair {u,v} and once for the pair {a,b}.
        # So total C4s = (1/2) * sum_{all pairs {u,v}} C(|N(u)∩N(v)|, 2)
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2
    
    # Helper: count C4s efficiently using adjacency matrix
    def build_adj_matrix(edge_set):
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Find best edge to add to base graph
    adj_lists = [sorted(base_adj[i]) for i in range(n)]
    
    # Build fast adjacency lookup
    is_adj = [[False]*n for _ in range(n)]
    for (u, v) in base_edge_set:
        is_adj[u][v] = True
        is_adj[v][u] = True
    
    # Compute cost of adding each non-edge
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in base_edge_set:
                non_edges.append((i, j))
    
    # Sort by common neighbors as heuristic
    edge_costs = []
    for (i, j) in non_edges:
        common = len(base_adj[i] & base_adj[j])
        edge_costs.append((common, i, j))
    edge_costs.sort()
    
    best_add = None
    best_add_cost = float('inf')
    
    for common, i, j in edge_costs:
        if time.time() > deadline - 30:
            break
        cost = 0
        for a in adj_lists[i]:
            for b in adj_lists[j]:
                if a == b:
                    continue
                if is_adj[a][b]:
                    cost += 1
        if cost < best_add_cost:
            best_add_cost = cost
            best_add = (i, j)
    
    # Start with best 91-edge graph
    current_edges = set(base_edge_set)
    if best_add:
        current_edges.add(best_add)
    
    current_adj = build_adj_matrix(current_edges)
    current_c4 = count_c4s(current_adj, current_edges)
    
    # Local search: try edge swaps (remove one, add another)
    # To reduce C4 count
    best_c4 = current_c4
    best_edges = set(current_edges)
    
    # Precompute all non-edges for current graph
    all_pairs = set()
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.add((i, j))
    
    def get_non_edges(edge_set):
        return all_pairs - edge_set
    
    non_edges_current = get_non_edges(current_edges)
    
    # Local search iterations
    max_iterations = 500
    for it in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        improved = False
        non_edges_list = list(non_edges_current)
        
        # Try each edge removal
        for rem_edge in list(current_edges):
            if time.time() > deadline - 5:
                break
            
            # Try adding each non-edge
            for add_edge in non_edges_list:
                if add_edge == rem_edge:
                    continue
                
                # Quick check: would this swap reduce C4s?
                # New C4s from adding add_edge minus C4s removed by removing rem_edge
                # This is complex, so let's just try it for promising candidates
                
                # Heuristic: only try if the added edge has fewer common neighbors
                # than the removed edge
                rem_u, rem_v = rem_edge
                add_u, add_v = add_edge
                
                rem_common = len(current_adj[rem_u] & current_adj[rem_v])
                add_common = len(current_adj[add_u] & current_adj[add_v])
                
                if add_common >= rem_common:
                    continue
                
                # Try the swap
                new_edges = set(current_edges)
                new_edges.remove(rem_edge)
                new_edges.add(add_edge)
                
                new_adj = build_adj_matrix(new_edges)
                new_c4 = count_c4s(new_adj, new_edges)
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_edges = new_edges
                    current_edges = new_edges
                    current_adj = new_adj
                    current_c4 = new_c4
                    non_edges_current = get_non_edges(current_edges)
                    non_edges_list = list(non_edges_current)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Return best edges found
    result = []
    for (u, v) in best_edges:
        result.append([u, v])
    
    return result[:91]
