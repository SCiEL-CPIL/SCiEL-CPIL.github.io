"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 16
    
    # Precompute Q3 structure
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            diff = bin(i ^ j).count('1')
            if diff == 1:
                q3_edges.add((i, j))
    
    # Build Q3 adjacency lists
    q3_adj = [set() for _ in range(8)]
    for (a, b) in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)
    
    def check_q3_free_fast(adj_sets, new_edge):
        """Fast check: only verify if adding new_edge creates a Q3 using structural properties."""
        u, v = new_edge
        
        # Get neighbors of u and v (excluding each other)
        nu = list(adj_sets[u])
        if v in nu:
            nu.remove(v)
        nv = list(adj_sets[v])
        if u in nv:
            nv.remove(u)
        
        # We assume the edge (u,v) maps to (0,1) in Q3.
        # Q3 structure relative to edge (0,1):
        # 0 is connected to 1, 2, 4.
        # 1 is connected to 0, 3, 5.
        # 2 connected to 0, 6.
        # 4 connected to 0, 6.
        # 3 connected to 1, 7.
        # 5 connected to 1, 7.
        # 6 connected to 2, 4, 7.
        # 7 connected to 3, 5, 6.
        
        # So we need:
        # x, y in N(u) (images of 2, 4)
        # z, w in N(v) (images of 3, 5)
        # p in N(x) ∩ N(y) (image of 6)
        # q in N(z) ∩ N(w) (image of 7)
        # p-q is an edge
        # All distinct: u, v, x, y, z, w, p, q
        
        for i in range(len(nu)):
            x = nu[i]
            for j in range(i+1, len(nu)):
                y = nu[j]
                # x, y distinct from u, v by construction of nu
                # x != y by loop
                for k in range(len(nv)):
                    z = nv[k]
                    if z == x or z == y:
                        continue
                    for l in range(k+1, len(nv)):
                        w = nv[l]
                        if w == x or w == y or w == z:
                            continue
                        
                        # Find candidates for p: neighbors of x and y
                        # p must be distinct from u, v, x, y, z, w
                        common_xy = adj_sets[x] & adj_sets[y]
                        forbidden = {u, v, x, y, z, w}
                        p_candidates = common_xy - forbidden
                        if not p_candidates:
                            continue
                        
                        common_zw = adj_sets[z] & adj_sets[w]
                        # q must be distinct from u, v, x, y, z, w and p
                        for p in p_candidates:
                            forbidden_q = {u, v, x, y, z, w, p}
                            q_candidates = common_zw - forbidden_q
                            for q in q_candidates:
                                if q in adj_sets[p]:
                                    return False
                            
        return True
    
    def compute_marginal_utility(adj, adj_sets, u, v):
        """Compute marginal utility for edge (u,v): ratio of degree to conflict size.
        
        Higher utility = more valuable to add. We prioritize edges that:
        1. Have high degree (connect well-connected vertices)
        2. Don't create many potential Q3 conflicts
        """
        deg_u = len(adj_sets[u])
        deg_v = len(adj_sets[v])
        avg_deg = (deg_u + deg_v) / 2.0
        
        # Count how many "partial Q3" structures this edge would complete
        # This is a heuristic: count common neighbors and their connections
        common = adj_sets[u] & adj_sets[v]
        conflict_score = len(common) * 2  # Each common neighbor creates potential conflicts
        
        # Utility: prefer high-degree edges with low conflict potential
        if conflict_score == 0:
            return avg_deg * 10
        return avg_deg / (1.0 + conflict_score * 0.5)
    
    def greedy_add_edges(adj, adj_sets, edges, candidate_edges, deadline):
        """Greedily add edges using marginal utility scoring, checking Q3-freeness."""
        # Sort candidates by marginal utility (highest first)
        scored_candidates = []
        for edge in candidate_edges:
            u, v = edge
            if not adj[u][v]:
                utility = compute_marginal_utility(adj, adj_sets, u, v)
                scored_candidates.append((utility, edge))
        
        # Sort by utility descending
        scored_candidates.sort(key=lambda x: -x[0])
        
        for utility, edge in scored_candidates:
            if time.time() > deadline - 10:
                break
            
            u, v = edge
            if adj[u][v]:
                continue
                
            # Try adding this edge
            adj[u][v] = 1
            adj[v][u] = 1
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            edges.append([u, v])
            
            # Check if Q3-free
            if not check_q3_free_fast(adj_sets, (u, v)):
                # Revert
                adj[u][v] = 0
                adj[v][u] = 0
                adj_sets[u].remove(v)
                adj_sets[v].remove(u)
                edges.pop()
        
        return edges
    
    def local_search_polish(adj, adj_sets, edges, deadline):
        """Try to re-add rejected edges after other edges have been added."""
        # Collect all edges that were rejected
        all_possible = set()
        for i in range(n):
            for j in range(i+1, n):
                all_possible.add((i, j))
        
        current_edges = set()
        for e in edges:
            current_edges.add((e[0], e[1]))
        
        rejected = list(all_possible - current_edges)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Score rejected edges by marginal utility
            scored_rejected = []
            for edge in rejected:
                u, v = edge
                if adj[u][v]:
                    continue
                utility = compute_marginal_utility(adj, adj_sets, u, v)
                scored_rejected.append((utility, edge))
            
            scored_rejected.sort(key=lambda x: -x[0])
            
            for utility, edge in scored_rejected:
                if time.time() > deadline - 5:
                    break
                    
                u, v = edge
                if adj[u][v]:
                    continue
                    
                # Try adding this edge
                adj[u][v] = 1
                adj[v][u] = 1
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                edges.append([u, v])
                
                # Check if Q3-free
                if check_q3_free_fast(adj_sets, (u, v)):
                    improved = True
                    rejected.remove(edge)
                else:
                    # Revert
                    adj[u][v] = 0
                    adj[v][u] = 0
                    adj_sets[u].remove(v)
                    adj_sets[v].remove(u)
                    edges.pop()
        
        return edges
    
    def final_polish_pass(adj, adj_sets, edges, deadline):
        """High-precision polish: try all possible single-edge additions and swaps."""
        if time.time() > deadline - 2:
            return edges
        
        all_possible = set()
        for i in range(n):
            for j in range(i+1, n):
                all_possible.add((i, j))
        
        current_edges = set()
        for e in edges:
            current_edges.add((e[0], e[1]))
        
        rejected = list(all_possible - current_edges)
        
        # Multiple passes with different orderings
        for pass_num in range(3):
            if time.time() > deadline - 2:
                break
            
            random.shuffle(rejected)
            
            for edge in rejected:
                if time.time() > deadline - 2:
                    break
                    
                u, v = edge
                if adj[u][v]:
                    continue
                    
                # Try adding this edge
                adj[u][v] = 1
                adj[v][u] = 1
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                edges.append([u, v])
                
                # Check if Q3-free
                if check_q3_free_fast(adj_sets, (u, v)):
                    rejected.remove(edge)
                else:
                    # Revert
                    adj[u][v] = 0
                    adj[v][u] = 0
                    adj_sets[u].remove(v)
                    adj_sets[v].remove(u)
                    edges.pop()
        
        return edges
    
    def build_initial_graph():
        """Start with K_{3,13}: vertices 0,1,2 in part A, vertices 3-15 in part B."""
        adj = [[0]*n for _ in range(n)]
        adj_sets = [set() for _ in range(n)]
        edges = []
        
        # Add all edges between part A (0,1,2) and part B (3-15)
        for a in range(3):
            for b in range(3, n):
                adj[a][b] = 1
                adj[b][a] = 1
                adj_sets[a].add(b)
                adj_sets[b].add(a)
                edges.append([a, b])
        
        return adj, adj_sets, edges
    
    # Generate all possible non-cross edges
    all_non_cross_edges = []
    for i in range(n):
        for j in range(i+1, n):
            in_a = lambda x: x < 3
            if in_a(i) and in_a(j):
                all_non_cross_edges.append((i, j))
            elif not in_a(i) and not in_a(j):
                all_non_cross_edges.append((i, j))
    
    best_edges = []
    best_count = 0
    
    # Try multiple random orderings
    for trial in range(25):
        if time.time() > deadline - 30:
            break
        
        # Create a random ordering with priority for within-A edges
        within_a_edges = [(i, j) for i, j in all_non_cross_edges if i < 3 and j < 3]
        within_b_edges = [(i, j) for i, j in all_non_cross_edges if i >= 3 and j >= 3]
        
        random.shuffle(within_a_edges)
        random.shuffle(within_b_edges)
        
        order = within_a_edges + within_b_edges
        
        adj, adj_sets, edges = build_initial_graph()
        edges = greedy_add_edges(adj, adj_sets, edges, order, deadline)
        edges = local_search_polish(adj, adj_sets, edges, deadline)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
    
    # Final high-precision polish on the best solution
    if best_edges and time.time() < deadline - 5:
        # Reconstruct adj from best_edges
        adj = [[0]*n for _ in range(n)]
        adj_sets = [set() for _ in range(n)]
        for u, v in best_edges:
            adj[u][v] = 1
            adj[v][u] = 1
            adj_sets[u].add(v)
            adj_sets[v].add(u)
        
        polished = final_polish_pass(adj, adj_sets, list(best_edges), deadline)
        if len(polished) > best_count:
            best_count = len(polished)
            best_edges = polished
    
    return best_edges
# EVOLVE-BLOCK-END
