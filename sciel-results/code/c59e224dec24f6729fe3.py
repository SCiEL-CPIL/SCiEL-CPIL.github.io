import time
import itertools
from collections import deque

def construct():
    """
    Constructs a dense graph on 15 vertices that avoids equal-degree 3-path endpoints.
    
    Strategy:
    1. Start with K_{7,8} (bipartite, 56 edges) which is valid.
    2. Try to add intra-partition edges (chords) one by one.
    3. Adding a chord creates potential violations for paths involving that edge.
    4. We use a local repair heuristic: if adding an edge creates violations,
       we try to remove other edges to fix the violations, or reject the addition.
    5. We use a time budget to explore as many improvements as possible.
    
    The key insight is that while K_{7,8} is a strong baseline, adding chords might
    allow for denser graphs if we can manage the degree constraints carefully.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 15
    # Initial graph: K_{7,8}
    # Partition A: 0..6, Partition B: 7..14
    edges = set()
    for u in range(7):
        for v in range(7, 15):
            edges.add((u, v))
    
    def get_adjacency(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]
    
    def check_violations(adj, degrees, edge_set):
        """
        Check if any simple 3-edge path s-u-v-t has deg(s) == deg(t).
        Returns list of violating edges (u,v) that are part of such paths.
        Actually, we need to know which edges to remove to fix it.
        Let's just check if the graph is valid.
        """
        adj_list = [list(adj[i]) for i in range(n)]
        for u, v in edge_set:
            # For each edge (u,v), check paths s-u-v-t
            for s in adj_list[u]:
                if s == v:
                    continue
                for t in adj_list[v]:
                    if t == u or t == s:
                        continue
                    # Path s-u-v-t exists
                    if degrees[s] == degrees[t]:
                        return False
        return True
    
    def try_add_edge(edge_set, adj, degrees, edge):
        """
        Try adding 'edge' to the graph. If it creates violations, try to repair
        by removing other edges. Returns new edge_set if valid, else None.
        """
        u, v = edge
        if u in adj[v]:
            return edge_set
        
        new_edges = set(edge_set)
        new_edges.add(edge)
        
        # Update adjacency and degrees temporarily
        new_adj = [set(adj[i]) for i in range(n)]
        new_adj[u].add(v)
        new_adj[v].add(u)
        new_degrees = degrees.copy()
        new_degrees[u] += 1
        new_degrees[v] += 1
        
        if check_violations(new_adj, new_degrees, new_edges):
            return new_edges
        
        # Try to repair: remove edges that are causing conflicts
        # Heuristic: Find all paths s-u-v-t where deg(s)==deg(t)
        # and try removing the edge (s,u) or (v,t)
        
        # This is a simplified repair. In a full implementation, we'd do local search.
        # Given time constraints, we'll just reject if simple check fails.
        # However, we can try a few removals.
        
        # Collect conflicting edges
        conflicts = set()
        adj_list_u = list(new_adj[u])
        adj_list_v = list(new_adj[v])
        
        for s in adj_list_u:
            if s == v:
                continue
            for t in adj_list_v:
                if t == u or t == s:
                    continue
                if new_degrees[s] == new_degrees[t]:
                    # Conflict on path s-u-v-t
                    # Options: remove (s,u) or (v,t)
                    conflicts.add((min(s,u), max(s,u)))
                    conflicts.add((min(v,t), max(v,t)))
        
        # Try removing one conflict edge at a time
        for rem_edge in conflicts:
            if rem_edge == edge:
                continue
            candidate_edges = new_edges - {rem_edge}
            if rem_edge not in candidate_edges:
                continue
            
            cand_adj = [set(new_adj[i]) for i in range(n)]
            a, b = rem_edge
            cand_adj[a].remove(b)
            cand_adj[b].remove(a)
            
            cand_degrees = new_degrees.copy()
            cand_degrees[a] -= 1
            cand_degrees[b] -= 1
            
            if check_violations(cand_adj, cand_degrees, candidate_edges):
                return candidate_edges
                
        return None

    # Main loop: try to add chords
    # Potential chords are intra-partition edges
    chords_a = [(i, j) for i in range(7) for j in range(i+1, 7)]
    chords_b = [(i, j) for i in range(7, 15) for j in range(i+1, 15)]
    all_chords = chords_a + chords_b
    
    # Shuffle chords for diverse attempts? Or try in order.
    # Let's try a greedy approach with backtracking-like repair.
    
    current_edges = edges
    adj = get_adjacency(current_edges)
    degrees = get_degrees(adj)
    
    best_edges = current_edges
    best_count = len(current_edges)
    
    # We will try to add chords in a random order to explore different possibilities
    import random
    random.seed(42)
    random.shuffle(all_chords)
    
    # We can also try removing some cross edges to allow more chords, but that's complex.
    # Let's stick to adding chords first.
    
    for chord in all_chords:
        if time.time() > deadline:
            break
        
        new_edges = try_add_edge(current_edges, adj, degrees, chord)
        if new_edges is not None and len(new_edges) > best_count:
            best_edges = new_edges
            best_count = len(new_edges)
            current_edges = new_edges
            adj = get_adjacency(current_edges)
            degrees = get_degrees(adj)
    
    # If we didn't find improvements, maybe try a different strategy:
    # Start with K_{7,8}, remove some edges to make room for more chords?
    # This is complex. Let's return the best found.
    
    final_edges = []
    for u, v in best_edges:
        if isinstance(u, int) and isinstance(v, int):
            final_edges.append([u, v])
    
    return final_edges
