import random
import itertools

def construct():
    n = 20
    vertices = list(range(n))
    all_edges = list(itertools.combinations(vertices, 2))
    
    best_edges = []
    best_count = 0
    
    # Precompute edge indices for quick lookup?
    # Or just use sets for adj.
    
    for restart in range(100):
        random.seed(123 + restart)
        
        adj = [set() for _ in range(n)]
        # common_count[u][v] is the number of common neighbors of u and v
        common_count = [[0]*n for _ in range(n)]
        
        current_edges = []
        
        # 1. Random Init
        for u, v in all_edges:
            if random.random() < 0.4: # Slightly less dense to start
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append((u, v))
                # Update common counts
                # For each w in adj[u] (excluding v), common(u, w) increases by 1?
                # No, adding (u,v) affects:
                # 1. common(u, v) is not directly defined as count of common neighbors of u and v?
                #    Wait, common_count[u][v] = len(adj[u] & adj[v]).
                #    Adding (u,v) doesn't change adj[u] & adj[v] as established before?
                #    Wait, adj[u] gains v, adj[v] gains u.
                #    adj[u] & adj[v] remains the same set?
                #    Let's re-verify.
                #    Old: N(u), N(v). Common = N(u) \cap N(v).
                #    New: N'(u) = N(u) \cup {v}, N'(v) = N(v) \cup {u}.
                #    New Common = (N(u) \cup {v}) \cap (N(v) \cup {u})
                #    = (N(u) \cap N(v)) \cup (N(u) \cap {u}) \cup ({v} \cap N(v)) \cup ({v} \cap {u})
                #    = (N(u) \cap N(v)) \cup \emptyset \cup \emptyset \cup \emptyset
                #    So common_count[u][v] does NOT change.
                
                #    However, for any w:
                #    If w is in N(u) and w != v:
                #       Does common(u, w) change?
                #       N'(u) has v. N'(w) has v? Only if (v,w) is an edge.
                #       If (v,w) is an edge, then v is in N(w).
                #       So v becomes a common neighbor of u and w.
                #       So common_count[u][w] increases by 1.
                #    Similarly, if w is in N(v) and w != u:
                #       If (u,w) is an edge, then u becomes a common neighbor of v and w.
                #       So common_count[v][w] increases by 1.
                
                # So we need to update:
                # For all w in adj[u] (old adj[u], before adding v? No, v is new):
                # Actually, we added v to adj[u].
                # For each w in adj[u] (which now includes v):
                #   If w == v: skip (common_count[u][v] unchanged)
                #   Else: 
                #      If v in adj[w]: common_count[u][w] += 1; common_count[w][u] += 1
                #      (Note: v in adj[w] means (v,w) is an edge)
                
                # For each w in adj[v] (which now includes u):
                #   If w == u: skip
                #   Else:
                #      If u in adj[w]: common_count[v][w] += 1; common_count[w][v] += 1
                
                # This is O(degree).
                
        # This update logic is complex to implement correctly in a tight loop without bugs.
        # Given n=20, calculating `len(adj[u] & adj[v])` is very fast (set intersection of size <= 19).
        # 100 restarts * (Prune + Add) might be fast enough with direct set intersection.
        
        # Let's stick to the simpler Set Intersection approach but optimize the loops.
        
        adj = [set() for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            if random.random() < 0.4:
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append((u, v))
        
        # Prune
        # We need to remove edges until no pair has > 2 common neighbors.
        # To make this efficient, we can find a violation and remove a random edge from it.
        
        for _ in range(500): # Max prune steps
            violated = False
            for u in range(n):
                for v in range(u+1, n):
                    if len(adj[u] & adj[v]) > 2:
                        violated = True
                        # Pick a random common neighbor w and remove (u, w)
                        common = list(adj[u] & adj[v])
                        w = random.choice(common)
                        # Remove edge (u, w)
                        adj[u].discard(w)
                        adj[w].discard(u)
                        # Remove from current_edges
                        # Find index
                        # Since list is small, linear search is fine
                        for i, edge in enumerate(current_edges):
                            if (edge[0] == u and edge[1] == w) or (edge[0] == w and edge[1] == u):
                                current_edges.pop(i)
                                break
                        break
                if violated:
                    break
            if not violated:
                break
        
        # Add
        improved = True
        while improved:
            improved = False
            # Try all non-edges
            for u, v in all_edges:
                if u in adj[v]:
                    continue
                
                # Check if adding (u,v) is valid
                # 1. len(adj[u] & adj[v]) <= 2
                if len(adj[u] & adj[v]) > 2:
                    continue
                
                valid = True
                # Check w in adj[v]
                for w in adj[v]:
                    if w == u: continue
                    # New common(u, w) = old + (1 if v in adj[w] else 0)
                    # v in adj[w] is equivalent to w in adj[v] (symmetry)
                    # So if w in adj[v], then v in adj[w].
                    # So for ALL w in adj[v] (w!=u), the count increases by 1.
                    # Wait.
                    # N'(u) = N(u) U {v}.
                    # N'(w) = N(w).
                    # Common'(u, w) = (N(u) U {v}) & N(w) = (N(u) & N(w)) U ({v} & N(w)).
                    # {v} & N(w) is {v} if v in N(w), else empty.
                    # v in N(w) iff (v,w) is an edge.
                    # We are iterating w in adj[v], which means (v,w) is an edge.
                    # So v IS in N(w).
                    # So Common'(u, w) = Common(u, w) + 1.
                    
                    if len(adj[u] & adj[w]) + 1 > 2:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check w in adj[u]
                for w in adj[u]:
                    if w == v: continue
                    # Similarly, for w in adj[u], u is in N(w).
                    # Common'(v, w) = Common(v, w) + 1.
                    if len(adj[v] & adj[w]) + 1 > 2:
                        valid = False
                        break
                
                if valid:
                    adj[u].add(v)
                    adj[v].add(u)
                    current_edges.append((u, v))
                    improved = True
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
    return [[u, v] for u, v in best_edges]
