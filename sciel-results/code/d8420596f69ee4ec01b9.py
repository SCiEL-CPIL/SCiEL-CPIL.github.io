import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    vertices = range(n)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Pre-compute 8-subsets
    subsets_8 = list(combinations(vertices, 8))
    
    def contains_q3_check(sub, adj_set):
        """
        Check if the induced subgraph on 'sub' contains a Q3.
        """
        # Count edges and collect them
        edges_in_sub = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = sub[i], sub[j]
                if (min(u,v), max(u,v)) in adj_set:
                    edges_in_sub.append((min(u,v), max(u,v)))
        
        if len(edges_in_sub) < 12:
            return False
        
        # If exactly 12 edges, check if it is Q3
        if len(edges_in_sub) == 12:
            # Check 3-regular
            degs = {v: 0 for v in sub}
            for u, v in edges_in_sub:
                degs[u] += 1
                degs[v] += 1
            if any(d != 3 for d in degs.values()):
                return False
            
            # Check bipartite and connected
            color = {}
            color[sub[0]] = 0
            queue = [sub[0]]
            while queue:
                v = queue.pop(0)
                for u in sub:
                    if u != v and (min(v,u), max(v,u)) in adj_set:
                        if u not in color:
                            color[u] = 1 - color[v]
                            queue.append(u)
                        elif color[u] == color[v]:
                            return False
            if len(color) != 8:
                return False
            return True
        
        # If > 12 edges, assume it contains a Q3 to be safe (conservative)
        # This might remove edges unnecessarily, but ensures validity.
        return True

    def is_q3_free(adj_set):
        for sub in subsets_8:
            if contains_q3_check(sub, adj_set):
                return False
        return True

    # Start with K16
    adj = set(all_edges)
    
    # Greedy removal
    while time.time() < deadline - 30:
        # Find a Q3
        q3_found_sub = None
        for sub in subsets_8:
            if contains_q3_check(sub, adj):
                q3_found_sub = sub
                break
        
        if q3_found_sub is None:
            break
        
        # Remove a random edge from this subgraph
        edges_in_sub = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = q3_found_sub[i], q3_found_sub[j]
                if (min(u,v), max(u,v)) in adj:
                    edges_in_sub.append((min(u,v), max(u,v)))
        
        if edges_in_sub:
            edge_to_remove = random.choice(edges_in_sub)
            adj.discard(edge_to_remove)
        
        # Check if we are out of time
        if time.time() > deadline - 30:
            break

    # Local search to add edges
    best_edges = len(adj)
    best_adj = set(adj)
    
    while time.time() < deadline:
        improved = False
        missing = [e for e in all_edges if e not in adj]
        random.shuffle(missing)
        
        for edge in missing:
            if time.time() > deadline:
                break
            
            adj.add(edge)
            if is_q3_free(adj):
                if len(adj) > best_edges:
                    best_edges = len(adj)
                    best_adj = set(adj)
                    improved = True
            else:
                adj.discard(edge)
        
        if not improved:
            # Random restart/perturbation
            adj = set(best_adj)
            # Remove a few random edges to escape local optimum
            for _ in range(5):
                if adj:
                    e = random.choice(list(adj))
                    adj.discard(e)
            
            # Try to add them back or different ones
            # The loop above will handle adding.
        
        # Check time
        if time.time() > deadline - 5:
            break

    return [[u, v] for u, v in best_adj]
