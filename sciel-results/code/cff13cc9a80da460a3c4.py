import time
import random

def construct():
    """
    Decomposes the edge sets of three fixed random graphs into cycles and single edges,
    minimizing the total number of pieces.
    """
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    # Global deadline guard: 110 seconds to be safe
    start_time = time.time()
    deadline = 110.0
    
    for spec in specs:
        if time.time() > deadline:
            # If we run out of time, return empty or best effort?
            # The task says "invalid result ... scores 0". 
            # We should try to return something valid if possible, but if we can't, 
            # we might as well return the baseline for the remaining graphs?
            # Actually, if we haven't finished graph 1, we can't do 2 and 3.
            # We will just process as many as we can.
            pass
            
        edges = instance(*spec)
        # Convert to set of tuples for fast lookup and removal
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        pieces = []
        
        # Decompose this graph
        while edge_set:
            if time.time() > deadline:
                # Fallback: add all remaining edges as single pieces
                for e in edge_set:
                    pieces.append(list(e))
                break
            
            best_cycle = None
            best_len = 0
            
            # Try to find a cycle by picking random edges as potential closing edges
            # We limit the number of attempts to keep time bounded
            # 50 attempts per cycle is a good tradeoff for n<=30
            edge_list = list(edge_set)
            random.shuffle(edge_list)
            
            for e in edge_list[:50]:
                u, v = e
                # Temporarily remove the edge to see if a path exists between u and v
                edge_set.remove(e)
                
                path = find_path_heuristic(u, v, edge_set)
                
                # Restore the edge
                edge_set.add(e)
                
                if path:
                    # Path is from u to v (or v to u). 
                    # In find_path_heuristic, we search from start to end.
                    # If path is found, it is a list of nodes [u, ..., v].
                    # The cycle is this path.
                    # Wait, if we removed (u,v), and found path u->v, 
                    # then the cycle is u->...->v->u.
                    # The piece is the cycle [u, ..., v].
                    # The closing edge (v,u) is implied.
                    # So the piece is the list of vertices in the path.
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path
            
            if best_cycle:
                # Add the cycle to pieces
                pieces.append(best_cycle)
                
                # Remove all edges in the cycle from edge_set
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = tuple(sorted((u, v)))
                    if edge in edge_set:
                        edge_set.remove(edge)
            else:
                # No cycle found, add a single edge
                e = next(iter(edge_set))
                edge_set.remove(e)
                pieces.append(list(e))
        
        results.append(pieces)

    return results

def find_path_heuristic(start, end, edges):
    """
    Finds a path from start to end in the graph defined by edges.
    Uses randomized DFS with multiple restarts to find a longer path.
    Returns the list of vertices in the path, or None if no path exists.
    """
    # Build adjacency list
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    
    if start not in adj or end not in adj:
        return None
        
    best_path = None
    
    # Try a few randomizations
    for _ in range(10):
        # Create a copy of adjacency with shuffled lists for randomness
        shuffled_adj = {}
        for node, neighbors in adj.items():
            n_list = list(neighbors)
            random.shuffle(n_list)
            shuffled_adj[node] = n_list
        
        path = []
        visited = set()
        # Stack for DFS: (node, iterator of neighbors)
        stack = [(start, iter(shuffled_adj.get(start, [])))]
        path.append(start)
        visited.add(start)
        
        found = False
        while stack:
            node, neighbors = stack[-1]
            
            if node == end and len(path) > 1:
                # Found a path
                if best_path is None or len(path) > len(best_path):
                    best_path = path[:]
                found = True
                break
            
            next_node = None
            for nb in neighbors:
                if nb not in visited:
                    next_node = nb
                    break
            
            if next_node is not None:
                visited.add(next_node)
                path.append(next_node)
                stack.append((next_node, iter(shuffled_adj.get(next_node, []))))
            else:
                # Backtrack
                stack.pop()
                path.pop()
                visited.remove(node)
        
        # If we found a path, we could stop, but we want the LONGEST.
        # However, DFS doesn't guarantee longest. 
        # Running multiple times helps.
        # If we found a path, we don't necessarily need to break immediately 
        # if we want to try more randomizations, but it's costly.
        # Given the small graph size, 10 restarts is cheap.
        
    return best_path

# We need to import Random from random for the instance generator
from random import Random
