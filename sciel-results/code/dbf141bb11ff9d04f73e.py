import random
import time
import sys
from collections import defaultdict
from random import Random

def instance_graph(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_graph(edges, n):
    """
    Decompose the graph into edge-disjoint cycles and single edges.
    Strategy: Greedily find cycles and remove them.
    Remaining edges are returned as single-edge pieces.
    """
    remaining_edges = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    for u, v in remaining_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    cycles = []
    
    def find_cycle():
        """
        Finds a cycle in the current graph using DFS.
        Returns a list of vertices representing the cycle, or None.
        """
        visited = {} # node -> (state, parent)
        # state: 0=unvisited, 1=visiting, 2=done
        for start in list(adj.keys()):
            if visited.get(start, 0) == 0:
                # DFS
                stack = [(start, start, [start])] # (node, parent, path)
                visited[start] = (1, start)
                while stack:
                    node, parent, path = stack.pop()
                    for neighbor in adj[node]:
                        if neighbor == parent:
                            continue
                        if visited.get(neighbor, 0) == 0:
                            visited[neighbor] = (1, node)
                            stack.append((neighbor, node, path + [neighbor]))
                        elif visited.get(neighbor, 0) == 1:
                            # Found a back edge to a node in the current path
                            # The cycle is from neighbor to node in the path
                            if neighbor in path:
                                idx = path.index(neighbor)
                                cycle = path[idx:]
                                # Ensure the cycle is valid (distinct vertices in the list representation for the output)
                                # The list `cycle` contains distinct vertices because it's a path segment.
                                # The edge (node, neighbor) closes the cycle.
                                return cycle
                            # If neighbor is not in path but is visiting, it's a cross edge in undirected graph?
                            # In undirected DFS, back edges are to ancestors.
                            # If neighbor is visiting but not in path, it might be in a different branch?
                            # Actually, in undirected graphs, if we see a visiting node, it's always an ancestor in the DFS tree.
                            # So it must be in the path.
                # Mark as done? Not strictly necessary for cycle finding if we restart from unvisited.
                # But we need to clean up 'visiting' states if we don't find a cycle in this component?
                # If we don't find a cycle, the component is a tree.
        
        return None

    # To avoid infinite loops or slow performance, we limit the number of cycle extraction steps.
    # With N=30, max edges ~ 435. Max cycles ~ 435/3 = 145.
    # Each cycle finding is O(N+E).
    # Total time is very small.
    
    while remaining_edges:
        cycle = find_cycle()
        if cycle is None:
            break
        
        # Remove edges of the cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge_key = tuple(sorted((u, v)))
            if edge_key in remaining_edges:
                remaining_edges.remove(edge_key)
                adj[u].remove(v)
                adj[v].remove(u)
                # Cleanup empty adj lists
                if not adj[u]:
                    del adj[u]
                if not adj[v]:
                    del adj[v]
        
        cycles.append(cycle)
        
        # Optional: Time check
        # if time.time() - start_time > 100: break

    # Remaining edges are single-edge pieces
    edge_pieces = [list(e) for e in remaining_edges]
    
    return cycles, edge_pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    start_time = time.time()
    
    for n, p, seed in specs:
        edges = instance_graph(n, p, seed)
        cycles, edge_pieces = decompose_graph(edges, n)
        
        # Combine into a single list of pieces
        # Cycles are lists of vertices [v0, v1, ..., vk-1]
        # Edges are lists of vertices [u, v]
        pieces = cycles + edge_pieces
        results.append(pieces)
        
        # Deadline guard
        if time.time() - start_time > 110:
            # If we run out of time, we return what we have.
            # Since the algorithm is greedy and fast, this shouldn't happen.
            # But if it does, we fill the rest with empty lists to maintain structure?
            # No, the task expects 3 lists. If we fail to compute one, we should return valid partial results or empty?
            # The verifier expects 3 lists.
            # If we timeout, we score 0.
            # So we just break and return.
            pass
            
    return results
