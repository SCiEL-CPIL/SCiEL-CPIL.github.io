"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

# Increase recursion limit for DFS
sys.setrecursionlimit(100000)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Copy instance generator from verifier
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycles(edges_set, n, max_time_per_graph):
        """
        Find a set of edge-disjoint cycles in the graph.
        Returns list of cycles (each cycle is a list of vertices).
        """
        cycles = []
        graph = defaultdict(set)
        for u, v in edges_set:
            graph[u].add(v)
            graph[v].add(u)
        
        # Use iterative DFS to find cycles
        def find_one_cycle():
            """Find one cycle using DFS."""
            for start in sorted(graph.keys()):
                if not graph[start]:
                    continue
                # Try to find a cycle starting from 'start'
                stack = [(start, [start], {start})]
                while stack:
                    node, path, visited = stack.pop()
                    for neighbor in sorted(graph[node]):
                        if neighbor == start and len(path) >= 3:
                            # Found a cycle
                            return path
                        if neighbor not in visited:
                            new_path = path + [neighbor]
                            new_visited = visited | {neighbor}
                            stack.append((neighbor, new_path, new_visited))
            return None
        
        # Repeatedly find cycles and remove their edges
        while True:
            if time.time() > max_time_per_graph:
                break
            cycle = find_one_cycle()
            if cycle is None:
                break
            # Remove edges of this cycle from the graph
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                if edge in edges_set:
                    edges_set.discard(edge)
                    graph[u].discard(v)
                    graph[v].discard(u)
                    # Clean up empty nodes
                    if not graph[u]:
                        del graph[u]
                    if not graph[v]:
                        del graph[v]
            cycles.append(cycle)
        
        return cycles

    def decompose(edges, n, max_time_per_graph):
        """
        Decompose the graph into cycles and single edges.
        Returns a list of pieces (each piece is a list of vertices for a cycle, or [u, v] for an edge).
        """
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        # First, find cycles
        cycles = find_cycles(edges_set, n, max_time_per_graph)
        
        # Remaining edges become single-edge pieces
        pieces = [list(cycle) for cycle in cycles]
        for u, v in edges_set:
            pieces.append([u, v])
        
        return pieces

    # Process each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for i, (n, p, seed) in enumerate(specs):
        if time.time() > deadline:
            # Not enough time, return what we have
            break
        
        edges = instance(n, p, seed)
        
        # Allocate remaining time to this graph
        remaining_time = deadline - time.time()
        time_per_graph = remaining_time / (len(specs) - i)
        
        pieces = decompose(edges, n, time_per_graph)
        results.append(pieces)
    
    # If we ran out of time, fill remaining with trivial decomposition
    while len(results) < len(specs):
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([list(e) for e in edges])
    
    return results
# EVOLVE-BLOCK-END
