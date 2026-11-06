"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, remaining_edges, max_len=None):
        """Find a long cycle in the remaining graph using DFS with backtracking."""
        if max_len is None:
            max_len = len(remaining_edges)
        
        best_cycle = []
        
        # Try to find cycles starting from each vertex
        for start in sorted(adj.keys()):
            if len(adj[start]) < 2:
                continue
            
            # DFS to find longest cycle starting and ending at start
            def dfs(node, path, visited_edges, visited_nodes):
                nonlocal best_cycle
                if len(path) > len(best_cycle):
                    best_cycle = path[:]
                
                if max_len and len(path) >= max_len:
                    return
                
                for neighbor in adj[node]:
                    edge = tuple(sorted((node, neighbor)))
                    if edge in remaining_edges and edge not in visited_edges:
                        if neighbor == start and len(path) >= 3:
                            # Found a cycle
                            if len(path) > len(best_cycle):
                                best_cycle = path[:]
                        elif neighbor not in visited_nodes:
                            visited_edges.add(edge)
                            visited_nodes.add(neighbor)
                            path.append(neighbor)
                            dfs(neighbor, path, visited_edges, visited_nodes)
                            path.pop()
                            visited_nodes.remove(neighbor)
                            visited_edges.remove(edge)
            
            dfs(start, [start], set(), {start})
            
            if best_cycle and (max_len is None or len(best_cycle) >= max_len):
                break
        
        return best_cycle

    def decompose_greedy(edges, time_limit=30):
        """Decompose edges into cycles and single edges using greedy longest-cycle selection."""
        if not edges:
            return []
        
        deadline_local = time.time() + time_limit
        pieces = []
        remaining = set(tuple(sorted(e)) for e in edges)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        while remaining:
            if time.time() > deadline_local:
                # Fallback: add all remaining edges as single edges
                for e in remaining:
                    pieces.append(list(e))
                break
            
            # Find longest cycle
            cycle = find_longest_cycle(adj, remaining)
            
            if cycle and len(cycle) >= 3:
                # Extract edges from cycle
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    cycle_edges.add(edge)
                
                # Verify all edges exist in remaining
                if cycle_edges.issubset(remaining):
                    pieces.append(cycle)
                    remaining -= cycle_edges
                    # Update adjacency
                    for u, v in cycle_edges:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        if not adj[u]:
                            del adj[u]
                        if not adj[v]:
                            del adj[v]
                else:
                    # Cycle invalid, treat as single edge
                    pieces.append(list(cycle[0], cycle[1]))
                    edge = tuple(sorted((cycle[0], cycle[1])))
                    remaining.discard(edge)
                    adj[cycle[0]].discard(cycle[1])
                    adj[cycle[1]].discard(cycle[0])
            else:
                # No cycle found, take a single edge
                u, v = next(iter(remaining))
                pieces.append([u, v])
                remaining.discard((u, v))
                adj[u].discard(v)
                adj[v].discard(u)
                if not adj[u]:
                    del adj[u]
                if not adj[v]:
                    del adj[v]
        
        return pieces

    results = []
    for n, p, seed in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(n, p, seed)
        pieces = decompose_greedy(edges, time_limit=30)
        results.append(pieces)
        if time.time() > deadline:
            # Emergency: return what we have, pad with single edges if needed
            for i in range(len(results), 3):
                results.append([])
            break
    
    return results
# EVOLVE-BLOCK-END
