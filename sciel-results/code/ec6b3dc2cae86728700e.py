import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We need to return well before.
    # Let's aim for 140s max to be safe.
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def greedy_decompose(edges, rng_seed):
        """
        Performs a greedy cycle peeling decomposition.
        Uses a specific random seed to determine the order of neighbors and start vertices,
        allowing different runs to explore different decomposition paths.
        """
        if not edges:
            return []

        rng = random.Random(rng_seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set(edges)
        pieces = []

        def find_cycle():
            """
            Finds a cycle in the remaining graph.
            Strategy: Pick a random starting vertex with degree >= 2.
            Perform a randomized DFS to find a cycle.
            To encourage longer cycles, we can try to avoid closing the loop too early,
            but for simplicity and speed, we just find *a* cycle.
            Actually, finding the *longest* cycle is better.
            Let's use a bounded DFS that tries to find a long cycle.
            """
            if not remaining_edges:
                return None

            # Get vertices with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                return None

            # Shuffle candidates to vary the start point
            rng.shuffle(candidates)

            # Try a few start vertices
            for start in candidates[:5]:
                # Reset visited for this start
                visited = {start}
                path = [start]
                
                # DFS to find a cycle
                # We want to find a cycle that closes back to start.
                # To find a "good" (long) cycle, we can use a heuristic:
                # Prefer neighbors with higher degree? Or just random?
                # Randomness allows exploration.
                
                def dfs(current):
                    # Check if we can close the cycle
                    # If path length >= 3 and current is connected to start
                    if len(path) >= 3:
                        if start in adj[current]:
                            return list(path)
                    
                    # Get neighbors
                    neighbors = list(adj[current])
                    rng.shuffle(neighbors) # Randomize order
                    
                    for neighbor in neighbors:
                        if neighbor == start:
                            continue
                        if neighbor not in visited:
                            visited.add(neighbor)
                            path.append(neighbor)
                            res = dfs(neighbor)
                            if res:
                                return res
                            path.pop()
                            visited.remove(neighbor)
                            # Optional: Prune if path gets too long? 
                            # For small graphs, no need.
                    return None

                cycle = dfs(start)
                if cycle:
                    return cycle
            
            # Fallback: Find any triangle if DFS failed (shouldn't happen if deg>=2 and connected comp has cycle)
            # But if the graph is a tree, no cycle.
            # If we are here, we might have missed a cycle due to DFS pathing?
            # Unlikely for small graphs.
            return None

        def remove_cycle_edges(cycle):
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i+1]
                e = tuple(sorted((u, v)))
                if e in remaining_edges:
                    remaining_edges.remove(e)
                # Update adjacency
                adj[u].discard(v)
                adj[v].discard(u)
            
            u, v = cycle[-1], cycle[0]
            e = tuple(sorted((u, v)))
            if e in remaining_edges:
                remaining_edges.remove(e)
            adj[u].discard(v)
            adj[v].discard(u)
            
            # Clean up empty adj entries
            for u in cycle:
                if not adj[u]:
                    del adj[u]

        while remaining_edges:
            cycle = find_cycle()
            if cycle:
                pieces.append(cycle)
                remove_cycle_edges(cycle)
            else:
                # No cycle found, must be a tree or forest.
                # Pick an edge
                edge = next(iter(remaining_edges))
                pieces.append(list(edge))
                remaining_edges.remove(edge)
                u, v = edge
                adj[u].discard(v)
                adj[v].discard(u)
                if not adj[u]:
                    del adj[u]
                if not adj[v]:
                    del adj[v]

        return pieces

    def decompose_with_restarts(edges, time_limit):
        """
        Runs greedy_decompose multiple times with different seeds and returns the best.
        """
        best_pieces = None
        best_count = float('inf')
        
        # Estimate time per run
        # We don't know the exact time, so we run until time limit.
        # Small graphs are fast.
        
        seed_counter = 0
        while time.time() < time_limit:
            # Use a pseudo-random seed based on counter to ensure variety
            run_seed = seed_counter + 12345
            pieces = greedy_decompose(edges, run_seed)
            
            # Validate pieces quickly (optional, assume greedy is valid)
            # Count pieces
            count = len(pieces)
            
            if count < best_count:
                best_count = count
                best_pieces = pieces
                # If we found a decomposition with only 1 piece (impossible for non-cycle) 
                # or very few, we might stop early? 
                # Theoretical lower bound: ceil(E / (n-1))? No, max cycle length is n.
                # So min pieces is roughly E/n.
                # For n=30, E~135, min pieces ~ 5.
                # If we get close to that, we can stop.
                # But let's just run as many as possible.
            
            seed_counter += 1
            # Small sleep to avoid CPU thrashing if runs are very fast?
            # No, CPU bound is fine.
            
        return best_pieces if best_pieces is not None else [list(e) for e in edges]

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time for this graph
        # We have 3 graphs.
        # Let's give each graph 1/3 of the available time, minus a buffer.
        time_remaining = deadline - time.time()
        time_per_graph = time_remaining / (len(graphs) - i)
        # Reserve 1 second for overhead
        time_limit = time.time() + max(1.0, time_per_graph * 0.95)
        
        best_pieces = decompose_with_restarts(edges, time_limit)
        results.append(best_pieces)

        # Check if we are running out of time for subsequent graphs
        if time.time() > deadline - 2.0:
            # Fallback for remaining graphs: trivial decomposition
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break

    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)

        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break

        if valid and used_edges != edge_set:
            valid = False

        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
