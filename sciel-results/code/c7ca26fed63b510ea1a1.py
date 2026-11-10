"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def decompose_graph(edges, time_limit, trial_seed):
        """
        Decompose edges into cycles and single edges.
        Uses a greedy strategy to find long cycles, then tries to improve via local search.
        """
        remaining = set()
        for e in edges:
            remaining.add(edge_key(e[0], e[1]))
        
        if not remaining:
            return []
        
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        # Track edge usage for potential local search
        edge_to_piece = {}
        
        def commit_cycle(cycle_vertices):
            for i in range(len(cycle_vertices)):
                u = cycle_vertices[i]
                v = cycle_vertices[(i + 1) % len(cycle_vertices)]
                ek = edge_key(u, v)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
                edge_to_piece[ek] = len(pieces)
            pieces.append(cycle_vertices[:])
        
        def commit_single_edge(u, v):
            ek = edge_key(u, v)
            remaining.discard(ek)
            adj[u].discard(v)
            adj[v].discard(u)
            edge_to_piece[ek] = len(pieces)
            pieces.append([u, v])
        
        # Phase 1: Greedy cycle extraction
        # We want to find long cycles to minimize the number of pieces.
        # Strategy: Repeatedly pick a vertex with high degree, try to find a long cycle through it.
        
        while remaining:
            if time.time() > time_limit:
                break
            
            # Find vertices with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                break
            
            # Sort candidates by degree descending to prioritize dense areas
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            # Randomize the top candidates to avoid getting stuck in local optima
            top_k = min(5, len(candidates))
            random.seed(trial_seed) # Local seed for this trial's randomization
            random.shuffle(candidates[:top_k])
            
            found_cycle = None
            
            for start in candidates[:top_k]:
                if time.time() > time_limit:
                    break
                
                # Try to find a cycle starting from 'start'
                # We use a randomized DFS with a limit on depth and branching
                path = [start]
                visited = {start}
                
                # Use a stack for iterative DFS to avoid recursion limits and overhead
                # But we need to track the path for cycle reconstruction.
                # Let's stick to recursive for clarity but with strict depth limits.
                
                def find_long_cycle(current, depth, max_depth=15):
                    if time.time() > time_limit:
                        return None
                    if depth >= max_depth:
                        return None
                    
                    # Get neighbors, prioritize unvisited
                    neighbors = list(adj[current])
                    # Shuffle neighbors for randomization
                    random.shuffle(neighbors)
                    
                    for nb in neighbors:
                        if nb == start and len(path) >= 3:
                            return path[:]
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            result = find_long_cycle(nb, depth + 1, max_depth)
                            if result is not None:
                                return result
                            path.pop()
                            visited.discard(nb)
                    return None
                
                cycle = find_long_cycle(start, 0)
                if cycle is not None:
                    found_cycle = cycle
                    # If we found a very long cycle, break early
                    if len(cycle) >= 8:
                        break
                else:
                    # Try a simpler approach: just find ANY cycle if the long one fails
                    # This might happen if the graph is sparse or structured in a tricky way
                    pass
            
            if found_cycle:
                commit_cycle(found_cycle)
            else:
                # If no cycle found via the heuristic, but there are edges with deg >= 2,
                # there must be a cycle. We might have missed it due to depth limit.
                # Fallback: Find a cycle using a standard algorithm (e.g., DFS on the remaining graph)
                # For speed, we just pick an edge and try to close a loop.
                # If we can't find one, we commit a single edge to ensure progress.
                # This is a safety net.
                if remaining:
                    # Try to find a cycle by following a path
                    # Pick a random edge
                    u, v = next(iter(remaining))
                    # Try to extend from v to u
                    path = [u, v]
                    visited = {u, v}
                    
                    def find_cycle_from_path(current, depth, max_depth=10):
                        if time.time() > time_limit:
                            return None
                        if depth >= max_depth:
                            return None
                        neighbors = list(adj[current])
                        random.shuffle(neighbors)
                        for nb in neighbors:
                            if nb == u and len(path) >= 3:
                                return path[:]
                            if nb not in visited:
                                visited.add(nb)
                                path.append(nb)
                                result = find_cycle_from_path(nb, depth + 1, max_depth)
                                if result is not None:
                                    return result
                                path.pop()
                                visited.discard(nb)
                        return None
                    
                    cycle = find_cycle_from_path(v, 0)
                    if cycle:
                        commit_cycle(cycle)
                    else:
                        # Truly stuck or no cycle found quickly, take single edge
                        commit_single_edge(u, v)
        
        # Phase 2: Local Search / Improvement
        # Try to merge two single edges into a cycle if possible? No, they must be edge-disjoint.
        # Try to replace a short cycle + single edge with a longer cycle?
        # Or replace two single edges with a cycle?
        # A cycle of length k uses k edges. Two single edges use 2 edges.
        # If we can form a cycle using those 2 edges and some others, we reduce piece count by 1 (2 pieces -> 1 piece).
        # This is complex. Let's stick to the greedy approach which is already quite good.
        
        # Any remaining edges become single-edge pieces
        for u, v in list(remaining):
            commit_single_edge(u, v)
            
        return pieces

    # Main: decompose each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for spec in specs:
        if time.time() > deadline - 15:
            # Not enough time, use trivial solution
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        
        # Try multiple random seeds and keep the best
        best_pieces = None
        best_count = float('inf')
        
        # Allocate time for this instance
        time_per_instance = min(40, deadline - time.time() - 15)
        if time_per_instance < 5:
            time_per_instance = 5
            
        # Run multiple trials with different random seeds
        # We want to maximize the number of trials, so we make each trial fast
        # The decompose_graph function has internal time checks.
        
        num_trials = 30
        trial_time = time_per_instance / num_trials
        
        for t in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Set a specific seed for reproducibility within the trial
            # But we pass it to decompose_graph for internal shuffling
            
            # Time limit for this specific trial
            trial_deadline = time.time() + trial_time
            
            pieces = decompose_graph(edges, trial_deadline, 1000 + t)
            count = len(pieces)
            
            if count < best_count:
                best_count = count
                best_pieces = pieces
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results
# EVOLVE-BLOCK-END
