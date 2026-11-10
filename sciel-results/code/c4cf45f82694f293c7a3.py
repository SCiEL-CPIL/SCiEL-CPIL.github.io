"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
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
    
    def decompose_graph(edges, time_limit):
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
        
        def commit_cycle(cycle_vertices):
            for i in range(len(cycle_vertices)):
                u = cycle_vertices[i]
                v = cycle_vertices[(i + 1) % len(cycle_vertices)]
                ek = edge_key(u, v)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle_vertices[:])
        
        def commit_single_edge(u, v):
            ek = edge_key(u, v)
            remaining.discard(ek)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
        
        # Repeatedly find and remove cycles
        while remaining:
            if time.time() > time_limit:
                break
            
            # Find vertices with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                # No cycles left, commit all remaining as singles
                for u, v in list(remaining):
                    commit_single_edge(u, v)
                break
            
            # Try to find a long cycle starting from a random candidate
            best_cycle = None
            best_len = 0
            
            # Shuffle candidates to randomize search
            random.shuffle(candidates)
            
            # Limit the number of attempts to avoid getting stuck
            attempts = 0
            max_attempts = 10
            
            for start in candidates:
                if time.time() > time_limit:
                    break
                if attempts >= max_attempts:
                    break
                attempts += 1
                
                # Randomized DFS to find a cycle
                path = [start]
                visited = {start}
                
                def dfs(current, depth):
                    if time.time() > time_limit:
                        return None
                    if depth >= 30: # Max depth
                        return None
                    
                    # Get neighbors
                    neighbors = list(adj[current])
                    random.shuffle(neighbors)
                    
                    for nb in neighbors:
                        if nb == start and len(path) >= 3:
                            return path[:]
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            result = dfs(nb, depth + 1)
                            if result is not None:
                                return result
                            path.pop()
                            visited.discard(nb)
                    return None
                
                cycle = dfs(start, 0)
                if cycle is not None:
                    if len(cycle) > best_len:
                        best_cycle = cycle
                        best_len = len(cycle)
                        # If we found a very long cycle, stop early
                        if best_len >= 10:
                            break
            
            if best_cycle:
                commit_cycle(best_cycle)
            else:
                # If we couldn't find a cycle despite deg >= 2, 
                # it might be a bug or time issue. 
                # Fallback: take a single edge to ensure progress.
                # This shouldn't happen if the graph has a cycle.
                if remaining:
                    u, v = list(remaining)[0]
                    commit_single_edge(u, v)
        
        # Any remaining edges become single-edge pieces
        for u, v in list(remaining):
            commit_single_edge(u, v)
            
        return pieces

    # Main: decompose each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Not enough time, use trivial solution
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        
        # Try multiple random seeds and keep the best
        best_pieces = None
        best_count = float('inf')
        
        # Allocate time for this instance
        time_per_instance = min(40, deadline - time.time() - 10)
        if time_per_instance < 5:
            time_per_instance = 5
            
        # Run multiple trials with different random seeds
        num_trials = 20
        trial_time = time_per_instance / num_trials
        
        for t in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Set a specific seed for reproducibility within the trial
            random.seed(1000 + t)
            
            # Time limit for this specific trial
            trial_deadline = time.time() + trial_time
            
            pieces = decompose_graph(edges, trial_deadline)
            count = len(pieces)
            
            if count < best_count:
                best_count = count
                best_pieces = pieces
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results
# EVOLVE-BLOCK-END
