import time
import random
from collections import defaultdict

_DEADLINE_SECONDS = 145.0

def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle(adj, start):
    """
    Find a simple cycle starting from 'start' using DFS.
    Returns a list of vertices forming the cycle, or None.
    """
    visited = {start}
    path = [start]
    
    def dfs(u):
        for v in adj[u]:
            if v == start:
                if len(path) >= 3:
                    return path[:]
                continue
            if v not in visited:
                visited.add(v)
                path.append(v)
                result = dfs(v)
                if result:
                    return result
                path.pop()
                visited.discard(v)
        return None
    
    return dfs(start)

def _greedy_decompose(edges, n, rng, strategy='random'):
    """
    Greedily find edge-disjoint cycles.
    strategy: 'random', 'longest', 'shortest'
    """
    if not edges:
        return []
    
    remaining = set((min(u, v), max(u, v)) for u, v in edges)
    cycles = []
    
    while remaining:
        # Build adjacency list for remaining edges
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find a cycle
        best_cycle = None
        best_len = -1 if strategy == 'longest' else float('inf')
        
        # Try to find a cycle
        # To be efficient, we can pick a random start node or iterate
        # For small graphs, iterating all nodes is fine, but we can randomize
        nodes = list(range(n))
        if strategy == 'random':
            rng.shuffle(nodes)
        
        found = False
        for start in nodes:
            if not adj[start]:
                continue
            
            cycle = _find_cycle(adj, start)
            if cycle:
                if strategy == 'longest' and len(cycle) > best_len:
                    best_cycle = cycle
                    best_len = len(cycle)
                elif strategy == 'shortest' and len(cycle) < best_len:
                    best_cycle = cycle
                    best_len = len(cycle)
                elif strategy == 'random':
                    best_cycle = cycle
                    found = True
                    break
        
        if best_cycle:
            cycles.append(best_cycle)
            # Remove edges of the cycle from remaining
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
        else:
            break
            
    return cycles

def _score_cycles(cycles):
    """Calculate the objective score: sum(|c| - 1) for all cycles."""
    return sum(len(c) - 1 for c in cycles)

def _local_search(initial_cycles, edges, n, rng, time_limit):
    """
    Improve the set of cycles using local search.
    Move: Remove one cycle, refill with greedy cycles from the recovered edges.
    """
    current_cycles = list(initial_cycles)
    current_score = _score_cycles(current_cycles)
    start_time = time.time()
    improved = True
    
    # Strategies for refilling
    strategies = ['random', 'longest', 'shortest']
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Shuffle order of cycles to try to avoid getting stuck in a local minimum
        indices = list(range(len(current_cycles)))
        rng.shuffle(indices)
        
        for i in indices:
            if (time.time() - start_time) > time_limit:
                break
            
            c1 = current_cycles[i]
            
            # Remove c1
            temp_cycles = current_cycles[:i] + current_cycles[i+1:]
            
            # Calculate edges covered by temp_cycles
            covered = set()
            for c in temp_cycles:
                for j in range(len(c)):
                    u = c[j]
                    v = c[(j + 1) % len(c)]
                    covered.add((min(u, v), max(u, v)))
            
            # Available edges are all edges NOT in covered
            # Note: The edges of c1 are now available.
            available_edges = set((min(u, v), max(u, v)) for u, v in edges) - covered
            
            if not available_edges:
                continue
                
            # Try to find a better set of cycles in available_edges
            best_new_set = None
            best_new_score = -1
            
            # Try multiple strategies and random restarts for the greedy refill
            for strat in strategies:
                # Run greedy a few times with randomization if strategy is random
                attempts = 3 if strat == 'random' else 1
                for _ in range(attempts):
                    new_cycles = _greedy_decompose(available_edges, n, rng, strategy=strat)
                    new_score = _score_cycles(new_cycles)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_set = new_cycles
                        
            # Calculate total score if we replace c1 with best_new_set
            # Score of temp_cycles + Score of best_new_set
            temp_score = _score_cycles(temp_cycles)
            total_new_score = temp_score + best_new_score
            
            if total_new_score > current_score:
                current_cycles = temp_cycles + best_new_set
                current_score = total_new_score
                improved = True
                # Break to restart the loop with the new configuration
                break
                
    return current_cycles

def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    # Distribute time among the 3 graphs
    time_per_graph = (_DEADLINE_SECONDS - 5) / 3.0
    
    for idx, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        remaining_time = _DEADLINE_SECONDS - elapsed
        if remaining_time < 5:
            # Fallback: just output all edges as single-edge pieces
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        
        # Reserve some time for the next graphs
        # Use 80% of the allocated time for this graph to be safe
        allowed_time = min(time_per_graph * 0.8, remaining_time - 5)
        
        rng = random.Random(42 + idx)
        
        # 1. Initial Greedy Decomposition
        initial_cycles = _greedy_decompose(edges, n, rng, strategy='random')
        
        # 2. Local Search
        if allowed_time > 2:
            improved_cycles = _local_search(initial_cycles, edges, n, rng, allowed_time)
        else:
            improved_cycles = initial_cycles
            
        # 3. Construct Pieces
        pieces = []
        covered_edges = set()
        
        for cycle in improved_cycles:
            # Add cycle as a piece
            # Ensure vertices are distinct (they are, by construction of simple cycle)
            pieces.append(cycle)
            
            # Mark edges as covered
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                covered_edges.add(e)
                
        # Add remaining edges as single-edge pieces
        for u, v in edges:
            e = (min(u, v), max(u, v))
            if e not in covered_edges:
                pieces.append([u, v])
                
        results.append(pieces)
        
    return results

# Verification helper (not used in final output, but good for debugging)
def _validate(pieces, edges, n):
    used_edges = set()
    for piece in pieces:
        if len(piece) == 2:
            u, v = piece
            e = (min(u, v), max(u, v))
            if e in used_edges:
                return False
            used_edges.add(e)
        elif len(piece) >= 3:
            if len(set(piece)) < len(piece):
                return False
            for i in range(len(piece)):
                u = piece[i]
                v = piece[(i + 1) % len(piece)]
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    return False
                used_edges.add(e)
        else:
            return False
            
    all_edges = set((min(u, v), max(u, v)) for u, v in edges)
    return used_edges == all_edges
