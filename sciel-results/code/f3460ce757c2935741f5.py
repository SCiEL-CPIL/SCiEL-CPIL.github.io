"""Rewritten program: Local search on a pool of candidate cycles."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycles_randomized(adj, rem_edges, num_cycles, max_len, rng):
    """
    Find a set of random cycles in the graph defined by rem_edges.
    Returns a list of cycles (lists of vertices).
    """
    cycles = []
    # Build adjacency from rem_edges for fast neighbor lookup
    rem_adj = defaultdict(set)
    for u, v in rem_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    for _ in range(num_cycles):
        # Pick a random start vertex with degree > 0
        candidates = [v for v in rem_adj if rem_adj[v]]
        if not candidates:
            break
        start = rng.choice(candidates)
        
        # Random DFS to find a cycle
        path = [start]
        visited = {start}
        current = start
        found = False
        
        # Limit depth to avoid infinite loops or very long paths in dense graphs
        max_depth = max_len if max_len else 20
        for _ in range(max_depth):
            neighbors = list(rem_adj[current])
            if not neighbors:
                break
            # Shuffle neighbors for randomness
            rng.shuffle(neighbors)
            next_node = None
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    # Close cycle
                    found = True
                    break
                if nb not in visited:
                    next_node = nb
                    break
            
            if found:
                break
            if next_node is None:
                break
            
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        if found and len(path) >= 3:
            # Check if this cycle is already in the list (simple check by edge set)
            cycle_edges = set()
            valid = True
            for i in range(len(path)):
                u, v = path[i], path[(i + 1) % len(path)]
                e = (u, v) if u < v else (v, u)
                if e in cycle_edges:
                    valid = False
                    break
                cycle_edges.add(e)
            if valid:
                # Check if edges are in rem_edges
                if all(e in rem_edges for e in cycle_edges):
                    cycles.append(path)
    
    return cycles

def _decompose(edges, deadline, rng):
    """
    Decompose edge set into cycles and single edges, minimizing piece count.
    Strategy: 
    1. Generate a pool of candidate cycles.
    2. Use local search to find a good subset of edge-disjoint cycles.
    3. Remaining edges are single pieces.
    """
    rem = set(edges)
    adj = defaultdict(set)
    for u, v in rem:
        adj[u].add(v)
        adj[v].add(u)
    
    # 1. Generate Candidate Pool
    # We want a diverse set of cycles. 
    # n is small (20-30), so we can generate a few hundred.
    num_candidates = 500
    candidates = []
    
    # Initial batch
    c1 = _find_cycles_randomized(adj, rem, num_candidates // 2, 15, rng)
    candidates.extend(c1)
    
    # Second batch with different max_len or just more randomness
    c2 = _find_cycles_randomized(adj, rem, num_candidates // 2, 25, rng)
    candidates.extend(c2)
    
    # Deduplicate candidates based on edge set
    unique_candidates = []
    seen_edge_sets = set()
    for c in candidates:
        edge_set = frozenset((min(u,v), max(u,v)) for u,v in zip(c, c[1:]+c[:1]))
        if edge_set not in seen_edge_sets:
            seen_edge_sets.add(edge_set)
            unique_candidates.append(c)
    
    candidates = unique_candidates
    
    if not candidates:
        # No cycles found, return all as edges
        return [[u, v] for u, v in edges]
    
    # 2. Local Search
    # State: set of selected cycles
    selected_cycles = []
    used_edges = set()
    current_score = 0 # Sum of (len(c)-1)
    
    def get_score(cycle):
        return len(cycle) - 1
        
    def is_disjoint(cycle, used_edge_set):
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            e = (u, v) if u < v else (v, u)
            if e in used_edge_set:
                return False
        return True
        
    def add_cycle(cycle):
        nonlocal used_edges, current_score, selected_cycles
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            e = (u, v) if u < v else (v, u)
            used_edges.add(e)
        selected_cycles.append(cycle)
        current_score += get_score(cycle)
        
    def remove_cycle(cycle):
        nonlocal used_edges, current_score, selected_cycles
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            e = (u, v) if u < v else (v, u)
            used_edges.discard(e)
        selected_cycles.remove(cycle)
        current_score -= get_score(cycle)
        
    # Greedy Initialization
    # Sort candidates by length descending (longer cycles are better)
    candidates.sort(key=lambda x: -len(x))
    
    for c in candidates:
        if is_disjoint(c, used_edges):
            add_cycle(c)
            
    # Local Search Improvements
    # We try to swap out a selected cycle for a better combination from the pool
    # Or just add any remaining disjoint cycle from the pool
    
    steps = 0
    max_steps = 2000
    no_improve_count = 0
    best_score = current_score
    
    while steps < max_steps and time.time() < deadline - 5:
        steps += 1
        
        # Strategy 1: Try to add a cycle from candidates that is disjoint
        added = False
        for c in candidates:
            if is_disjoint(c, used_edges):
                add_cycle(c)
                added = True
                break
        
        if added:
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
            continue
            
        # Strategy 2: Try to swap a selected cycle with a different one
        # Pick a random selected cycle
        if selected_cycles:
            idx = rng.randint(0, len(selected_cycles) - 1)
            c_old = selected_cycles[idx]
            
            # Remove it
            old_score = current_score
            remove_cycle(c_old)
            
            # Try to find a replacement cycle from candidates that fits
            # and yields a higher score
            best_new_score = current_score
            best_new_cycle = None
            
            # We can also try to find a NEW cycle in the freed space
            # But for now, just look in the candidate pool
            for c_new in candidates:
                if c_new == c_old:
                    continue
                if is_disjoint(c_new, used_edges):
                    score_new = current_score + get_score(c_new)
                    if score_new > best_new_score:
                        best_new_score = score_new
                        best_new_cycle = c_new
            
            if best_new_cycle is not None:
                add_cycle(best_new_cycle)
                if current_score > best_score:
                    best_score = current_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # No improvement, put it back
                add_cycle(c_old)
                no_improve_count += 1
                
        else:
            break
            
        # Early stop if stuck
        if no_improve_count > 100:
            break
            
    # 3. Construct Result
    # The remaining edges are those not in used_edges
    remaining_edges = rem - used_edges
    
    pieces = []
    # Add cycles
    for c in selected_cycles:
        pieces.append(c)
        
    # Add remaining edges as single pieces
    for u, v in remaining_edges:
        pieces.append([u, v])
        
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = _instance(*spec)
        rng = random.Random(42)
        # Allocate time per graph
        time_per_graph = (deadline - time.time()) / (3 - len(results))
        graph_deadline = time.time() + time_per_graph
        
        pieces = _decompose(edges, graph_deadline, rng)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
