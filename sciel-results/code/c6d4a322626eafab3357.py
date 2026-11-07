"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = 75  # safety margin before 90s kill

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycles(S, adj, max_cycles=100):
    """Find Hamiltonian cycles on vertex set S. Returns list of edge-sets."""
    S_set = frozenset(S)
    S_list = list(S)
    k = len(S_list)
    
    if k < 3:
        return []
    
    adj_S = {}
    for v in S_list:
        adj_S[v] = adj[v] & S_set
    
    # Check if subgraph is connected enough
    # Each vertex needs degree >= 2 in the induced subgraph
    for v in S_list:
        if len(adj_S[v]) < 2:
            return []
    
    cycles = []
    start = S_list[0]
    
    # DFS to find all Hamiltonian cycles (with a cap)
    def dfs(path, visited, used_edges):
        if len(cycles) >= max_cycles:
            return
        
        if len(path) == k:
            last = path[-1]
            if start in adj_S[last]:
                edge = frozenset([last, start])
                if edge not in used_edges:
                    cycle_edges = set(used_edges) | {edge}
                    cycles.append((list(path), cycle_edges))
            return
        
        for neighbor in sorted(adj_S[path[-1]]):
            if neighbor not in visited:
                edge = frozenset([path[-1], neighbor])
                if edge not in used_edges:
                    visited.add(neighbor)
                    used_edges.add(edge)
                    dfs(path + [neighbor], visited, used_edges)
                    used_edges.discard(edge)
                    visited.discard(neighbor)
                    if len(cycles) >= max_cycles:
                        return
    
    visited = {start}
    used_edges = set()
    dfs([start], visited, used_edges)
    
    return cycles

def _select_edge_disjoint(cycles_with_edges, max_count=10):
    """Select up to max_count edge-disjoint cycles from the list."""
    selected = []
    used_edges = set()
    
    # Sort by number of edges (all same for Hamiltonian, but just in case)
    # Try to find a maximal set of edge-disjoint cycles
    # Greedy: try each cycle, add if no edge conflict
    for cycle, edges in cycles_with_edges:
        if len(selected) >= max_count:
            break
        if not (edges & used_edges):
            selected.append(cycle)
            used_edges |= edges
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    adj = _build_adjacency()
    n = P['n']
    all_vertices = list(range(n))
    
    best_cycles = None
    best_count = 0
    
    # Strategy: Try vertex sets of size 4, 5, 6, 7, 8
    # For each, find all Hamiltonian cycles, then select edge-disjoint ones
    # Focus on highly connected subgraphs
    
    # Precompute degree for ranking
    degree = [len(adj[v]) for v in range(n)]
    
    # For small sets, try all combinations; for larger, try promising ones
    for s in [4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 10:
            break
        
        max_possible = min(10, 70 // s)
        if max_possible <= best_count:
            continue
        
        # Generate candidate vertex sets
        # Prioritize sets with high minimum degree in induced subgraph
        if s <= 6:
            candidates = list(itertools.combinations(all_vertices, s))
        elif s <= 8:
            candidates = list(itertools.combinations(all_vertices, s))
        else:
            # For larger sets, try subsets of high-degree vertices
            high_deg = sorted(range(n), key=lambda v: -degree[v])
            candidates = [tuple(high_deg[:s])]
            # Add some random ones
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                rand_set = random.sample(all_vertices, s)
                candidates.append(tuple(sorted(rand_set)))
        
        for candidate in candidates:
            if time.time() > deadline - 5:
                break
            
            S = list(candidate)
            
            # Quick check: each vertex must have degree >= 2 in induced subgraph
            S_set = set(S)
            valid = True
            for v in S:
                if len(adj[v] & S_set) < 2:
                    valid = False
                    break
            if not valid:
                continue
            
            # Find Hamiltonian cycles
            h_cycles = _find_hamiltonian_cycles(S, adj, max_cycles=200)
            
            if len(h_cycles) < 2:
                continue
            
            # Select edge-disjoint cycles
            selected = _select_edge_disjoint(h_cycles, max_count=max_possible)
            
            if len(selected) > best_count:
                best_cycles = [list(c) for c in selected]
                best_count = len(selected)
                
                if best_count >= 10:
                    break
        
        if best_count >= 10:
            break
    
    # If we didn't find enough, try a different approach:
    # Use randomized search for edge-disjoint cycles on promising vertex sets
    if best_count < 10:
        # Try to improve by finding more cycles on the best vertex set found
        # Or try different vertex sets with randomized cycle finding
        if time.time() < deadline - 10:
            # Try vertex sets of size 4-8 with randomized DFS
            for s in [4, 5, 6]:
                if time.time() > deadline - 5:
                    break
                
                max_possible = min(10, 70 // s)
                if max_possible <= best_count:
                    continue
                
                # Try many random vertex sets
                for attempt in range(200):
                    if time.time() > deadline - 3:
                        break
                    
                    S = random.sample(all_vertices, s)
                    S_set = set(S)
                    
                    # Quick check
                    valid = True
                    for v in S:
                        if len(adj[v] & S_set) < 2:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Find cycles with randomized DFS
                    h_cycles = _find_hamiltonian_cycles(S, adj, max_cycles=100)
                    if len(h_cycles) < 2:
                        continue
                    
                    selected = _select_edge_disjoint(h_cycles, max_count=max_possible)
                    
                    if len(selected) > best_count:
                        best_cycles = [list(c) for c in selected]
                        best_count = len(selected)
                        
                        if best_count >= 10:
                            break
                
                if best_count >= 10:
                    break
    
    # Ensure we have at least one valid cycle
    if best_cycles is None or len(best_cycles) == 0:
        # Fallback: find any triangle
        edge_set = set(frozenset([u, v]) for u, v in P['edges'])
        for u, v in P['edges']:
            if time.time() > deadline:
                break
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    best_cycles = [[u, v, w]]
                    best_count = 1
                    break
            if best_cycles:
                break
    
    if best_cycles is None or len(best_cycles) == 0:
        return [list(range(P['n']))]
    
    # Validate and convert types
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result

# EVOLVE-BLOCK-END
