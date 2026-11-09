"""Triangle-free diameter-four augmentation via optimized search."""
# EVOLVE-BLOCK-START

import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = 24
BASE_EDGE_SET = set((min(u, v), max(u, v)) for u, v in P['edges'])
ALL_E_O_EDGES = []
for u in range(0, N, 2):
    for v in range(1, N, 2):
        key = (min(u, v), max(u, v))
        if key not in BASE_EDGE_SET:
            ALL_E_O_EDGES.append((u, v))

def _build_adj_bits(added_edges):
    """Build adjacency as list of integers (bitsets)."""
    adj = [0] * N
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    for u, v in added_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _diameter_bits(adj_bits):
    """Compute diameter using bitset BFS. Returns -1 if disconnected."""
    max_d = 0
    for start in range(N):
        dist = [-1] * N
        dist[start] = 0
        visited = 1 << start
        frontier = 1 << start
        d = 0
        while frontier:
            d += 1
            # Expand frontier: neighbors of frontier not in visited
            # For each bit in frontier, get its neighbors
            new_frontier = 0
            f = frontier
            while f:
                # Extract lowest set bit
                lsb = f & (-f)
                idx = lsb.bit_length() - 1
                new_frontier |= adj_bits[idx]
                f ^= lsb
            new_frontier &= ~visited
            if not new_frontier:
                break
            # Update distances for new_frontier
            nf = new_frontier
            while nf:
                lsb = nf & (-nf)
                idx = lsb.bit_length() - 1
                dist[idx] = d
                nf ^= lsb
            
            visited |= new_frontier
            frontier = new_frontier
            
        # Check connectivity
        if visited != (1 << N) - 1:
            return -1
        
        # Update max distance
        for d_val in dist:
            if d_val > max_d:
                max_d = d_val
                if max_d > 4:
                    return -1
    return max_d

def _validate(added_edges):
    """Check if added edges result in valid graph."""
    # Triangle-free is guaranteed by bipartite nature (E-O edges only)
    # Just check diameter and connectivity
    adj_bits = _build_adj_bits(added_edges)
    diam = _diameter_bits(adj_bits)
    return diam != -1

def construct():
    deadline = time.time() + 50  # Safety margin
    
    best_edges = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                  [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    best_count = 10
    
    # Pre-compute candidate edges list for swapping
    candidates = ALL_E_O_EDGES
    
    # Local Search Loop
    while time.time() < deadline:
        current_edges = best_edges[:]
        current_set = set((min(u, v), max(u, v)) for u, v in current_edges)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to remove an edge
            for i in range(len(current_edges)):
                if time.time() > deadline:
                    break
                trial_edges = current_edges[:i] + current_edges[i+1:]
                if _validate(trial_edges):
                    current_edges = trial_edges
                    current_set.remove((min(current_edges[i][0], current_edges[i][1]), max(current_edges[i][0], current_edges[i][1])))
                    improved = True
                    break
            
            if not improved:
                # Phase 2: Try to swap an edge
                # Pick a random edge to remove
                if current_edges:
                    i_remove = random.randint(0, len(current_edges) - 1)
                    edge_to_remove = current_edges[i_remove]
                    key_remove = (min(edge_to_remove[0], edge_to_remove[1]), max(edge_to_remove[0], edge_to_remove[1]))
                    
                    # Try adding a random candidate
                    # We want to explore the space. 
                    # Pick a candidate not in current_set
                    available = [e for e in candidates if (min(e[0], e[1]), max(e[0], e[1])) not in current_set]
                    if available:
                        # Try a few random swaps
                        for _ in range(10):
                            if time.time() > deadline:
                                break
                            e_add = random.choice(available)
                            key_add = (min(e_add[0], e_add[1]), max(e_add[0], e_add[1]))
                            
                            trial_edges = current_edges[:i_remove] + current_edges[i_remove+1:] + [list(e_add)]
                            if _validate(trial_edges):
                                current_edges = trial_edges
                                current_set.remove(key_remove)
                                current_set.add(key_add)
                                improved = True
                                break
        
        # Update best
        if len(current_edges) < best_count:
            best_edges = current_edges
            best_count = len(current_edges)
            
        # If we found a very good solution, we might want to restart from scratch with fewer edges?
        # Or just continue local search.
        
    # Final validation
    if not _validate(best_edges):
        # Fallback
        best_edges = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                      [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]

    return [[int(u), int(v)] for u, v in best_edges]

# EVOLVE-BLOCK-END
