"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_hamiltonian_cycle(n, adj, used_edges, start, deadline):
    """Find a Hamiltonian cycle starting and ending at 'start', avoiding used_edges."""
    # used_edges is a set of frozensets {u,v}
    path = [start]
    visited = {start}
    
    def dfs():
        if time.time() > deadline:
            return False
        if len(path) == n:
            # Check if last vertex connects back to start
            last = path[-1]
            edge = frozenset([last, start])
            if edge not in used_edges:
                return True
            return False
        
        current = path[-1]
        # Try neighbors in a deterministic but varied order
        neighbors = list(adj[current])
        # Sort by degree to prune earlier (fewer options first)
        neighbors.sort(key=lambda x: len([nb for nb in adj[x] if frozenset([x, nb]) not in used_edges]))
        
        for nb in neighbors:
            if nb in visited:
                continue
            edge = frozenset([current, nb])
            if edge in used_edges:
                continue
            # Pruning: check if nb still has enough unused edges to reach all remaining
            remaining_unvisited = n - len(path) - 1
            if remaining_unvisited > 0:
                # nb must be able to reach at least remaining_unvisited other unvisited vertices
                available = [x for x in adj[nb] if x not in visited and frozenset([nb, x]) not in used_edges]
                if len(available) < remaining_unvisited:
                    continue
            
            path.append(nb)
            visited.add(nb)
            used_edges.add(edge)
            if dfs():
                return True
            used_edges.discard(edge)
            visited.discard(nb)
            path.pop()
        
        return False
    
    if dfs():
        return list(path)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s/90s kill
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(n, edges)
    
    # Edge set for quick lookup
    all_edges = set()
    for u, v in edges:
        all_edges.add(frozenset([u, v]))
    
    # Try to find up to 2 edge-disjoint Hamiltonian cycles
    # Max possible is floor(62/21) = 2
    
    best_cycles = []
    used_edges = set()
    
    # Strategy: try different starting vertices and orderings
    # First, try to find one Hamiltonian cycle
    # Then try to find a second one avoiding the first's edges
    
    # Try multiple approaches with different heuristics
    found_first = None
    found_second = None
    
    # Attempt 1: Greedy + backtracking with vertex 0 as start
    used_edges_try = set()
    cycle1 = _find_hamiltonian_cycle(n, adj, used_edges_try, 0, deadline)
    if cycle1:
        for i in range(len(cycle1)):
            u = cycle1[i]
            v = cycle1[(i + 1) % len(cycle1)]
            used_edges_try.add(frozenset([u, v]))
        found_first = cycle1
        best_cycles.append(list(cycle1))
    
    # Attempt 2: Find second cycle avoiding first cycle's edges
    if found_first and time.time() < deadline:
        cycle2 = _find_hamiltonian_cycle(n, adj, used_edges_try, 0, deadline)
        if cycle2:
            for i in range(len(cycle2)):
                u = cycle2[i]
                v = cycle2[(i + 1) % len(cycle2)]
                used_edges_try.add(frozenset([u, v]))
            found_second = cycle2
            best_cycles.append(list(cycle2))
    
    # If we didn't find 2 cycles, try alternative first cycles
    if len(best_cycles) < 2 and time.time() < deadline:
        # Try different starting points for the first cycle
        for start_v in range(1, n):
            if time.time() > deadline - 5:
                break
            used_edges_try2 = set()
            cycle1_alt = _find_hamiltonian_cycle(n, adj, used_edges_try2, start_v, deadline)
            if cycle1_alt:
                # Mark its edges
                for i in range(len(cycle1_alt)):
                    u = cycle1_alt[i]
                    v = cycle1_alt[(i + 1) % len(cycle1_alt)]
                    used_edges_try2.add(frozenset([u, v]))
                # Try to find a second cycle
                cycle2_alt = _find_hamiltonian_cycle(n, adj, used_edges_try2, 0, deadline)
                if cycle2_alt:
                    # Verify edge-disjointness
                    edges1 = set()
                    for i in range(len(cycle1_alt)):
                        u = cycle1_alt[i]
                        v = cycle1_alt[(i + 1) % len(cycle1_alt)]
                        edges1.add(frozenset([u, v]))
                    edges2 = set()
                    for i in range(len(cycle2_alt)):
                        u = cycle2_alt[i]
                        v = cycle2_alt[(i + 1) % len(cycle2_alt)]
                        edges2.add(frozenset([u, v]))
                    if edges1.isdisjoint(edges2):
                        best_cycles = [list(cycle1_alt), list(cycle2_alt)]
                        break
    
    # Validate and return
    if not best_cycles:
        # Fallback: return a single valid Hamiltonian cycle if we have one
        # Try one more time with a simpler approach
        used_edges_fb = set()
        c = _find_hamiltonian_cycle(n, adj, used_edges_fb, 0, deadline)
        if c:
            return [list(c)]
        return []
    
    # Final validation
    for cyc in best_cycles:
        assert len(cyc) == n, f"Cycle length {len(cyc)} != {n}"
        assert len(set(cyc)) == n, "Cycle has repeated vertices"
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            assert frozenset([u, v]) in all_edges, f"Edge {u}-{v} not in graph"
    
    # Check pairwise edge-disjointness
    for i in range(len(best_cycles)):
        for j in range(i + 1, len(best_cycles)):
            ei = set()
            for k in range(len(best_cycles[i])):
                u = best_cycles[i][k]
                v = best_cycles[i][(k + 1) % len(best_cycles[i])]
                ei.add(frozenset([u, v]))
            ej = set()
            for k in range(len(best_cycles[j])):
                u = best_cycles[j][k]
                v = best_cycles[j][(k + 1) % len(best_cycles[j])]
                ej.add(frozenset([u, v]))
            assert ei.isdisjoint(ej), f"Cycles {i} and {j} share an edge"
    
    return best_cycles
# EVOLVE-BLOCK-END
