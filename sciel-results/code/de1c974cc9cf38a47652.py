"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
ALL_VERTS = set(range(N))

# Build adjacency
adj = [set() for _ in range(N)]
edge_set = set()
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)
    edge_set.add((min(u, v), max(u, v)))

# Precompute edge keys for quick lookup
def edge_key(u, v):
    return (min(u, v), max(u, v))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # We'll use a backtracking search to find edge-disjoint Hamiltonian cycles.
    # A Hamiltonian cycle on 21 vertices uses 21 edges.
    # Total edges = 72. Max possible cycles = 72 // 21 = 3.
    
    # Strategy: Find 3 edge-disjoint Hamiltonian cycles using backtracking.
    # To make this tractable, we use constraint propagation and heuristics.
    
    # Represent used edges as a set of edge keys
    used_edges = set()
    cycles = []
    
    # For efficiency, we'll try to find cycles in a specific order.
    # We can use the fact that the graph is quite dense.
    
    # Let's use a smarter approach: 
    # 1. Find a Hamiltonian cycle
    # 2. Remove its edges
    # 3. Find another Hamiltonian cycle in the remaining graph
    # 4. Remove its edges
    # 5. Find another Hamiltonian cycle in the remaining graph
    
    def find_ham_cycle(remaining_edges, deadline):
        """Find a Hamiltonian cycle using the remaining edges."""
        # Build adjacency from remaining edges
        rem_adj = [set() for _ in range(N)]
        for (u, v) in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Check if every vertex has degree >= 2 in remaining graph
        for i in range(N):
            if len(rem_adj[i]) < 2:
                return None
        
        # Use backtracking with MRV heuristic
        # We fix vertex 0 as the start to break symmetry
        # A Hamiltonian cycle is a path from 0 visiting all vertices and returning to 0
        
        # State: current path, used vertices, used edges in this cycle
        path = [0]
        used_verts = {0}
        used_cycle_edges = set()
        
        # For the return edge, we need edge (last, 0)
        
        def backtrack(path, used_verts, used_cycle_edges, deadline):
            if time.time() > deadline:
                return None
            
            if len(path) == N:
                # Check if we can close the cycle: edge from last to 0
                last = path[-1]
                e = edge_key(last, 0)
                if e in remaining_edges and e not in used_cycle_edges:
                    return list(path)
                return None
            
            # MRV: choose next vertex with fewest options
            last = path[-1]
            candidates = []
            for v in rem_adj[last]:
                if v not in used_verts:
                    e = edge_key(last, v)
                    if e in remaining_edges and e not in used_cycle_edges:
                        # Count options for v (how many unvisited neighbors it has)
                        opts = 0
                        for w in rem_adj[v]:
                            if w not in used_verts:
                                ew = edge_key(v, w)
                                if ew in remaining_edges and ew not in used_cycle_edges:
                                    opts += 1
                        # Also need to be able to eventually return to 0
                        candidates.append((opts, v))
            
            if not candidates:
                return None
            
            # Sort by fewest options first (MRV)
            candidates.sort()
            
            for _, v in candidates:
                e = edge_key(last, v)
                new_path = path + [v]
                new_used = used_verts | {v}
                new_edges = used_cycle_edges | {e}
                
                result = backtrack(new_path, new_used, new_edges, deadline)
                if result is not None:
                    return result
            
            return None
        
        return backtrack(path, used_verts, used_cycle_edges, deadline)
    
    # Try to find 3 cycles
    remaining = set(edge_set)
    found_cycles = []
    
    for cycle_idx in range(3):
        if time.time() > deadline:
            break
        
        cycle = find_ham_cycle(remaining, deadline)
        if cycle is None:
            break
        
        # Convert cycle to edge set
        cycle_edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            cycle_edges.add(edge_key(u, v))
        
        # Verify all edges are in remaining
        if not cycle_edges.issubset(remaining):
            break
        
        remaining -= cycle_edges
        found_cycles.append(cycle)
    
    # If we couldn't find 3, try alternative strategies
    if len(found_cycles) < 3:
        # Try different starting points or orders
        # For now, let's try a few different heuristics
        
        # Alternative: try fixing different vertices as start
        for start_vertex in range(1, 5):
            if time.time() > deadline:
                break
            if len(found_cycles) >= 3:
                break
            
            # Reset and try again with different start
            remaining2 = set(edge_set)
            found2 = []
            
            for cycle_idx in range(3):
                if time.time() > deadline:
                    break
                
                # Find cycle starting from start_vertex
                rem_adj = [set() for _ in range(N)]
                for (u, v) in remaining2:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)
                
                # Quick check
                valid = True
                for i in range(N):
                    if len(rem_adj[i]) < 2:
                        valid = False
                        break
                if not valid:
                    break
                
                path = [start_vertex]
                used_verts = {start_vertex}
                used_cycle_edges = set()
                
                def backtrack2(path, used_verts, used_cycle_edges, deadline):
                    if time.time() > deadline:
                        return None
                    
                    if len(path) == N:
                        last = path[-1]
                        e = edge_key(last, start_vertex)
                        if e in remaining2 and e not in used_cycle_edges:
                            return list(path)
                        return None
                    
                    last = path[-1]
                    candidates = []
                    for v in rem_adj[last]:
                        if v not in used_verts:
                            e = edge_key(last, v)
                            if e in remaining2 and e not in used_cycle_edges:
                                opts = 0
                                for w in rem_adj[v]:
                                    if w not in used_verts:
                                        ew = edge_key(v, w)
                                        if ew in remaining2 and ew not in used_cycle_edges:
                                            opts += 1
                                candidates.append((opts, v))
                    
                    if not candidates:
                        return None
                    
                    candidates.sort()
                    
                    for _, v in candidates:
                        e = edge_key(last, v)
                        new_path = path + [v]
                        new_used = used_verts | {v}
                        new_edges = used_cycle_edges | {e}
                        
                        result = backtrack2(new_path, new_used, new_edges, deadline)
                        if result is not None:
                            return result
                    
                    return None
                
                cycle = backtrack2(path, used_verts, used_cycle_edges, deadline)
                if cycle is None:
                    break
                
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    cycle_edges.add(edge_key(u, v))
                
                if not cycle_edges.issubset(remaining2):
                    break
                
                remaining2 -= cycle_edges
                found2.append(cycle)
            
            if len(found2) > len(found_cycles):
                found_cycles = found2
    
    # Ensure we return a valid result
    if not found_cycles:
        # Fallback: return a single known cycle if we can find one
        remaining = set(edge_set)
        cycle = find_ham_cycle(remaining, deadline)
        if cycle is not None:
            return [cycle]
        else:
            # Last resort: try to construct any valid cycle
            # The seed program returns list(range(N)), which is a Hamiltonian cycle
            # if consecutive vertices are adjacent. Let's check.
            seed_cycle = list(range(N))
            valid = True
            for i in range(N):
                u = seed_cycle[i]
                v = seed_cycle[(i + 1) % N]
                if edge_key(u, v) not in edge_set:
                    valid = False
                    break
            if valid:
                return [seed_cycle]
            else:
                # Find any Hamiltonian cycle
                rem_adj = [set() for _ in range(N)]
                for (u, v) in edge_set:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)
                
                path = [0]
                used_verts = {0}
                
                def backtrack3(path, used_verts, deadline):
                    if time.time() > deadline:
                        return None
                    if len(path) == N:
                        last = path[-1]
                        if edge_key(last, 0) in edge_set:
                            return list(path)
                        return None
                    last = path[-1]
                    for v in sorted(rem_adj[last]):
                        if v not in used_verts:
                            result = backtrack3(path + [v], used_verts | {v}, deadline)
                            if result is not None:
                                return result
                    return None
                
                cycle = backtrack3(path, used_verts, deadline)
                if cycle is not None:
                    return [cycle]
                else:
                    # Absolute last resort
                    return [list(range(N))]
    
    return found_cycles

# EVOLVE-BLOCK-END
