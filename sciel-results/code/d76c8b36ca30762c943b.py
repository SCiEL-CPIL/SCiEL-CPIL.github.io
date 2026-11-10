"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import time
import sys
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build adjacency structure
    edge_map = {}
    for i, (u, v) in enumerate(edges):
        edge_map[frozenset((u, v))] = i
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i
        
    adj = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute: for each edge, the edge index
    def get_edge_idx(u, v):
        return edge_map[(u, v)]
    
    def find_hamiltonian_cycle(used_edges, seed_vertex=0):
        """Find a Hamiltonian cycle in the remaining graph (edges not in used_edges)."""
        # Build adjacency of available edges
        avail_adj = defaultdict(set)
        for i, (u, v) in enumerate(edges):
            if i not in used_edges:
                avail_adj[u].add(v)
                avail_adj[v].add(u)
        
        # Check degree 2 for all vertices
        for v in range(n):
            if len(avail_adj[v]) != 2:
                return None
        
        # The graph is now 2-regular. Find the cycle(s).
        # Start from seed_vertex
        visited = set()
        path = []
        curr = seed_vertex
        prev = -1
        while True:
            visited.add(curr)
            path.append(curr)
            if len(path) == n:
                # Check if we can close the cycle
                if seed_vertex in avail_adj[curr]:
                    return path
                else:
                    return None
            # Choose next vertex
            nexts = list(avail_adj[curr])
            next_node = None
            for nb in nexts:
                if nb not in visited and nb != prev:
                    next_node = nb
                    break
            if next_node is None:
                # Only one unvisited neighbor left?
                for nb in nexts:
                    if nb not in visited:
                        next_node = nb
                        break
            if next_node is None:
                return None
            prev = curr
            curr = next_node
        
        return None
    
    def try_find_k_cycles(k, random_seed=0):
        """Try to find k edge-disjoint Hamiltonian cycles using backtracking with randomization."""
        rng = random.Random(random_seed)
        used_edges = set()
        cycles = []
        
        # Order edges for potential backtracking - we'll use a simple approach:
        # Greedily find cycles, and if stuck, backtrack.
        
        def backtrack(cycles_so_far, used_so_far):
            if time.time() > deadline:
                return None
            if len(cycles_so_far) == k:
                return cycles_so_far
            
            # Find a Hamiltonian cycle in the remaining graph
            # To make this work, we need the remaining graph to be 2-regular
            # OR we need to search for any Hamiltonian cycle.
            
            # Actually, for the remaining graph to contain a Hamiltonian cycle,
            # it doesn't need to be 2-regular. We need to search for one.
            
            # Let's use a DFS-based Hamiltonian cycle finder with pruning.
            # This is expensive, so let's first check if the graph is "close" to 2-regular.
            
            # Build available adjacency
            avail_adj = defaultdict(set)
            for i, (u, v) in enumerate(edges):
                if i not in used_so_far:
                    avail_adj[u].add(v)
                    avail_adj[v].add(u)
            
            # Check if any vertex has degree < 2
            for v in range(n):
                if len(avail_adj[v]) < 2:
                    return None
            
            # If all vertices have degree exactly 2, the graph is a union of cycles.
            # Check if it's a single Hamiltonian cycle.
            all_deg2 = all(len(avail_adj[v]) == 2 for v in range(n))
            if all_deg2:
                # Find the cycle starting from 0
                path = [0]
                visited = {0}
                curr = 0
                while True:
                    nexts = list(avail_adj[curr])
                    next_node = None
                    for nb in nexts:
                        if nb not in visited:
                            next_node = nb
                            break
                    if next_node is None:
                        if len(path) == n and 0 in avail_adj[curr]:
                            return cycles_so_far + [path]
                        else:
                            return None
                    visited.add(next_node)
                    path.append(next_node)
                    curr = next_node
            
            # If not all degree 2, we need to search for a Hamiltonian cycle.
            # This is NP-hard, so we use a heuristic DFS.
            # For efficiency, let's try a few random orderings.
            
            # Actually, let's just do a direct DFS for Hamiltonian cycle.
            # We fix the first vertex as 0 to break symmetry.
            
            def dfs_ham(path, visited, curr):
                if time.time() > deadline:
                    return None
                if len(path) == n:
                    if 0 in avail_adj[curr]:
                        return path
                    return None
                # Try neighbors in some order
                # Heuristic: prefer neighbors that lead to dead ends later
                nexts = list(avail_adj[curr])
                # Sort by degree (lower degree first) for better pruning
                nexts.sort(key=lambda x: len(avail_adj[x]))
                for nb in nexts:
                    if nb not in visited and nb != 0:  # Don't visit 0 until the end
                        visited.add(nb)
                        path.append(nb)
                        result = dfs_ham(path, visited, nb)
                        if result is not None:
                            return result
                        path.pop()
                        visited.remove(nb)
                return None
            
            result = dfs_ham([0], {0}, 0)
            if result is None:
                return None
            
            # Add this cycle
            new_used = set(used_so_far)
            for i in range(len(result)):
                u = result[i]
                v = result[(i + 1) % len(result)]
                new_used.add(get_edge_idx(u, v))
            
            return backtrack(cycles_so_far + [result], new_used)
        
        return backtrack([], used_edges)
    
    # First, try to find 2 cycles (since graph is 5-regular, max 2 disjoint Ham cycles)
    # Wait, the task says "Maximize number of cycles" and "At most 10 cycles may be listed."
    # But the graph is 5-regular on 21 vertices. Each Hamiltonian cycle uses 21 edges.
    # Total edges = 55 (since 21*5/2 = 52.5? No, 21*5=105/2=52.5, so it's not regular?)
    
    # Let me recount edges.
    # From the list: 
    # 0: 1,2,5,16,19,20 -> degree 6
    # So it's not 5-regular. Let me check the degree of each vertex.
    
    degrees = [0] * n
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    
    # Max cycles is limited by min_degree / 2 (since each cycle uses 2 edges per vertex)
    # and by m // n (since each cycle uses n edges)
    
    max_possible_by_degree = min(degrees) // 2
    max_possible_by_edges = m // n
    
    max_cycles = min(max_possible_by_degree, max_possible_by_edges, 10)
    
    # Try to find as many cycles as possible
    # Start with the maximum and work down
    
    # For a 6-regular graph, we can have up to 3 Hamiltonian cycles.
    # Let's try to find 3, then 2, then 1.
    
    best_result = None
    
    # Try with different random seeds for robustness
    for k in range(min(max_cycles, 3), 0, -1):
        for seed in range(20):
            if time.time() > deadline:
                break
            result = try_find_k_cycles(k, seed)
            if result is not None:
                best_result = result
                break
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback: try to find at least 1 cycle
        for seed in range(50):
            if time.time() > deadline:
                break
            result = try_find_k_cycles(1, seed)
            if result is not None:
                best_result = result
                break
    
    if best_result is None:
        # Last resort: return empty or a dummy
        # But we must return a nonempty list.
        # Let's just return a simple cycle if we can find one manually.
        # For now, return an empty list to indicate failure, but the task says nonempty.
        # We need to ensure we return something valid.
        
        # Let's try a very simple DFS one more time
        avail_adj = defaultdict(set)
        for u, v in edges:
            avail_adj[u].add(v)
            avail_adj[v].add(u)
        
        def dfs_simple(path, visited, curr):
            if len(path) == n:
                if 0 in avail_adj[curr]:
                    return path
                return None
            for nb in sorted(avail_adj[curr], key=lambda x: len(avail_adj[x])):
                if nb not in visited and nb != 0:
                    visited.add(nb)
                    path.append(nb)
                    result = dfs_simple(path, visited, nb)
                    if result is not None:
                        return result
                    path.pop()
                    visited.remove(nb)
            return None
        
        result = dfs_simple([0], {0}, 0)
        if result is not None:
            best_result = [result]
    
    if best_result is None:
        # This should not happen, but we need a fallback
        # Return a dummy cycle (will be invalid, but better than exception)
        return list(range(n))
    
    # Convert to correct types
    result = [list(c) for c in best_result]
    for c in result:
        for v in c:
            if not isinstance(v, int) or isinstance(v, bool):
                v = int(v)
    
    if not result:
        return list(range(n))
        
    return result
# EVOLVE-BLOCK-END
