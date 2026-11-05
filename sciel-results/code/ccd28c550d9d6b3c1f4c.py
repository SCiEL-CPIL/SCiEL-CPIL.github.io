"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Helper to check if a graph contains a cycle of length L
    # Uses DFS with pruning. For N=40, this is feasible for a limited number of calls.
    def has_cycle_of_length(adj, L):
        if L < 3 or L > N:
            return False
        if L == 3:
            # Triangle check
            for u in range(N):
                neighbors = adj[u]
                for i, v in enumerate(neighbors):
                    if i == 0: continue
                    for w in neighbors[i+1:]:
                        if w in neighbors:
                            return True
            return False
        
        # General cycle check
        # We fix a start node 0 to avoid symmetry duplicates in search, 
        # but for checking existence, we can just try all starts if needed.
        # However, for speed, we try to find any cycle.
        
        # Simple DFS
        # State: (current_node, path_list, visited_set)
        # To speed up, we only explore paths that can potentially close to length L.
        
        # Optimization: If L == N, it's Hamiltonian.
        # For L < N, we can stop if remaining nodes < L - len(path).
        
        found = [False]
        
        def dfs(u, path, visited):
            if found[0]:
                return
            path_len = len(path)
            if path_len == L:
                # Check if closes to start
                start = path[0]
                if start in adj[u]:
                    found[0] = True
                    return
            
            if path_len >= L:
                return
            
            # Pruning: if we can't reach length L with remaining unvisited nodes
            # (rough estimate)
            
            for v in adj[u]:
                if v == path[0] and path_len < L - 1:
                    continue # Can't close early
                if v in visited:
                    continue
                
                # Symmetry breaking: if we are at depth 0, we can restrict neighbors?
                # Not strictly necessary for correctness, but helps speed.
                
                visited.add(v)
                path.append(v)
                dfs(v, path, visited)
                path.pop()
                visited.remove(v)
                
                if found[0]:
                    return

        # Try starting from each node? Or just 0?
        # A graph might have a cycle not involving 0.
        # But for sparse graphs constructed via local search, cycles are usually global.
        # To be safe, we check from all nodes, but we can short-circuit.
        
        # Actually, for the specific structures we generate (connected, dense enough),
        # checking from node 0 is often sufficient for small L, but not for large L.
        # Let's check from all nodes but stop early.
        
        for start in range(N):
            # We don't need to try all starts if we found one.
            # But we need to ensure we don't miss a cycle that doesn't touch 0.
            # However, in a connected graph with enough edges, cycles likely touch most nodes.
            # Let's just run DFS from 0 first. If not found, try others?
            # For L close to N, the cycle must touch most nodes.
            
            # Let's just run from 0 for speed in the inner loop.
            # If the graph is disconnected, it's invalid anyway.
            # If the graph is connected, a cycle of length L might not touch 0.
            # Example: Two triangles connected by a bridge. Cycle length 3 doesn't touch 0 if 0 is on the bridge.
            # So we must check all starts or use a more global method.
            
            # Given the time limit, we will check from all starts but with a strict time guard?
            # No, DFS is fast enough for N=40 if the graph is sparse.
            
            visited = {start}
            path = [start]
            dfs(start, path, visited)
            if found[0]:
                return True
            
            # Optimization: If we didn't find it starting at 0, try 1, etc?
            # This is expensive. 
            # Alternative: Use a randomized walk?
            # Let's stick to checking from 0 first. If it fails, we might have a false negative.
            # To be robust, we check from a few representative nodes: 0, N//2, N-1.
            
        # If we are very strict, we check all. But for local search, we assume connectivity.
        # Let's add a check for a few other starts if 0 fails.
        for start in [N//2, N-1, 1, 2]:
            if start == 0: continue
            visited = {start}
            path = [start]
            dfs(start, path, visited)
            if found[0]:
                return True
                
        return False

    def is_pancyclic(edges):
        if len(edges) < N:
            return False
        adj = [set() for _ in range(N)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check connectivity (simple BFS)
        visited = [False] * N
        stack = [0]
        visited[0] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if not all(visited):
            return False

        for L in range(3, N + 1):
            if not has_cycle_of_length(adj, L):
                return False
        return True

    def get_cycles_for_edges(edges):
        """
        Returns a list of cycles for lengths 3..40.
        This is expensive, so we only call it on the final best candidate.
        """
        adj = [set() for _ in range(N)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = []
        for L in range(3, N + 1):
            found_cycle = None
            
            # Re-use the DFS logic but return the path
            def find_cycle(adj, L):
                for start in range(N):
                    visited = {start}
                    path = [start]
                    
                    def dfs(u):
                        if len(path) == L:
                            if start in adj[u]:
                                return path[:]
                            return None
                        if len(path) >= L:
                            return None
                        for v in adj[u]:
                            if v == start and len(path) < L - 1:
                                continue
                            if v in visited:
                                continue
                            visited.add(v)
                            path.append(v)
                            res = dfs(v)
                            if res:
                                return res
                            path.pop()
                            visited.remove(v)
                        return None
                    res = dfs(start)
                    if res:
                        return res
                return None

            found_cycle = find_cycle(adj, L)
            if found_cycle:
                cycles.append(found_cycle)
            else:
                # Should not happen if is_pancyclic passed
                cycles.append([])
        return cycles

    # Strategy:
    # 1. Generate a good initial candidate.
    # 2. Local search: Try to remove edges.
    
    # Initial Candidate 1: The Seed (Fan)
    # Edges: (0, i) for i=1..39, (i, i+1) for i=1..38
    seed_edges = set([(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)])
    
    # Initial Candidate 2: Symmetric Ladder
    # Two hubs: 0 and 20.
    # Backbone: 0-1-...-39 (cycle? No, let's make it a cycle 0-1-...-39-0)
    # If we use a Hamiltonian cycle as base (40 edges), we need 38 more cycle lengths.
    # Chords from 0 to even nodes?
    
    # Let's try a "Double Fan" with shared backbone.
    # Vertices 0..39.
    # Base: Cycle (0, 1, ..., 39, 0). Edges: 40.
    # Add chords to create shorter cycles.
    # Chord (0, k) creates cycles of length k+1 and 40-k+1.
    # If we add (0, 2), (0, 4), ..., (0, 20)?
    # (0, 2) -> len 3, 39
    # (0, 4) -> len 5, 37
    # ...
    # This covers odd lengths and 40. Even lengths 4..38 are missing.
    # Add (1, 3), (1, 5)...?
    
    # Let's just start with the seed and try to improve it via local search.
    # The seed has 77 edges.
    
    best_edges = list(seed_edges)
    best_score = len(best_edges) - N
    
    # Local Search Loop
    # We will try to remove random edges and check if the graph remains pancyclic.
    # If yes, we keep the removal (improving score).
    # If no, we revert.
    
    # To speed up, we can check only the "critical" lengths first.
    # But for now, full check.
    
    # Optimization: 
    # Instead of full is_pancyclic every time, we can track which lengths are "fragile".
    # But that's complex. Let's rely on the speed of the DFS for N=40.
    
    current_edges = list(seed_edges)
    
    # Sort edges to remove: try removing "redundant" ones first?
    # Edges in the backbone (i, i+1) are critical for long cycles.
    # Chords (0, i) are critical for specific short/long cycles.
    # Let's just shuffle.
    
    random.seed(42)
    
    max_iterations = 1000
    for i in range(max_iterations):
        if time.time() > deadline:
            break
        
        if len(current_edges) <= N:
            break
            
        # Pick a random edge to try removing
        edge_idx = random.randint(0, len(current_edges) - 1)
        edge_to_remove = current_edges[edge_idx]
        
        # Tentatively remove
        new_edges = current_edges[:]
        new_edges.pop(edge_idx)
        
        # Check validity
        # To save time, we can do a quick check:
        # Does the graph still have enough edges? Yes.
        # Is it connected?
        adj = [set() for _ in range(N)]
        for u, v in new_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Quick connectivity check
        visited = [False] * N
        stack = [0]
        visited[0] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if not all(visited):
            continue
            
        # Full pancyclic check
        # We can optimize has_cycle_of_length by passing adj directly
        valid = True
        for L in range(3, N + 1):
            if not has_cycle_of_length(adj, L):
                valid = False
                break
                
        if valid:
            current_edges = new_edges
            best_edges = current_edges
            best_score = len(current_edges) - N
            
            # If we improved, we might want to restart the random order or just continue
            # Continuing is fine.
            
    # If we found a better graph, we need to generate the cycles for it.
    # If not, we return the seed cycles.
    
    # Check if current_edges is the seed
    if set(current_edges) == seed_edges:
        # Return seed result
        return {
            'edges': list(seed_edges),
            'cycles': [list(range(k)) for k in range(3, N+1)]
        }
    else:
        # Generate cycles for the new graph
        cycles = get_cycles_for_edges(current_edges)
        # Verify cycles are valid
        for i, cyc in enumerate(cycles):
            L = i + 3
            if len(cyc) != L:
                # Fallback to seed if generation failed
                return {
                    'edges': list(seed_edges),
                    'cycles': [list(range(k)) for k in range(3, N+1)]
                }
            # Check edges in cycle exist
            for j in range(L):
                u = cyc[j]
                v = cyc[(j+1)%L]
                if (u, v) not in set(current_edges) and (v, u) not in set(current_edges):
                     return {
                         'edges': list(seed_edges),
                         'cycles': [list(range(k)) for k in range(3, N+1)]
                     }
        
        return {
            'edges': current_edges,
            'cycles': cycles
        }

# EVOLVE-BLOCK-END
