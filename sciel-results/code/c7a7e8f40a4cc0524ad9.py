"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 40
DEADLINE = 80.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a sparse pancyclic graph on 40 vertices.
    Strategy: 
    1. Start with a 40-cycle (40 edges).
    2. Use a fixed set of chords (additional edges) to generate cycles of all lengths 3..39.
    3. The chords are chosen from a set that allows efficient verification of pancyclicity.
    4. If the fixed set fails verification (which is extremely unlikely for the known good set), 
       fall back to a greedy search or a larger set.
    
    The seed program used a fan graph with ~77 edges.
    This program uses 40 (cycle) + K (chords) edges.
    We aim for K ~ 5-8, giving total edges ~ 45-48.
    Score = SeedEdges / CurrentEdges. If Seed=77 and Current=45, Score=1.71.
    """
    start_time = time.time()
    
    # Base cycle: 0-1-2-...-39-0
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    base_edges_set = set(frozenset(e) for e in base_edges)
    
    # We need to find a set of chords such that the graph is pancyclic.
    # A known good set of chords for N=40:
    # Chords connecting 0 to other vertices.
    # Let's try a set of chords from a specific construction.
    # For N=40, chords (0, 13), (0, 21), (0, 29) might not be enough.
    # Let's use a more robust set.
    
    # Heuristic: Use chords that are relatively prime to N and spread out.
    # Chords: (0, 7), (0, 13), (0, 21), (0, 27), (0, 33)
    # Let's verify if this set works.
    
    chord_candidates = [
        [(0, 7), (0, 13), (0, 21), (0, 27), (0, 33)],
        [(0, 5), (0, 11), (0, 17), (0, 23), (0, 29), (0, 35)],
        [(0, 3), (0, 11), (0, 19), (0, 27), (0, 35)],
        [(0, 4), (0, 10), (0, 16), (0, 22), (0, 28), (0, 34)],
        [(0, 6), (0, 14), (0, 22), (0, 30), (0, 38)],
    ]
    
    best_edges = base_edges[:]
    best_cycles = None
    
    def is_pancyclic(edges):
        """
        Verify if the graph with given edges is pancyclic.
        Returns a list of cycles (vertex lists) for lengths 3..40 if valid, else None.
        """
        # Build adjacency list
        adj = {i: set() for i in range(N)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # We need to find a cycle of length k for each k in 3..N
        cycles = {}
        
        # Helper to find a simple cycle of length k containing vertex 0
        # Using DFS with backtracking and pruning.
        # To speed up, we can use the fact that the graph is relatively sparse.
        
        # Optimization: If we find a cycle of length k, we can try to extend it 
        # or use it as a base. But for simplicity and robustness, let's run a separate search for each k.
        # Since N=40 is small, and the graph is sparse, this should be fast enough.
        
        # Even better: Find ONE Hamiltonian cycle (length 40). 
        # If we have a Hamiltonian cycle, we can often derive shorter cycles by "chord shortcuts".
        # But deriving them requires the chords to be present.
        # Actually, the task requires returning explicit cycles.
        
        # Let's try to find cycles for each length k.
        for k in range(3, N + 1):
            if k == N:
                # Try to find Hamiltonian cycle
                # This is the hardest part.
                pass
            
            # Search for cycle of length k
            # We will use a DFS that tracks path and visited set.
            # To make it faster, we can guide the search.
            
            found_cycle = None
            
            # Heuristic: Start DFS from vertex 0.
            # Pruning: If remaining edges < k - current_path_len, fail.
            # But simple DFS might be slow for k=40.
            
            # Alternative for k=N:
            # If the graph is constructed with a specific structure (cycle + chords),
            # we might know the Hamiltonian cycle explicitly or find it easily.
            # For our specific chord sets, the Hamiltonian cycle might just be the base cycle 
            # IF the chords don't break it? No, the base cycle is always there.
            # Wait, the base cycle IS a Hamiltonian cycle!
            # 0-1-2-...-39-0 is a cycle of length 40.
            # So cycles[40] = [0, 1, ..., 39]
            
            if k == N:
                cycles[N] = list(range(N))
                continue
                
            # For k < N, we need to find a cycle of length k.
            # Since the base cycle is there, we can "cut" it using chords.
            # Example: Chord (0, d). Cycle 0-1-...-d-0 has length d+1.
            # Also 0-d-...-N-1-0 has length N-d+1.
            # Chord (a, b) can create cycles by combining parts of the base cycle and the chord.
            
            # Let's just do a fast search for small k.
            # For k close to N, we can use the base cycle and "shorten" it.
            
            # Simple DFS with timeout per k? No, global deadline.
            # Let's assume the chord sets are good and use a specific algorithm.
            
            # Algorithm for finding cycle of length k:
            # Try all pairs of chords?
            # Or just DFS with strong pruning.
            
            # Let's implement a optimized DFS for each k.
            # Since N=40, max depth 40.
            
            path = [0]
            visited = {0}
            
            def dfs(start_idx, current_len):
                if current_len == k:
                    # Check if last node connects to start (0)
                    if 0 in adj[path[-1]]:
                        return path[:]
                    return None
                if time.time() - start_time > DEADLINE - 5:
                    return "TIMEOUT"
                
                # Pruning:
                # 1. Remaining nodes to reach k
                remaining = k - current_len
                # If remaining > degree of current node + 1? No, that's not quite right.
                
                for nxt in sorted(adj[path[-1]]): # Sort for determinism
                    if nxt == 0 and current_len < k:
                        continue # Can't close cycle early
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    path.append(nxt)
                    res = dfs(start_idx, current_len + 1)
                    if res == "TIMEOUT":
                        return "TIMEOUT"
                    if res:
                        return res
                    path.pop()
                    visited.remove(nxt)
                return None
            
            res = dfs(0, 1)
            if res == "TIMEOUT":
                return None
            if res:
                cycles[k] = res
            else:
                return None
                
        return cycles

    # Test the chord candidates
    for chords in chord_candidates:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        current_edges = base_edges + chords
        # Remove duplicates just in case (chords might overlap with base, though we chose them not to)
        # (0,7) is not in base.
        
        cycles = is_pancyclic(current_edges)
        if cycles:
            best_edges = current_edges
            best_cycles = cycles
            break # Found a valid sparse graph
    
    if not best_cycles:
        # Fallback: Use the seed's fan graph if the fast search fails or times out
        # This ensures we return a valid result, even if score is lower.
        # Seed: Edges (0, i) for i in 1..39 and (i, i+1) for i in 1..38.
        # Total edges: 39 + 38 = 77.
        # Cycles: For length k, cycle 0-1-...-(k-1)-0? 
        # Wait, in fan graph, edge (0, k-1) exists. Path 1-2-...-(k-1) exists.
        # So cycle 0-1-2-...-(k-1)-0 is valid.
        # Length k.
        
        edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        cycles = [list(range(k)) for k in range(3, N + 1)]
        return {'edges': edges, 'cycles': cycles}
    
    # Format output
    # Edges should be distinct pairs.
    # Cycles should be lists of vertices.
    
    return {
        'edges': best_edges,
        'cycles': [best_cycles[k] for k in range(3, N + 1)]
    }

# EVOLVE-BLOCK-END
