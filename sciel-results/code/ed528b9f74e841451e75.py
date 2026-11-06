import time
import itertools
from collections import deque

N = 40

def construct():
    """
    Sparse pancyclic graph on 40 vertices.
    Returns {'edges': ..., 'cycles': ...}
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Base Structure: Two complete graphs K_20 on V1 (0..19) and V2 (20..39)
    #    Connected by a perfect matching: i <-> i+20
    #    This is a candidate for the "twin" structure mentioned in the prompt.
    
    edges = set()
    cycles = []
    
    # Build V1 clique
    for i in range(20):
        for j in range(i+1, 20):
            edges.add((i, j))
            
    # Build V2 clique
    for i in range(20, 40):
        for j in range(i+1, 40):
            edges.add((i, j))
            
    # Add matching edges
    for i in range(20):
        edges.add((i, i+20))

    # 2. Cycle Construction Strategy
    # We need cycles of length k for k in 3..40.
    # 
    # Observation:
    # - Since V1 and V2 are cliques, any subset of vertices within V1 induces a clique, 
    #   so we can form cycles of any length up to 20 entirely within V1? 
    #   Wait, a clique K_m is not necessarily having a cycle of every length 3..m?
    #   Yes, K_m contains cycles of all lengths 3..m.
    #   So for k in 3..20, we can just use k vertices from V1.
    #   Example: for k=3, {0,1,2}. For k=20, {0..19}.
    #
    # - What about k > 20?
    #   We need cycles that use vertices from both V1 and V2.
    #   Let a cycle use $a$ vertices from V1 and $b$ vertices from V2, with $a+b=k$.
    #   Since we have matching edges $i \leftrightarrow i+20$, we can "bridge" between the two cliques.
    #   
    #   A simple construction for a cycle of length $k > 20$:
    #   Let $a = 20$ (all of V1) and $b = k - 20$ (some of V2).
    #   We need to form a cycle alternating or bridging.
    #   
    #   Let's try a specific pattern for $k=20+m$ where $1 \le m \le 20$.
    #   Vertices: $V_{1\_all} = [0, 1, ..., 19]$ and $V_{2\_subset} = [20, 21, ..., 19+m]$.
    #   
    #   How to connect them into a single cycle?
    #   We have all edges within V1, all within V2, and matching $i \leftrightarrow i+20$.
    #   
    #   Consider the cycle:
    #   Start at $0 \in V1$.
    #   Go to $1 \in V1$, ..., to $19 \in V1$.
    #   From $19 \in V1$, go to $19+20 = 39 \in V2$ (matching).
    #   Now we are in V2. We need to visit $m$ vertices in V2 and return to V1?
    #   No, we need a single cycle.
    #   
    #   Let's try a different pattern.
    #   For $k=21$:
    #   Use all 20 from V1 and 1 from V2.
    #   Cycle: $0-1-2-...-19-39-20-...$? No.
    #   
    #   Let's use the property that the matching connects $i$ and $i+20$.
    #   If we take a subset $A \subset V1$ and $B \subset V2$ such that $|A|=a, |B|=b$,
    #   and we ensure that for every $i \in A$, $i+20 \in B$ or similar, we can weave them.
    #   
    #   Actually, a simpler approach:
    #   The graph contains $K_{20} \cup K_{20}$ plus matching.
    #   Is this graph pancyclic?
    #   
    #   Let's check small cases mentally or via code simulation later if needed, but logically:
    #   For $k \le 20$: Use $K_k$ in V1.
    #   For $k = 20 + m$:
    #   We can use all 20 vertices of V1 and $m$ vertices of V2.
    #   Let $V2_{used} = \{20, 21, ..., 19+m\}$.
    #   Note that $V2_{used}$ is a clique.
    #   We need to merge $V1$ (clique) and $V2_{used}$ (clique) into one cycle.
    #   The only edges between them are the matching edges $i \leftrightarrow i+20$.
    #   The matching edges available between $V1$ and $V2_{used}$ are:
    #   $0 \leftrightarrow 20, 1 \leftrightarrow 21, ..., (m-1) \leftrightarrow (19+m)$.
    #   Wait, if $m < 20$, only the first $m$ pairs are fully connected?
    #   No, the matching is defined for all $i \in 0..19$.
    #   So edges exist: $(i, i+20)$ for $i=0..m-1$ are between $V1$ and $V2_{used}$.
    #   Edges $(i, i+20)$ for $i=m..19$ connect $V1$ to $V2 \setminus V2_{used}$, which are not in our cycle.
    #   
    #   So we have $m$ "bridge" edges.
    #   Can we form a cycle of length $20+m$ using $m$ bridges?
    #   
    #   If $m=1$: 20 from V1, 1 from V2 (say 20).
    #   Bridges: $0-20$.
    #   We need a cycle using 0..19 and 20.
    #   Path in V1: $1-2-...-19-0$.
    #   Bridge: $0-20$.
    #   Bridge back? We only have one bridge $0-20$.
    #   We need to return from 20 to V1. But 20 is only connected to 0 in V1 (via matching) and to other V2 vertices (not in cycle).
    #   So 20 has degree 1 in the induced subgraph (connected to 0).
    #   So we cannot form a cycle with a single bridge if the vertex in V2 has no other connections to V1.
    #   Therefore, $K_{20} \cup K_{20}$ + matching is NOT pancyclic. It fails for $k=21$.
    
    # Strategy Adjustment:
    # We need more edges between V1 and V2 to support cycles of length > 20.
    # A common sparse pancyclic construction is a "ladder" or "prism" like structure.
    # 
    # Alternative Base:
    # A single Hamiltonian cycle $C_{40}$ is not enough (needs chords for shorter cycles).
    # 
    # Let's try a "Double Cycle" or "Prism" graph $Y_4$? No, $Y_4$ is specific.
    # 
    # Let's try:
    # Vertices $0..39$.
    # Edges:
    # 1. Cycle $C_{40}$: $(i, i+1 \pmod{40})$.
    # 2. Add chords to ensure smaller cycles.
    # 
    # A known sparse pancyclic graph is the graph formed by a cycle $C_n$ and a matching that creates triangles, etc.
    # 
    # Let's use a constructive approach with a deadline.
    # Start with $C_{40}$.
    # Add chords greedily to cover missing cycle lengths.
    # 
    # However, the "Twin" idea in the prompt was specific.
    # "Twin" $K_{20}$ failed for $k=21$.
    # What if we add a Hamiltonian cycle on the *matching* vertices?
    # i.e., connect $20-21-...-39-20$? That's already in $K_{20}$.
    # 
    # Let's try a different split:
    # $V1 = \{0, 2, 4, ..., 38\}$ (even)
    # $V2 = \{1, 3, 5, ..., 39\}$ (odd)
    # This is bipartite. No triangles. So not pancyclic.
    
    # Let's stick to the Greedy Chord Addition on $C_{40}$.
    # It is robust and likely to succeed within time.
    
    # Reset to Greedy Plan
    edges = set()
    # Add C40
    for i in range(N):
        edges.add((i, (i+1) % N))
        
    # We need cycles of length 3..40.
    # C40 covers length 40.
    # We need to find cycles of length k for k=3..39.
    
    # Helper to find a cycle of length k in the graph
    def find_cycle(graph, k):
        """
        Finds a simple cycle of length k in an undirected graph.
        Returns list of vertices or None.
        Uses DFS with pruning.
        """
        if k < 3:
            return None
        if k == 40:
            return list(range(40)) # Assuming C40 is present
        
        # Try starting from each vertex? Or just 0?
        # For symmetry breaking, we can fix 0 in the cycle if k>1?
        # Actually, any cycle is isomorphic to its rotations.
        # Let's just try to find *any* cycle.
        
        # Optimized search:
        # Since the graph is sparse, we can use a recursive backtracking.
        # To avoid redundant checks, we can enforce an ordering?
        # No, cycles can be in any order.
        
        # Simple DFS
        def dfs(path, visited):
            if len(path) == k:
                if (path[-1], path[0]) in graph or (path[0], path[-1]) in graph:
                    return path
                return None
            
            current = path[-1]
            for neighbor in graph.get(current, []):
                if neighbor == path[0] and len(path) == k:
                    return path
                if neighbor in visited:
                    continue
                # Pruning: if remaining nodes + (k - len(path)) < ... ?
                # Simple pruning: if we have visited k-1 nodes, check if last connects to first.
                
                visited.add(neighbor)
                path.append(neighbor)
                res = dfs(path, visited)
                if res:
                    return res
                path.pop()
                visited.remove(neighbor)
            return None

        # Try starting from vertex 0
        # To make it faster, we can try to find a cycle containing 0 first.
        # If the graph is pancyclic, it's likely to have many cycles.
        
        # We need adjacency list
        adj = {i: [] for i in range(N)}
        for u, v in graph:
            adj[u].append(v)
            adj[v].append(u)
            
        # Sort neighbors to make search deterministic and potentially faster
        for i in range(N):
            adj[i].sort()
            
        # Try starting from 0
        # Note: This might miss cycles that don't include 0, but for a pancyclic graph,
        # if we add chords carefully, we can ensure 0 is in many cycles.
        # Actually, to be safe, we should try a few start vertices if 0 fails.
        
        start_candidates = [0, 1, 2]
        for start in start_candidates:
            visited = {start}
            path = [start]
            res = dfs(path, visited)
            if res:
                return res
                
        return None

    # Greedy Addition
    # We need cycles for lengths 3..39.
    # 40 is covered by C40.
    
    missing_lengths = list(range(3, 40))
    
    # Candidate chords: all pairs (i,j) not in edges, i<j
    # Heuristic: Add chords that create small cycles first?
    # Or just iterate.
    
    # To speed up, we can pre-compute candidate chords.
    # Number of possible chords is ~ 40*39/2 - 40 = 740.
    # We can test each one.
    
    # To be efficient, we only test a chord if it helps with a missing length.
    
    # Let's maintain a set of current edges.
    current_edges = set(edges)
    
    # Helper to check if adding a specific edge helps
    # We can just add the edge, check all missing lengths, if any are now present, keep it?
    # No, we want to MINIMIZE edges.
    # So we should only add an edge if it *enables* a previously missing cycle to be found,
    # AND that cycle is not found by other existing edges?
    # Actually, if a cycle of length k was missing, and adding edge e makes it present,
    # we should add e.
    # But what if multiple edges are needed?
    # Greedy:
    # While there are missing lengths:
    #   Find a missing length k.
    #   Find a chord e such that adding e creates a cycle of length k.
    #   Add e.
    #   Update missing lengths.
    
    # How to find such an e efficiently?
    # For a given k, iterate all possible chords e.
    # For each e, check if cycle of length k exists in G+e.
    # This is expensive.
    
    # Alternative:
    # Just add chords in a "smart" order.
    # Chords like (0,2), (0,3), ... create triangles, etc.
    
    # Let's try a specific sequence of chords known to be effective:
    # Connect 0 to all even numbers?
    
    # Better: Use the time budget to run the greedy search properly.
    
    # Generate all potential chords
    all_chords = []
    for i in range(N):
        for j in range(i+1, N):
            if (i,j) not in current_edges:
                all_chords.append((i,j))
                
    # Shuffle? No, deterministic is better for reproducibility.
    # Sort by distance? Chords with small distance create small cycles.
    def chord_dist(e):
        i,j = e
        d1 = (j-i) % N
        d2 = (i-j) % N
        return min(d1, d2)
        
    all_chords.sort(key=chord_dist)
    
    # Now, iterate through chords
    # We need to check if adding a chord completes a cycle for a missing length.
    
    # To make this fast, we will maintain the graph and re-run find_cycle for missing lengths.
    # But find_cycle is expensive.
    
    # Optimization:
    # Only check for a specific missing length k when we add a chord.
    # If we add a chord, we check if it helps *any* missing length.
    # We can prioritize the smallest missing length.
    
    # Let's refine the loop:
    
    # We will keep a list of missing lengths.
    # At each step, pick the smallest missing length k_min.
    # Try to find a chord e that enables a cycle of length k_min.
    # If found, add it, remove k_min from missing, and update.
    # If no single chord enables k_min, we might be stuck or need to add a chord that helps a *different* length first?
    # No, if k_min is missing, it must be enabled by some sequence of chords.
    # Adding a chord that doesn't help k_min immediately might be a waste if we can just focus on k_min.
    
    # So:
    while missing_lengths and time.time() < deadline:
        k = missing_lengths[0]
        
        # Try to find a chord that creates a cycle of length k
        found_chord = None
        for e in all_chords:
            if e in current_edges:
                continue
            # Test if adding e creates a cycle of length k
            test_edges = current_edges | {e}
            
            # Only check for length k
            # To speed up, we can pass k to find_cycle?
            # My find_cycle is hardcoded to find *a* cycle of length k.
            
            # We can optimize find_cycle by early termination if it finds *any* cycle of length k.
            
            # But calling find_cycle for every chord is O(V^k) which is too slow.
            
            # Heuristic:
            # Just add the chord and see if it reduces the number of missing cycles?
            # No, we want minimal edges.
            
            # Let's try a different tactic:
            # Add chords in the sorted order (small distance first).
            # After adding each, check if the smallest missing cycle is now present.
            # If yes, remove it from missing list.
            # If no, keep the chord?
            # This might add useless chords.
            
            # Better:
            # Add a chord ONLY IF it enables a new cycle for a currently missing length.
            # How to check this efficiently?
            # We can check if the new edge is part of *any* cycle of the missing lengths?
            # That's also hard.
            
            # Let's just add chords in order and verify at the end?
            # No, that might lead to many edges.
            
            # Compromise:
            # Add the chord. Then check if the smallest missing length k is now satisfiable.
            # If yes, remove k from missing.
            # If no, keep the chord anyway? 
            # This risks adding too many edges.
            
            # Let's try: Add chord e. 
            # Check if k is now present. 
            # If yes, update missing.
            # If no, but k is still missing, we might have wasted an edge.
            # To avoid wasting, we can only add e if we *expect* it to help.
            # 
            # Actually, for a cycle of length k, we need k edges.
            # If we are adding chords to a C40, we are increasing connectivity.
            
            # Let's just do the simple thing:
            # 1. Add next chord from sorted list.
            # 2. Check if the current smallest missing length is now covered.
            # 3. If yes, pop it.
            # 4. If we cover all, stop.
            # 5. If we run out of chords, stop.
            
            # This is a heuristic. It might not be optimal but should be sparse.
            
            # Add the chord
            current_edges.add(e)
            
            # Check if k is now covered
            if find_cycle(current_edges, k):
                missing_lengths.pop(0)
                
            # Time check
            if time.time() > deadline - 5:
                break
                
    # If we still have missing cycles, we might need to add more chords or return seed.
    # Since we are minimizing edges, returning the current graph if it's valid is good.
    # But if it's invalid, we must return the seed (which is valid).
    
    # Verify validity
    valid = True
    if missing_lengths:
        valid = False
    else:
        # Double check all cycles
        for k in range(3, N+1):
            if not find_cycle(current_edges, k):
                valid = False
                break
    
    if valid:
        # Construct the cycles list
        cycles = []
        for k in range(3, N+1):
            cyc = find_cycle(current_edges, k)
            if cyc is None:
                valid = False
                break
            cycles.append(cyc)
            
        if valid:
            return {
                'edges': list(current_edges),
                'cycles': cycles
            }
            
    # Fallback to Seed
    return {'edges': [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)],
            'cycles': [list(range(k)) for k in range(3,N+1)]}

# EVOLVE-BLOCK-END
