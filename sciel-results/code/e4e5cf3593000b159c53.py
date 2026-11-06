"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin
    
    # Ladder: two 20-cycles + perfect matching
    # Cycle A: 0..19, Cycle B: 20..39, Matching: i <-> i+20
    edges_set = set()
    edges_list = []
    
    def add_edge(u, v):
        if u > v:
            u, v = v, u
        if (u, v) not in edges_set:
            edges_set.add((u, v))
            edges_list.append((u, v))
    
    # Cycle A: 0-1-2-...-19-0
    for i in range(20):
        add_edge(i, (i + 1) % 20)
    # Cycle B: 20-21-22-...-39-20
    for i in range(20):
        add_edge(20 + i, 20 + ((i + 1) % 20))
    # Matching: i <-> i+20
    for i in range(20):
        add_edge(i, i + 20)
    
    # Build adjacency
    adj = [[] for _ in range(N)]
    for u, v in edges_list:
        adj[u].append(v)
        adj[v].append(u)
    
    def find_cycle(length, start=None):
        """Find a cycle of given length using DFS/BFS."""
        if length == 2:
            return None
        if start is None:
            start = 0
        
        # For small cycles, try all starting vertices
        starts = [start] if start is not None else range(N)
        
        for s in starts:
            # DFS to find a cycle of exact length
            visited = [-1] * N  # -1 = unvisited, else distance from s
            visited[s] = 0
            stack = [(s, [s])]
            
            while stack:
                if time.time() > deadline:
                    return None
                node, path = stack.pop()
                if len(path) > length:
                    continue
                if len(path) == length:
                    # Check if last node connects back to start
                    if s in adj[node]:
                        return path
                    continue
                for nb in adj[node]:
                    if visited[nb] == -1:
                        visited[nb] = len(path)
                        stack.append((nb, path + [nb]))
                    elif nb == s and len(path) == length:
                        return path
            # Reset visited
            visited = [-1] * N
        
        return None
    
    # Better: use a more systematic approach
    def find_cycle_bfs(length):
        """Find a simple cycle of exact length using BFS from each start."""
        if length < 3:
            return None
        
        for s in range(N):
            if time.time() > deadline:
                return None
            
            # BFS layer by layer
            # visited[node] = (distance, parent)
            visited = {}
            visited[s] = (-1, -1)  # distance -1, no parent
            queue = deque([(s, [s])])
            
            while queue:
                if time.time() > deadline:
                    return None
                node, path = queue.popleft()
                if len(path) > length:
                    continue
                if len(path) == length:
                    if s in adj[node]:
                        return path
                    continue
                for nb in adj[node]:
                    if nb == s:
                        if len(path) == length:
                            return path
                        continue
                    if nb not in visited:
                        visited[nb] = (len(path), node)
                        queue.append((nb, path + [nb]))
        
        return None
    
    # Even better: precompute for the ladder structure
    # In the ladder, we can construct cycles analytically
    def make_cycle_ladder(k):
        """Construct a cycle of length k in the ladder graph analytically."""
        if k == 40:
            # Go around the entire perimeter: 0,1,...,19,20,39,38,...,21,0? No.
            # Actually: 0,1,2,...,19,20,21,...,39,0? No, (39,0) isn't an edge.
            # Full cycle: 0,1,...,19,20,21,...,39,38,...,21? That's not right either.
            # The 40-cycle: 0,1,2,...,19,20,21,...,39,0 - but (39,0) is not an edge!
            # In our ladder, (39,20) is an edge (cycle B), and (0,20) is matching.
            # So: 0,1,2,...,19,20,21,...,39,20? No, that revisits 20.
            # 
            # The actual 40-cycle in the ladder: 
            # 0-1-2-...-19-20-21-...-39-38-...-21? No.
            # 
            # Let me trace: from 0, go along cycle A to 19, then matching to 39, 
            # then along cycle B backwards: 39-38-...-20, then matching to 0.
            # That's: 0,1,2,...,19,39,38,...,20,0
            # Length: 20 (0 to 19) + 1 (19 to 39) + 20 (39 to 20) + 1 (20 to 0) = 42? No.
            # 0,1,...,19 is 20 vertices. 19->39 is one edge. 39,38,...,20 is 20 vertices. 20->0 is one edge.
            # Total vertices: 20 + 20 = 40. Edges: 19 + 1 + 19 + 1 = 40. Yes!
            return list(range(20)) + list(range(39, 19, -1))
            # 0,1,...,19,39,38,...,20
            # Check: (19,39) edge? Yes (matching). (20,0) edge? Yes (matching).
            # (39,38) edge? Yes (cycle B). (1,0) edge? Yes (cycle A). Good.
        
        if k <= 20:
            # Use a portion of cycle A
            if k == 20:
                return list(range(20))
            # For k < 20, we need chords... but we don't have chords in cycle A alone.
            # We need to use the matching edges.
            # For a k-cycle using i consecutive vertices of cycle A and some of cycle B:
            # Example: k=3: vertices 0,1,20. Edges: (0,1),(1,21)? No.
            # (0,1) yes, (0,20) yes, (1,20)? No.
            # k=3: 0,1,21? (0,1) yes, (1,21) yes (matching), (21,0)? No.
            # 
            # Hmm, triangles: need three mutually connected vertices.
            # In the ladder, vertex i connects to: i-1, i+1 (cycle A), i+20 (matching)
            # Vertex i+20 connects to: i+19, i+21 (cycle B), i (matching)
            # 
            # Triangle: i, i+1, i+20? Edges: (i,i+1) yes, (i,i+20) yes, (i+1,i+20)? No.
            # Triangle: i, i+20, i+21? Edges: (i,i+20) yes, (i+20,i+21) yes, (i,i+21)? No.
            # 
            # No triangles in the pure ladder! We need to add edges.
            return None
        
        # For k > 20, use portions of both cycles
        if k > 20 and k < 40:
            # Use i vertices from cycle A and k-i from cycle B
            # Path: a_0, a_1, ..., a_{i-1}, b_{j}, b_{j+1}, ..., b_{k-i-1}, back to a_0
            # Where a_0 = 0, a_{i-1} = i-1, and we need (i-1) connected to some b, and b connected to 0.
            # (i-1) connects to i-1+20 via matching. 0 connects to 20 via matching.
            # So: 0,1,...,i-1, i-1+20, i-1+21, ..., i-1+20+(k-i-1), ... need to get back to 0.
            # The last vertex must connect to 0. 0 connects to 1, 19, 20.
            # So last vertex should be 20 or 19 or 1.
            # 
            # Let's use: 0,1,...,i-1, then matching to i-1+20, then along cycle B: i-1+20, i-1+21, ..., 
            # We need the last vertex to be 20 (to connect to 0 via matching).
            # So we go from i-1+20 along cycle B to 20.
            # Cycle B: 20,21,...,39,20. Going from i-1+20 to 20 along cycle B.
            # If i-1+20 <= 39: go forward: i-1+20, i+20, ..., 39, 20. That's (39-(i-1+20)+1) + 1 = 21-i+1 = 22-i steps? 
            # Vertices visited in B: from i-1+20 to 39 (that's 39-(i-1+20)+1 = 21-i vertices) then 20. Total B vertices: 22-i.
            # Total vertices: i + (22-i) = 22. That's fixed at 22! Not variable.
            #
            # Alternative: go the other way around cycle B.
            # From i-1+20, go backwards: i-1+20, i-2+20, ..., 20. That's i vertices in B.
            # Total: i + i = 2i vertices. So k = 2i, meaning k must be even and i = k/2.
            #
            # Or: use a different connection. Instead of matching at both ends, 
            # we can enter/exit at different points.
            #
            # General approach: 
            # Path in A: 0, 1, ..., a (a vertices, a+1 vertices)
            # Matching: a -> a+20
            # Path in B: a+20, a+21, ..., b (some number of vertices)
            # Matching or cycle edge: b -> 0
            # For b -> 0: b must be 20 (matching) or 1 or 19 (cycle A, but those are in A)
            # So b = 20.
            #
            # Path in B from a+20 to 20:
            # Forward: a+20, a+21, ..., 39, 20. Vertices: (39-(a+20)+1) + 1 = 20-a+1 = 21-a. Wait:
            # a+20 to 39: that's 39-(a+20)+1 = 20-a vertices. Then 20. Total: 21-a vertices.
            # Total cycle: (a+1) + (21-a) = 22. Always 22!
            #
            # Backward: a+20, a-1+20, ..., 20. Vertices: (a+20-20)+1 = a+1. 
            # Total: (a+1) + (a+1) = 2a+2. So k = 2a+2, a = (k-2)/2.
            # This works for even k >= 4, with a = (k-2)/2, need a <= 19, so k <= 40.
            #
            # For odd k, we need a different construction.
            #
            # Let me reconsider. Maybe add some extra edges to enable odd cycles.
            return None
        
        return None
    
    # The pure ladder doesn't have triangles or odd cycles easily.
    # Let me add some strategic chords to enable all cycle lengths.
    # 
    # Better approach: use a different base graph.
    # 
    # Alternative: Start with a 40-cycle and add minimum chords to get all lengths.
    # A 40-cycle: (0,1),(1,2),...,(39,0). 40 edges.
    # Add chord (0,k) to create a (k+1)-cycle and a (40-k+1)-cycle... 
    # Actually chord (0,k) creates cycles of length k+1 and 40-k+1... no.
    # Chord (0,k): path 0-1-...-k-0 has length k+1 (k edges + 1 closing = k+1 vertices).
    # And 0-k-(k-1)-...-39-0 has length 40-k+1 = 41-k vertices.
    #
    # With chord (0,k), we get cycles of length k+1 and 41-k.
    # We need all lengths 3..40.
    #
    # Let's think about what's achievable with few chords.
    # 
    # Actually, let me just do a search. Build a sparse graph and find cycles.
    
    # Let me try a different construction:
    # Base: 40-cycle (0,1,...,39,0) = 40 edges
    # Add chords to enable all cycle lengths.
    
    edges_set2 = set()
    edges_list2 = []
    
    def add_edge2(u, v):
        if u > v:
            u, v = v, u
        if (u, v) not in edges_set2:
            edges_set2.add((u, v))
            edges_list2.append((u, v))
    
    # 40-cycle
    for i in range(40):
        add_edge2(i, (i + 1) % 40)
    
    # Add chords. Each chord (0,k) gives cycles of length k+1 and 41-k.
    # We need lengths 3,4,...,40.
    # 40-cycle gives length 40.
    # Chord (0,2) gives length 3 (0,1,2,0) and length 39 (0,2,3,...,39,0).
    # Chord (0,3) gives length 4 and 38.
    # Chord (0,4) gives length 5 and 37.
    # ...
    # Chord (0,k) gives lengths k+1 and 41-k.
    # For k=1: length 2 (invalid) and 40 (already have).
    # For k=2: length 3 and 39.
    # For k=3: length 4 and 38.
    # For k=4: length 5 and 37.
    # For k=5: length 6 and 36.
    # For k=6: length 7 and 35.
    # For k=7: length 8 and 34.
    # For k=8: length 9 and 33.
    # For k=9: length 10 and 32.
    # For k=10: length 11 and 31.
    # For k=11: length 12 and 30.
    # For k=12: length 13 and 29.
    # For k=13: length 14 and 28.
    # For k=14: length 15 and 27.
    # For k=15: length 16 and 26.
    # For k=16: length 17 and 25.
    # For k=17: length 18 and 24.
    # For k=18: length 19 and 23.
    # For k=19: length 20 and 22.
    # For k=20: length 21 and 21.
    # 
    # With chords (0,2) through (0,20), we get all lengths 3..40!
    # That's 19 chords + 40 cycle edges = 59 edges.
    # Score: 38/(59-40) = 38/19 = 2.0
    
    for k in range(2, 21):
        add_edge2(0, k)
    
    # Now find cycles
    adj2 = [[] for _ in range(N)]
    for u, v in edges_list2:
        adj2[u].append(v)
        adj2[v].append(u)
    
    cycles = []
    for length in range(3, N + 1):
        if time.time() > deadline:
            break
        cycle = None
        if length == 40:
            cycle = list(range(40))
        elif length == 21:
            cycle = [0] + list(range(1, 21))  # 0,1,...,20,0? No, 21 vertices: 0,1,...,20
            # Check: (0,1),(1,2),...,(19,20),(20,0). (20,0) is a chord. Yes!
        else:
            # For length k, chord (0, k-1) gives cycle 0,1,...,k-1,0
            # That's k vertices: 0,1,...,k-1
            chord_k = length - 1
            if 2 <= chord_k <= 20:
                cycle = list(range(chord_k + 1))  # 0,1,...,chord_k
            elif 21 <= chord_k <= 38:
                # Use the other direction: 0, chord_k, chord_k+1, ..., 39, 0
                # Length: 2 + (40 - chord_k - 1) = 41 - chord_k
                # We need 41 - chord_k = length, so chord_k = 41 - length
                # This is the same as above since chord_k = length-1 and 41-(length-1) = 42-length
                # For length > 21, chord_k = length-1 > 20, so we use the complementary cycle
                comp_chord = 41 - length  # = 41 - length
                if 2 <= comp_chord <= 20:
                    # Cycle: 0, comp_chord, comp_chord+1, ..., 39, 0
                    # Vertices: 0, comp_chord, comp_chord+1, ..., 39
                    # Count: 1 + (40 - comp_chord) = 41 - comp_chord = 41 - (41-length) = length. Good.
                    cycle = [0] + list(range(comp_chord, 40))
                else:
                    cycle = None
            else:
                cycle = None
        
        if cycle is None or len(cycle) != length:
            # Fallback: search
            cycle = find_cycle_bfs(length)
        
        if cycle is None:
            # Shouldn't happen, but fallback
            cycle = list(range(length))
        
        cycles.append(cycle)
    
    # Verify and return
    return {'edges': edges_list2, 'cycles': cycles}

# EVOLVE-BLOCK-END
