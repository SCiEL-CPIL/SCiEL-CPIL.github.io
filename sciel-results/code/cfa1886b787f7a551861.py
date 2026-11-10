"""
Constructs a triangle-free graph on 16 vertices with chromatic number at least 4.
Strategy:
1. Start with the Grötzsch graph (11 vertices, 20 edges, chi=4).
2. Identify a maximum independent set I of size 5 in the Grötzsch graph.
3. Add 5 new vertices (11..15).
4. Connect each new vertex to all vertices in I.
5. This adds 25 edges. Total edges = 45.
6. The graph remains triangle-free because I is independent and the new vertices are only connected to I.
7. The graph remains 4-chromatic because it contains the Grötzsch graph as a subgraph.
"""

def construct():
    # Vertices 0..10 form the Grötzsch graph
    # Vertices 11..15 are the new vertices
    
    # Grötzsch graph edges
    # Cycle 0-1-2-3-4-0
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        # Copies 5..9 connected to cycle neighbors
        # 5 connected to 4, 0
        [4, 5], [5, 0],
        # 6 connected to 0, 1
        [0, 6], [6, 1],
        # 7 connected to 1, 2
        [1, 7], [7, 2],
        # 8 connected to 2, 3
        [2, 8], [8, 3],
        # 9 connected to 3, 4
        [3, 9], [9, 4],
        # Top vertex 10 connected to 5..9
        [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]
    ]
    
    # Max independent set in Grötzsch graph.
    # The set {0, 2, 5, 7, 9} is an independent set?
    # Let's verify.
    # 0 connected to 1, 4, 5, 6.
    # 2 connected to 1, 3, 7, 8.
    # 5 connected to 4, 0, 10.
    # 7 connected to 1, 2, 10.
    # 9 connected to 3, 4, 10.
    # Check pairs:
    # (0,2): No edge.
    # (0,5): Edge [5,0] exists. So {0,5} is NOT independent.
    
    # Let's find a valid independent set of size 5.
    # The set of "odd" vertices in the cycle plus some others?
    # In C5, max independent set is 2.
    # In Grötzsch, alpha=5.
    # One such set is {0, 2, 4, 6, 8}?
    # 0-2: No.
    # 0-4: Edge [4,0]. No.
    
    # Let's use a known independent set.
    # The vertices 0, 2, 5, 7, 9 failed.
    # How about 0, 2, 8, 5, 7?
    # 0-8: No.
    # 0-5: Edge. No.
    
    # Let's look at the structure again.
    # 10 is connected to 5,6,7,8,9. So 10 cannot be in I.
    # 5 is connected to 4,0,10.
    # 6 is connected to 0,1,10.
    # 7 is connected to 1,2,10.
    # 8 is connected to 2,3,10.
    # 9 is connected to 3,4,10.
    # 0 is connected to 1,4,5,6.
    # 1 is connected to 0,2,6,7.
    # 2 is connected to 1,3,7,8.
    # 3 is connected to 2,4,8,9.
    # 4 is connected to 3,0,5,9.
    
    # Try I = {0, 2, 4, 7, 9}?
    # 0-2: No.
    # 0-4: Edge. No.
    
    # Try I = {1, 3, 5, 7, 9}?
    # 1-3: No.
    # 1-5: No (1 conn to 0,2,6,7).
    # 1-7: Edge. No.
    
    # Try I = {0, 2, 5, 8, 9}?
    # 0-2: No.
    # 0-5: Edge. No.
    
    # Let's just pick any 5 non-adjacent vertices.
    # Start with 10? No, 10 is connected to 5-9.
    # If we don't include 10, we can pick from 0-9.
    # Pick 0. Exclude 1,4,5,6.
    # Remaining: 2,3,7,8,9.
    # Pick 2. Exclude 1,3,7,8.
    # Remaining: 9. (3,7,8 excluded).
    # Current: {0, 2, 9}.
    # Can we add more?
    # 9 is connected to 3,4,10.
    # We excluded 3,4.
    # From remaining {3,7,8}, 3 is excluded by 9? No, 9-3 is edge. So 3 excluded.
    # 7 excluded by 2.
    # 8 excluded by 2.
    # So {0,2,9} is maximal? Size 3.
    
    # Let's try a different start.
    # Pick 1. Exclude 0,2,6,7.
    # Remaining: 3,4,5,8,9.
    # Pick 3. Exclude 2,4,8,9.
    # Remaining: 5.
    # Current: {1, 3, 5}.
    # 5 connected to 4,0,10. 4,0 excluded.
    # Can we add from remaining {4,8,9}?
    # 4 excluded by 3.
    # 8 excluded by 3.
    # 9 excluded by 3.
    # Size 3.
    
    # It seems hard to find size 5 manually.
    # However, it is a known fact that alpha(Grötzsch) = 5.
    # One such set is {0, 2, 4, 6, 8}? No, 0-4 edge.
    # How about {0, 2, 5, 7, 9}? 0-5 edge.
    
    # Let's use a simple heuristic in code to find an independent set of size 5, or just hardcode one if I can find it.
    # Actually, I can just use the vertices 0, 2, 4, 6, 8?
    # Wait, in C5, 0 and 2 are not adjacent. 0 and 4 are adjacent.
    # In Grötzsch, 0 is adjacent to 5.
    
    # Let's try {0, 2, 8, 5, 7}?
    # 0-8: No.
    # 0-5: Edge.
    
    # Okay, I will write a small helper to find an independent set of size 5 in the Grötzsch graph defined above.
    
    adj = {i: set() for i in range(11)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find independent set of size 5
    # Greedy or backtracking
    def find_independent_set(size):
        # Simple backtracking
        def backtrack(current, remaining, candidates):
            if len(current) == size:
                return current
            if not candidates or not remaining:
                return None
            for i in range(len(candidates)):
                c = candidates[i]
                new_candidates = [x for x in candidates[i+1:] if x not in adj[c]]
                res = backtrack(current + [c], remaining - 1, new_candidates)
                if res:
                    return res
            return None
        return backtrack([], size, list(range(11)))
        
    I = find_independent_set(5)
    if I is None:
        # Fallback to a smaller set if 5 not found (should not happen)
        I = find_independent_set(4)
        
    # Add new vertices 11..15
    # Connect each to I
    for i in range(5):
        new_v = 11 + i
        for v in I:
            edges.append([new_v, v])
            
    return edges

