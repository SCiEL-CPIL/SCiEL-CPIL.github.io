"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 12, 'minimum_degree': 3, 'girth': 4}

def _count_distinct_cycle_lengths(n, edges):
    """Count the number of distinct simple cycle lengths in an undirected graph."""
    if not edges:
        return 0
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    lengths = set()
    
    # Enumerate all simple cycles using a DFS-based approach
    # To avoid duplicate cycles, we enforce that the smallest vertex in the cycle
    # is the starting point, and the second smallest is the second vertex.
    
    # Actually, a simpler approach: for each pair (start, end), find all simple paths
    # from start to end that don't use start or end as internal nodes.
    # But that's complicated. Let's use a standard cycle enumeration.
    
    # Standard approach: for each edge (u,v), find all simple paths from u to v
    # that don't use that edge. Each such path + the edge forms a cycle.
    # But this overcounts (each cycle is counted multiple times).
    
    # Better: enumerate all simple cycles by fixing the smallest vertex as the start,
    # and the second-smallest vertex as the second vertex in the cycle.
    
    # For each vertex s (smallest in cycle), for each neighbor n1 of s (second smallest),
    # find all simple paths from n1 to s that only use vertices > s, with n1 being the
    # second smallest vertex in the path.
    
    # This is complex. Let's use a simpler approach for n=12:
    # Enumerate all subsets of vertices of size 4, 5, 6, ... and check if they form a cycle.
    # But that's 2^12 = 4096 subsets, which is fine.
    
    # Actually, let's just do a DFS from each vertex, enforcing that the start vertex
    # is the smallest in the cycle, and track the second smallest to avoid duplicates.
    
    # Simpler: for each starting vertex s, do DFS only visiting vertices >= s,
    # and only allow the cycle to close back to s. But this still overcounts.
    
    # Let's use the approach: for each edge (u,v), find all simple paths from u to v
    # that don't include edge (u,v). Each such path forms a unique cycle.
    # To avoid duplicates, we only consider the path where u is the smallest vertex
    # in the cycle, and v is the second smallest.
    
    # Actually, for n=12, let's just use a brute-force approach:
    # For each subset S of vertices with |S| >= 4, check if the induced subgraph
    # on S contains a cycle. But we need to count distinct cycle lengths, not just
    # whether a cycle exists.
    
    # Let me use a different approach: for each possible length L from 4 to n,
    # check if there exists a simple cycle of length L.
    
    for L in range(4, n + 1):
        found = False
        # Try all starting vertices
        for start in range(n):
            # DFS to find a simple path of length L-1 from start to a neighbor of start
            # that doesn't revisit vertices
            def dfs(current, path, visited):
                if len(path) == L:
                    # Check if current is connected to start
                    if start in adj[current]:
                        return True
                    return False
                for nb in adj[current]:
                    if nb not in visited:
                        # Pruning: if nb < start and len(path) > 1, skip
                        # (to ensure start is the smallest in the cycle)
                        if nb < start and len(path) > 1:
                            continue
                        visited.add(nb)
                        path.append(nb)
                        if dfs(nb, path, visited):
                            return True
                        path.pop()
                        visited.remove(nb)
                return False
            
            visited = {start}
            path = [start]
            if dfs(start, path, visited):
                found = True
                break
        if found:
            lengths.add(L)
    
    return len(lengths)

def construct():
    n = P['n']
    min_deg = P['minimum_degree']
    
    # Start with two disjoint K_{3,3} components, which have cycle lengths {4, 6}
    # This gives 2 distinct cycle lengths.
    
    # Can we do better? Can we have only 1 distinct cycle length?
    # A graph where all cycles have the same length L.
    # For L=4: all cycles are 4-cycles. This is possible (e.g., a grid-like structure).
    # But we need min degree 3.
    
    # Let's try to construct a graph with all cycles of length 4.
    # One example: the complete bipartite graph K_{3,3} has cycles of lengths 4 and 6.
    # So K_{3,3} doesn't work for L=4 only.
    
    # What about a graph that's a union of 4-cycles?
    # If we take two disjoint 4-cycles, each vertex has degree 2. Not enough.
    
    # We need min degree 3. Let's think about a 3-regular graph where all cycles are 4.
    # The smallest such graph might be the utility graph minus some edges?
    
    # Actually, let's try a different approach: search for a graph with min distinct cycle lengths.
    
    # Let's try to find a graph with only 4-cycles.
    # Consider the graph formed by taking two 4-cycles and connecting them with 3 edges.
    # Cycle 1: 0-1-2-3-0
    # Cycle 2: 4-5-6-7-4
    # Connect: 0-4, 1-5, 2-6
    # Now vertex 0 has degree 3 (1, 3, 4)
    # Vertex 1 has degree 3 (0, 2, 5)
    # Vertex 2 has degree 3 (1, 3, 6)
    # Vertex 3 has degree 2 (0, 2) - not enough
    # Vertex 4 has degree 3 (5, 7, 0)
    # Vertex 5 has degree 3 (4, 6, 1)
    # Vertex 6 has degree 3 (5, 7, 2)
    # Vertex 7 has degree 2 (4, 6) - not enough
    
    # We need to add more edges to make 3 and 7 have degree 3.
    # Add 3-7: now 3 has degree 3, 7 has degree 3.
    # But this creates a new cycle: 0-3-7-4-0, which is a 4-cycle. Good.
    # And 1-3-7-5-1, which is a 4-cycle. Good.
    # And 2-3-7-6-2, which is a 4-cycle. Good.
    # And 0-1-5-4-0, which is a 4-cycle. Good.
    # And 0-1-2-6-5-4-0, which is a 6-cycle! Bad.
    
    # So this doesn't work.
    
    # Let's just go with the K_{3,3} approach which gives 2 distinct cycle lengths.
    
    # Two disjoint K_{3,3} on 12 vertices
    edges = []
    # First K_{3,3}: vertices 0,1,2 connected to 3,4,5
    for i in range(3):
        for j in range(3, 6):
            edges.append([i, j])
    
    # Second K_{3,3}: vertices 6,7,8 connected to 9,10,11
    for i in range(6, 9):
        for j in range(9, 12):
            edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
