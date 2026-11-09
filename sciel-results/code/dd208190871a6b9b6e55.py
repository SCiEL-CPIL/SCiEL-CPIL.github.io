"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

def construct():
    """
    Construct a triangle-free graph on 12 vertices (0..11) with minimum degree >= 3
    and girth >= 4, minimizing the number of distinct simple cycle lengths.
    
    Approach: Bipartite graph with two partitions of 6 vertices each.
    We use a randomized greedy algorithm to select edges ensuring:
    1. Each vertex has degree at least 3
    2. No triangles (guaranteed by bipartite structure)
    3. Girth >= 4 (guaranteed by triangle-freeness in simple graphs)
    4. Minimize distinct cycle lengths
    
    For K_{3,3} x K_{3,3} (disjoint), cycle lengths are {4, 6} -> 2 distinct lengths.
    We try to do better or at least match this.
    
    Actually, let's think about this more carefully. In a bipartite graph, all cycles
    have even length. The minimum possible girth is 4. If we can make all cycles
    length 4, we'd have just 1 distinct cycle length. But that's only possible if
    the graph is a union of 4-cycles (C4).
    
    A 3-regular graph where all cycles are length 4: this would be a complete bipartite
    graph K_{3,3}? No, K_{3,3} has cycles of length 4 and 6.
    
    Let's try a different approach: use the complete bipartite graph K_{4,4} plus two
    isolated vertices? No, we need all 12 vertices to have degree >= 3.
    
    Let me reconsider. The best we can do is probably 2 distinct cycle lengths (like K_{3,3}
    components giving {4, 6}). Can we achieve 1?
    
    For 1 distinct cycle length L, all cycles must have length L. In a bipartite graph,
    L must be even. If L=4, the graph must be a union of C4s. A 3-regular graph that is
    a union of C4s: each C4 has 4 vertices of degree 2, but we need degree 3. So this
    doesn't work directly.
    
    Actually, if we take multiple C4s and add edges between them to raise degrees to 3,
    we'll create longer cycles. So achieving exactly 1 distinct cycle length with degree >= 3
    seems very hard or impossible.
    
    Let's stick with the K_{3,3} approach which gives score 2.5. But let me try a
    randomized search to see if we can find a better configuration.
    """
    n = 12
    best_edges = None
    best_distinct_lengths = float('inf')
    
    # Use bipartite structure: left = 0..5, right = 6..11
    left = list(range(6))
    right = list(range(6, 12))
    
    # Try multiple random attempts
    for attempt in range(200):
        random.seed(attempt)
        
        # Start with a random 3-regular bipartite graph (each vertex degree exactly 3)
        # This is a random 3-regular bipartite graph on 6+6 vertices
        
        # Generate a random 3-regular bipartite graph
        # Each left vertex connects to 3 random right vertices
        # Each right vertex must also have degree 3
        
        # Use configuration model for bipartite 3-regular graphs
        # Left side: 6 vertices, each needs 3 edges -> 18 edge-ends
        # Right side: 6 vertices, each needs 3 edges -> 18 edge-ends
        
        left_ends = []
        for v in left:
            for _ in range(3):
                left_ends.append(v)
        
        right_ends = []
        for v in right:
            for _ in range(3):
                right_ends.append(v)
        
        # Shuffle and pair
        random.shuffle(left_ends)
        random.shuffle(right_ends)
        
        edges_set = set()
        valid = True
        for u, v in zip(left_ends, right_ends):
            if u == v:  # Shouldn't happen since they're in different partitions
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges_set:
                valid = False
                break
            edges_set.add(edge)
        
        if not valid:
            continue
        
        edges = list(edges_set)
        
        # Check degrees
        deg = {v: 0 for v in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        if any(d < 3 for d in deg.values()):
            continue
        
        # Find all simple cycles and their lengths
        # Use DFS-based cycle enumeration
        adj = {v: [] for v in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        cycle_lengths = set()
        
        # Enumerate all simple cycles using a modified DFS
        # For each pair (start, end), find all simple paths from start to end
        # where start < end (to avoid duplicates), and add path length + 1
        
        def find_cycles(adj, n):
            lengths = set()
            for start in range(n):
                for end in range(start + 1, n):
                    # Find all simple paths from start to end
                    # This is expensive but n=12 is small
                    # Use DFS with backtracking
                    visited = [False] * n
                    path = [start]
                    
                    def dfs(current, target, path, visited):
                        if current == target and len(path) >= 3:
                            # Found a cycle of length len(path)
                            lengths.add(len(path))
                            return
                        for neighbor in adj[current]:
                            if neighbor == start and len(path) >= 2:
                                # Can only go back to start if forming a cycle
                                continue
                            if not visited[neighbor] and neighbor != start:
                                visited[neighbor] = True
                                path.append(neighbor)
                                dfs(neighbor, target, path, visited)
                                path.pop()
                                visited[neighbor] = False
                    
                    # Actually, let me use a cleaner approach
                    # Enumerate all simple cycles by fixing the smallest vertex in the cycle
                    pass
            
            return lengths
        
        # Better approach: enumerate cycles by picking each edge and finding cycles containing it
        # Or use networkx-style approach
        
        # Let's just count distinct cycle lengths by enumerating all simple cycles
        # For n=12, this is feasible
        
        def enumerate_simple_cycles(adj, n):
            """Enumerate all simple cycles in the graph."""
            cycles = []
            for start in range(n):
                # Fix start as the smallest vertex in the cycle
                for first_neighbor in adj[start]:
                    if first_neighbor < start:
                        continue
                    # DFS from first_neighbor, trying to get back to start
                    visited = [False] * n
                    visited[start] = True
                    path = [start, first_neighbor]
                    
                    def dfs(current, target, path, visited):
                        if current == target:
                            # Found a cycle
                            cycles.append(len(path))
                            return
                        for neighbor in adj[current]:
                            if neighbor == start and len(path) >= 3:
                                # Complete the cycle
                                cycles.append(len(path))
                                return
                            if not visited[neighbor] and neighbor > start:
                                # Only allow vertices > start to avoid duplicates
                                # Wait, this is wrong. We need to allow all vertices
                                # but only count cycles where start is the minimum
                                # Let me rethink this
                                pass
                    
                    # This is getting complicated. Let me use a simpler but slower approach.
                    pass
            
            return cycles
        
        # Simpler: just find all cycle lengths by checking all subsets? No, too slow.
        # For n=12, let's use a standard cycle enumeration algorithm.
        
        def get_cycle_lengths(adj, n):
            """Get set of all simple cycle lengths in the graph."""
            lengths = set()
            for start in range(n):
                # Only consider cycles where start is the smallest vertex
                for end in adj[start]:
                    if end < start:
                        continue
                    # Find all simple paths from end to start where all internal vertices > start
                    visited = [False] * n
                    visited[start] = True
                    path = [start]
                    
                    def dfs(current, target, path, visited):
                        if current == target and len(path) >= 3:
                            lengths.add(len(path))
                            return
                        for neighbor in adj[current]:
                            if neighbor == start and len(path) >= 3:
                                lengths.add(len(path))
                                return
                            if not visited[neighbor] and neighbor > start:
                                visited[neighbor] = True
                                path.append(neighbor)
                                dfs(neighbor, target, path, visited)
                                path.pop()
                                visited[neighbor] = False
                    
                    visited[end] = True
                    path.append(end)
                    dfs(end, start, path, visited)
                    path.pop()
                    visited[end] = False
            
            return lengths
        
        distinct_lengths = get_cycle_lengths(adj, n)
        
        if len(distinct_lengths) < best_distinct_lengths:
            best_distinct_lengths = len(distinct_lengths)
            best_edges = edges
        
        if best_distinct_lengths == 1:
            break
    
    # If random search didn't find anything better, fall back to K_{3,3} x K_{3,3}
    if best_edges is None or len(best_edges) == 0:
        best_edges = []
        for offset in (0, 6):
            l = [offset, offset + 1, offset + 2]
            r = [offset + 3, offset + 4, offset + 5]
            for u in l:
                for v in r:
                    best_edges.append([int(u), int(v)])
    else:
        best_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return best_edges

# EVOLVE-BLOCK-END
