import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin

    # Best result found
    best_edges = None
    best_count = float('inf')

    # Known good: two K_{3,3} -> {4,6} -> count=2
    def make_k33():
        edges = []
        for offset in (0, 6):
            part_a = [offset, offset+1, offset+2]
            part_b = [offset+3, offset+4, offset+5]
            for u in part_a:
                for v in part_b:
                    edges.append([u, v])
        return edges

    best_edges = make_k33()
    best_count = 2

    # Function to count distinct cycle lengths
    def count_distinct_cycle_lengths(edges):
        n = 12
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        all_cycles = []

        def dfs_cycle(current, start, path, visited):
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 3:
                    all_cycles.append(len(path))
                elif neighbor not in visited and neighbor > start:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs_cycle(neighbor, start, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        for start in range(n):
            visited = {start}
            path = [start]
            dfs_cycle(start, start, path, visited)

        return len(set(all_cycles))

    # Try to find a graph with only 1 distinct cycle length
    # A graph with only 4-cycles: e.g., a grid or similar
    # Let's try to construct such a graph

    # First, let's try a 3x4 grid (12 vertices, all degrees 2-4)
    # But we need min degree >= 3
    # Let's try other structures

    # A complete bipartite graph K_{4,4} has 8 vertices
    # K_{3,3} x K_{1,1} doesn't work
    # Let's think: what graphs have all cycles the same length?
    # - Cycles only: C_n has 1 cycle of length n
    # - But we need 12 vertices, min degree 3
    # - A graph made of disjoint cycles won't work (degree 2)
    # - What about a graph where all cycles are 4-cycles?
    #   E.g., a "ladder" or grid structure

    # Let's try: two K_{3,3} but modify to eliminate 6-cycles
    # K_{3,3} has 4-cycles and 6-cycles
    # If we add edges within parts, we break bipartiteness but may create triangles

    # Alternative: try to find a graph with only 4-cycles
    # A 4-regular graph on 12 vertices where all cycles are 4-cycles
    # This is related to "cage" graphs or specific constructions

    # Let's try random search more aggressively
    random.seed(42)

    # Also try some specific constructions
    # 1. Complete graph K_4 has all cycles of length 3 and 4 -> not triangle-free
    # 2. Octahedron graph (6 vertices, 4-regular) -> has 3-cycles
    # 3. Icosahedron has triangles

    # Let's try: a graph that is a "thickened" cycle
    # Or: use the fact that disconnected graphs are allowed

    # What about: one component is K_{3,3} (cycles 4,6) and another is... 
    # No, that still has 2 lengths

    # What if we can make a graph with ONLY 4-cycles?
    # Consider: a 3-regular bipartite graph where all cycles are 4-cycles
    # This would be a "C4-free" complement... no, we want ONLY C4s
    # A graph where every cycle is a 4-cycle is called a "C4-graph" or similar

    # Actually, let's try: the 3x4 grid graph (12 vertices)
    # Vertices: (i,j) for i in 0..2, j in 0..3
    # Edges: horizontal and vertical
    # Degrees: corners have degree 2, others have degree 3 or 4
    # Corners have degree 2 -> violates min degree 3

    # Let's add diagonal edges to fix corners
    # Or use a different structure

    # Try: two disjoint K_{3,3} but with some edges removed/added to eliminate 6-cycles
    # In K_{3,3}, 6-cycles go through all 6 vertices
    # If we remove one edge from each K_{3,3}, do we eliminate all 6-cycles?
    # No, there are still 6-cycles

    # Let's just do more random search with better heuristics
    # Focus on 3-regular bipartite graphs (most likely to have few cycle lengths)

    for attempt in range(200000):
        if time.time() > deadline:
            break

        # Generate random 3-regular bipartite graph on 6+6 vertices
        part_a = list(range(6))
        part_b = list(range(6, 12))

        a_stubs = [v for v in part_a for _ in range(3)]
        b_stubs = [v for v in part_b for _ in range(3)]

        random.shuffle(b_stubs)

        edges = set()
        valid = True
        for a, b in zip(a_stubs, b_stubs):
            if a == b:
                valid = False
                break
            pair = (min(a,b), max(a,b))
            if pair in edges:
                valid = False
                break
            edges.add(pair)

        if not valid or len(edges) != 18:
            continue

        edge_list = [[u, v] for u, v in edges]

        # Quick check: count distinct cycle lengths
        cnt = count_distinct_cycle_lengths(edge_list)

        if cnt < best_count:
            best_count = cnt
            best_edges = edge_list
            if best_count == 1:
                break

    # Ensure types are correct
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])

    return result
