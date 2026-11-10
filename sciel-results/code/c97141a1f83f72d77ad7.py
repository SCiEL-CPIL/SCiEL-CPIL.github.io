"""Improved program: targeted search for graphs with minimal distinct cycle lengths."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _count_distinct_cycle_lengths_fast(edges, n):
    """Fast count of distinct simple cycle lengths. Returns None if invalid."""
    if not edges:
        return None
    
    edge_set = set()
    for e in edges:
        u, v = e
        if u == v or u < 0 or u >= n or v < 0 or v >= n:
            return None
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
    
    degree = [0] * n
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        degree[u] += 1
        degree[v] += 1
        adj[u].add(v)
        adj[v].add(u)
    
    for d in degree:
        if d < 3:
            return None
    
    # Triangle check
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return None
    
    # Enumerate cycles using DFS with canonical ordering
    cycle_lengths = set()
    
    for start in range(n):
        stack = [(start, {start}, 1)]
        while stack:
            current, visited, path_len = stack.pop()
            if path_len >= 3 and start in adj[current]:
                cycle_lengths.add(path_len)
            for next_node in adj[current]:
                if next_node == start:
                    continue
                if next_node > start and next_node not in visited:
                    stack.append((next_node, visited | {next_node}, path_len + 1))
    
    if not cycle_lengths:
        return None
    
    # Check girth >= 4 (already ensured by triangle-free, but verify min cycle len)
    min_cl = min(cycle_lengths)
    if min_cl < 4:
        return None
    
    return len(cycle_lengths)


def construct():
    """Construct a graph with minimum degree 3, girth at least 4, on 12 vertices.
    Minimize the number of distinct simple cycle lengths."""
    
    start_time = time.time()
    DEADLINE = 85.0
    n = 12
    
    best_score = float('inf')
    best_edges = None
    
    def try_graph(edge_list):
        nonlocal best_score, best_edges
        if time.time() > DEADLINE - 5:
            return
        score = _count_distinct_cycle_lengths_fast(edge_list, n)
        if score is not None and score < best_score:
            best_score = score
            best_edges = [list(e) for e in edge_list]
            if best_score == 1:
                return
    
    # Predefined candidate graphs
    
    # 1. K_{6,6} - has many cycle lengths
    edges_k66 = [[i, j + 6] for i in range(6) for j in range(6)]
    try_graph(edges_k66)
    
    # 2. Two K_{3,3} connected by matching
    edges_two_k33 = []
    # First K_{3,3}: vertices 0,1,2 and 3,4,5
    for i in range(3):
        for j in range(3):
            edges_two_k33.append([i, j + 3])
    # Second K_{3,3}: vertices 6,7,8 and 9,10,11
    for i in range(3):
        for j in range(3):
            edges_two_k33.append([i + 6, j + 9])
    # Connect with matching: 0-6, 1-7, 2-8, 3-9, 4-10, 5-11
    for i in range(6):
        edges_two_k33.append([i, i + 6])
    try_graph(edges_two_k33)
    
    # 3. 12-cycle with perfect matching chords (distance 6)
    edges_cycle_match = [[i, (i + 1) % 12] for i in range(12)]
    for i in range(6):
        edges_cycle_match.append([i, i + 6])
    try_graph(edges_cycle_match)
    
    # 4. 12-cycle with chords at distance 3
    edges_d3 = [[i, (i + 1) % 12] for i in range(12)]
    for i in range(12):
        edges_d3.append([i, (i + 3) % 12])
    try_graph(edges_d3)
    
    # 5. 12-cycle with chords at distance 4
    edges_d4 = [[i, (i + 1) % 12] for i in range(12)]
    for i in range(12):
        edges_d4.append([i, (i + 4) % 12])
    try_graph(edges_d4)
    
    # 6. Cubic bipartite graph: 3-regular bipartite on 12 vertices
    # Use the 3-prism bipartite version or a known cubic bipartite graph
    # The 3-cube Q3 has 8 vertices. We need 12.
    # Let's try: two 6-cycles with cross-edges
    edges_bip = []
    # Cycle 1: 0-2-4-6-8-10-0
    for i in range(6):
        edges_bip.append([2*i, 2*(i+1)%6])
    # Cycle 2: 1-3-5-7-9-11-1
    for i in range(6):
        edges_bip.append([2*i+1, 2*(i+1)%6+1])
    # Cross edges: 0-1, 2-3, 4-5, 6-7, 8-9, 10-11
    for i in range(6):
        edges_bip.append([2*i, 2*i+1])
    try_graph(edges_bip)
    
    # 7. Another cubic bipartite: 12 vertices
    # Part A: {0,1,2,3,4,5}, Part B: {6,7,8,9,10,11}
    # Each vertex in A connects to 3 in B, and vice versa
    edges_cubic_bip = []
    # 0: 6,7,8
    edges_cubic_bip += [[0,6],[0,7],[0,8]]
    # 1: 6,9,10
    edges_cubic_bip += [[1,6],[1,9],[1,10]]
    # 2: 7,9,11
    edges_cubic_bip += [[2,7],[2,9],[2,11]]
    # 3: 8,10,11
    edges_cubic_bip += [[3,8],[3,10],[3,11]]
    # 4: 7,10,6
    edges_cubic_bip += [[4,7],[4,10],[4,6]]
    # 5: 8,11,9
    edges_cubic_bip += [[5,8],[5,11],[5,9]]
    try_graph(edges_cubic_bip)
    
    # 8. Systematic search over chord patterns on 12-cycle
    base_cycle = [[i, (i + 1) % 12] for i in range(12)]
    
    # Try adding k chords from a set of candidate chords
    candidate_chords = []
    for dist in [3, 4, 5]:
        for i in range(12):
            candidate_chords.append((i, (i + dist) % 12))
    
    # Deduplicate
    seen_chords = set()
    unique_chords = []
    for c in candidate_chords:
        key = (min(c), max(c))
        if key not in seen_chords:
            seen_chords.add(key)
            unique_chords.append(list(key))
    
    # Try single chords
    for chord in unique_chords:
        if time.time() > DEADLINE - 5:
            break
        edges_test = base_cycle + [chord]
        # Check degrees: vertices in chord get +1, others stay at 2
        # Need all >= 3, so this won't work with single chord
        pass
    
    # Try pairs of chords that cover all vertices
    for i in range(len(unique_chords)):
        if time.time() > DEADLINE - 5:
            break
        for j in range(i + 1, len(unique_chords)):
            if time.time() > DEADLINE - 10:
                break
            c1 = unique_chords[i]
            c2 = unique_chords[j]
            # Check no shared vertices (for simplicity, try disjoint first)
            if len(set(c1) & set(c2)) > 0:
                continue
            # Degrees: 4 vertices get degree 3, 8 get degree 2 -> not enough
            pass
    
    # Try 6 chords forming a perfect matching
    all_matchings = []
    def gen_matchings(remaining, current):
        if not remaining:
            all_matchings.append(current[:])
            return
        if time.time() > DEADLINE - 10:
            return
        first = remaining[0]
        for idx in range(1, len(remaining)):
            second = remaining[idx]
            new_remaining = remaining[1:idx] + remaining[idx+1:]
            current.append((first, second))
            gen_matchings(new_remaining, current)
            current.pop()
    
    # Only generate matchings from the 12 vertices
    # This is 11! = 39916800, too many. Let's be smarter.
    
    # Instead, try structured matchings
    # Matching type: i connected to i+6
    edges_m1 = base_cycle + [[i, i+6] for i in range(6)]
    try_graph(edges_m1)
    
    # Matching type: i connected to (i+3)%12 for i in 0..5
    edges_m2 = base_cycle + [[i, (i+3)%12] for i in range(6)]
    try_graph(edges_m2)
    
    # Matching type: i connected to (i+4)%12 for i in 0..5
    edges_m3 = base_cycle + [[i, (i+4)%12] for i in range(6)]
    try_graph(edges_m3)
    
    # Matching type: i connected to (i+5)%12 for i in 0..5
    edges_m4 = base_cycle + [[i, (i+5)%12] for i in range(6)]
    try_graph(edges_m4)
    
    # 9. Random search with better heuristics
    random.seed(42)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    iterations = 0
    while time.time() < DEADLINE - 5 and iterations < 50000:
        iterations += 1
        
        # Generate random bipartite graph with parts of 6
        # This ensures triangle-free
        part_a = list(range(6))
        part_b = list(range(6, 12))
        
        edge_set = set()
        degree = [0] * n
        
        # Start with random edges
        num_edges = random.randint(18, 30)
        bip_edges = [(a, b) for a in part_a for b in part_b]
        random.shuffle(bip_edges)
        
        for u, v in bip_edges[:num_edges]:
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                edge_set.add(key)
                degree[u] += 1
                degree[v] += 1
        
        # Fix min degree
        valid = True
        for vertex in range(n):
            while degree[vertex] < 3:
                if vertex < 6:
                    candidates = [u for u in part_b if (vertex, u) not in edge_set]
                else:
                    candidates = [u for u in part_a if (u, vertex) not in edge_set]
                if not candidates:
                    valid = False
                    break
                u = random.choice(candidates)
                key = (min(vertex, u), max(vertex, u))
                edge_set.add(key)
                degree[vertex] += 1
                degree[u] += 1
            if not valid:
                break
        
        if not valid:
            continue
        
        if len(edge_set) > 36:
            continue
        
        edges_list = [[u, v] for u, v in edge_set]
        try_graph(edges_list)
        
        if best_score == 1:
            break
    
    if best_edges is None:
        best_edges = edges_k66
    
    # Ensure proper types
    final_edges = []
    for e in best_edges:
        u, v = e
        final_edges.append([int(u), int(v)])
    
    final_edges.sort()
    
    return final_edges

# EVOLVE-BLOCK-END
