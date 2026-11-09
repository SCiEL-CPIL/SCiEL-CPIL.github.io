"""
Four-colourings of complete graphs without monochromatic C6 (R_4(C6)).
Goal: maximize n such that K_n can be 4-coloured with no monochromatic 6-cycle.
"""

import time
import numpy as np
from itertools import combinations

# Deadline guard: evaluator kills at 150s wall-clock.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin

def _elapsed():
    return time.time() - _START_TIME

def _build_adj(color_array, n):
    """
    Build 4 adjacency matrices (n x n) from the color array.
    color_array is a flat array of length n*(n-1)//2 in lexicographic order.
    """
    adj = [np.zeros((n, n), dtype=np.uint8) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = color_array[idx]
            adj[c][i, j] = 1
            adj[c][j, i] = 1
            idx += 1
    return adj

def _count_c6_in_adj(adj, n):
    """
    Count the number of (not necessarily induced) 6-cycles in the adjacency matrix adj.
    A 6-cycle is a sequence v0-v1-v2-v3-v4-v5-v0 where all edges exist.
    We count each cycle once by fixing v0 as the smallest vertex and requiring
    v1 < v3 < v5 (to avoid rotations and reflections double-counting).
    
    Actually, let's use a more direct approach: enumerate all 6-tuples
    (a,b,c,d,e,f) with a < b < c < d < e < f is too restrictive.
    
    Better: use the standard formula. Number of C6 in a graph =
    sum over all 6-tuples of distinct vertices forming a cycle.
    
    For efficiency, we'll use: for each pair of opposite vertices in the hexagon,
    count paths of length 3 between them.
    
    A 6-cycle v0-v1-v2-v3-v4-v5-v0 has opposite pairs (v0,v3), (v1,v4), (v2,v5).
    Number of C6 = (1/6) * sum over all unordered pairs {u,v} of (number of paths of length 3 from u to v that use only internal vertices distinct from u,v) * ... 
    
    This is getting complex. Let me use a direct but optimized enumeration.
    
    For n up to ~17, the number of 6-cycles to check per color class is manageable.
    Total 6-tuples: C(n,6) * 5! / 2 = C(n,6) * 60. For n=17, C(17,6)=12376, so 12376*60/2 = 371280 per color class, times 4 = ~1.5M. That's fine.
    
    Actually, let's just check for existence (not count) to be faster.
    """
    # For each color class, check if any C6 exists.
    # A C6 is a 6-cycle: v0,v1,v2,v3,v4,v5 with edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0).
    # To avoid overcounting, fix v0 as the smallest vertex in the cycle.
    # Then we need v1..v5 to be a path of length 5 from v0 back to v0, with all vertices distinct and v0 < all others.
    
    # Alternative: use the fact that a C6 exists iff there exist two vertices u,v with at least
    # one common neighbor structure forming a 6-cycle. 
    # 
    # Simplest robust method: for each color, check all 6-combinations of vertices and all 15 ways to form a cycle.
    # For a set of 6 vertices, a 6-cycle exists if the induced subgraph contains a Hamiltonian cycle.
    # There are (6-1)!/2 = 60 distinct cycles on 6 labeled vertices.
    
    for c in range(4):
        A = adj[c]
        # For each 6-subset, check if it contains a C6
        for combo in combinations(range(n), 6):
            # Check if the induced subgraph on these 6 vertices has a Hamiltonian cycle
            # Try all 60 permutations (cycles)
            found = False
            # Generate all cyclic permutations of the 6 vertices
            # Fix first vertex, permute the rest, divide by 2 for reflection
            first = combo[0]
            rest = list(combo[1:])
            from itertools import permutations
            for perm in permutations(rest):
                cycle = (first,) + perm
                # Check if all consecutive pairs (and last-first) are edges
                ok = True
                for k in range(6):
                    u = cycle[k]
                    v = cycle[(k+1) % 6]
                    if A[u, v] == 0:
                        ok = False
                        break
                if ok:
                    found = True
                    break
            if found:
                return True
    return False

def _has_c6(color_array, n):
    """Check if any color class contains a C6."""
    adj = _build_adj(color_array, n)
    return _count_c6_in_adj(adj, n)

def _random_coloring(n):
    """Generate a random 4-coloring of K_n."""
    m = n * (n - 1) // 2
    return np.random.randint(0, 4, size=m)

def _local_search(n, max_iters=5000):
    """
    Local search: start with random coloring, try to reduce C6 count.
    """
    # For local search, we need to count C6s efficiently.
    # Let's count the number of monochromatic C6s.
    
    def count_total_c6s(colors):
        """Count total monochromatic C6s."""
        adj = _build_adj(colors, n)
        total = 0
        for c in range(4):
            A = adj[c]
            # Count C6s in this color class
            # Use: for each 6-subset, count Hamiltonian cycles in induced subgraph
            for combo in combinations(range(n), 6):
                first = combo[0]
                rest = list(combo[1:])
                from itertools import permutations
                count = 0
                for perm in permutations(rest):
                    cycle = (first,) + perm
                    ok = True
                    for k in range(6):
                        u = cycle[k]
                        v = cycle[(k+1) % 6]
                        if A[u, v] == 0:
                            ok = False
                            break
                    if ok:
                        count += 1
                total += count
        return total
    
    # This is too slow for local search. Let me use a different approach.
    # Instead of counting all C6s, use a heuristic: try to find a valid coloring by
    # starting from a known good structure and perturbing.
    
    # Known: the best known is n=17. Let me try to construct one.
    # Actually, let me use a smarter approach: use the fact that for small n,
    # we can use backtracking with constraint propagation.
    
    # For now, let's just do random restarts with local improvement using
    # a faster C6 check.
    
    pass

def _backtrack_search(n):
    """
    Backtracking search for a valid 4-coloring of K_n with no monochromatic C6.
    Use constraint propagation: when assigning a color to edge (i,j), check if
    it creates a C6 in that color class.
    """
    m = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    colors = [-1] * m
    
    # Precompute for each edge, which other edges would form a C6 with it
    # This is complex. Let's use a simpler approach: assign edges in order,
    # and after each assignment, check if any C6 is completed.
    
    # A C6 is completed when 6 specific edges are all assigned the same color.
    # Precompute all possible C6s (as sets of 6 edge indices).
    
    # For n up to ~12, this is feasible. For larger n, we need pruning.
    
    c6_sets = []
    for combo in combinations(range(n), 6):
        first = combo[0]
        rest = list(combo[1:])
        from itertools import permutations
        seen = set()
        for perm in permutations(rest):
            cycle = (first,) + perm
            # Normalize: rotate so smallest vertex is first, and reflect
            # The cycle edges are (cycle[k], cycle[k+1]) for k=0..5
            edge_set = frozenset()
            valid = True
            for k in range(6):
                u = cycle[k]
                v = cycle[(k+1) % 6]
                if u > v:
                    u, v = v, u
                # Find edge index
                # Edge (i,j) has index: sum of (n-1-i) for i' < i... 
                # Actually, let's precompute edge index lookup
                edge_set.add((u, v))
            if len(edge_set) == 6:
                # Convert to frozenset of edge indices
                edge_idx_set = frozenset(_edge_index(u, v, n) for u, v in edge_set)
                if edge_idx_set not in seen:
                    seen.add(edge_idx_set)
                    c6_sets.append(edge_idx_set)
    
    # This precomputation is O(C(n,6) * 60) which for n=12 is C(12,6)=924 * 60 = 55440, fine.
    # For n=17, C(17,6)=12376 * 60 = 742560, still feasible.
    
    # Now backtrack
    def _edge_index(i, j, n):
        if i > j:
            i, j = j, i
        # Index of edge (i,j) in lexicographic order
        # Edges are (0,1),(0,2),...,(0,n-1),(1,2),(1,3),...,(1,n-1),...
        # Index = i*(n-1) - i*(i+1)//2 + (j - i - 1)
        # Let me verify: for i=0: 0 + (j-0-1) = j-1. Edge (0,1) -> 0, (0,2) -> 1. Correct.
        # For i=1: 1*(n-1) - 1*2//2 + (j-1-1) = (n-1) - 1 + j - 2 = n + j - 4.
        # Edge (1,2): n+2-4 = n-2. After (0,1)..(0,n-1) which are n-1 edges, next is (1,2) at index n-1. 
        # Hmm, n-2 vs n-1. Let me recompute.
        # Number of edges with first vertex < i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # Then within vertex i, edge (i,j) is at position (j-i-1)
        # So index = i*(n-1) - i*(i-1)//2 + (j-i-1)
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Recompute c6_sets with correct edge indices
    c6_sets = []
    for combo in combinations(range(n), 6):
        first = combo[0]
        rest = list(combo[1:])
        from itertools import permutations
        seen = set()
        for perm in permutations(rest):
            cycle = (first,) + perm
            edge_set = set()
            for k in range(6):
                u = cycle[k]
                v = cycle[(k+1) % 6]
                if u > v:
                    u, v = v, u
                edge_set.add((u, v))
            if len(edge_set) == 6:
                edge_idx_set = frozenset(_edge_index(u, v, n) for u, v in edge_set)
                if edge_idx_set not in seen:
                    seen.add(edge_idx_set)
                    c6_sets.append(edge_idx_set)
    
    # For each edge, precompute which C6s it belongs to
    edge_to_c6s = [[] for _ in range(m)]
    for idx, c6 in enumerate(c6_sets):
        for e in c6:
            edge_to_c6s[e].append(idx)
    
    # Backtracking with forward checking
    # State: colors array, and for each C6, how many of its edges are assigned and to which colors
    
    # This is getting very complex. Let me simplify: use a greedy/random approach
    # with restarts, and use the C6 check only at the end or periodically.
    
    # Given time constraints, let me just do random restarts with the full C6 check.
    # For n <= 12, this should be fast enough.
    
    return None  # Not implemented fully

def construct():
    """
    Construct a 4-coloring of K_n with no monochromatic C6, maximizing n.
    """
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to find larger n using random restarts
    # For each n from 10 upward, try random colorings and check for C6-free
    
    for n in range(10, 21):
        if _elapsed() > _DEADLINE - 5:
            break
        
        found = False
        max_attempts = 200  # Limit attempts per n
        
        for attempt in range(max_attempts):
            if _elapsed() > _DEADLINE - 5:
                break
            
            colors = _random_coloring(n)
            if not _has_c6(colors, n):
                best_n = n
                best_colors = colors.tolist()
                found = True
                break
        
        if found:
            # Continue to see if we can go higher
            continue
        else:
            # If we couldn't find one for this n, try a bit more with local search
            # For now, just break
            break
    
    return {'n': best_n, 'colors': best_colors}
