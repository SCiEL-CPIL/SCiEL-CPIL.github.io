"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _count_c6(adj):
    """Count 6-cycles in a graph using numpy. adj is an n x n boolean adjacency matrix."""
    n = adj.shape[0]
    # A 6-cycle v0-v1-v2-v3-v4-v5-v0 exists if there are 6 distinct vertices
    # with consecutive edges. We count using matrix powers.
    # Number of closed walks of length 6 / 6 / 2 (each cycle counted 12 times: 6 starting points * 2 directions)
    # But this counts walks, not simple cycles. For sparse graphs this overcounts.
    # Better: use the formula for counting 6-cycles via matrix multiplication.
    # For now, use a simpler approach: count triangles in the "square" graph won't work.
    # Let's use: number of C6 = (1/12) * sum over all 6-tuples with proper structure.
    # Actually, let's just do a direct but vectorized approach.
    
    # adj is symmetric boolean matrix
    # A 6-cycle: v0,v1,v2,v3,v4,v5 all distinct, edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0)
    # Count: for each ordered pair (v0,v1) with edge, count paths of length 5 from v1 back to v0
    # where all intermediate vertices are distinct from v0,v1.
    
    # This is complex. Let's use a different approach: 
    # Count C6 by iterating over all 6-element subsets is O(C(n,6)) which for n=20 is 38760.
    # For each subset, check if it contains a C6. But checking if a 6-vertex graph contains C6 is doable.
    
    # Actually for n up to 20, C(20,6) = 38760, and for each we need to check ~15 Hamiltonian cycles.
    # That's about 580K checks, each O(6) = manageable.
    
    # But we have 4 color classes, so we need to check each one.
    # Let's precompute for a given coloring.
    
    # For efficiency, let's use the fact that we're checking color classes separately.
    # For a color class with adjacency matrix A (boolean), count C6.
    
    # Use: number of C6 in a graph = (1/12) * trace(A^6) - corrections for smaller cycles and walks
    # This is getting complicated. Let's just do the direct enumeration for n <= 20.
    
    # For each 6-subset, check if the induced subgraph contains a C6.
    # A 6-vertex graph contains a C6 iff it has a Hamiltonian cycle.
    # For 6 vertices, there are (6-1)!/2 = 60 undirected Hamiltonian cycles.
    
    # But actually, we can check more efficiently.
    # Let's just iterate over all permutations of 6 vertices and check if they form a cycle.
    # For a given 6-subset S, check if there exists a permutation (v0..v5) of S such that
    # all consecutive pairs (including v5,v0) are edges in the color class.
    
    # For speed, let's precompute all 60 cycle orderings for 6 labeled positions.
    
    return _count_c6_direct(adj)

def _count_c6_direct(adj):
    """Direct count of 6-cycles in a boolean adjacency matrix."""
    n = adj.shape[0]
    if n < 6:
        return 0
    
    # Precompute all 60 undirected cycles on 6 labeled vertices
    # A cycle is defined by its edge set. Two permutations give the same cycle if they're
    # rotations or reversals of each other.
    # We'll generate all permutations and deduplicate by edge set.
    
    from itertools import permutations
    
    # Generate all distinct 6-cycles on vertices 0-5
    # Fix vertex 0 at position 0 to avoid rotational symmetry
    # And fix vertex 1 at position 1 to avoid reversal symmetry (partially)
    # Actually, let's just generate all and use a set of frozensets of edges
    
    cycles = []
    seen = set()
    for perm in permutations(range(1, 6)):
        cycle = (0,) + perm
        # Edges: (cycle[i], cycle[(i+1)%6]) for i in 0..5
        edges = frozenset(frozenset((cycle[i], cycle[(i+1)%6])) for i in range(6))
        if edges not in seen:
            seen.add(edges)
            cycles.append(cycle)
    
    # Should have 60 cycles
    count = 0
    # For each 6-subset of vertices
    from itertools import combinations
    for subset in combinations(range(n), 6):
        # Check each cycle pattern
        for cycle in cycles:
            # Map cycle positions to actual vertices
            # cycle[k] is the position in the cycle (0-5), subset[cycle[k]] is the vertex
            # Wait, cycle is a permutation of (0,1,2,3,4,5)
            # So for position k in the cycle, the vertex is subset[cycle[k]]
            ok = True
            for k in range(6):
                v = subset[cycle[k]]
                w = subset[cycle[(k+1)%6]]
                if not adj[v, w]:
                    ok = False
                    break
            if ok:
                count += 1
                break  # Only count once per subset
    
    return count

def _has_c6(adj):
    """Check if a boolean adjacency matrix contains a 6-cycle."""
    n = adj.shape[0]
    if n < 6:
        return False
    
    from itertools import permutations, combinations
    
    # Precompute cycle patterns once (global cache)
    if not hasattr(_has_c6, '_cycles'):
        cycles = []
        seen = set()
        for perm in permutations(range(1, 6)):
            cycle = (0,) + perm
            edges = frozenset(frozenset((cycle[i], cycle[(i+1)%6])) for i in range(6))
            if edges not in seen:
                seen.add(edges)
                cycles.append(cycle)
        _has_c6._cycles = cycles
    
    cycles = _has_c6._cycles
    
    for subset in combinations(range(n), 6):
        for cycle in cycles:
            ok = True
            for k in range(6):
                v = subset[cycle[k]]
                w = subset[cycle[(k+1)%6]]
                if not adj[v, w]:
                    ok = False
                    break
            if ok:
                return True
    return False

def _build_adj_matrices(colors, n):
    """Build 4 adjacency matrices from color list."""
    adjs = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adjs[c][i, j] = True
            adjs[c][j, i] = True
            idx += 1
    return adjs

def _is_valid(colors, n):
    """Check if coloring is valid (no C6 in any color class)."""
    adjs = _build_adj_matrices(colors, n)
    for adj in adjs:
        if _has_c6(adj):
            return False
    return True

def _random_coloring(n):
    """Generate a random 4-coloring."""
    m = n*(n-1)//2
    return np.random.randint(0, 4, size=m).tolist()

def _local_search(colors, n, deadline):
    """Local search: try to fix violations by flipping edges."""
    adjs = _build_adj_matrices(colors, n)
    
    # Check initial validity
    if _is_valid(colors, n):
        return colors
    
    # Local search: find violating C6s and try to fix them
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Check if valid
        if _is_valid(colors, n):
            return colors
        
        # Find a violating color class and C6
        adjs = _build_adj_matrices(colors, n)
        found = False
        for c in range(4):
            if _has_c6(adjs[c]):
                # Find a specific C6 in this color class
                from itertools import combinations
                cycles = _has_c6._cycles
                for subset in combinations(range(n), 6):
                    for cycle in cycles:
                        ok = True
                        for k in range(6):
                            v = subset[cycle[k]]
                            w = subset[cycle[(k+1)%6]]
                            if not adjs[c][v, w]:
                                ok = False
                                break
                        if ok:
                            # Found a C6. Try flipping one edge to break it.
                            # Pick a random edge in the cycle and flip its color.
                            k = np.random.randint(6)
                            v = subset[cycle[k]]
                            w = subset[cycle[(k+1)%6]]
                            if v > w:
                                v, w = w, v
                            idx = _edge_index(v, w, n)
                            old_color = colors[idx]
                            new_color = np.random.randint(0, 4)
                            while new_color == old_color:
                                new_color = np.random.randint(0, 4)
                            colors[idx] = new_color
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        
        if not found:
            # No C6 found? Shouldn't happen if _is_valid said invalid
            # Maybe race condition, just return
            break
    
    return colors

def _edge_index(i, j, n):
    """Get the index in the lexicographic list for edge (i,j) with i<j."""
    # Index = sum of (n-1-k) for k=0 to i-1, plus (j-i-1)
    # For vertex i, the number of edges before it is sum_{k=0}^{i-1} (n-1-k)
    # = i*(n-1) - i*(i-1)//2
    # Then within vertex i's edges, j is at position j-i-1
    if i > j:
        i, j = j, i
    return i*(n-1) - i*(i-1)//2 + (j - i - 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with the known K9 solution
    n_base = 9
    colors_base = [min(j - i, n_base - j + i) - 1 for i in range(n_base) for j in range(i + 1, n_base)]
    
    best_n = n_base
    best_colors = colors_base
    
    # Try to extend to larger n
    for target_n in range(10, 21):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts
        for restart in range(50):
            if time.time() > deadline - 2:
                break
            
            # Start from random coloring
            colors = _random_coloring(target_n)
            
            # Local search
            colors = _local_search(colors, target_n, deadline)
            
            if _is_valid(colors, target_n):
                best_n = target_n
                best_colors = colors
                break  # Found valid coloring for this n, try larger
        
        if best_n >= target_n:
            continue  # Already found valid for this n
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
