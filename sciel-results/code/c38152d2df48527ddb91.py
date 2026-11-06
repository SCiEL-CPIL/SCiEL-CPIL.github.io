import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Known construction: 8 disjoint K4s gives n=32, C5-free, alpha=8.
    # We try to improve by adding vertices with careful edge selection.
    #
    # Base: 8 K4s on vertices 0..31
    # Try adding vertices 32, 33, 34 with edges to base vertices such that:
    #   - No C5 is created
    #   - Independence number stays < 9

    n = 32
    edges = []

    # Build 8 disjoint K4s
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])

    # Adjacency set for quick lookup
    adj = set()
    for e in edges:
        adj.add((e[0], e[1]))
        adj.add((e[1], e[0]))

    def has_edge(u, v):
        return (u, v) in adj

    def add_edge(u, v):
        edges.append([u, v])
        adj.add((u, v))
        adj.add((v, u))

    def has_c5(adj_set, n_verts):
        """Check if graph has a C5 (5-cycle as subgraph)."""
        # For small graphs, brute force check all 5-tuples
        # A C5 exists if there are 5 distinct vertices v0,v1,v2,v3,v4
        # such that all consecutive edges exist (v0-v1, v1-v2, v2-v3, v3-v4, v4-v0)
        # To avoid checking too many, use a smarter approach:
        # Check for cycles of length exactly 5 in the adjacency structure.
        
        # Build adjacency list
        adj_list = [[] for _ in range(n_verts)]
        for (u, v) in adj_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # For each starting vertex, try to find a 5-cycle
        # To reduce redundancy, only consider cycles where v0 < v1 < ... or use canonical form
        # Actually, let's just check all 5-tuples with v0 < v1 to reduce by factor of 10
        # But even that is O(n^5) which for n=35 is 35^5 = 52M, too slow
        
        # Better: use BFS-based cycle detection for length 5
        # Or: for each pair (u,v) with edge, count common neighbors etc.
        # 
        # Simpler approach for small n: check all combinations of 5 vertices
        # C(35,5) = 324632, for each check if any ordering forms a C5.
        # For each 5-set, check if the induced subgraph contains a C5.
        # A 5-vertex graph has a C5 iff it is exactly a 5-cycle or has a 5-cycle as subgraph.
        # With 5 vertices, a C5 subgraph means the 5 vertices form a cycle (possibly with extra edges).
        # So: for each 5-set, check if there exists a permutation (v0..v4) such that
        # all 5 edges (v0v1, v1v2, v2v3, v3v4, v4v0) exist.
        
        import itertools
        
        for combo in itertools.combinations(range(n_verts), 5):
            # Check if this 5-set contains a C5
            # Try all 12 distinct cycles on 5 vertices (5!/2/5 = 12? No, (5-1)!/2 = 12)
            # Actually number of distinct cycles on 5 labeled vertices = (5-1)!/2 = 12
            # But we can just check: does the induced subgraph on these 5 vertices
            # have minimum degree >= 2 and is it a single cycle? No, it could have extra edges.
            # A C5 as subgraph means there exist 5 vertices in cycle order where all 5 edges exist.
            # With exactly 5 vertices, this means the cycle uses all 5 vertices.
            # So we need to check if there's a Hamiltonian cycle in the induced subgraph.
            
            verts = list(combo)
            # Build adjacency for these 5 verts
            adj5 = set()
            for u in verts:
                for v in adj_list[u]:
                    if v in combo:
                        adj5.add((u, v))
                        adj5.add((v, u))
            
            # Check for Hamiltonian cycle: try all permutations starting with verts[0]
            # Fix v0 = verts[0], permute remaining 4: 4! = 24
            found = False
            for perm in itertools.permutations(verts[1:]):
                cycle = [verts[0]] + list(perm)
                ok = True
                for i in range(5):
                    u = cycle[i]
                    v = cycle[(i + 1) % 5]
                    if (u, v) not in adj5:
                        ok = False
                        break
                if ok:
                    found = True
                    break
            if found:
                return True
        return False

    def max_independent_set_size(adj_set, n_verts):
        """Find the size of the maximum independent set."""
        import itertools
        
        # Build non-edges set
        # For each 9-subset, check if it's independent
        # C(35,9) = 70607460, too many.
        # Use a greedy + backtracking approach.
        
        # Actually, for n up to 35, we can use a simple recursive approach with pruning.
        # Or use the fact that we want to check if alpha >= 9.
        
        # Let's use a simple branch and bound:
        # Pick a vertex, either include it (then exclude its neighbors) or don't.
        
        # Build adjacency list
        adj_list = [[] for _ in range(n_verts)]
        for (u, v) in adj_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # Convert to sets for fast lookup
        adj_set_list = [set() for _ in range(n_verts)]
        for u in range(n_verts):
            for v in adj_list[u]:
                adj_set_list[u].add(v)
        
        best = 0
        
        def backtrack(candidates, current_size):
            nonlocal best
            if current_size + len(candidates) <= best:
                return
            if not candidates:
                best = max(best, current_size)
                return
            
            # Pick a vertex with max degree in the remaining candidate set
            # for better pruning
            cands = list(candidates)
            # Simple heuristic: pick first
            v = cands[0]
            
            # Option 1: include v
            new_candidates = set(candidates) - {v} - adj_set_list[v]
            backtrack(new_candidates, current_size + 1)
            
            # Option 2: exclude v
            new_candidates2 = set(candidates) - {v}
            backtrack(new_candidates2, current_size)
        
        backtrack(set(range(n_verts)), 0)
        return best

    # Try to add vertex 32
    # For a new vertex, its neighbors among the base vertices determine:
    # 1. Whether C5s are created
    # 2. Whether alpha increases
    
    # Strategy: for vertex 32, try different neighbor sets.
    # The base graph has 8 K4s. An independent set in the base graph picks
    # at most 1 vertex from each K4, so alpha = 8.
    # When we add vertex 32 with neighbor set N32, an independent set in the
    # new graph can either include 32 (then pick independent vertices from
    # V \\ N32) or not include 32 (then pick from base, alpha=8).
    # To keep alpha < 9, we need: max(alpha(base), 1 + alpha(V \\ N32)) < 9
    # i.e., alpha(V \\ N32) <= 7 (if 32 is included, total = 1 + alpha(V\N32) <= 8)
    # and alpha(base) = 8 < 9 already holds.
    
    # So we need alpha(V \ N32) <= 7 where V \ N32 is the set of base vertices
    # NOT adjacent to 32.
    
    # Also need C5-free: no 5-cycle involving vertex 32.
    # A 5-cycle through 32 would be: 32 - a - b - c - d - 32
    # where a,d in N32, and a-b-c-d is a path of length 3 in the base graph.
    # So we need: no path of length 3 in the base graph between two vertices
    # both in N32.
    
    # Let's try to add vertex 32 with a neighbor set that satisfies both constraints.
    
    # For the base graph (8 K4s), paths of length 3:
    # Within a K4: any 3-vertex path exists (e.g., 0-1-2-3 is a path of length 3)
    # Across K4s: no edges, so no paths across K4s.
    
    # So a path of length 3 exists within a single K4.
    # To avoid C5 through 32: for each K4, 32 cannot be adjacent to two vertices
    # that are endpoints of a length-3 path in that K4.
    # In a K4, any two distinct vertices are endpoints of some length-3 path.
    # (In K4 on {a,b,c,d}: path a-b-c-d has endpoints a,d; path a-c-b-d has endpoints a,d, etc.)
    # Actually, in K4, for any two distinct vertices x,y, there is a path of length 3 from x to y
    # using the other two vertices.
    # So if 32 is adjacent to two vertices in the same K4, there's a C5.
    
    # Therefore: 32 can be adjacent to at most 1 vertex per K4.
    # Max neighbors of 32 in base: 8 (one per K4).
    # Then V \ N32 has 32 - 8 = 24 vertices.
    # alpha(V \ N32): each K4 contributes 3 non-neighbors (if 32 is adjacent to 1 in that K4),
    # so we can pick at most 3 from each K4... wait no, independent set in V\N32:
    # In each K4, if 1 vertex is removed (adjacent to 32), the remaining 3 form a K3,
    # so independent set picks at most 1 from each K4.
    # So alpha(V \ N32) = 8 (one from each K4's 3 remaining vertices).
    # Then 1 + 8 = 9, which means alpha >= 9. Not good.
    
    # If 32 is adjacent to 0 vertices per some K4s:
    # If 32 is adjacent to k vertices total (at most 1 per K4), then in K4s where 32 has 0 neighbors,
    # all 4 vertices are in V\N32, alpha contribution = 1.
    # In K4s where 32 has 1 neighbor, 3 vertices remain, alpha contribution = 1.
    # So alpha(V\N32) = 8 always (one per K4).
    # Thus 1 + 8 = 9, alpha = 9. Violates constraint.
    
    # So we can't add a single vertex to 8xK4 while maintaining C5-free and alpha < 9.
    # This confirms n=32 is likely optimal for this family.
    
    # Let's try a different base construction.
    # What about using a different C5-free graph with higher alpha capacity?
    
    # Alternative: Use the complement approach.
    # G is C5-free and alpha(G) < 9.
    # Complement: Gbar has no independent set of size 5 (since G is C5-free, Gbar has no K5? No.)
    # Actually: G is C5-free means G has no 5-cycle. Gbar has no independent set of size 5?
    # No: an independent set of size 5 in G is a K5 in Gbar. That's different.
    # alpha(G) < 9 means Gbar has no K9.
    # So Gbar has no K9 and... G C5-free means Gbar has no... hmm, this is getting complex.
    
    # Let's just return the 32-vertex construction.
    # It's known to be optimal or near-optimal.
    
    return {'n': n, 'edges': edges}
