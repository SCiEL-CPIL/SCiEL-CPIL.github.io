"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def edge_idx(i, j, n):
        """Lexicographic index for edge (i,j) with i<j in K_n."""
        # Sum of (n-1-k) for k=0..i-1 gives offset, then j-1 within row i
        # Row i has edges (i, i+1), ..., (i, n-1) => n-1-i edges
        offset = i * (2 * n - 1 - i) // 2
        return offset + (j - i - 1)

    def precompute_c6s(n):
        """Precompute all C6s in K_n. Each C6 is a frozenset of 6 edges (as indices).
        Returns list of tuples of edge indices."""
        c6_list = []
        # A C6: 6 distinct vertices in a cycle. Enumerate all 6-subsets and all cycles on them.
        for verts in combinations(range(n), 6):
            # Number of distinct C6s on 6 labeled vertices: (6-1)!/2 = 60
            # Generate all 60 Hamiltonian cycles on these 6 vertices
            # Fix vertex 0 as start to avoid rotation duplicates, fix second vertex < last to avoid reversal
            v = verts
            # All permutations of v[1:] where v[1] < v[5] (to fix reversal)
            rest = v[1:]
            for perm in product(range(5), repeat=5):
                pass  # too slow; use permutations
            # Use itertools.permutations
            from itertools import permutations
            seen = set()
            for perm in permutations(v[1:]):
                # Fix reversal: require perm[0] < perm[4]
                if perm[0] >= perm[4]:
                    continue
                cycle = (v[0],) + perm
                # Build edge set
                edges = set()
                for k in range(6):
                    a = cycle[k]
                    b = cycle[(k + 1) % 6]
                    if a < b:
                        edges.add(edge_idx(a, b, n))
                    else:
                        edges.add(edge_idx(b, a, n))
                c6_list.append(tuple(sorted(edges)))
        return c6_list

    def verify_coloring(colors, n):
        """Verify that no color class contains a C6."""
        # Build per-color edge sets
        color_edges = [set() for _ in range(4)]
        for idx, c in enumerate(colors):
            color_edges[c].add(idx)
        
        # Check each color for C6
        for c in range(4):
            es = color_edges[c]
            # For each C6, check if all 6 edges are in this color
            # We'll check during search, but for verification:
            pass
        # Actually just check: for each C6 in precomputed list, all edges not same color
        # But we don't have the C6 list here. Let's do a direct check.
        # Build adjacency per color
        for c in range(4):
            adj = [set() for _ in range(n)]
            for idx in color_edges[c]:
                # Decode edge index back to (i,j)
                i = 0
                while True:
                    row_end = (i + 1) * (2 * n - 1 - i) // 2
                    if idx < row_end:
                        j = i + 1 + (idx - i * (2 * n - 1 - i) // 2)
                        break
                    i += 1
                adj[i].add(j)
                adj[j].add(i)
            # Check for C6 in this adjacency
            # Use the fast check
            for a0 in range(n):
                for a1 in adj[a0]:
                    if a1 <= a0:
                        continue
                    for a2 in adj[a1]:
                        if a2 == a0:
                            continue
                        for a3 in adj[a2]:
                            if a3 in (a0, a1):
                                continue
                            for a4 in adj[a3]:
                                if a4 in (a0, a1, a2):
                                    continue
                                for a5 in adj[a4]:
                                    if a5 in (a0, a1, a2, a3):
                                        continue
                                    if a0 in adj[a5]:
                                        return False
        return True

    def decode_edge(idx, n):
        """Decode edge index to (i, j) with i < j."""
        i = 0
        while True:
            row_end = (i + 1) * (2 * n - 1 - i) // 2
            if idx < row_end:
                j = i + 1 + (idx - i * (2 * n - 1 - i) // 2)
                return (i, j)
            i += 1

    def sat_backtrack(n, max_time):
        """Backtracking with forward checking using precomputed C6 constraints.
        Assigns colors to edges in order. Uses constraint propagation:
        when an edge is assigned color c, any C6 that contains this edge and
        has all other 5 edges already assigned color c is a violation.
        Also, if a C6 has 5 edges assigned color c and 1 unassigned, the
        unassigned edge cannot be color c."""
        
        num_edges = n * (n - 1) // 2
        
        # Precompute C6s as tuples of edge indices
        print(f"Precomputing C6s for n={n}...", flush=True)
        c6s = precompute_c6s(n)
        print(f"Found {len(c6s)} C6s", flush=True)
        
        # For each edge, list of C6 indices it participates in
        edge_to_c6s = [[] for _ in range(num_edges)]
        for ci, c6 in enumerate(c6s):
            for e in c6:
                edge_to_c6s[e].append(ci)
        
        # For each C6, which edges are in it
        # c6s[ci] = tuple of 6 edge indices
        
        # State: colors array (None = unassigned)
        colors = [None] * num_edges
        
        # For each C6, track how many edges are assigned color c for each c
        # c6_color_count[ci][c] = number of edges in C6 ci assigned color c
        c6_color_count = [[0, 0, 0, 0] for _ in range(len(c6s))]
        # c6_assigned[ci] = number of edges in C6 ci that are assigned (any color)
        c6_assigned = [0] * len(c6s)
        # c6_forbidden[ci][c] = True if color c is forbidden for the remaining unassigned edges
        # Actually: if c6_color_count[ci][c] == 5, then the 6th edge cannot be color c
        # if c6_color_count[ci][c] == 6, violation (shouldn't happen if we check before assigning)
        
        # For forward checking: for each unassigned edge, what colors are still possible?
        # edge_possible[edge][c] = True if color c is still possible for this edge
        edge_possible = [[True, True, True, True] for _ in range(num_edges)]
        
        # Count of assigned edges per C6 and per color
        # When we assign edge e color c:
        #   For each C6 ci containing e:
        #     c6_assigned[ci] += 1
        #     c6_color_count[ci][c] += 1
        #     If c6_color_count[ci][c] == 6: violation
        #     If c6_color_count[ci][c] == 5: for the one unassigned edge in ci, forbid color c
        #     If c6_color_count[ci][c] == 4: for the two unassigned edges, at least one must not be c
        #       (handled implicitly)
        
        # We need to know, for each C6, which edges are unassigned
        # This is expensive to maintain. Alternative: just check at assignment time.
        
        # Simpler approach: at each step, when assigning edge e color c,
        # check all C6s containing e: if any has all 6 edges now color c, fail.
        # Also do forward check: for each C6 containing e, if 5 edges are color c
        # and one is unassigned, mark that unassigned edge as not-allowed-c.
        
        # Let's use a simpler but effective backtracking:
        # Order edges, try colors 0-3, check for immediate C6 completion.
        
        # Variable ordering: process edges in a fixed order (lexicographic)
        # but use MRV (minimum remaining values) for better pruning
        
        assignment = [None] * num_edges
        # c6_edge_colors[ci] = list of (edge_idx, color) for assigned edges in this C6
        # Simpler: just re-check by looking at assignment array
        
        def check_c6_violation(ci, e, c):
            """Check if assigning edge e color c completes a C6 ci."""
            for e2 in c6s[ci]:
                if e2 == e:
                    continue
                if assignment[e2] is None or assignment[e2] != c:
                    return False
            return True  # All other 5 edges are color c, so adding e makes 6
        
        def forward_check(e, c):
            """After assigning edge e color c, check if any unassigned edge is forced.
            Returns (ok, forced_forbidden) where forced_forbidden is dict edge->set of forbidden colors."""
            # For each C6 containing e:
            #   Count how many edges in this C6 are now color c
            #   If count == 5 and one edge is unassigned, that edge cannot be color c
            #   If count == 6, violation (shouldn't happen)
            for ci in edge_to_c6s[e]:
                count_c = 1  # edge e is color c
                unassigned_edges = []
                for e2 in c6s[ci]:
                    if e2 == e:
                        continue
                    if assignment[e2] is None:
                        unassigned_edges.append(e2)
                    elif assignment[e2] == c:
                        count_c += 1
                if count_c >= 6:
                    return (False, None)
                if count_c == 5 and len(unassigned_edges) == 1:
                    ue = unassigned_edges[0]
                    # ue cannot be color c
                    if not edge_possible[ue][c]:
                        return (False, None)  # already forbidden, no new info
                    # Mark it - but we don't want to modify state yet, just check
                    # Actually for simplicity, let's just check at assignment time
                    pass
            return (True, None)
        
        # Given the complexity, let's use a simpler but effective approach:
        # Backtracking with immediate C6 check (no full forward checking)
        # but with good variable ordering and symmetry breaking
        
        # Symmetry breaking: first edge must be color 0
        # Second edge (0,1) and (0,2): break more symmetry
        
        # Let's just do straightforward backtracking with C6 check
        # The key insight: for n=17, there are C(17,6)*60 C6s = 12376*60 = 742560 C6s
        # That's a lot. Let's check feasibility.
        
        # Actually, let me reconsider. The number of C6s in K_17:
        # C(17,6) = 12376, each 6-set has 60 distinct cycles, so 742560 C6s.
        # For each edge, it's in C(15,4)*4*5 = 1365*20 = 27300 C6s... that's a lot.
        # Actually: an edge (u,v) is in a C6 if we pick 4 more vertices from the remaining 15,
        # and arrange them in a cycle with u,v. The number of C6s containing a specific edge:
        # Choose 4 from 15: C(15,4) = 1365. For each such 4-set, the number of C6s
        # containing edge (u,v): we need to arrange 6 vertices in a cycle with u,v adjacent.
        # Total C6s on 6 vertices = 60. Fraction with u,v adjacent: 
        # In a random cycle of 6, probability u,v adjacent = 2*4!/4! = 2/5... 
        # Actually: fix u at position 0. v can be at position 1 or 5 (2 choices out of 5 remaining positions).
        # So 2/5 of 60 = 24 C6s per 6-set contain a given edge.
        # So edge is in 1365 * 24 = 32760 C6s. That's too many to check per assignment.
        
        # Alternative: don't precompute all C6s. Instead, check for C6 completion locally.
        # When assigning edge (u,v) color c, check if there exists a path u-a-b-d-e-v of length 5
        # where all edges are color c. This is equivalent to: in the subgraph of color-c edges,
        # is there a path of length 5 between u and v?
        
        # Even better: maintain the color-c subgraph incrementally.
        # When adding edge (u,v) to color c, check if there's already a path of length 4
        # from u to v in the color-c graph (which would make a C6 when we add (u,v)).
        # Path of length 4: u-a-b-d-v where all edges exist in color-c graph.
        
        # This is: for each neighbor a of u in color-c graph, for each neighbor b of a (b!=u),
        # for each neighbor d of b (d!=u,a), check if d is neighbor of v in color-c graph.
        # That's potentially expensive but much less than checking all C6s.
        
        # Let me implement this approach:
        
        # adj_c[c][v] = set of neighbors of v in color-c subgraph
        adj_c = [set() for _ in range(4)]  # will be lists of sets
        
        def would_create_c6(c, u, v):
            """Check if adding edge (u,v) to color-c subgraph creates a C6.
            A C6 is created if there's already a path of length 5 from u to v,
            i.e., a path u-a-b-d-e-v where all edges are in color-c subgraph."""
            # Path of length 5 from u to v: u-a-b-d-e-v
            # Equivalently: there exists a path of length 4 from u to some neighbor of v
            # Let's do: for each a in adj_c[c][u], for each b in adj_c[c][a] (b != u),
            #   for each d in adj_c[c][b] (d != u, a), for each e in adj_c[c][d] (e != u, a, b),
            #     check if e == v or v in adj_c[c][e]
            # Wait, that's path length 4 from u to e, then e-v makes length 5.
            # Actually we need path u-a-b-d-e-v of exactly 5 edges, all distinct vertices.
            
            # More carefully: we need 5 distinct intermediate... no, 6 distinct vertices total.
            # u, a, b, d, e, v all distinct, and edges (u,a),(a,b),(b,d),(d,e),(e,v) all in color-c.
            
            # For efficiency: BFS/DFS of depth 4 from u in color-c graph, avoiding v,
            # and check if we can reach a neighbor of v.
            
            # Depth 1: neighbors of u (excluding v)
            # Depth 2: neighbors of depth-1 nodes (excluding u, v, and other depth-1 nodes... 
            #   actually just excluding previously visited)
            # Depth 3, 4 similarly.
            # At depth 4, check if current node is adjacent to v.
            
            visited = {u, v}  # v is the target, don't go through it
            # Level 1
            level1 = set()
            for a in adj_c[c][u]:
                if a != v and a not in visited:
                    level1.add(a)
                    visited.add(a)
            if not level1:
                return False
            # Level 2
            level2 = set()
            for a in level1:
                for b in adj_c[c][a]:
                    if b not in visited:
                        level2.add(b)
                        visited.add(b)
            if not level2:
                return False
            # Level 3
            level3 = set()
            for b in level2:
                for d in adj_c[c][b]:
                    if d not in visited:
                        level3.add(d)
                        visited.add(d)
            if not level3:
                return False
            # Level 4
            for d in level3:
                for e in adj_c[c][d]:
                    if e not in visited:
                        if e == v or v in adj_c[c][e]:
                            return True
            return False
        
        # This is O(degree^4) per check which could be expensive for dense graphs.
        # But early in the search, graphs are sparse.
        
        # Let me try a different approach: use the known structure.
        # For n=17, we know a solution exists. Let me try a more targeted search.
        
        # Actually, let me just implement a clean backtracking with the local C6 check.
        
        def backtrack(n):
            num_edges = n * (n - 1) // 2
            assignment = [-1] * num_edges
            adj_c = [set() for _ in range(4)]
            
            # Edge list in order
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append((i, j))
            
            def check_and_add(c, u, v, eidx):
                """Check if adding edge (u,v) with color c creates a C6. If not, add it."""
                if would_create_c6(c, u, v):
                    return False
                adj_c[c].add(u)  # wrong, need to add to sets
                # Fix:
                return True
            
            # Let me redo this more carefully
            adj_c = [set() for _ in range(4)]  # adj_c[c] is set of edges? No, adj_c[c][v] = neighbors
            
            # Better: adj_c[c] is a list of sets
            adj_c = [set() for _ in range(4)]
            # Actually let's use: adj_c[c] = [set() for _ in range(n)]
            
            def would_create_c6(c, u, v):
                """Check if adding edge (u,v) to color-c graph creates a C6."""
                # Need path of length 5 from u to v in current color-c graph
                # with all 6 vertices distinct
                # BFS from u, depth up to 4, avoiding v
                # At depth 4, check adjacency to v
                
                from collections import deque
                # State: (vertex, depth, visited_set) - but visited_set is expensive
                # Alternative: since max depth is 4 and n <= 19, we can do DFS with visited
                # Use iterative DFS with explicit stack
                
                # For each neighbor a of u (a != v):
                #   For each neighbor b of a (b != u, v):
                #     For each neighbor d of b (d != u, a, v):
                #       For each neighbor e of d (e != u, a, b, v):
                #         If e == v or v in adj_c[c][e]: return True
                #         (e == v means path u-a-b-d-e-v where e=v, but we excluded v from neighbors)
                #         Actually we need e adjacent to v, and e != v
                #         So: if v in adj_c[c][e] and e not in {u,a,b,d}: return True
                #         Wait, e is already not in {u,a,b} by construction, and e != d (neighbor)
                
                for a in adj_c[c][u]:
                    if a == v:
                        continue
                    for b in adj_c[c][a]:
                        if b == u or b == v or b == a:
                            continue
                        for d in adj_c[c][b]:
                            if d == u or d == a or d == b or d == v:
                                continue
                            for e in adj_c[c][d]:
                                if e == u or e == a or e == b or e == d or e == v:
                                    continue
                                if v in adj_c[c][e]:
                                    return True
                return False
            
            # This is O(deg^4) which for n=17 with ~4 colors means avg degree ~ (17-1)/4 * ... 
            # Actually each vertex has (n-1) edges total, split among 4 colors, so avg degree per color ~ 4
            # So deg^4 ~ 256 per check, times num_edges ~ 136, total ~ 35000 per level.
            # With backtracking depth 136, this could be very slow.
            
            # Let me try a completely different approach: use the known answer for n=17
            # and verify it, then try to find n=18 with a smarter method.
            
            # Actually, let me just try the cyclic construction more carefully for n=17,
            # and use a SAT-like approach for n=18.
            
            return None
        
        # Let me simplify: try multiple strategies in order of likelihood
        # 1. Cyclic construction for n=17 (fast if it works)
        # 2. Backtracking with local C6 check for n=17
        # 3. Backtracking for n=18 (if time permits)
        
        return None
    
    def try_cyclic(n, num_trials=5000):
        """Try cyclic 4-coloring of K_n where n is odd."""
        if n % 2 == 0:
            return None
        
        m = (n - 1) // 2  # each color gets m distances
        # We need to partition {1, ..., n-1} into 4 sets of size m
        # such that no color class contains a C6
        
        # In the cyclic coloring, color c connects i to i±s for s in S_c
        # A C6 in color c exists if there are s1,s2,s3,s4,s5,s6 in S_c (with signs)
        # such that s1±s2±s3±s4±s5±s6 ≡ 0 mod n with all partial sums distinct
        
        # Simpler: just build the graph and check for C6
        
        elements = list(range(1, n))
        
        for trial in range(num_trials):
            random.shuffle(elements)
            sets = [frozenset(elements[i*m:(i+1)*m]) for i in range(4)]
            
            valid = True
            for c in range(4):
                S = sets[c]
                # Build adjacency
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        j1 = (i + s) % n
                        j2 = (i - s) % n
                        adj[i].add(j1)
                        adj[i].add(j2)
                # Check for C6
                found = False
                for a0 in range(n):
                    if found:
                        break
                    for a1 in adj[a0]:
                        if found or a1 <= a0:
                            continue
                        for a2 in adj[a1]:
                            if found or a2 == a0:
                                continue
                            for a3 in adj[a2]:
                                if found or a3 in (a0, a1):
                                    continue
                                for a4 in adj[a3]:
                                    if found or a4 in (a0, a1, a2):
                                        continue
                                    for a5 in adj[a4]:
                                        if a5 in (a0, a1, a2, a3):
                                            continue
                                        if a0 in adj[a5]:
                                            found = True
                                            break
                if found:
                    valid = False
                    break
            
            if valid:
                # Build color assignment
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = (j - i) % n
                        for c in range(4):
                            if d in sets[c] or (n - d) in sets[c]:
                                colors.append(c)
                                break
                return colors
        return None
    
    def backtrack_local(n, time_limit):
        """Backtracking with local C6 check using adjacency sets."""
        num_edges = n * (n - 1) // 2
        assignment = [-1] * num_edges
        adj_c = [set() for _ in range(4)]  # adj_c[c] = list of sets? No.
        # Let's use: adj_c[c] is a list of sets, adj_c[c][v] = set of neighbors of v in color c
        adj_c = [ [set() for _ in range(n)] for _ in range(4) ]
        
        def would_create_c6(c, u, v):
            """Check if adding edge (u,v) to color-c graph creates a C6."""
            # Need a path of length 5 from u to v with 6 distinct vertices
            # u-a-b-d-e-v, all edges in color c, all vertices distinct
            for a in adj_c[c][u]:
                if a == v:
                    continue
                for b in adj_c[c][a]:
                    if b in (u, v, a):
                        continue
                    for d in adj_c[c][b]:
                        if d in (u, a, b, v):
                            continue
                        for e in adj_c[c][d]:
                            if e in (u, a, b, d, v):
                                continue
                            if v in adj_c[c][e]:
                                return True
            return False
        
        # Edge order: try to use a good ordering
        # Process edges in lexicographic order
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Symmetry breaking: first few edges
        # Edge (0,1) = 0, edge (0,2) in {0,1}, etc.
        
        def solve(pos):
            if time.time() > time_limit:
                return None
            if pos == num_edges:
                return assignment[:]
            
            u, v = edges[pos]
            
            # Try colors 0-3
            for c in range(4):
                if would_create_c6(c, u, v):
                    continue
                # Assign
                assignment[pos] = c
                adj_c[c][u].add(v)
                adj_c[c][v].add(u)
                
                result = solve(pos + 1)
                if result is not None:
                    return result
                
                # Unassign
                assignment[pos] = -1
                adj_c[c][u].discard(v)
                adj_c[c][v].discard(u)
            
            return None
        
        # Add symmetry breaking: force edge (0,1) to be color 0
        # This reduces search by factor of 4
        assignment[0] = 0
        adj_c[0][0].add(1)
        adj_c[0][1].add(0)
        
        result = solve(1)
        return result
    
    # Strategy: try n=18 first with backtracking (if we find it, great),
    # then n=17 with cyclic + backtracking, then fallback
    
    best_result = None
    
    # Try n=18 with limited backtracking
    if time.time() < deadline - 60:
        print("Trying n=18 with backtracking...", flush=True)
        colors = backtrack_local(18, deadline - 60)
        if colors is not None and verify_coloring(colors, 18):
            return {'n': 18, 'colors': colors}
    
    # Try n=17
    if time.time() < deadline - 30:
        print("Trying n=17 cyclic...", flush=True)
        colors = try_cyclic(17, num_trials=3000)
        if colors is not None and verify_coloring(colors, 17):
            return {'n': 17, 'colors': colors}
    
    if time.time() < deadline - 10:
        print("Trying n=17 backtracking...", flush=True)
        colors = backtrack_local(17, deadline - 10)
        if colors is not None and verify_coloring(colors, 17):
            return {'n': 17, 'colors': colors}
    
    # Try n=16
    if time.time() < deadline - 5:
        colors = try_cyclic(15, num_trials=2000)  # 15 is odd
        if colors is not None and verify_coloring(colors, 15):
            return {'n': 15, 'colors': colors}
        colors = backtrack_local(16, deadline - 5)
        if colors is not None and verify_coloring(colors, 16):
            return {'n': 16, 'colors': colors}
    
    # Fallback: n=9 cyclic
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
