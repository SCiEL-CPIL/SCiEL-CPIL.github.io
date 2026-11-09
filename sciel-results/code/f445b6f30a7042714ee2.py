import sys
import time
import random
from itertools import combinations, permutations

P = {'n': 15}

def _construct_graph(n):
    """
    Build a valid graph on n vertices (15) maximizing edge count.
    Constraint: no simple 3-edge path s-u-v-t (4 distinct vertices) has
    deg(s) == deg(t) in the full graph.
    
    Strategy:
    1. Structured construction: partition vertices into degree classes where
       no two vertices of the same degree are at distance 3.
       - Within a class: make it a clique (dist 1) or independent set (dist >= 2)
       - Between classes: ensure no distance-3 path connects same-degree pairs
    2. Local search: remove edges causing violations, then add edges greedily.
    """
    start_time = time.time()
    deadline = 82.0  # 8s safety margin before 90s kill
    
    def compute_deg(adj_bits, n):
        return [adj_bits[i].bit_count() for i in range(n)]
    
    def count_edges(adj_bits, n):
        total = 0
        for i in range(n):
            total += adj_bits[i].bit_count()
        return total // 2
    
    def find_violation(adj_bits, n):
        """Return (s, t, u, v) if a violation exists, else None."""
        deg = compute_deg(adj_bits, n)
        from collections import defaultdict
        deg_groups = defaultdict(list)
        for i in range(n):
            deg_groups[deg[i]].append(i)
        
        for d, group in deg_groups.items():
            if len(group) < 2:
                continue
            for s, t in combinations(group, 2):
                # Check if there's a path s-u-v-t with all 4 distinct
                # s adjacent to u, u adjacent to v, v adjacent to t
                # u != s, u != t, v != s, v != t, u != v
                for u in range(n):
                    if u == s or u == t:
                        continue
                    if not (adj_bits[s] >> u) & 1:
                        continue
                    # v must be neighbor of both u and t
                    candidates_v = adj_bits[u] & adj_bits[t]
                    candidates_v &= ~(1 << s)
                    candidates_v &= ~(1 << t)
                    candidates_v &= ~(1 << u)
                    if candidates_v:
                        v = (candidates_v & -candidates_v).bit_length() - 1
                        return (s, t, u, v)
        return None
    
    def is_valid(adj_bits, n):
        return find_violation(adj_bits, n) is None
    
    def remove_edge(adj_bits, i, j):
        adj_bits[i] &= ~(1 << j)
        adj_bits[j] &= ~(1 << i)
    
    def add_edge(adj_bits, i, j):
        adj_bits[i] |= (1 << j)
        adj_bits[j] |= (1 << i)
    
    def greedy_remove_until_valid(adj_bits, n, max_iter=500):
        """Remove edges until valid. Returns (adj_bits, valid)."""
        for _ in range(max_iter):
            if time.time() > deadline:
                return adj_bits, is_valid(adj_bits, n)
            viol = find_violation(adj_bits, n)
            if viol is None:
                return adj_bits, True
            s, t, u, v = viol
            # Remove edge (s,u) to break the path
            remove_edge(adj_bits, s, u)
        return adj_bits, is_valid(adj_bits, n)
    
    def greedy_add_edges(adj_bits, n):
        """Try to add edges while maintaining validity. Returns adj_bits."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    return adj_bits
                for j in range(i+1, n):
                    if (adj_bits[i] >> j) & 1:
                        continue
                    # Try adding edge (i, j)
                    add_edge(adj_bits, i, j)
                    if is_valid(adj_bits, n):
                        improved = True
                    else:
                        remove_edge(adj_bits, i, j)
        return adj_bits
    
    best_edges = []
    best_count = -1
    
    def update_best(adj_bits):
        nonlocal best_edges, best_count
        if is_valid(adj_bits, n):
            c = count_edges(adj_bits, n)
            if c > best_count:
                best_count = c
                best_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj_bits[i] >> j) & 1:
                            best_edges.append([i, j])
    
    # === Structured Construction ===
    # Key insight: if we partition vertices into groups where each group has
    # a distinct degree, and within each group the vertices form either a clique
    # or independent set (so no two same-degree vertices are at distance 2 within
    # the group), we need to ensure cross-group edges don't create distance-3
    # paths between same-degree vertices.
    
    # Approach: Try different degree assignments.
    # For 15 vertices, assign degrees so that:
    # - Vertices with the same degree are "far apart" (not connected by 3-edge path)
    
    # Pattern 1: Two groups - a clique of size a and independent set of size (15-a)
    # Clique vertices have high degree, independent set vertices have low degree
    # If all clique vertices have the same degree and all independent set vertices
    # have the same degree, we need:
    #   - No 3-edge path between two clique vertices
    #   - No 3-edge path between two independent set vertices
    
    # For clique of size a: each clique vertex connects to all other (a-1) clique vertices
    #   and some independent set vertices. Two clique vertices s,t:
    #   Path s-u-v-t: u must be neighbor of s. If u is in clique, u is also neighbor of t.
    #   Then v must be neighbor of u and t. If v is in clique, v is neighbor of both.
    #   So s-u-v-t exists if there are clique vertices u,v != s,t.
    #   This means: if a >= 4, two clique vertices s,t have path s-u-v-t where u,v are
    #   other clique vertices. So clique size must be <= 3 to avoid this.
    
    # Actually wait: s-u-v-t requires u != v, u != s, u != t, v != s, v != t.
    # If clique has size a, and s,t are in clique, then for u,v in clique:
    #   u != s, u != t, v != s, v != t, u != v
    #   We need at least 2 other clique vertices besides s and t, so a >= 4.
    #   If a <= 3, no such u,v exist in the clique.
    #   But u could be in independent set!
    #   s-u-v-t: s in clique, u in indep set, v must be neighbor of u and t.
    #   u is in indep set, so u's neighbors are only in clique (if we connect them).
    #   v must be neighbor of u (so v in clique) and neighbor of t (t in clique, so v in clique works).
    #   So v in clique, v != s, v != t, v != u (u in indep set so v != u automatically).
    #   We need v in clique, v != s, v != t. If a >= 3, such v exists.
    #   So even with a=3, if s connects to some indep vertex u, and u connects to some clique vertex v != s,t,
    #   then s-u-v-t is a valid path.
    
    # This is getting complex. Let me try a different structured approach.
    
    # Pattern 2: All vertices have distinct degrees.
    # If all 15 vertices have distinct degrees, the constraint is trivially satisfied!
    # Degrees range from 0 to 14. Can we have a graph where all 15 vertices have distinct degrees?
    # In a graph on 15 vertices, degrees are 0..14. A graph with all distinct degrees would have
    # degrees 0,1,2,...,14. But this is impossible: if vertex 0 has degree 0 (isolated) and
    # vertex 14 has degree 14 (connected to all), then vertex 14 is connected to vertex 0,
    # contradicting vertex 0 having degree 0.
    
    # So we can't have all distinct. But we can have at most 14 distinct degrees.
    # Let's try: degrees 0,1,2,...,13,13 (one pair with same degree 13).
    # Or: 1,2,3,...,14,14 (one pair with same degree 14).
    # Or: 0,1,2,...,12,13,13,14 (two pairs).
    
    # Actually, the key constraint is: no two vertices of the SAME degree are at distance 3.
    # If we can ensure that vertices with the same degree are at distance 1 (adjacent) or
    # distance 2 (share a common neighbor) but NOT distance 3, we're fine.
    
    # Simplest: make all same-degree vertices adjacent (clique). Then they're at distance 1.
    # But we also need to check cross-group interactions.
    
    # Let me try: partition into groups by degree, make each group a clique.
    # Between groups, connect freely but check for violations.
    
    # Try: 15 vertices, assign degrees as evenly as possible but with distinct values where possible.
    # Actually, let's just try several structured patterns and use local search.
    
    # Pattern A: Complete bipartite-like structure
    # Split into two sets A and B. Connect all A-A (clique), all B-B (clique), 
    # and some A-B edges.
    
    # Pattern B: Random with guided degree assignment
    
    # Let's try multiple structured starts:
    
    # Start 1: Two cliques (sizes 7 and 8) with no cross edges
    # Clique of 7: each vertex has degree 6 within clique
    # Clique of 8: each vertex has degree 7 within clique
    # All degrees distinct between groups (6 vs 7), so no same-degree pairs across groups.
    # Within clique of 7: all have degree 6. Check distance-3 paths within clique:
    #   s-u-v-t in K7: need 4 distinct vertices. K7 has 7 vertices, so yes possible.
    #   s,u,v,t all in K7: s adjacent u, u adjacent v, v adjacent t. All edges exist.
    #   So this is INVALID if any clique has size >= 4.
    
    # So cliques of size >= 4 are problematic for same-degree vertices.
    # Solution: make same-degree vertices NOT form a clique, but rather an independent set.
    # If same-degree vertices form an independent set, they're at distance >= 2.
    # Distance 2: s-w-t where w is common neighbor. That's fine (not distance 3).
    # Distance 3: s-u-v-t. Need to avoid this.
    
    # Pattern: Independent sets for each degree class, with careful cross connections.
    
    # Let me try a concrete construction:
    # 15 vertices. Assign degrees: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
    # But as noted, 0 and 14 can't coexist.
    # Try: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,14
    # Or: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,13
    
    # Let me try a different approach: use the fact that we can have at most
    # a few vertices per degree class, and make them independent sets.
    
    # Construction attempt:
    # - 15 vertices
    # - Give them degrees: try to maximize total edges = sum(degrees)/2
    # - Subject to: no two same-degree vertices at distance 3
    
    # Let's try: 1 vertex of each degree 0-14, but replace 0 and 14 with something valid.
    # Actually, let's just do local search from several good starts.
    
    # Start from various structured graphs:
    
    starts = []
    
    # Start 1: Path graph (all degrees <= 2, mostly distinct)
    # Path: 0-1-2-...-14. Degrees: 1,2,2,2,2,2,2,2,2,2,2,2,2,2,1
    # Many vertices with degree 2. Need to check distance-3 paths.
    # In a path, distance between vertices is |i-j|. Distance 3 means |i-j|=3.
    # Vertices with degree 2: indices 1-13. Two such vertices at distance 3: e.g., 1 and 4.
    # Path 1-2-3-4: s=1(deg2), u=2, v=3, t=4(deg2). This is a violation!
    # So a path is invalid.
    
    # Start 2: Star (one center degree 14, 14 leaves degree 1)
    # Center: degree 14. Leaves: degree 1.
    # Two leaves s,t: path s-center-?-t. Need u=center, v=? where v adjacent to center and t.
    # v must be a leaf (only leaves adjacent to center besides center itself, but center not adjacent to itself).
    # v adjacent to t: t is a leaf, only adjacent to center. So v must be center, but v != u=center. Contradiction.
    # So no 3-edge path between two leaves. Valid!
    # Edges: 14. Not great but valid.
    
    adj_star = [0] * n
    for v in range(1, n):
        add_edge(adj_star, 0, v)
    starts.append(adj_star.copy())
    
    # Start 3: Two stars (centers 0 and 1, shared no leaves)
    # 0 connected to 2,3,4,5,6,7,8 (7 edges)
    # 1 connected to 9,10,11,12,13,14 (6 edges)
    # Plus edge (0,1)
    # Degrees: 0:8, 1:7, 2-8:1, 9-14:1
    # Leaves 2-14 all degree 1. Check distance-3 between two leaves.
    # Leaves in same star: e.g., 2 and 3. Path 2-0-?-3. ? must be adj to 0 and 3.
    # 3 only adj to 0. So ? = 0, but ? != u=0. No.
    # Leaves in different stars: e.g., 2 and 9. Path 2-0-1-9. s=2(deg1), u=0, v=1, t=9(deg1).
    # All distinct: 2,0,1,9. Yes! This is a violation.
    # So this is invalid.
    
    # Start 4: Complete graph K15 (all degree 14) - definitely invalid, but good for removal search
    adj_complete = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            add_edge(adj_complete, i, j)
    starts.append(adj_complete.copy())
    
    # Start 5: Random dense
    random.seed(123)
    for density in [0.7, 0.8, 0.9]:
        adj_r = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    add_edge(adj_r, i, j)
        starts.append(adj_r.copy())
    
    # Start 6: Structured - try to create graph with few degree classes
    # Idea: make a bipartite graph with parts of size 7 and 8.
    # Part A: 0-6, Part B: 7-14
    # Connect all A-B edges (complete bipartite K7,8)
    # Degrees: A vertices: 8, B vertices: 7
    # Two A vertices s,t (both degree 8): path s-u-v-t where u in B, v in B.
    # s adj u (yes, complete bipartite), u adj v? u,v both in B, no edges within B.
    # So no path. Valid within same part!
    # Two B vertices s,t (both degree 7): path s-u-v-t where u in A, v in A.
    # s adj u (yes), u adj v? u,v both in A, no edges within A. No path. Valid!
    # Cross-part: A vertex (deg 8) and B vertex (deg 7): different degrees, no constraint.
    # So K7,8 is VALID! Edges: 7*8 = 56.
    
    adj_bip = [0] * n
    for i in range(7):
        for j in range(7, 15):
            add_edge(adj_bip, i, j)
    starts.append(adj_bip.copy())
    
    # Start 7: K7,8 plus some within-part edges
    # Add edges within A: make A a clique. Then A vertices have degree 8+6=14.
    # But then two A vertices s,t: path s-u-v-t. u,v in A: s adj u (clique), u adj v (clique), v adj t (clique).
    # All in A, 4 distinct. Violation!
    # So can't make A a clique.
    # Add edges within B: make B a clique. B vertices: degree 7+7=14.
    # Two B vertices: similar violation.
    # Add one edge within A: say (0,1). Now 0 and 1 have degree 9, others in A have degree 8.
    # 0 and 1 (both degree 9): path 0-u-v-1. u,v in A or B.
    # If u,v in A: 0 adj u, u adj v, v adj 1. In A (size 7), with edge (0,1) added:
    #   0 adj 1, and all B. 1 adj 0, and all B.
    #   u in A, u != 0,1: 0 adj u? u in A, 0 only adj 1 in A. So 0 not adj u. No.
    #   u in B: 0 adj u (yes). v in B: u adj v? B is independent. No.
    #   v in A: u in B, v in A: u adj v (yes, complete bipartite). v adj 1? v in A, 1 adj 0 in A and all B.
    #   1 not adj v (v in A, v != 0,1). No.
    # So no violation for 0,1. Good.
    # But now 0,1 have degree 9, other A vertices have degree 8.
    # Two A vertices with degree 8: e.g., 2 and 3. Path 2-u-v-3.
    # u in B: 2 adj u. v in B: u adj v? No (B independent).
    # v in A: u in B, v in A: u adj v (yes). v adj 3? v in A, 3 adj all B and no A (except if we added edges).
    # 3 not adj v (v in A, v != 3, no within-A edges for 3). No.
    # u in A: 2 adj u? 2 only adj B and no A. No.
    # So no violation. 
    # Let's add all within-A edges except those creating violations.
    # Actually, let's just use local search to optimize.
    
    # Start 8: K7,8 with a matching within A and within B
    adj_k78_match = adj_bip.copy()
    # Add matching in A: (0,1), (2,3), (4,5)
    for i in [0,2,4]:
        add_edge(adj_k78_match, i, i+1)
    # Add matching in B: (7,8), (9,10), (11,12), (13,14)
    for i in [7,9,11,13]:
        add_edge(adj_k78_match, i, i+1)
    starts.append(adj_k78_match.copy())
    
    # Start 9: K7,8 with full matching in A and B
    adj_k78_full = adj_bip.copy()
    # A: (0,1),(2,3),(4,5) - 6th vertex 6 has no match
    for i in [0,2,4]:
        add_edge(adj_k78_full, i, i+1)
    # B: (7,8),(9,10),(11,12),(13,14)
    for i in [7,9,11,13]:
        add_edge(adj_k78_full, i, i+1)
    starts.append(adj_k78_full.copy())
    
    # Process each start: remove violations, then add edges greedily
    for idx, adj_start in enumerate(starts):
        if time.time() > deadline:
            break
        
        adj_work = adj_start.copy()
        
        # Phase 1: Remove edges to make valid
        adj_work, valid = greedy_remove_until_valid(adj_work, n)
        
        if valid:
            # Phase 2: Add edges greedily
            adj_work = greedy_add_edges(adj_work, n)
            update_best(adj_work)
    
    # Additional random restarts with local search
    random.seed(456)
    for trial in range(100):
        if time.time() > deadline:
            break
        
        # Random graph with varying density
        density = 0.5 + random.random() * 0.4  # 0.5 to 0.9
        adj_r = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    add_edge(adj_r, i, j)
        
        # Remove violations
        adj_r, valid = greedy_remove_until_valid(adj_r, n)
        
        if valid:
            # Add edges
            adj_r = greedy_add_edges(adj_r, n)
            update_best(adj_r)
    
    # Try more structured patterns
    if time.time() < deadline:
        # Pattern: Complete multipartite
        # K_{a,b,c} where a+b+c=15
        # Vertices in same part: independent set (no edges)
        # Vertices in different parts: all edges
        # Degrees: vertex in part of size a has degree (15-a)
        # Two vertices in same part: same degree. Distance between them: 2 (via any vertex in other part)
        # Distance 3: s-u-v-t. s,t in same part. u in other part, v in other part.
        # s adj u (yes, different parts). u adj v? If u,v in same other part, no. If different, yes.
        # v adj t (yes, different parts).
        # So if there are at least 2 other parts, and u,v are in different parts, we get a violation.
        # For 3 parts: s in part1, u in part2, v in part3, t in part1. s adj u, u adj v, v adj t. All distinct. Violation!
        # For 2 parts (bipartite): s,t in part1. u in part2, v in part2. u adj v? No (same part). No violation.
        # So complete bipartite is valid, complete tripartite is not.
        
        # Try complete bipartite with various splits:
        for a in range(1, 8):
            b = n - a
            adj_cp = [0] * n
            for i in range(a):
                for j in range(a, n):
                    add_edge(adj_cp, i, j)
            adj_cp, valid = greedy_remove_until_valid(adj_cp, n)
            if valid:
                adj_cp = greedy_add_edges(adj_cp, n)
                update_best(adj_cp)
    
    if best_count > 0:
        return best_edges
    
    # Fallback: star
    return [[0, v] for v in range(1, n)]

def construct():
    edges = _construct_graph(P['n'])
    # Ensure types
    result = []
    for e in edges:
        u, v = int(e[0]), int(e[1])
        result.append([u, v])
    return result
