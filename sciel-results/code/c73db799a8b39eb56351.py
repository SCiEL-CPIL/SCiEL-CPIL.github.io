"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve_one(n, k, p, seed):
        if time.time() > DEADLINE - 5:
            # Fallback: return edge list
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in edges]
        
        rng = Random(seed)
        # Build adjacency
        adj = [set() for _ in range(n)]
        # C-C edges
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # All edges
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))
        
        # Separate C-C and cross edges
        cc_edges = set()
        cross_edges = set()
        for (u, v) in all_edges:
            if u < k and v < k:
                cc_edges.add((u, v))
            elif u < k and v >= k:
                cross_edges.add((u, v))
            elif v < k and u >= k:
                cross_edges.add((v, u))
        
        # I-vertices sorted by neighborhood size in C (ascending)
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda v: len(adj[v] & set(range(k))))
        
        # For each I-vertex, its neighborhood in C
        i_neigh = {}
        for v in i_vertices:
            i_neigh[v] = adj[v] & set(range(k))
        
        # Greedy: assign I-vertices to cliques
        # Each clique is (C_part: set of C-vertices, I_part: set of I-vertices)
        # A clique is valid if C_part is a clique in G[C] and all I-vertices in I_part
        # are adjacent to all C-vertices in C_part
        
        cliques = []  # list of (C_set, I_set)
        
        def can_add_i(clique_C, clique_I, v):
            """Check if I-vertex v can be added to this clique"""
            # v must be adjacent to all C-vertices in the clique
            c_set = set(clique_C)
            for c in c_set:
                if c not in adj[v]:
                    return False
            return True
        
        def get_clique_edges(clique_C, clique_I):
            """Get all edges covered by this clique"""
            edges = set()
            c_list = list(clique_C)
            i_list = list(clique_I)
            # C-C edges
            for i in range(len(c_list)):
                for j in range(i+1, len(c_list)):
                    edges.add((min(c_list[i], c_list[j]), max(c_list[i], c_list[j])))
            # Cross edges
            for c in c_list:
                for iv in i_list:
                    edges.add((min(c, iv), max(c, iv)))
            return edges
        
        # Phase 1: Greedily assign I-vertices
        # For each I-vertex (sorted by neigh size asc), try to add to existing clique
        # If no existing clique works, create a new one with the largest possible C-subset
        
        unassigned_i = set(i_vertices)
        
        for v in i_vertices:
            if time.time() > DEADLINE - 10:
                break
            neigh = i_neigh[v]
            
            # Try to add to existing cliques (prefer cliques where it fits and maximizes edges)
            best_clique = -1
            best_gain = 0
            for idx, (C_set, I_set) in enumerate(cliques):
                if can_add_i(C_set, I_set, v):
                    gain = len(C_set)  # number of new cross edges
                    if gain > best_gain:
                        best_gain = gain
                        best_clique = idx
            
            if best_clique >= 0:
                cliques[best_clique] = (cliques[best_clique][0], cliques[best_clique][1] | {v})
            else:
                # Create new clique: use all of neigh as C-part (they form a clique since C is a clique)
                cliques.append((set(neigh), {v}))
        
        # Phase 2: Handle C-C edges not yet covered
        covered_edges = set()
        for C_set, I_set in cliques:
            covered_edges |= get_clique_edges(C_set, I_set)
        
        # Find uncovered C-C edges
        uncovered_cc = cc_edges - covered_edges
        
        # For uncovered C-C edges, we need to add them to cliques or create new ones
        # A C-C edge (u,v) can be added to a clique if both u,v are in the clique's C-part
        # or we create a new clique {u, v}
        
        # Try to merge: for each uncovered C-C edge, see if we can extend an existing clique
        for (u, v) in sorted(uncovered_cc):
            if time.time() > DEADLINE - 10:
                break
            # Check if u and v are already in some clique's C-part
            found = False
            for idx, (C_set, I_set) in enumerate(cliques):
                if u in C_set and v in C_set:
                    # Edge already covered? Shouldn't be uncovered then
                    found = True
                    break
                elif u in C_set or v in C_set:
                    # Can we add the other?
                    if u in C_set and v not in C_set:
                        # Check v is adjacent to all in C_set (it is, since C is clique)
                        cliques[idx] = (C_set | {v}, I_set)
                        found = True
                        break
                    elif v in C_set and u not in C_set:
                        cliques[idx] = (C_set | {u}, I_set)
                        found = True
                        break
            
            if not found:
                # Create new clique with just {u, v}
                cliques.append(({u, v}, set()))
        
        # Phase 3: Recompute covered edges and fix any issues
        # Verify all edges are covered exactly once
        # Actually, we need to ensure NO edge is covered more than once
        # This is the hard part - edge-disjointness
        
        # Let's restart with a cleaner approach:
        # Build the solution more carefully
        
        return None  # Placeholder, will rebuild
    
    # Actually, let me rethink the whole approach.
    # The key constraint is edge-disjointness. Let me use a different strategy.
    
    def solve_instance(n, k, p, seed):
        if time.time() > DEADLINE - 5:
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in edges]
        
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))
        
        cc_edges = set((u,v) for u in range(k) for v in range(u+1,k))
        cross_edges = set()
        for (u,v) in all_edges:
            if not (u < k and v < k):
                cross_edges.add((u,v))
        
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda v: len(adj[v] & set(range(k))))
        
        # Strategy: 
        # 1. For cross edges: group I-vertices by their C-neighborhood
        # 2. For each group, the C-neighborhood forms a clique
        # 3. We can make one clique per (C-subset, I-subset) pair
        # 4. C-C edges are covered by the C-parts of these cliques
        # 5. Any remaining C-C edges get their own 2-cliques
        
        # Better: use the structure directly
        # Each I-vertex v has neighborhood N(v) in C
        # A clique containing v can only use C-vertices from N(v)
        # If we put I-vertices {v1, v2, ...} in a clique, the C-part must be subset of intersection of N(vi)
        
        # Greedy: 
        # - Start with no cliques
        # - For each I-vertex (sorted by |N(v)| ascending), assign to a clique
        # - The clique's C-part is the intersection of neighborhoods of all I-vertices in it
        # - Plus we can add C-vertices to the C-part that are in the intersection
        
        # Actually simpler: 
        # Each clique is determined by (S_C, S_I) where S_C is a clique in C, S_I subset of I
        # such that all of S_I is adjacent to all of S_C
        # The edges covered are: C-C edges within S_C, and cross edges between S_C and S_I
        
        # We need to partition all edges into such cliques.
        
        # Key insight: if we put all I-vertices with the same neighborhood into one clique,
        # and use the full neighborhood as C-part, we cover many edges at once.
        
        # Group I-vertices by their C-neighborhood
        from collections import defaultdict
        groups = defaultdict(list)
        for v in i_vertices:
            key = frozenset(adj[v] & set(range(k)))
            groups[key].append(v)
        
        cliques = []
        covered = set()
        
        # Process groups: for each group with neighborhood N, create clique (N, group)
        # This covers C-C edges in N and cross edges between N and group
        for N, group in groups.items():
            if time.time() > DEADLINE - 10:
                break
            if not N:
                # I-vertex with no C-neighbors: isolated, no edges to cover
                continue
            if not group:
                continue
            cliques.append((set(N), set(group)))
        
        # Now find uncovered edges
        for C_set, I_set in cliques:
            c_list = list(C_set)
            i_list = list(I_set)
            for i in range(len(c_list)):
                for j in range(i+1, len(c_list)):
                    covered.add((min(c_list[i], c_list[j]), max(c_list[i], c_list[j])))
            for c in c_list:
                for iv in i_list:
                    covered.add((min(c, iv), max(c, iv)))
        
        # Uncovered C-C edges
        uncovered_cc = cc_edges - covered
        for (u, v) in sorted(uncovered_cc):
            if time.time() > DEADLINE - 10:
                break
            cliques.append(({u, v}, set()))
        
        # Uncovered cross edges (shouldn't happen if groups are correct, but check)
        uncovered_cross = cross_edges - covered
        for (u, v) in sorted(uncovered_cross):
            if time.time() > DEADLINE - 10:
                break
            # Find which group v is in, check if u is in that group's neighborhood
            # If not, we need a separate clique
            cliques.append(({u, v}, set()))
        
        # Now check for edge overlaps! We need edge-disjointness.
        # The problem: if two cliques share C-vertices, their C-C edges overlap.
        # We need to fix this.
        
        # Better approach: ensure C-parts are disjoint or nested properly.
        # Actually, the simplest correct approach:
        # Make all cliques have disjoint C-parts where possible.
        
        # Let me use a completely different, guaranteed-correct approach:
        # 1. Cover all cross edges first using I-vertex-based cliques
        # 2. Cover remaining C-C edges with 2-cliques
        # 3. Ensure no edge is in two cliques
        
        # For cross edges: each cross edge (c, i) must be in exactly one clique
        # If we put I-vertex i in a clique with C-part S, then all edges (c,i) for c in S are covered
        # So we need to partition the cross edges such that for each I-vertex i,
        # the set of C-vertices paired with i across all cliques forms a partition of N(i)
        
        # Simplest: for each I-vertex i, one clique per C-vertex in N(i)? No, too many.
        # Better: for each I-vertex i, one clique with C-part = N(i)
        # But then C-C edges in N(i) are covered, and if N(i1) and N(i2) overlap,
        # the C-C edges in the overlap are covered twice!
        
        # So we need: for any two cliques, their C-parts must have no common C-C edge,
        # i.e., |C1 ∩ C2| <= 1 for any two cliques with C-parts C1, C2.
        
        # This is getting complex. Let me use a simpler but valid approach:
        # Each clique has C-part of size at most 2 (so C-C edges don't overlap between cliques
        # unless they share exactly one C-vertex, which is fine since a single vertex has no C-C edge)
        
        # Wait, if C1 = {a,b} and C2 = {b,c}, they share vertex b but no C-C edge.
        # If C1 = {a,b} and C2 = {a,b}, they share edge (a,b) - BAD.
        
        # So: no two cliques can have the same pair of C-vertices.
        # If all C-parts have size <= 2, then we just need no repeated pairs.
        
        # New strategy:
        # - For cross edges: for each I-vertex i, create cliques (c, i) for each c in N(i)
        #   This uses |N(i)| cliques per I-vertex, each with C-part {c} (size 1)
        #   C-parts of size 1 have no C-C edges, so no overlap issues
        #   Total: sum of |N(i)| cliques for cross edges
        # - For C-C edges: each gets its own 2-clique
        # - Total: sum(|N(i)|) + k*(k-1)/2
        # This is valid but not minimal.
        
        # Better: group I-vertices that share the same single C-neighbor
        # If i1 and i2 both have N(i) = {c}, we can make clique ({c}, {i1, i2})
        # This saves cliques.
        
        # Even better: for each C-vertex c, collect all I-vertices that are adjacent to c
        # But they might be adjacent to other C-vertices too...
        
        # Let me think about this differently. The constraint is edge-disjointness.
        # A clique (S_C, S_I) covers:
        # - C-C edges: all pairs in S_C
        # - Cross edges: all (c, i) for c in S_C, i in S_I
        
        # Two cliques (S_C1, S_I1) and (S_C2, S_I2) have overlapping C-C edges
        # iff |S_C1 ∩ S_C2| >= 2.
        # They have overlapping cross edges iff there exists c in S_C1 ∩ S_C2 and i in S_I1 ∩ S_I2.
        
        # So to avoid overlap: for any two cliques, either |S_C1 ∩ S_C2| <= 1,
        # or S_I1 ∩ S_I2 = empty (for the C-vertices in the intersection).
        
        # Simplest valid construction:
        # All C-parts are singletons or empty. Then no C-C edges are covered by cross cliques.
        # Cross cliques: (c, {i1, i2, ...}) where all i's are adjacent to c.
        # C-C edges: separate 2-cliques.
        
        # For cross: for each c in C, let I_c = {i in I : c adjacent to i}
        # One clique per c: ({c}, I_c)
        # This covers all cross edges, each exactly once (since each cross edge (c,i) is only in clique c)
        # Total cross cliques: k (one per C-vertex, but only if I_c is nonempty)
        
        # For C-C: k*(k-1)/2 cliques of size 2.
        
        # Total: k + k*(k-1)/2 = k*(k+1)/2
        
        # Can we do better? Yes, by using larger C-parts.
        # If we use C-part {c1, c2} for some I-vertices, we cover edge (c1,c2) and
        # cross edges (c1,i) and (c2,i) for those i's.
        # But then we can't use {c1} or {c2} for those same i's.
        
        # The optimal approach is complex. Let me implement the simple valid one first,
        # then try to improve.
        
        # Simple valid: one clique per C-vertex for cross, 2-cliques for C-C
        result = []
        
        # Cross edges: one clique per C-vertex
        for c in range(k):
            i_neighbors = [v for v in range(k, n) if c in adj[v]]
            if i_neighbors:
                result.append([c] + i_neighbors)
        
        # C-C edges: 2-cliques
        for u in range(k):
            for v in range(u+1, k):
                result.append([u, v])
        
        # Verify: every edge covered exactly once
        # Cross edge (c, i): in clique for c, and i is in i_neighbors for c. 
        #   Is it in any other clique? Only if i is in i_neighbors for another c'.
        #   But each cross edge (c,i) is only in the clique for c. Good.
        # C-C edge (u,v): in the 2-clique [u,v]. Is it in any cross clique?
        #   Cross cliques have C-part of size 1, so no C-C edges. Good.
        
        return result
    
    # Try to improve: use larger C-parts for cross cliques
    # Idea: if two C-vertices c1, c2 have the same set of I-neighbors,
    # we can merge their cliques into ({c1,c2}, I_shared)
    # This saves one clique and covers the C-C edge (c1,c2) for free.
    
    def solve_improved(n, k, p, seed):
        if time.time() > DEADLINE - 5:
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in edges]
        
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # For each C-vertex c, its I-neighborhood
        c_i_neigh = [set() for _ in range(k)]
        for c in range(k):
            for v in range(k, n):
                if c in adj[v]:
                    c_i_neigh[c].add(v)
        
        # Group C-vertices by their I-neighborhood
        from collections import defaultdict
        groups = defaultdict(list)
        for c in range(k):
            key = frozenset(c_i_neigh[c])
            groups[key].append(c)
        
        result = []
        covered_cc = set()
        
        # For each group of C-vertices with same I-neighborhood:
        # If group has size >= 2, make one clique with all of them
        # This covers C-C edges within the group
        for neigh_set, c_list in groups.items():
            if time.time() > DEADLINE - 10:
                break
            if not neigh_set:
                # No I-neighbors: these C-vertices only have C-C edges
                # Handle C-C edges separately
                continue
            if len(c_list) == 1:
                c = c_list[0]
                result.append([c] + list(neigh_set))
            else:
                # Multiple C-vertices with same I-neighborhood
                # Make one clique with all of them
                clique = c_list + list(neigh_set)
                result.append(clique)
                # Mark C-C edges within this group as covered
                for i in range(len(c_list)):
                    for j in range(i+1, len(c_list)):
                        covered_cc.add((min(c_list[i], c_list[j]), max(c_list[i], c_list[j])))
        
        # C-C edges not covered
        all_cc = set((u,v) for u in range(k) for v in range(u+1,k))
        uncovered_cc = all_cc - covered_cc
        for (u, v) in sorted(uncovered_cc):
            if time.time() > DEADLINE - 10:
                break
            result.append([u, v])
        
        # Verify validity
        # Check every clique is a valid clique
        for clique in result:
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    if clique[j] not in adj[clique[i]]:
                        # Invalid!
                        pass  # Shouldn't happen
        
        return result
    
    # Even better: try to merge C-vertices with overlapping (not identical) I-neighborhoods
    # If c1 and c2 have I-neighbors I1 and I2, and I1 == I2, merge.
    # If I1 ⊂ I2, we could put c1 with I1's clique and c2 with I2's clique,
    # but the C-C edge (c1,c2) needs to be covered.
    
    # Let me try a local search approach:
    # Start with the simple solution, then try to merge cliques.
    
    def solve_with_local_search(n, k, p, seed, time_limit=50):
        if time.time() > DEADLINE - 5:
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in edges]
        
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Build initial solution: one clique per C-vertex + 2-cliques for C-C
        cliques = []
        for c in range(k):
            i_neighbors = [v for v in range(k, n) if c in adj[v]]
            if i_neighbors:
                cliques.append([c] + i_neighbors)
        
        for u in range(k):
            for v in range(u+1, k):
                cliques.append([u, v])
        
        # Local search: try to merge pairs of cliques
        # Two cliques can be merged if:
        # 1. The union is a clique (all pairs are edges)
        # 2. No edge is covered by both (edge-disjointness)
        #    Since we're merging, the new clique covers all edges of both.
        #    We need to ensure no OTHER clique covers any of these edges.
        
        def is_clique(vertices, adj):
            vs = list(vertices)
            for i in range(len(vs)):
                for j in range(i+1, len(vs)):
                    if vs[j] not in adj[vs[i]]:
                        return False
            return True
        
        def clique_edges(clique, adj):
            edges = set()
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    if clique[j] in adj[clique[i]]:
                        edges.add((min(clique[i], clique[j]), max(clique[i], clique[j])))
            return edges
        
        # Compute edges for each clique
        clique_edge_sets = [clique_edges(c, adj) for c in cliques]
        
        # Check initial validity: all edges covered exactly once
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))
        
        covered = set()
        for es in clique_edge_sets:
            covered |= es
        
        # Try local search
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(cliques)):
                if time.time() > DEADLINE - 10:
                    break
                for j in range(i+1, len(cliques)):
                    if time.time() > DEADLINE - 10:
                        break
                    # Try to merge cliques i and j
                    union = list(set(cliques[i]) | set(cliques[j]))
                    if not is_clique(union, adj):
                        continue
                    new_edges = clique_edges(union, adj)
                    # Check: new_edges must be subset of (clique_edges[i] ∪ clique_edges[j])
                    # i.e., no new edges are introduced
                    if not new_edges.issubset(clique_edge_sets[i] | clique_edge_sets[j]):
                        continue
                    # Check: no other clique covers any edge in new_edges
                    conflict = False
                    for m in range(len(cliques)):
                        if m == i or m == j:
                            continue
                        if clique_edge_sets[m] & new_edges:
                            conflict = True
                            break
                    if conflict:
                        continue
                    # Merge!
                    cliques[i] = union
                    clique_edge_sets[i] = new_edges
                    del cliques[j]
                    del clique_edge_sets[j]
                    improved = True
                    break
                if improved:
                    break
        
        return [list(c) for c in cliques]
    
    # Use the improved solver
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE - 20:
            # Fallback
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
        else:
            try:
                sol = solve_with_local_search(spec[0], spec[1], spec[2], spec[3], time_limit=30)
                results.append(sol)
            except:
                edges = instance(*spec)
                results.append([[u, v] for u, v in edges])
    
    # Final validation
    for idx, spec in enumerate(((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))):
        n, k, p, seed = spec
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))
        
        cliques = results[idx]
        
        # Validate
        valid = True
        edge_count = [0] * len(all_edges)
        edge_list = list(all_edges)
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        for clique in cliques:
            if len(clique) < 2:
                valid = False
                break
            # Check it's a clique
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    if clique[j] not in adj[clique[i]]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            # Count edges
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    if e in edge_to_idx:
                        edge_count[edge_to_idx[e]] += 1
                    else:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid or any(c != 1 for c in edge_count):
            # Fallback
            edges = instance(*spec)
            results[idx] = [[u, v] for u, v in edges]
    
    # Convert to proper types
    final = []
    for cliques in results:
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(v) for v in c])
        final.append(final_cliques)
    
    return final

# EVOLVE-BLOCK-END
