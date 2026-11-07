import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def maximal_clique_in_subgraph(start_node, candidates, adj_set, vertices):
        """
        Greedy maximal clique: start with start_node, add candidates in order if adjacent to all current.
        adj_set: set of edges as (min, max) tuples
        candidates: list of vertices to consider
        """
        clique = {start_node}
        for c in candidates:
            if c in clique:
                continue
            # Check if c is adjacent to all in clique
            ok = True
            for x in clique:
                e = (min(c, x), max(c, x))
                if e not in adj_set:
                    ok = False
                    break
            if ok:
                clique.add(c)
        return clique

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Build internal adjacency: K is a clique, so all pairs in K are edges
        # We need to cover all edges of the graph with cliques.
        # Edges are:
        # 1. Internal edges of K: C(k,2) edges
        # 2. Cross edges: between K and I, where v in I is adjacent to u in K with prob p
        
        # Strategy:
        # - Each v in I has a neighborhood N(v) subset of K.
        # - We can form cliques {v} union S where S is a clique in K (which is any subset since K is a clique).
        #   But the clique must only contain edges that exist. Since K is a clique, any subset S of K is a clique.
        #   And v is adjacent to all in S iff S subset of N(v).
        #   So {v} union S is a valid clique iff S subset of N(v).
        # - We want to partition all edges into cliques.
        # - Each edge (u, v) where u in K, v in I must be in exactly one clique.
        #   That clique must contain both u and v, so it must be of the form {v} union S where u in S.
        #   So for each v in I, the edges from v to K are partitioned among cliques that include v.
        #   Each such clique is {v} union S_i where S_i is a subset of N(v), and the S_i partition N(v).
        #   Wait, not necessarily partition of N(v), because we could have multiple cliques including v,
        #   and the union of their S_i's must be exactly N(v) (since every edge (v, u) for u in N(v) must be covered).
        #   And they must be disjoint? No, they can overlap? No, because each edge must be in exactly one clique.
        #   Edge (v, u) is in a clique {v} union S iff u in S. So for a fixed v, the sets S_i (for cliques containing v)
        #   must be disjoint and their union is N(v).
        # - Additionally, internal edges of K: each edge (u, w) in K must be in exactly one clique.
        #   A clique containing v in I and subset S of K covers all internal edges within S.
        #   So the internal edges covered by cliques involving v are all edges within the union of S_i for v.
        #   But wait, if two different vertices v1, v2 in I both have cliques that include edge (u, w) in K,
        #   that would be double-covering. So we must ensure that for each internal edge (u, w),
        #   at most one clique contains both u and w.
        #   A clique {v} union S contains internal edges within S. So we need: for each pair (u, w) in K,
        #   there is at most one v in I such that u, w in S_v (where S_v is the set of K-vertices in the clique with v).
        #   But actually, multiple cliques for the same v could contain u and w? No, because for a fixed v,
        #   the S_i's are disjoint, so u and w can be in at most one S_i for a given v.
        #   But across different v's, we need u and w to appear together in S_v for at most one v.
        #   Actually, the internal edge (u, w) must be covered by exactly one clique total.
        #   So the collection of all S_v (across all v and all cliques for v) must have the property that
        #   each pair {u, w} is contained in exactly one S_v.
        
        # This is getting complex. Let me think of a simpler approach.
        # 
        # Alternative: Treat each v in I independently. For v, we need to partition N(v) into subsets S_1, S_2, ...
        # such that each S_i is a clique (trivially true since K is a clique) and we create cliques {v} union S_i.
        # The internal edges covered by v's cliques are all edges within each S_i.
        # We need to choose the partitions for all v's such that no internal edge is covered twice.
        # 
        # This is like: assign each internal edge of K to at most one v in I (specifically, to a subset S_i of v's partition
        # that contains both endpoints).
        # 
        # Greedy approach:
        # 1. For each v in I, sort N(v) by some criterion.
        # 2. Process v's in order of |N(v)| descending.
        # 3. For each v, find a partition of N(v) into subsets such that no subset contains a pair (u, w)
        #    that has already been "claimed" by a previous v's clique.
        #    Actually, we want to minimize the number of cliques, so we want large subsets.
        #    But we must avoid claiming pairs that are already claimed.
        # 
        # Let's maintain a set of "available" internal edges (pairs in K not yet covered).
        # For v, we want to partition N(v) into as few subsets as possible, where each subset S
        # has the property that all pairs within S are available (not yet covered).
        # Then we mark all pairs within each S as covered.
        # 
        # To minimize the number of subsets for v, we want to make each S as large as possible,
        # subject to all pairs within S being available.
        # This is equivalent to: find a maximal clique in the graph induced by N(v) using only available edges.
        # But K is a clique, so the induced graph on N(v) is a complete graph. The available edges are a subset.
        # So we want to find a clique in the "available" graph induced by N(v).
        # 
        # Greedy: for v, repeatedly find a maximal clique in the available graph induced by remaining N(v),
        # add it as a subset, remove its internal edges from available, and continue.
        
        # Build set of all internal edges
        all_internal = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_internal.add((u, w))
        
        available = set(all_internal)
        cliques = []
        
        # Process v in I, sorted by degree descending
        vs = sorted(range(k, n), key=lambda v: len(neighbors[v]), reverse=True)
        
        for v in vs:
            N_v = set(neighbors[v])
            if not N_v:
                continue
            
            remaining = set(N_v)
            while remaining:
                # Find maximal clique in 'available' graph induced by 'remaining'
                # Start with node of highest degree in available graph within remaining
                deg_in_avail = {node: 0 for node in remaining}
                for node in remaining:
                    for other in remaining:
                        if node != other:
                            e = (min(node, other), max(node, other))
                            if e in available:
                                deg_in_avail[node] += 1
                
                if not any(d > 0 for d in deg_in_avail.values()):
                    # No available edges among remaining, so each node forms a singleton subset
                    # But a singleton subset with v gives clique {v, node} of size 2
                    node = next(iter(remaining))
                    subset = {node}
                else:
                    start = max(deg_in_avail, key=lambda x: deg_in_avail[x])
                    # Greedy maximal clique
                    subset = {start}
                    candidates = sorted([x for x in remaining if x != start], 
                                        key=lambda x: deg_in_avail.get(x, 0), reverse=True)
                    for c in candidates:
                        if c in subset:
                            continue
                        # Check if c is connected to all in subset via available edges
                        ok = True
                        for x in subset:
                            e = (min(c, x), max(c, x))
                            if e not in available:
                                ok = False
                                break
                        if ok:
                            subset.add(c)
                
                full_clique = [v] + sorted(subset)
                cliques.append(full_clique)
                
                # Remove internal edges within subset from available
                for u, w in combinations(subset, 2):
                    e = (min(u, w), max(u, w))
                    available.discard(e)
                
                remaining -= subset
        
        # Cover remaining available internal edges with cliques on K
        if available:
            # Find cliques covering all remaining edges
            # Greedy: repeatedly find maximal clique in available graph
            remaining_edges = set(available)
            vertices_in_rem = set()
            for u, v in remaining_edges:
                vertices_in_rem.add(u)
                vertices_in_rem.add(v)
            
            while remaining_edges:
                # Find maximal clique in remaining_edges
                deg = {v: 0 for v in vertices_in_rem}
                for u, v in remaining_edges:
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
                
                start = max(deg, key=lambda x: deg[x])
                clique = {start}
                candidates = sorted([x for x in vertices_in_rem if x != start], 
                                    key=lambda x: deg.get(x, 0), reverse=True)
                for c in candidates:
                    if c in clique:
                        continue
                    ok = True
                    for x in clique:
                        e = (min(c, x), max(c, x))
                        if e not in remaining_edges:
                            ok = False
                            break
                    if ok:
                        clique.add(c)
                
                if len(clique) >= 2:
                    cliques.append(sorted(clique))
                    for u, w in combinations(clique, 2):
                        e = (min(u, w), max(u, w))
                        remaining_edges.discard(e)
                else:
                    # Should not happen if remaining_edges is non-empty
                    e = next(iter(remaining_edges))
                    cliques.append([e[0], e[1]])
                    remaining_edges.discard(e)
        
        clean_cliques = [[int(x) for x in c] for c in cliques]
        return clean_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results
