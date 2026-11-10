import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def is_clique_bitset(vertices, adj_bitsets):
        """Check if vertices form a clique using bitsets for speed."""
        if len(vertices) < 2:
            return False
        # Pick first vertex, check all others are connected to it
        first = vertices[0]
        mask = adj_bitsets[first]
        for v in vertices[1:]:
            if not (mask & (1 << v)):
                return False
            # For efficiency, we can also check pairwise, but for small cliques 
            # (max size ~10-15 in dense random graphs), checking against first is 
            # not enough. We need full pairwise.
            # Let's do a proper check: for each pair, check adjacency.
            # But bitsets make this fast.
            # Actually, let's just do the full check but with bit operations.
            # For a clique, every vertex must be connected to every other.
            # We can check: for each v in vertices, (adj_bitsets[v] & vertex_mask) == (vertex_mask - (1<<v))
            # But constructing vertex_mask each time is costly.
            # Given small clique sizes, let's just do the pairwise check with bit tests.
            pass
        
        # Proper full check
        for i in range(len(vertices)):
            vi = vertices[i]
            for j in range(i + 1, len(vertices)):
                vj = vertices[j]
                if not (adj_bitsets[vi] & (1 << vj)):
                    return False
        return True

    def solve_instance(n, p, seed):
        edges, adj = random_graph(n, p, seed)
        edge_list = list(edges)
        m = len(edge_list)
        if m == 0:
            return []
        
        # Build bitsets for adjacency
        adj_bitsets = [0] * n
        for u in range(n):
            for v in adj[u]:
                adj_bitsets[u] |= (1 << v)
        
        # Precompute edge index for quick lookup
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        def get_clique_edges(clique):
            """Return list of edge indices in the clique."""
            es = []
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v:
                        u, v = v, u
                    es.append(edge_to_idx[(u, v)])
            return es
        
        def is_valid_partition(clique_vertices_list):
            """Check if partition is valid: each clique is a clique, covers all edges exactly once."""
            covered = [False] * m
            for clique in clique_vertices_list:
                if len(clique) < 2:
                    return False
                if not is_clique_bitset(clique, adj_bitsets):
                    return False
                for e_idx in get_clique_edges(clique):
                    if covered[e_idx]:
                        return False
                    covered[e_idx] = True
            return all(covered)
        
        # Strategy: Multi-start local search
        # Representation: list of cliques (each clique is a list of vertices)
        # Initial solution: greedy clique extraction
        
        def greedy_initial():
            remaining = set(edge_list)
            cliques = []
            while remaining:
                # Find a clique from remaining edges
                # Pick a random edge, try to expand
                edge = random.choice(list(remaining))
                u, v = edge
                clique = {u, v}
                # Try to add more vertices
                changed = True
                while changed:
                    changed = False
                    # Find candidates: vertices connected to all in clique
                    candidates = set(adj[u]) & set(adj[v])
                    # Intersect with neighbors of all current clique members
                    for w in clique:
                        candidates &= adj[w]
                    candidates -= clique
                    if candidates:
                        # Pick the one with most connections within candidates
                        best_w = max(candidates, key=lambda w: len(adj[w] & candidates))
                        clique.add(best_w)
                        changed = True
                # Remove edges covered by this clique
                for e in get_clique_edges(list(clique)):
                    e_edge = edge_list[e]
                    if e_edge in remaining:
                        remaining.discard(e_edge)
                cliques.append(list(clique))
            return cliques
        
        def local_search(cliques, max_iters=5000):
            """Local search: try to merge cliques or split/merge to reduce count."""
            cliques = [list(c) for c in cliques]
            improved = True
            iters = 0
            while improved and iters < max_iters and time.time() < DEADLINE:
                improved = False
                iters += 1
                # Try all pairs of cliques to merge
                for i in range(len(cliques)):
                    for j in range(i + 1, len(cliques)):
                        merged = set(cliques[i]) | set(cliques[j])
                        if is_clique_bitset(list(merged), adj_bitsets):
                            # Check if merged edges are subset of union of their edges
                            merged_edges = set(get_clique_edges(list(merged)))
                            union_edges = set(get_clique_edges(cliques[i])) | set(get_clique_edges(cliques[j]))
                            if merged_edges <= union_edges:
                                # Safe merge
                                cliques = [c for k, c in enumerate(cliques) if k != i and k != j]
                                cliques.append(list(merged))
                                improved = True
                                break
                    if improved:
                        break
                if not improved:
                    # Try moving a vertex from one clique to another
                    for i in range(len(cliques)):
                        if len(cliques[i]) < 3:
                            continue
                        for v in cliques[i]:
                            new_i = [x for x in cliques[i] if x != v]
                            if len(new_i) < 2:
                                continue
                            if not is_clique_bitset(new_i, adj_bitsets):
                                continue
                            for j in range(len(cliques)):
                                if i == j:
                                    continue
                                new_j = list(set(cliques[j]) | {v})
                                if not is_clique_bitset(new_j, adj_bitsets):
                                    continue
                                # Check edge coverage
                                old_i_edges = set(get_clique_edges(cliques[i]))
                                old_j_edges = set(get_clique_edges(cliques[j]))
                                new_i_edges = set(get_clique_edges(new_i))
                                new_j_edges = set(get_clique_edges(new_j))
                                
                                # Edges removed from i: old_i - new_i
                                removed_from_i = old_i_edges - new_i_edges
                                # Edges added to j: new_j - old_j
                                added_to_j = new_j_edges - old_j_edges
                                # Edges added to i: new_i - old_i (should be empty)
                                added_to_i = new_i_edges - old_i_edges
                                # Edges removed from j: old_j - new_j (should be empty)
                                removed_from_j = old_j_edges - new_j_edges
                                
                                # For validity:
                                # 1. added_to_i must be empty (we're only removing v from i)
                                # 2. removed_from_j must be empty (we're only adding v to j)
                                # 3. removed_from_i (edges incident to v in clique i) must be covered by new_j
                                # 4. added_to_j (edges from v to clique j) must not be covered elsewhere
                                #    Actually, we need the total coverage to remain exact.
                                #    The edges removed_from_i were in clique i, now they must be in new_j
                                #    The edges added_to_j were not in any clique, now they are in new_j
                                #    But wait: the edges from v to vertices in clique j were NOT covered before
                                #    (since v wasn't in clique j). So adding them creates new coverage.
                                #    This is only valid if those edges were already covered by another clique.
                                #    This makes simple vertex moves very restricted.
                                #    Let's only allow moves where added_to_j is empty, which means
                                #    v was already connected to all in clique j and those edges were covered.
                                #    But if v is connected to all in clique j, and those edges are covered,
                                #    then v could have been in clique j already.
                                #    This is getting complex. Let's stick to merges for now.
                                pass
            return cliques
        
        best_cliques = None
        best_count = float('inf')
        
        # Multi-start
        for restart in range(30):
            if time.time() > DEADLINE:
                break
            cliques = greedy_initial()
            if not is_valid_partition(cliques):
                continue
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = [list(c) for c in cliques]
            # Local search
            improved_cliques = local_search(cliques)
            if is_valid_partition(improved_cliques):
                count = len(improved_cliques)
                if count < best_count:
                    best_count = count
                    best_cliques = [list(c) for c in improved_cliques]
        
        if best_cliques is None:
            # Fallback
            best_cliques = [list(e) for e in edge_list]
        
        return best_cliques
    
    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    return [result1, result2]
