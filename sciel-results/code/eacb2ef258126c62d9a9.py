"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        """Solve minimum clique partition for a split graph."""
        # Generate the graph
        rng = Random(seed)
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)
        # Cross edges
        cross_adj = {}
        for v in range(k, n):
            neighbors_in_C = set()
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    neighbors_in_C.add(u)
            cross_adj[v] = neighbors_in_C

        # The clique C = {0, ..., k-1} forms one big clique.
        # Each I-vertex v can be added to a clique containing a subset of its C-neighbors.
        # We need to partition ALL edges into cliques.
        #
        # Key insight: Every edge within C must be covered. The C-clique itself covers all C-C edges.
        # But we can also split C into smaller cliques if it helps cover C-I edges better.
        #
        # Actually, the optimal strategy:
        # - Each I-vertex v must have its edges to C covered. These edges go to cross_adj[v].
        # - A clique containing v can include v and any subset of cross_adj[v] (the C-neighbors of v).
        # - But the clique must also be a clique in the graph, so all C-vertices in it must be mutually adjacent (they are, since C is a clique).
        # - C-C edges must be covered by cliques that contain those C-vertices.
        #
        # Strategy: We'll try to assign each I-vertex to one or more cliques, and partition C-C edges.
        #
        # A better approach: Think of it as covering all edges.
        # Edges are: C-C edges (k*(k-1)/2) and C-I edges.
        #
        # In a split graph, a clique can be:
        # 1. A subset of C only
        # 2. A subset of C plus one I-vertex (since I is independent, at most one I-vertex per clique)
        #
        # So each clique has at most one I-vertex.
        #
        # Let's say we have m cliques. Some cliques are pure C-subsets, others contain one I-vertex.
        # For an I-vertex v, if it's in a clique with C-subset S_v ⊆ cross_adj[v], then all edges (v, u) for u in S_v are covered.
        # But v must have ALL its edges covered. So v must be in cliques whose C-subsets together cover cross_adj[v].
        # Wait, no - v can be in MULTIPLE cliques. Each clique containing v covers a subset of v's C-neighbor edges.
        #
        # Actually, re-reading the problem: "every edge must lie in exactly one listed clique"
        # So each edge is in exactly one clique. This is a partition of edges into cliques.
        #
        # For I-vertex v with C-neighbors N(v): the edges (v, u) for u in N(v) must each be in some clique.
        # Since v can be in multiple cliques, we can split N(v) among different cliques containing v.
        # But each such clique must be a clique: {v} ∪ S where S ⊆ N(v) and S is a clique in C (always true since C is a clique).
        #
        # For C-C edges: they must be covered by cliques. A C-C edge (a,b) can be in:
        # - A pure C-clique containing a and b
        # - A clique containing an I-vertex v that is adjacent to both a and b (so a,b ∈ N(v))
        #
        # This is complex. Let me think of a greedy/heuristic approach.
        #
        # Simple baseline: One big clique C covers all C-C edges. Each I-vertex v gets its own clique {v} ∪ N(v).
        # Total cliques: 1 + (n-k)
        # But this might not be optimal if we can merge.
        #
        # Better: We can have the big C clique, and then for I-vertices, we can group their C-neighbor edges.
        # But each I-vertex can only be in cliques with its neighbors.
        #
        # Actually, the seed solution just returns each edge as a clique of size 2. That's terrible.
        # Let's do much better.
        #
        # Approach:
        # 1. Start with one clique = all of C. This covers all C-C edges.
        # 2. For each I-vertex v, we need to cover edges (v, u) for u in N(v).
        #    We can create cliques {v} ∪ S where S ⊆ N(v).
        #    To minimize cliques, we want to cover all of N(v) with as few subsets as possible.
        #    But each subset must form a clique with v, which is automatic.
        #    So one clique {v} ∪ N(v) covers all of v's edges. That's 1 clique per I-vertex.
        #    Total: 1 + (n-k) cliques.
        #
        # But wait - can we do better? Can we share C-vertices across I-vertex cliques?
        # The constraint is that each EDGE is in exactly one clique. C-C edges are already covered by the big C clique.
        # C-I edges (v,u) are only in the clique that contains both v and u.
        # So for each I-vertex v, all its C-neighbor edges must be in cliques containing v.
        # Since v can be in multiple cliques, we could split N(v) into parts, but that would use MORE cliques.
        # So the minimum for I-vertex v is 1 clique (if N(v) is non-empty) or 0 (if N(v) is empty).
        #
        # So total minimum = 1 (for C) + number of I-vertices with non-empty N(v).
        #
        # But wait - what if we DON'T use one big C clique? What if we split C into smaller cliques to potentially
        # reduce the number of I-vertex cliques? That doesn't make sense because I-vertex cliques are independent.
        #
        # Hmm, but there's a subtlety. If we use the big C clique, it covers ALL C-C edges. Then each I-vertex
        # needs its own clique(s). But what if an I-vertex has a very large N(v)? We still only need 1 clique for it.
        #
        # Let me verify: can we do better than 1 + |{v in I : N(v) ≠ ∅}|?
        # Each I-vertex v with N(v) ≠ ∅ needs at least one clique containing v (to cover its edges to C).
        # These cliques for different I-vertices are disjoint in their I-vertex content (each has exactly one I-vertex).
        # C-C edges need to be covered. They can be covered by the C-clique or by I-vertex cliques that contain both endpoints.
        #
        # If we use 1 big C-clique, we cover all C-C edges, and need 1 clique per non-isolated I-vertex.
        # Alternative: don't use a big C-clique. Then C-C edges must be covered by I-vertex cliques or smaller C-cliques.
        # This seems harder, not easier.
        #
        # So the optimal is: 1 + |{v in I : N(v) ≠ ∅}|
        #
        # Let me just compute this.

        non_isolated_I = sum(1 for v in range(k, n) if len(cross_adj[v]) > 0)
        num_cliques = 1 + non_isolated_I

        # Build the partition
        cliques = []
        # Big C clique
        C_list = list(range(k))
        if len(C_list) >= 2:
            cliques.append(C_list)

        # For each non-isolated I-vertex, create clique {v} ∪ N(v)
        for v in range(k, n):
            if len(cross_adj[v]) > 0:
                clique = [v] + sorted(cross_adj[v])
                cliques.append(clique)

        # Verify: every edge is in exactly one clique
        # Build edge -> clique mapping
        edge_to_clique = {}
        for i, clique in enumerate(cliques):
            for a_idx in range(len(clique)):
                for b_idx in range(a_idx+1, len(clique)):
                    u, v = clique[a_idx], clique[b_idx]
                    edge = (min(u,v), max(u,v))
                    if edge in edge_to_clique:
                        # Edge appears in two cliques - invalid!
                        return None
                    edge_to_clique[edge] = i

        # Check all edges are covered
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))

        if len(edge_to_clique) != len(all_edges):
            return None

        # Check each clique is actually a clique in the graph
        for clique in cliques:
            for a_idx in range(len(clique)):
                for b_idx in range(a_idx+1, len(clique)):
                    u, v = clique[a_idx], clique[b_idx]
                    if v not in adj[u]:
                        return None

        # Check each clique has at least 2 vertices
        for clique in cliques:
            if len(clique) < 2:
                return None

        return cliques

    def solve_split_optimized(n, k, p, seed):
        """
        Optimized: The key realization is that in a split graph, the minimum clique partition
        has a specific structure. Let's think more carefully.
        
        Actually, I realize the above might not be optimal. Consider: if we have the big C-clique,
        it covers all C-C edges. But some C-C edges could be "freed" by not having the big clique,
        allowing I-vertex cliques to cover them instead. But that doesn't reduce the count.
        
        The real question: can we cover C-C edges AND C-I edges with fewer total cliques?
        
        Each clique contains at most 1 I-vertex. Let's say we have t cliques containing I-vertices
        and s cliques that are pure C-subsets.
        
        The t I-vertex cliques cover: all C-I edges (each C-I edge (v,u) is in the clique for v that contains u).
        Wait, no - v can be in multiple cliques. So C-I edges of v are split among v's cliques.
        
        The s pure C-cliques cover: some C-C edges.
        The t I-vertex cliques also cover: C-C edges among their C-vertices.
        
        Total C-C edges = k(k-1)/2. These are covered by pure C-cliques and by C-C edges within I-vertex cliques.
        
        To minimize t+s, we want to maximize the C-C edges covered by I-vertex cliques (so we need fewer pure C-cliques).
        
        This is getting complex. For the given sizes (k=10,20,30), let me just use the simple approach
        and see if it's good enough, then try to improve.
        """
        pass

    def solve_simple(n, k, p, seed):
        """Simple: big C clique + one clique per non-isolated I-vertex."""
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u+1, k):
                adj[u].add(v)
                adj[v].add(u)
        cross_adj = {}
        for v in range(k, n):
            neighbors_in_C = set()
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    neighbors_in_C.add(u)
            cross_adj[v] = neighbors_in_C

        cliques = []
        C_list = list(range(k))
        if len(C_list) >= 2:
            cliques.append(C_list)

        for v in range(k, n):
            if len(cross_adj[v]) > 0:
                clique = [v] + sorted(cross_adj[v])
                cliques.append(clique)

        # Validate
        edge_to_clique = {}
        for i, clique in enumerate(cliques):
            for a_idx in range(len(clique)):
                for b_idx in range(a_idx+1, len(clique)):
                    u, v = clique[a_idx], clique[b_idx]
                    edge = (min(u,v), max(u,v))
                    if edge in edge_to_clique:
                        return None
                    edge_to_clique[edge] = i

        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))

        if len(edge_to_clique) != len(all_edges):
            return None

        for clique in cliques:
            if len(clique) < 2:
                return None
            for a_idx in range(len(clique)):
                for b_idx in range(a_idx+1, len(clique)):
                    u, v = clique[a_idx], clique[b_idx]
                    if v not in adj[u]:
                        return None

        return cliques

    def solve_greedy(n, k, p, seed):
        """
        Greedy edge-based approach: repeatedly find the largest clique that covers uncovered edges.
        This might find better solutions than the simple structural approach.
        """
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

        # Get all edges
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))

        uncovered = set(all_edges)
        cliques = []

        # Precompute: for each vertex, its neighbors
        # Greedy: find max clique in the graph of uncovered edges
        # Since n is small (90), we can do this efficiently.

        while uncovered and (time.time() - t0) < DEADLINE:
            # Find the vertex with highest degree in uncovered subgraph
            deg = {v: 0 for v in range(n)}
            for (u, v) in uncovered:
                deg[u] += 1
                deg[v] += 1

            if max(deg.values()) == 0:
                break

            # Pick vertex with max degree
            start_v = max(range(n), key=lambda x: deg[x])

            # Greedy max clique starting from start_v
            # Use Bron-Kerbosch style or simple greedy
            candidates = set(adj[start_v])
            # Filter to only vertices connected by uncovered edges
            for v in list(candidates):
                if (min(start_v, v), max(start_v, v)) not in uncovered:
                    candidates.discard(v)

            # Greedy: add vertices that connect to all already selected
            clique = [start_v]
            selected = {start_v}
            for v in sorted(candidates, key=lambda x: -len(adj[x] & candidates)):
                if all((min(v, w), max(v, w)) in uncovered for w in selected):
                    # Check v is connected to all in selected via uncovered edges
                    clique.append(v)
                    selected.add(v)

            if len(clique) < 2:
                # Just take an edge
                edge = next(iter(uncovered))
                clique = list(edge)
                selected = set(edge)

            # Remove covered edges
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (min(u,v), max(u,v))
                    if e in uncovered:
                        uncovered.discard(e)

            cliques.append(clique)

        if uncovered:
            # Couldn't cover all - fallback
            return None

        # Validate
        edge_to_clique = {}
        for i, clique in enumerate(cliques):
            for a_idx in range(len(clique)):
                for b_idx in range(a_idx+1, len(clique)):
                    u, v = clique[a_idx], clique[b_idx]
                    edge = (min(u,v), max(u,v))
                    if edge in edge_to_clique:
                        return None
                    edge_to_clique[edge] = i

        if len(edge_to_clique) != len(all_edges):
            return None

        for clique in cliques:
            if len(clique) < 2:
                return None
            for a_idx in range(len(clique)):
                for b_idx in range(a_idx+1, len(clique)):
                    u, v = clique[a_idx], clique[b_idx]
                    if v not in adj[u]:
                        return None

        return cliques

    def solve_multi(n, k, p, seed):
        """
        Try multiple strategies and return the best (fewest cliques).
        """
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

        # Strategy 1: Simple structural
        result1 = solve_simple(n, k, p, seed)

        # Strategy 2: Greedy
        result2 = solve_greedy(n, k, p, seed)

        # Strategy 3: Improved greedy with different orderings
        # Try greedy starting from different vertices
        best = None
        for strat_result in [result1, result2]:
            if strat_result is not None:
                if best is None or len(strat_result) < len(best):
                    best = strat_result

        # Strategy 4: More sophisticated - for split graphs, use the structure
        # Each I-vertex can be in multiple cliques. Try to optimize.
        #
        # Key insight: In a split graph, the optimal clique partition can be found by:
        # - Deciding how to partition C-C edges among pure C-cliques and I-vertex cliques
        # - For each I-vertex, deciding which of its C-neighbor edges go in which clique
        #
        # Since k is at most 30, we can try a smarter approach.
        #
        # Actually, let me think about this differently. The problem is equivalent to:
        # Given the graph, partition edges into minimum number of cliques.
        # This is the "clique partition number" or "clique cover of edges."
        #
        # For split graphs, there's a known result. Let me just try the structural approach
        # and a few variations.

        # Strategy 5: Don't use one big C-clique. Instead, use multiple C-cliques
        # to potentially share with I-vertex cliques.
        #
        # Actually, the simplest good solution is: 1 big C-clique + 1 clique per non-isolated I-vertex.
        # Let's also try: for each I-vertex, if it has few neighbors, maybe merge with C differently.
        #
        # Let me try a local improvement: after getting an initial solution, try to merge/split cliques.

        if best is not None:
            # Try local search: for each pair of cliques, see if merging them (if they share a vertex)
            # or restructuring helps. Actually, cliques in a valid partition can't share edges,
            # but they can share vertices.
            #
            # Try: for each I-vertex v, if it's in a clique with C-subset S, can we move some
            # of S's C-C edges to another clique? This is complex.
            #
            # For now, let's just return the best we have.
            return best

        return None

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if (time.time() - t0) > DEADLINE:
            # Emergency: return what we have
            break
        n, k, p, seed = spec
        result = solve_multi(n, k, p, seed)
        if result is None:
            # Fallback to simple
            result = solve_simple(n, k, p, seed)
            if result is None:
                # Last resort: individual edges
                rng = Random(seed)
                edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
                for v in range(k, n):
                    for u in range(k):
                        if rng.random() < p:
                            edges.append((u, v))
                result = [list(e) for e in edges]
        # Ensure types are correct
        clean_result = []
        for clique in result:
            clean_clique = [int(v) for v in clique]
            clean_result.append(clean_clique)
        results.append(clean_result)

    return results
# EVOLVE-BLOCK-END
