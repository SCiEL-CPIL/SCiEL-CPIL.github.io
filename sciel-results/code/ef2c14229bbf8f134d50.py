"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split(n, k, p, seed):
        """Find a minimal clique partition for a split graph.
        
        Key structural insight: In a split graph, any clique contains at most one
        vertex from the independent set I. So cliques are either:
        - Subsets of C (clique vertices)
        - {i} ∪ S where i ∈ I and S ⊆ N_C(i) forms a clique in C
        
        Strategy:
        1. For each I-vertex, find the largest clique in its C-neighborhood (uncovered)
           and form a clique {i} ∪ S.
        2. Handle remaining C-C edges with cliques within C.
        """
        edges = instance(n, k, p, seed)
        edge_set = set()
        for e in edges:
            if e[0] < e[1]:
                edge_set.add(e)
            else:
                edge_set.add((e[1], e[0]))
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        uncovered = set(edge_set)
        cliques = []
        
        # C vertices: 0..k-1, I vertices: k..n-1
        C = list(range(k))
        I = list(range(k, n))
        
        # Helper: find a maximal clique in a graph given by adjacency sets
        # Using Bron-Kerbosch with pivot for small graphs
        def find_maximal_clique(candidates, adj_sets):
            """Find a maximal clique in the graph induced by candidates using adj_sets."""
            if not candidates:
                return set()
            # Greedy: pick vertex with max degree in subgraph, recurse
            best = None
            best_size = 0
            # Try greedy from each starting vertex
            for start in candidates:
                clique = {start}
                remaining = candidates - {start}
                # Iteratively add vertices connected to all in clique
                changed = True
                while changed:
                    changed = False
                    for v in list(remaining):
                        if all(v in adj_sets.get(c, set()) for c in clique):
                            clique.add(v)
                            remaining.remove(v)
                            changed = True
                if len(clique) > best_size:
                    best_size = len(clique)
                    best = clique
            return best if best else set()

        # Phase 1: For each I-vertex, try to form a clique {i} ∪ S
        # Process I-vertices in order of decreasing number of uncovered C-neighbors
        i_order = sorted(I, key=lambda i: -len([c for c in C if (min(i,c), max(i,c)) in uncovered]))
        
        for i in i_order:
            if time.time() > deadline:
                break
            # Find uncovered C-neighbors of i
            c_neighbors = [c for c in C if (min(i, c), max(i, c)) in uncovered]
            if not c_neighbors:
                continue
            
            # Find a maximal clique in the subgraph induced by c_neighbors
            # where edges are uncovered C-C edges
            # Build adj for this subgraph
            sub_adj = {c: set() for c in c_neighbors}
            for idx, a in enumerate(c_neighbors):
                for b in c_neighbors[idx+1:]:
                    if (min(a,b), max(a,b)) in uncovered:
                        sub_adj[a].add(b)
                        sub_adj[b].add(a)
            
            # Greedy maximal clique in sub_adj
            clique_C = find_maximal_clique(set(c_neighbors), sub_adj)
            
            if len(clique_C) >= 1:
                # Form clique {i} ∪ clique_C
                full_clique = [i] + list(clique_C)
                cliques.append(full_clique)
                # Remove all edges within this clique
                vlist = full_clique
                for idx in range(len(vlist)):
                    for jdx in range(idx+1, len(vlist)):
                        a, b = vlist[idx], vlist[jdx]
                        uncovered.discard((min(a,b), max(a,b)))
        
        # Phase 2: Handle remaining C-C edges
        # These are edges within C that weren't covered
        remaining_cc = [(u,v) for (u,v) in uncovered if u < k and v < k]
        if remaining_cc:
            # Greedy: repeatedly pick an uncovered C-C edge, expand to maximal clique in C
            while remaining_cc:
                if time.time() > deadline:
                    break
                u, v = remaining_cc[0]
                # Find all C vertices that form a clique with u and v (uncovered edges)
                candidates = [c for c in C if c != u and c != v]
                valid = []
                for c in candidates:
                    ok = True
                    for x in [u, v]:
                        if (min(c,x), max(c,x)) not in uncovered:
                            ok = False
                            break
                    if ok:
                        valid.append(c)
                
                # Find maximal clique among valid + {u,v}
                # Build subgraph on valid where edges are uncovered
                sub_adj = {c: set() for c in valid}
                for idx, a in enumerate(valid):
                    for b in valid[idx+1:]:
                        if (min(a,b), max(a,b)) in uncovered:
                            sub_adj[a].add(b)
                            sub_adj[b].add(a)
                
                clique_C = find_maximal_clique(set(valid), sub_adj)
                full_clique = [u, v] + list(clique_C)
                cliques.append(full_clique)
                
                # Remove edges
                vlist = full_clique
                for idx in range(len(vlist)):
                    for jdx in range(idx+1, len(vlist)):
                        a, b = vlist[idx], vlist[jdx]
                        uncovered.discard((min(a,b), max(a,b)))
                
                # Update remaining_cc
                remaining_cc = [(a,b) for (a,b) in remaining_cc if (a,b) in uncovered]
        
        # Phase 3: Any remaining edges (should be C-C only, but handle generically)
        while uncovered:
            if time.time() > deadline:
                break
            u, v = next(iter(uncovered))
            clique = {u, v}
            # Try to expand
            candidates = [x for x in range(n) if x != u and x != v]
            valid = []
            for c in candidates:
                ok = True
                for x in [u, v]:
                    if (min(c,x), max(c,x)) not in uncovered:
                        ok = False
                        break
                if ok:
                    valid.append(c)
            
            # Greedy expand
            for c in valid:
                if all((min(c,x), max(c,x)) in uncovered for x in clique):
                    clique.add(c)
            
            cliques.append(list(clique))
            vlist = list(clique)
            for idx in range(len(vlist)):
                for jdx in range(idx+1, len(vlist)):
                    a, b = vlist[idx], vlist[jdx]
                    uncovered.discard((min(a,b), max(a,b)))
        
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            pass
        cliques = solve_split(*spec)
        result.append(cliques)

    return result
# EVOLVE-BLOCK-END
