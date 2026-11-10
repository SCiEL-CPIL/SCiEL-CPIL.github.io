"""Clique partition of split graphs using matching-based pair packing."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Separate C-edges and cross-edges
        c_edges = []
        cross_edges = []
        for (u, v) in edges:
            if u < k and v < k:
                c_edges.append((u, v))
            else:
                if u < k:
                    c, i = u, v
                else:
                    c, i = v, u
                cross_edges.append((c, i))

        c_edge_set = set(c_edges)
        cliques = []
        covered_c = set()
        
        # Phase 1: Process each I-vertex using matching-based pair packing
        # For each i in I, group its C-neighbors into pairs that form C-edges.
        # Each pair (u, v) where (u,v) is a C-edge becomes a 3-clique {i, u, v}.
        # Remaining single C-neighbors become 2-cliques {i, u}.
        # This selectively consumes C-edges, preserving denser residual structure.
        
        # Group cross edges by I-vertex
        i_to_c = {}
        for (c, i) in cross_edges:
            if i not in i_to_c:
                i_to_c[i] = []
            i_to_c[i].append(c)
        
        # Process each I-vertex
        for i in sorted(i_to_c.keys()):
            neighbors = i_to_c[i]
            # Find pairs (u, v) in neighbors such that (u, v) is a C-edge and not covered
            used = set()
            pairs = []
            singles = []
            
            # Try to form pairs
            for idx_u, u in enumerate(neighbors):
                if u in used:
                    continue
                for v in neighbors[idx_u+1:]:
                    if v in used:
                        continue
                    e = (min(u, v), max(u, v))
                    if e in c_edge_set and e not in covered_c:
                        # Form a pair
                        pairs.append((u, v))
                        used.add(u)
                        used.add(v)
                        covered_c.add(e)
                        break  # u is now used, move to next u
            
            # Remaining neighbors are singles
            for u in neighbors:
                if u not in used:
                    singles.append(u)
            
            # Create cliques
            for (u, v) in pairs:
                cliques.append(sorted([i, u, v]))
            for u in singles:
                cliques.append(sorted([i, u]))
        
        # Phase 2: Handle remaining C-edges with greedy maximal clique packing
        uncovered_c = set(e for e in c_edges if e not in covered_c)
        
        while uncovered_c:
            # Find vertex with max degree in uncovered_c
            deg = {}
            for (u, v) in uncovered_c:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            if not deg:
                break
                
            # Pick vertex with max degree
            u_max = max(deg.keys(), key=lambda x: deg[x])
            
            # Build maximal clique starting with u_max
            S_c = {u_max}
            
            # Get candidates: neighbors of u_max in uncovered_c
            candidates = set()
            for (u, v) in uncovered_c:
                if u == u_max:
                    candidates.add(v)
                elif v == u_max:
                    candidates.add(u)
            
            # Greedily add candidates
            changed = True
            while changed:
                changed = False
                for c in sorted(candidates):
                    # Check if c is connected to all in S_c via uncovered edges
                    ok = True
                    for v in S_c:
                        if v == c:
                            continue
                        e = (min(c, v), max(c, v))
                        if e not in uncovered_c:
                            ok = False
                            break
                    if ok:
                        S_c.add(c)
                        candidates.discard(c)
                        # Remove edges from c to S_c
                        for v in S_c:
                            if v == c:
                                continue
                            e = (min(c, v), max(c, v))
                            uncovered_c.discard(e)
                        changed = True
                        break  # Restart with updated S_c
            
            # Create clique
            clique = sorted(list(S_c))
            if len(clique) >= 2:
                cliques.append(clique)
            
            # Safety: if we made no progress, break
            if len(S_c) == 1:
                # Should not happen, but break to avoid infinite loop
                break
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback: trivial partition (all edges as cliques)
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        # Validate and convert types
        final_cliques = []
        for c in cliques:
            c_ints = list(set(int(x) for x in c))
            if len(c_ints) >= 2:
                final_cliques.append(sorted(c_ints))
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
