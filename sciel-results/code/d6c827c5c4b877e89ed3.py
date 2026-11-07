import time
import random

_DEADLINE = 140.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        # Core: 0..k-1
        # Independent: k..n-1
        adj_core = [0] * k
        adj_cross = [0] * (n - k) # adj_cross[i] is bitmask of core neighbors of k+i
        
        core_edges = []
        cross_edges = []
        
        for (u, v) in edges:
            if u < k and v < k:
                adj_core[u] |= (1 << v)
                adj_core[v] |= (1 << u)
                core_edges.append((u, v))
            else:
                if u >= k:
                    u, v = v, u
                # u < k, v >= k
                idx_v = v - k
                adj_cross[idx_v] |= (1 << u)
                cross_edges.append((u, v))

        # Strategy:
        # 1. Partition core edges into matchings assigned to independent vertices.
        # 2. For each assigned matching edge (u, w) to v, create clique [u, w, v].
        # 3. Cover remaining core edges with 2-cliques.
        # 4. Cover remaining cross edges with 2-cliques.
        
        # Track covered edges
        # Core edges are identified by (u, v) with u < v < k
        # Cross edges are identified by (u, v) with u < k, v >= k
        
        covered_core = set()
        covered_cross = set()
        
        cliques = []
        
        # Map core edges to a set of independent vertices that can cover them
        # edge_to_vs: dict from (u, v) to list of idx_v
        edge_to_vs = {e: [] for e in core_edges}
        
        for idx_v in range(n - k):
            mask = adj_cross[idx_v]
            # Iterate through core edges
            # To be efficient, we can just check adjacency for each core edge
            for (u, v) in core_edges:
                if (mask >> u) & 1 and (mask >> v) & 1:
                    edge_to_vs[(u, v)].append(idx_v)

        # Greedy assignment of core edges to independent vertices
        # We want to form matchings for each v.
        # Sort independent vertices by number of available core edges (descending)
        # Or just iterate and pick best matching.
        
        # To ensure we cover as many core edges as possible with extensions:
        # We will maintain a list of uncovered core edges.
        uncovered_core = list(core_edges)
        
        # Sort vertices in I by potential (number of edges they touch)
        vs_sorted = sorted(range(n - k), key=lambda idx: sum(1 for e in core_edges if (adj_cross[idx] >> e[0]) & 1 and (adj_cross[idx] >> e[1]) & 1), reverse=True)
        
        for idx_v in vs_sorted:
            if _deadline_hit():
                break
            mask = adj_cross[idx_v]
            
            # Find a maximal matching in the subgraph of uncovered_core where v is adjacent to both
            # Vertices in this subgraph are those in K that are neighbors of v
            # Edges are those in uncovered_core connecting two neighbors of v
            
            # Simple greedy matching:
            # Iterate through uncovered_core, if both endpoints are neighbors of v and not used in this matching, add to matching.
            matching = []
            used_in_matching = set()
            
            for (u, v) in uncovered_core:
                if (mask >> u) & 1 and (mask >> v) & 1:
                    if u not in used_in_matching and v not in used_in_matching:
                        matching.append((u, v))
                        used_in_matching.add(u)
                        used_in_matching.add(v)
            
            if matching:
                # Create cliques
                for (u, v) in matching:
                    cliques.append([u, v, k + idx_v])
                    covered_core.add((u, v))
                    covered_cross.add((u, k + idx_v))
                    covered_cross.add((v, k + idx_v))
                
                # Remove these core edges from uncovered list
                uncovered_core = [e for e in uncovered_core if e not in covered_core]
        
        # Cover remaining core edges
        for (u, v) in uncovered_core:
            cliques.append([u, v])
            covered_core.add((u, v))
            
        # Cover remaining cross edges
        for (u, v) in cross_edges:
            if (u, v) not in covered_cross:
                cliques.append([u, v])
                covered_cross.add((u, v))
        
        # Convert to proper format
        result = []
        for c in cliques:
            result.append([int(x) for x in c])
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            out.append([list(e) for e in instance(*spec)])
        else:
            out.append(_solve_split(*spec))
    return out
