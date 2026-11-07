import time
import sys
from collections import defaultdict
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))

    m = len(edge_set)

    # Build edge index
    edge_to_idx = {e: i for i, e in enumerate(edge_set)}
    edge_list = list(edge_set)

    # For each edge, precompute which other edges are "compatible" (can be in same biclique)
    # Two edges (a,b) and (c,d) can be in the same biclique if they are "cross-compatible"
    # i.e., all cross pairs are edges. For edges sharing a vertex, they can be in same biclique
    # if the non-shared endpoints are connected. For disjoint edges, all 4 cross pairs must be edges.

    def edges_compatible(e1, e2):
        a, b = e1
        c, d = e2
        # They share no vertex or share one
        if a == c or a == d or b == c or b == d:
            # Share at most one vertex (can't share both in a simple graph)
            # Find the non-shared endpoints
            if a == c:
                return adj[b][d]
            elif a == d:
                return adj[b][c]
            elif b == c:
                return adj[a][d]
            else:  # b == d
                return adj[a][c]
        else:
            # Disjoint edges: all 4 cross pairs must be edges
            return (adj[a][c] and adj[a][d] and adj[b][c] and adj[b][d])

    # Precompute compatibility bitmasks for edges
    # For each edge i, a bitmask of edges j that are compatible with i
    edge_compat = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            if edges_compatible(edge_list[i], edge_list[j]):
                edge_compat[i] |= (1 << j)
                edge_compat[j] |= (1 << i)

    # Generate candidate bicliques using a pivot-based maximal clique search
    # on the edge-compatibility graph. Each maximal clique in this graph
    # corresponds to a set of edges that can all be in one biclique.
    # But we need to extract the actual (L, R) from the edge set.

    def extract_biclique(edge_subset):
        """Given a set of edges that form a valid biclique edge set,
        extract (L, R) such that all edges are cross edges between L and R."""
        if not edge_subset:
            return None
        # Collect all vertices
        verts = set()
        for u, v in edge_subset:
            verts.add(u)
            verts.add(v)
        verts = list(verts)
        # Try to find a bipartition of the induced subgraph on these edges
        # such that all edges are cross edges.
        # This is equivalent to: the subgraph formed by these edges must be bipartite,
        # and the biclique must be complete bipartite on L, R.
        # Actually, if all edges in the subset are pairwise compatible (as defined),
        # then they form a valid biclique. We just need to find L and R.
        # 
        # Approach: pick any edge (a,b). Then L contains a, R contains b (or vice versa).
        # For any other edge (c,d) in the subset:
        #   - if c is connected to all in R and d to all in L, then c in L, d in R
        #   - etc.
        # 
        # Simpler: the vertices in the biclique are those incident to the edges.
        # L and R are the two parts of the bipartition.
        # Since all edges are cross edges, the subgraph is bipartite.
        # Find the bipartition.
        
        # Build subgraph
        sub_adj = {v: set() for v in verts}
        for u, v in edge_subset:
            sub_adj[u].add(v)
            sub_adj[v].add(u)
        
        # BFS to find bipartition
        color = {}
        for start in verts:
            if start in color:
                continue
            color[start] = 0
            queue = [start]
            while queue:
                cur = queue.pop()
                for nb in sub_adj[cur]:
                    if nb not in color:
                        color[nb] = 1 - color[cur]
                        queue.append(nb)
                    elif color[nb] == color[cur]:
                        return None  # Not bipartite
        
        L = [v for v in verts if color[v] == 0]
        R = [v for v in verts if color[v] == 1]
        
        # Verify it's a complete bipartite: all L-R pairs must be edges in edge_subset
        # Actually, we need all L-R pairs to be host edges, and the edge_subset must be exactly L x R
        # But the edge_subset might not include all L-R pairs. Let's check.
        # For a valid biclique, the cross edges must be exactly L x R (all of them must be host edges).
        # But our edge_subset is just a subset. The biclique's cross edges are ALL L-R pairs that are host edges.
        # So we need to expand: the actual biclique covers all host edges between L and R.
        
        # Compute all host edges between L and R
        biclique_edges = set()
        for u in L:
            for v in R:
                if adj[u][v]:
                    biclique_edges.add((min(u, v), max(u, v)))
        
        # The biclique is valid if biclique_edges is a superset of edge_subset
        # and all edges in biclique_edges are host edges (which they are by construction)
        if not edge_subset.issubset(biclique_edges):
            return None
        
        return (L, R, biclique_edges)

    # Generate candidate bicliques
    candidates = []  # list of (L, R, edge_set)
    candidate_edge_masks = []  # bitmask of edges for each candidate

    # Use Bron-Kerbosch with pivot on the edge-compatibility graph
    # to find all maximal cliques (each maximal clique = set of mutually compatible edges)
    
    def bron_kerbosch(R_set, P_set, X_set, depth=0):
        if time.time() > deadline:
            return
        if not P_set and not X_set:
            # R_set is a maximal clique
            if len(R_set) >= 2:  # At least 2 edges for a non-trivial biclique
                L, R, bic_edges = extract_biclique(frozenset(R_set))
                if L is not None and len(L) > 0 and len(R) > 0:
                    candidates.append((L, R, bic_edges))
                    mask = 0
                    for e in bic_edges:
                        mask |= (1 << edge_to_idx[e])
                    candidate_edge_masks.append(mask)
            elif len(R_set) == 1:
                # Single edge biclique
                e = list(R_set)[0]
                L, R, bic_edges = extract_biclique(frozenset(R_set))
                if L is not None:
                    candidates.append((L, R, bic_edges))
                    mask = (1 << edge_to_idx[e])
                    candidate_edge_masks.append(mask)
            return
        
        # Choose pivot
        if P_set:
            pivot = next(iter(P_set))
            # P \ N(pivot)
            non_neighbors = P_set - set()  # We need to compute neighbors of pivot in the compatibility graph
            # Actually, in the edge-compatibility graph, neighbors of edge i are edges compatible with i
            # Let's restructure: use the edge_compat bitmasks
            pass
        
        # Simpler approach: just iterate
        for edge in list(P_set):
            new_R = R_set | {edge}
            new_P = P_set & set(_get_compat_edges(edge))
            new_X = X_set & set(_get_compat_edges(edge))
            bron_kerbosch(new_R, new_P, new_X, depth + 1)
            P_set.remove(edge)
            X_set.add(edge)

    def _get_compat_edges(edge_idx):
        """Get set of edge indices compatible with edge_idx"""
        result = set()
        mask = edge_compat[edge_idx]
        for j in range(m):
            if mask & (1 << j):
                result.add(j)
        return result

    # Actually, let's use a simpler candidate generation: 
    # For each edge, try to build a maximal biclique by greedy expansion,
    # but with multiple random orderings. Also include single-edge bicliques.
    
    # Clear previous candidates
    candidates = []
    candidate_edge_masks = []

    def greedy_biclique_from_edge(start_edge, remaining_mask, order=None):
        """Build a maximal biclique starting from start_edge, using only edges in remaining_mask."""
        L = set()
        R = set()
        u, v = edge_list[start_edge]
        L.add(u)
        R.add(v)
        used_mask = (1 << start_edge)
        
        changed = True
        while changed:
            changed = False
            # Try adding vertices to L
            for x in range(n):
                if x in L:
                    continue
                ok = True
                for y in R:
                    if not adj[x][y]:
                        ok = False
                        break
                    e = (min(x, y), max(x, y))
                    ei = edge_to_idx[e]
                    if not (remaining_mask & (1 << ei)):
                        ok = False
                        break
                if ok:
                    L.add(x)
                    changed = True
            # Try adding vertices to R
            for y in range(n):
                if y in R:
                    continue
                ok = True
                for x in L:
                    if not adj[x][y]:
                        ok = False
                        break
                    e = (min(x, y), max(x, y))
                    ei = edge_to_idx[e]
                    if not (remaining_mask & (1 << ei)):
                        ok = False
                        break
                if ok:
                    R.add(y)
                    changed = True
        
        # Compute all edges in this biclique
        bic_edges = set()
        for u2 in L:
            for v2 in R:
                if adj[u2][v2]:
                    bic_edges.add((min(u2, v2), max(u2, v2)))
        
        # Only keep edges that are in remaining_mask
        bic_edges = {e for e in bic_edges if remaining_mask & (1 << edge_to_idx[e])}
        
        if not bic_edges:
            return None
        
        return (list(L), list(R), bic_edges)

    # Generate candidates: for each edge, do greedy expansion
    # Also try with random perturbations
    import random
    random.seed(123)
    
    # First pass: deterministic greedy from each edge
    for i in range(m):
        if time.time() > deadline:
            break
        result = greedy_biclique_from_edge(i, (1 << m) - 1)
        if result is not None:
            L, R, bic_edges = result
            mask = 0
            for e in bic_edges:
                mask |= (1 << edge_to_idx[e])
            candidates.append((L, R, bic_edges))
            candidate_edge_masks.append(mask)

    # Second pass: random orderings for diversity
    for attempt in range(50):
        if time.time() > deadline:
            break
        # Pick a random subset of edges to try
        for i in range(0, m, 3):  # Every 3rd edge
            if time.time() > deadline:
                break
            # Randomize the vertex order for expansion
            result = greedy_biclique_from_edge(i, (1 << m) - 1)
            if result is not None:
                L, R, bic_edges = result
                mask = 0
                for e in bic_edges:
                    mask |= (1 << edge_to_idx[e])
                candidates.append((L, R, bic_edges))
                candidate_edge_masks.append(mask)

    # Deduplicate candidates by their edge mask
    seen_masks = set()
    unique_candidates = []
    unique_masks = []
    for i, (L, R, bic_edges) in enumerate(candidates):
        if candidate_edge_masks[i] not in seen_masks:
            seen_masks.add(candidate_edge_masks[i])
            unique_candidates.append((L, R, bic_edges))
            unique_masks.append(candidate_edge_masks[i])
    
    candidates = unique_candidates
    candidate_edge_masks = unique_masks

    # Also ensure single-edge bicliques are in the pool
    for i in range(m):
        if time.time() > deadline:
            break
        mask = (1 << i)
        if mask not in seen_masks:
            u, v = edge_list[i]
            candidates.append(([u], [v], {edge_list[i]}))
            candidate_edge_masks.append(mask)
            seen_masks.add(mask)

    # Now solve minimum set cover: find minimum number of candidates whose edge sets cover all edges
    # Use branch and bound
    
    num_candidates = len(candidates)
    
    # Precompute: for each candidate, its edge mask
    # For each edge, which candidates cover it
    edge_to_candidates = [[] for _ in range(m)]
    for i in range(num_candidates):
        mask = candidate_edge_masks[i]
        for j in range(m):
            if mask & (1 << j):
                edge_to_candidates[j].append(i)

    best_cover = None
    best_count = m  # Upper bound: one biclique per edge

    # Greedy initial solution
    def greedy_cover():
        remaining = (1 << m) - 1
        cover = []
        while remaining:
            if time.time() > deadline:
                return None
            # Find candidate that covers most remaining edges
            best_i = -1
            best_count_edges = 0
            for i in range(num_candidates):
                count = bin(candidate_edge_masks[i] & remaining).count('1')
                if count > best_count_edges:
                    best_count_edges = count
                    best_i = i
            if best_i == -1 or best_count_edges == 0:
                return None
            cover.append(best_i)
            remaining &= ~candidate_edge_masks[best_i]
        return cover

    greedy_sol = greedy_cover()
    if greedy_sol is not None:
        best_count = len(greedy_sol)
        best_cover = greedy_sol[:]

    # Branch and bound
    # Order candidates by size (larger first)
    cand_order = sorted(range(num_candidates), key=lambda i: bin(candidate_edge_masks[i]).count('1'), reverse=True)

    def lower_bound(remaining_mask):
        """Lower bound on number of additional candidates needed to cover remaining_mask.
        Use: each candidate covers at most max_size edges, so need at least ceil(popcount(remaining)/max_size).
        Better: for each edge in remaining, it must be covered by some candidate. 
        Simple bound: popcount(remaining) / max_candidate_size_in_remaining"""
        if remaining_mask == 0:
            return 0
        pop = bin(remaining_mask).count('1')
        # Find max size candidate that has any overlap with remaining
        max_size = 0
        for i in range(num_candidates):
            overlap = bin(candidate_edge_masks[i] & remaining_mask).count('1')
            if overlap > max_size:
                max_size = overlap
        if max_size == 0:
            return float('inf')
        return (pop + max_size - 1) // max_size

    def branch_and_bound(idx, remaining_mask, current_cover, current_count):
        nonlocal best_cover, best_count
        if time.time() > deadline:
            return
        if remaining_mask == 0:
            if current_count < best_count:
                best_count = current_count
                best_cover = current_cover[:]
            return
        
        # Lower bound
        lb = lower_bound(remaining_mask)
        if current_count + lb >= best_count:
            return
        
        # If we've considered all candidates
        if idx >= num_candidates:
            return
        
        # Option 1: skip this candidate
        branch_and_bound(idx + 1, remaining_mask, current_cover, current_count)
        
        # Option 2: take this candidate
        mask = candidate_edge_masks[cand_order[idx]]
        if mask & remaining_mask:  # Only if it helps
            new_remaining = remaining_mask & ~mask
            current_cover.append(cand_order[idx])
            branch_and_bound(idx + 1, new_remaining, current_cover, current_count + 1)
            current_cover.pop()

    # Run B&B with time limit
    bnb_start = time.time()
    branch_and_bound(0, (1 << m) - 1, [], 0)
    bnb_time = time.time() - bnb_start

    # If B&B didn't finish well, try local search on the greedy solution
    if best_cover is not None and len(best_cover) > 2:
        # Try to improve by replacing pairs of candidates with a single one
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_cover)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_cover)):
                    if time.time() > deadline:
                        break
                    # Try replacing best_cover[i] and best_cover[j] with a single candidate
                    mask_i = candidate_edge_masks[best_cover[i]]
                    mask_j = candidate_edge_masks[best_cover[j]]
                    combined = mask_i | mask_j
                    # Find a candidate that covers all edges in combined
                    for k in range(num_candidates):
                        if candidate_edge_masks[k] == combined:
                            # Replace i and j with k
                            new_cover = best_cover[:i] + best_cover[i+1:j] + best_cover[j+1:] + [k]
                            if len(new_cover) < len(best_cover):
                                best_cover = new_cover
                                best_count = len(best_cover)
                                improved = True
                            break

    # Reconstruct result
    if best_cover is None:
        # Fallback
        best_cover = list(range(m))  # Single edge bicliques (last m candidates should be these)
        # Actually, let's just use the original edges
        result = [[[u], [v]] for u, v in edge_list]
    else:
        result = []
        covered = set()
        for idx in best_cover:
            L, R, bic_edges = candidates[idx]
            # Check no overlap
            overlap = bic_edges & covered
            if overlap:
                # This shouldn't happen if we're doing set cover correctly,
                # but let's handle it by filtering
                filtered_edges = bic_edges - covered
                if not filtered_edges:
                    continue
                # Re-extract L, R from filtered edges
                # This is tricky; for safety, just skip
                continue
            covered |= bic_edges
            result.append([[int(x) for x in L], [int(x) for x in R]])

    # Validate
    final_covered = set()
    valid = True
    for L, R in result:
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in final_covered:
                    valid = False
                    break
                final_covered.add(e)
                if not adj[u][v]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid or final_covered != edge_set:
        # Fallback to valid solution
        result = [[[u], [v]] for u, v in edge_list]

    return result


if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} bicliques")
