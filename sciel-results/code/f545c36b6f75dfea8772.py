"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_edge_set():
    es = set()
    for u, v in P['edges']:
        es.add((min(u, v), max(u, v)))
    return es

def _build_adj_mask(edge_set):
    n = P['n']
    adj_mask = [0] * n
    for u, v in edge_set:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj_mask

def _mask_to_list(mask, n):
    return [i for i in range(n) if (mask >> i) & 1]

def _expand_biclique(L_mask, R_mask, adj_mask, n, deadline):
    """Expand a biclique to maximal form."""
    all_mask = (1 << n) - 1
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        # Try adding to L: vertex x not in L∪R, connected to all in R
        candidates = all_mask & ~(L_mask | R_mask)
        temp = candidates
        while temp:
            x_bit = temp & (-temp)
            x = x_bit.bit_length() - 1
            temp -= x_bit
            if (adj_mask[x] & R_mask) == R_mask:
                L_mask |= x_bit
                changed = True
        # Try adding to R: vertex x not in L∪R, connected to all in L
        candidates = all_mask & ~(L_mask | R_mask)
        temp = candidates
        while temp:
            x_bit = temp & (-temp)
            x = x_bit.bit_length() - 1
            temp -= x_bit
            if (adj_mask[x] & L_mask) == L_mask:
                R_mask |= x_bit
                changed = True
    return L_mask, R_mask

def _find_all_maximal_bicliques(adj_mask, edge_set, deadline):
    """
    Find diverse maximal bicliques using multiple seeding strategies.
    """
    n = P['n']
    all_mask = (1 << n) - 1
    bicliques = []
    seen = set()
    
    def try_add(L_mask, R_mask):
        if L_mask and R_mask:
            # Canonicalize: smaller mask first
            if bin(L_mask).count('1') <= bin(R_mask).count('1'):
                key = (L_mask, R_mask)
            else:
                key = (R_mask, L_mask)
            if key not in seen:
                seen.add(key)
                bicliques.append(key)
    
    # Strategy 1: Seed from each edge, expand to maximal
    edges = list(edge_set)
    random.shuffle(edges)
    for seed_u, seed_v in edges:
        if time.time() > deadline:
            break
        L_mask, R_mask = _expand_biclique((1 << seed_u), (1 << seed_v), adj_mask, n, deadline)
        try_add(L_mask, R_mask)
    
    # Strategy 2: Seed from high-degree vertex pairs
    deg = [bin(adj_mask[u]).count('1') for u in range(n)]
    high_deg = sorted(range(n), key=lambda u: -deg[u])[:15]
    for i, u in enumerate(high_deg):
        if time.time() > deadline:
            break
        for v in high_deg[i+1:]:
            if time.time() > deadline:
                break
            # Try u on left, v on right
            L_mask, R_mask = _expand_biclique((1 << u), (1 << v), adj_mask, n, deadline)
            try_add(L_mask, R_mask)
            # Try v on left, u on right
            L_mask, R_mask = _expand_biclique((1 << v), (1 << u), adj_mask, n, deadline)
            try_add(L_mask, R_mask)
    
    # Strategy 3: Seed from vertex with its common neighbors
    for u in range(n):
        if time.time() > deadline:
            break
        # Find vertices adjacent to u
        neighbors = _mask_to_list(adj_mask[u], n)
        if len(neighbors) >= 2:
            # Pick a subset of neighbors for R
            for r_size in [2, 3, min(5, len(neighbors))]:
                if r_size > len(neighbors):
                    break
                # Use first r_size neighbors
                R_mask = 0
                for v in neighbors[:r_size]:
                    R_mask |= (1 << v)
                L_mask, R_mask2 = _expand_biclique((1 << u), R_mask, adj_mask, n, deadline)
                try_add(L_mask, R_mask2)
    
    # Strategy 4: Random edge subsets as seeds
    for _ in range(50):
        if time.time() > deadline:
            break
        # Pick 2-3 random edges, use their endpoints as seed
        seed_edges = random.sample(edges, min(3, len(edges)))
        L_mask = 0
        R_mask = 0
        for u, v in seed_edges:
            # Randomly assign to L or R
            if random.random() < 0.5:
                L_mask |= (1 << u)
                R_mask |= (1 << v)
            else:
                L_mask |= (1 << v)
                R_mask |= (1 << u)
        if L_mask and R_mask:
            L_mask, R_mask = _expand_biclique(L_mask, R_mask, adj_mask, n, deadline)
            try_add(L_mask, R_mask)
    
    return bicliques

def _biclique_edge_mask(L_mask, R_mask, edge_to_bit, edge_set):
    """Compute the bitmask of edges covered by this biclique."""
    edge_mask = 0
    for u in _mask_to_list(L_mask, P['n']):
        for v in _mask_to_list(R_mask, P['n']):
            e = (min(u, v), max(u, v))
            if e in edge_to_bit:
                edge_mask |= (1 << edge_to_bit[e])
    return edge_mask

def _greedy_partition_bitmask(adj_mask, edge_set, deadline):
    """
    Greedy: repeatedly find the largest biclique in remaining edges, remove those edges.
    Uses bitmasks for speed.
    """
    n = P['n']
    all_mask = (1 << n) - 1
    remaining = set(edge_set)
    bicliques = []
    
    # Build edge index
    edge_list = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    num_edges = len(edge_list)
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Build adjacency for remaining edges
        rem_adj = [0] * n
        rem_verts = set()
        for u, v in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
            rem_verts.add(u)
            rem_verts.add(v)
        
        # Find best biclique in remaining
        best_L = 0
        best_R = 0
        best_size = 0
        
        verts_list = list(rem_verts)
        # Try each vertex as anchor for left side
        for anchor in verts_list:
            if time.time() > deadline:
                break
            
            # Neighbors of anchor in remaining
            R_mask = rem_adj[anchor]
            L_mask = (1 << anchor)
            
            # Expand
            changed = True
            while changed:
                changed = False
                # Add to L
                candidates = rem_verts_mask = 0
                for v in rem_verts:
                    candidates |= (1 << v)
                candidates &= ~(L_mask | R_mask)
                temp = candidates
                while temp:
                    x_bit = temp & (-temp)
                    x = x_bit.bit_length() - 1
                    temp -= x_bit
                    if (rem_adj[x] & R_mask) == R_mask:
                        L_mask |= x_bit
                        changed = True
                
                # Add to R
                candidates = 0
                for v in rem_verts:
                    candidates |= (1 << v)
                candidates &= ~(L_mask | R_mask)
                temp = candidates
                while temp:
                    x_bit = temp & (-temp)
                    x = x_bit.bit_length() - 1
                    temp -= x_bit
                    if (rem_adj[x] & L_mask) == L_mask:
                        R_mask |= x_bit
                        changed = True
            
            size = bin(L_mask).count('1') * bin(R_mask).count('1')
            if size > best_size:
                best_size = size
                best_L = L_mask
                best_R = R_mask
        
        if best_L == 0 or best_R == 0:
            # Fallback: single edge
            u, v = next(iter(remaining))
            bicliques.append(((1 << u), (1 << v)))
            remaining.discard((u, v))
        else:
            bicliques.append((best_L, best_R))
            # Remove covered edges
            for u in _mask_to_list(best_L, n):
                for v in _mask_to_list(best_R, n):
                    remaining.discard((min(u, v), max(u, v)))
    
    return bicliques

def _solve_set_cover(biclique_masks, num_edges, deadline, initial_bound):
    """
    Solve minimum set cover using branch-and-bound with bitmasks.
    biclique_masks: list of int bitmasks, each representing edges covered by that biclique.
    Returns list of indices of selected bicliques.
    """
    full_mask = (1 << num_edges) - 1
    num_cand = len(biclique_masks)
    
    # Remove dominated candidates: if biclique i's edges are subset of biclique j's, drop i
    kept = []
    for i in range(num_cand):
        dominated = False
        for j in range(num_cand):
            if i != j and (biclique_masks[i] & ~biclique_masks[j]) == 0 and (biclique_masks[j] & ~biclique_masks[i]) != 0:
                dominated = True
                break
        if not dominated:
            kept.append(i)
    if kept:
        biclique_masks = [biclique_masks[i] for i in kept]
        num_cand = len(biclique_masks)
    
    # Precompute: for each edge, which bicliques cover it
    edge_to_bicliques = [0] * num_edges
    for i, bm in enumerate(biclique_masks):
        for e in range(num_edges):
            if (bm >> e) & 1:
                edge_to_bicliques[e] |= (1 << i)
    
    # Use int.bit_count() for speed (Python 3.8+)
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
    
    # Greedy for initial upper bound with multiple random restarts
    best_solution = None
    best_size = initial_bound
    
    def greedy_cover():
        nonlocal best_solution, best_size
        remaining = full_mask
        chosen = []
        while remaining:
            best_bi = -1
            best_count = 0
            for i in range(num_cand):
                cnt = popcount(biclique_masks[i] & remaining)
                if cnt > best_count:
                    best_count = cnt
                    best_bi = i
            if best_bi < 0:
                break
            chosen.append(best_bi)
            remaining &= ~biclique_masks[best_bi]
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_solution = list(chosen)
    
    greedy_cover()
    # A few random restarts
    for _ in range(5):
        if time.time() > deadline:
            break
        random.shuffle(biclique_masks)
        greedy_cover()
        random.shuffle(biclique_masks)  # restore
    
    if best_solution is None:
        # Fallback: greedy without shuffle
        remaining = full_mask
        chosen = []
        while remaining:
            best_bi = -1
            best_count = 0
            for i in range(num_cand):
                cnt = popcount(biclique_masks[i] & remaining)
                if cnt > best_count:
                    best_count = cnt
                    best_bi = i
            if best_bi < 0:
                break
            chosen.append(best_bi)
            remaining &= ~biclique_masks[best_bi]
        best_solution = chosen
        best_size = len(chosen)
    
    # Branch and bound
    node_count = [0]
    max_nodes = 3000000
    
    def lower_bound(uncovered):
        """Compute a lower bound on number of additional bicliques needed."""
        if uncovered == 0:
            return 0
        # Simple bound: ceiling of uncovered edges / max coverage
        max_cov = 0
        for i in range(num_cand):
            cov = popcount(biclique_masks[i] & uncovered)
            if cov > max_cov:
                max_cov = cov
        if max_cov == 0:
            return 1000000  # infeasible
        return (popcount(uncovered) + max_cov - 1) // max_cov
    
    def backtrack(uncovered, current, depth):
        nonlocal best_solution, best_size, node_count
        
        node_count[0] += 1
        if time.time() > deadline or node_count[0] > max_nodes:
            return
        
        if uncovered == 0:
            if depth < best_size:
                best_size = depth
                best_solution = list(current)
            return
        
        if depth >= best_size:
            return
        
        lb = lower_bound(uncovered)
        if depth + lb >= best_size:
            return
        
        # Find most constrained edge (fewest bicliques can cover it)
        min_count = num_cand + 1
        min_edge = -1
        temp = uncovered
        while temp:
            e_bit = temp & (-temp)
            e = e_bit.bit_length() - 1
            temp -= e_bit
            cnt = popcount(edge_to_bicliques[e])
            if cnt < min_count:
                min_count = cnt
                min_edge = e
        
        if min_edge < 0 or min_count == 0:
            return
        
        # Get candidates covering min_edge, sorted by coverage descending
        candidates_mask = edge_to_bicliques[min_edge]
        temp_cand = candidates_mask
        cand_list = []
        while temp_cand:
            bi_bit = temp_cand & (-temp_cand)
            bi = bi_bit.bit_length() - 1
            temp_cand -= bi_bit
            cov = popcount(biclique_masks[bi] & uncovered)
            if cov > 0:
                cand_list.append((cov, bi))
        cand_list.sort(reverse=True)
        
        for cov, bi in cand_list:
            new_uncovered = uncovered & ~biclique_masks[bi]
            current.append(bi)
            backtrack(new_uncovered, current, depth + 1)
            current.pop()
            if node_count[0] > max_nodes:
                break
    
    backtrack(full_mask, [], 0)
    
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    edge_set = _build_edge_set()
    n = P['n']
    num_edges = len(edge_set)
    
    # Build adjacency masks
    adj_mask = _build_adj_mask(edge_set)
    
    # Phase 1: Greedy partition for initial upper bound
    greedy_bicliques = _greedy_partition_bitmask(adj_mask, edge_set, deadline)
    greedy_count = len(greedy_bicliques)
    
    # Phase 2: Find candidate maximal bicliques (improved version)
    candidates = _find_all_maximal_bicliques(adj_mask, edge_set, deadline)
    
    # Also add the greedy bicliques as candidates
    all_candidates = list(candidates)
    existing_keys = set()
    for L, R in all_candidates:
        existing_keys.add((L, R))
        existing_keys.add((R, L))
    for L, R in greedy_bicliques:
        key = (L, R)
        if key not in existing_keys and (R, L) not in existing_keys:
            all_candidates.append((L, R))
            existing_keys.add((L, R))
            existing_keys.add((R, L))
    
    # Add star bicliques as additional candidates
    for u in range(n):
        neighbors = _mask_to_list(adj_mask[u], n)
        if neighbors:
            L_mask = (1 << u)
            R_mask = 0
            for v in neighbors:
                R_mask |= (1 << v)
            key = (L_mask, R_mask)
            if key not in existing_keys and (R_mask, L_mask) not in existing_keys:
                all_candidates.append((L_mask, R_mask))
                existing_keys.add((L_mask, R_mask))
                existing_keys.add((R_mask, L_mask))
    
    # Build edge-to-index mapping
    edge_list = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    
    # Convert candidates to edge bitmasks
    biclique_edge_masks = []
    biclique_vertices = []
    seen_edge_masks = set()
    
    for L_mask, R_mask in all_candidates:
        em = 0
        valid = True
        for u in _mask_to_list(L_mask, n):
            for v in _mask_to_list(R_mask, n):
                e = (min(u, v), max(u, v))
                if e in edge_to_idx:
                    em |= (1 << edge_to_idx[e])
                else:
                    valid = False
                    break
            if not valid:
                break
        if valid and em:
            if em not in seen_edge_masks:
                seen_edge_masks.add(em)
                biclique_edge_masks.append(em)
                biclique_vertices.append((L_mask, R_mask))
    
    # Phase 3: Solve set cover
    best_solution = None
    if biclique_edge_masks:
        best_indices = _solve_set_cover(biclique_edge_masks, num_edges, deadline, greedy_count)
        if best_indices:
            best_solution = best_indices
    
    # Build result
    if best_solution and len(best_solution) < greedy_count:
        result_masks = [biclique_vertices[i] for i in best_solution]
    else:
        result_masks = greedy_bicliques
    
    # Convert to proper format
    result = []
    for L_mask, R_mask in result_masks:
        L_list = [int(v) for v in _mask_to_list(L_mask, n)]
        R_list = [int(v) for v in _mask_to_list(R_mask, n)]
        if L_list and R_list:
            result.append([L_list, R_list])
    
    # Validate
    if not result:
        # Fallback: star decomposition
        for u in range(n):
            neighbors = _mask_to_list(adj_mask[u], n)
            if neighbors:
                result.append([neighbors, [u]])
    
    return result
# EVOLVE-BLOCK-END
