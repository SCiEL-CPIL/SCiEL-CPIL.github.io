"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_n = P['n']
_edges = P['edges']

# Build adjacency as bitsets for speed
adj_bits = [0] * _n
for u, v in _edges:
    adj_bits[u] |= (1 << v)
    adj_bits[v] |= (1 << u)

# Map each edge to an index
edge_to_idx = {}
idx_to_edge = []
for i, (u, v) in enumerate(_edges):
    a, b = min(u, v), max(u, v)
    edge_to_idx[(a, b)] = i
    idx_to_edge.append((a, b))
m = len(_edges)

# Precompute: for each vertex pair (u,v) where (u,v) is an edge, the set of edges in the "star" at u that go to neighbors of v
# Actually, let's think of it as: a biclique is defined by (L, R) where every (l,r) in L×R is an edge.
# We want to cover all m edges with minimum number of such bicliques.

# Strategy: Generate candidate bicliques, then solve min set cover via branch and bound.

def generate_candidates():
    """Generate a pool of candidate bicliques using various seeds."""
    candidates = []  # list of (frozenset_L, frozenset_R, frozenset_of_edge_indices)
    
    # 1. Star bicliques: for each vertex u, L={u}, R=N(u)
    for u in range(_n):
        R = [v for v in range(_n) if v != u and (adj_bits[u] >> v) & 1]
        if R:
            L = {u}
            Rf = frozenset(R)
            Lf = frozenset(L)
            # Compute which edges this covers
            edge_set = frozenset()
            for v in R:
                a, b = min(u, v), max(u, v)
                if (a, b) in edge_to_idx:
                    edge_set.add(edge_to_idx[(a, b)])
            if edge_set:
                candidates.append((Lf, Rf, edge_set))
    
    # 2. For each edge (u,v), try to expand into a biclique
    for (u, v) in _edges:
        # Start with L={u}, R={v}
        L = {u}
        R = {v}
        # Expand: add vertices to L that connect to all of R, and vice versa
        changed = True
        while changed:
            changed = False
            # Expand L
            new_L = set(L)
            for rv in R:
                # All neighbors of rv
                for x in range(_n):
                    if x != rv and (adj_bits[rv] >> x) & 1:
                        new_L.add(x)
            # Keep only those in new_L that connect to ALL of R
            L = set()
            for x in new_L:
                if x in R:
                    continue
                ok = True
                for rv in R:
                    if not ((adj_bits[x] >> rv) & 1):
                        ok = False
                        break
                if ok:
                    L.add(x)
            if len(L) > len({u}):
                changed = True
            # Expand R
            new_R = set(R)
            for lu in L:
                for x in range(_n):
                    if x != lu and (adj_bits[lu] >> x) & 1:
                        new_R.add(x)
            R = set()
            for x in new_R:
                if x in L:
                    continue
                ok = True
                for lu in L:
                    if not ((adj_bits[x] >> lu) & 1):
                        ok = False
                        break
                if ok:
                    R.add(x)
            if len(R) > len({v}):
                changed = True
        
        if L and R:
            Lf = frozenset(L)
            Rf = frozenset(R)
            edge_set = frozenset()
            for lu in L:
                for rv in R:
                    a, b = min(lu, rv), max(lu, rv)
                    if (a, b) in edge_to_idx:
                        edge_set.add(edge_to_idx[(a, b)])
            if edge_set:
                candidates.append((Lf, Rf, edge_set))
    
    # 3. For each pair of vertices (u,v) that are NOT connected, find common neighbors
    #    and form bicliques from subsets
    for u in range(_n):
        for v in range(u+1, _n):
            if (adj_bits[u] >> v) & 1:
                continue  # skip if they are connected (already covered by edge-based)
            # Common neighbors
            common = adj_bits[u] & adj_bits[v]
            if common:
                # L = {u,v}, R = common neighbors
                # But wait, u and v are not connected, so they can't both be in L if we need cross edges
                # Actually for a biclique, we need ALL cross pairs to be edges.
                # If u and v are both in L, and w is in R, then (u,w) and (v,w) must be edges.
                # That's fine. But (u,v) doesn't need to be an edge (it's within-side).
                L = {u, v}
                R = set()
                for w in range(_n):
                    if w != u and w != v and (adj_bits[u] >> w) & 1 and (adj_bits[v] >> w) & 1:
                        R.add(w)
                if R:
                    Lf = frozenset(L)
                    Rf = frozenset(R)
                    edge_set = frozenset()
                    for lu in L:
                        for rv in R:
                            a, b = min(lu, rv), max(lu, rv)
                            if (a, b) in edge_to_idx:
                                edge_set.add(edge_to_idx[(a, b)])
                    if edge_set:
                        candidates.append((Lf, Rf, edge_set))
    
    # Deduplicate by (Lf, Rf)
    seen = set()
    unique_candidates = []
    for Lf, Rf, es in candidates:
        key = (Lf, Rf)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((Lf, Rf, es))
    
    return unique_candidates

def solve_set_cover(candidates, m, deadline):
    """Solve min set cover using branch and bound with bitsets."""
    full_mask = (1 << m) - 1
    
    # Sort candidates by size (largest first) for better pruning
    sorted_cands = sorted(candidates, key=lambda x: len(x[2]), reverse=True)
    
    # Convert edge sets to bitmasks
    cand_masks = []
    for _, _, es in sorted_cands:
        mask = 0
        for e in es:
            mask |= (1 << e)
        cand_masks.append(mask)
    
    best_count = m + 1  # Upper bound: one biclique per edge
    best_solution = None
    
    # Greedy upper bound
    covered = 0
    greedy_sol = []
    remaining_cands = list(range(len(sorted_cands)))
    while covered != full_mask and remaining_cands:
        best_i = -1
        best_gain = 0
        for i in remaining_cands:
            gain = bin(cand_masks[i] & ~covered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i == -1:
            break
        greedy_sol.append(best_i)
        covered |= cand_masks[best_i]
        remaining_cands.remove(best_i)
    if covered == full_mask:
        best_count = len(greedy_sol)
        best_solution = list(greedy_sol)
    
    # Branch and bound
    def bnb(idx, covered, count, solution):
        nonlocal best_count, best_solution
        if count >= best_count:
            return
        if covered == full_mask:
            best_count = count
            best_solution = list(solution)
            return
        
        # Lower bound: number of uncovered edges / max coverage per remaining candidate
        uncovered = full_mask & ~covered
        if uncovered == 0:
            return
        # Simple LB: at least ceil(uncovered_edges / max_possible_coverage)
        # But for now, just use the fact that each candidate covers at most some edges
        # Better LB: for each uncovered edge, we need at least one candidate that covers it
        # Use a simple greedy LB
        rem_uncovered = uncovered
        lb = 0
        while rem_uncovered:
            # Find candidate that covers the most of rem_uncovered
            best_gain = 0
            for i in range(idx, len(sorted_cands)):
                gain = bin(cand_masks[i] & rem_uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
            if best_gain == 0:
                lb += 1  # Can't cover, but this shouldn't happen if we have full coverage
                break
            lb += 1
            # Remove the best candidate's coverage
            # This is an approximation
            break  # Just use 1 as a weak LB for now, or compute properly
        
        if count + lb >= best_count:
            return
        
        # Find next uncovered edge
        # Pick the edge with fewest covering candidates (most constrained)
        best_edge = -1
        best_options = None
        best_opt_count = m + 1
        for e in range(m):
            if not (covered >> e) & 1:
                # Count how many candidates from idx onwards cover this edge
                opts = []
                for i in range(idx, len(sorted_cands)):
                    if (cand_masks[i] >> e) & 1:
                        opts.append(i)
                if len(opts) < best_opt_count:
                    best_opt_count = len(opts)
                    best_edge = e
                    best_options = opts
                    if best_opt_count <= 1:
                        break
        
        if best_options is None or len(best_options) == 0:
            return  # Can't cover this edge
        
        # Branch on each option for best_edge
        for i in best_options:
            solution.append(i)
            bnb(i + 1, covered | cand_masks[i], count + 1, solution)
            solution.pop()
    
    start_time = time.time()
    bnb(0, 0, 0, [])
    
    if best_solution is not None:
        return [sorted_cands[i] for i in best_solution]
    return None

def _validate_partition(bicliques, original_edges):
    """Check if bicliques form a valid partition of original_edges."""
    seen = set()
    edge_set_local = set()
    for u, v in original_edges:
        edge_set_local.add((min(u, v), max(u, v)))
    for L, R in bicliques:
        if not L or not R:
            return False
        for u in L:
            for v in R:
                key = (min(u, v), max(u, v))
                if key not in edge_set_local:
                    return False
                if key in seen:
                    return False
                seen.add(key)
    return seen == edge_set_local

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Generate candidates
    candidates = generate_candidates()
    
    # Solve set cover
    result = solve_set_cover(candidates, m, deadline)
    
    if result is not None:
        # Convert to required format
        final = []
        for Lf, Rf, _ in result:
            L = [int(x) for x in sorted(Lf)]
            R = [int(x) for x in sorted(Rf)]
            if L and R:
                final.append([L, R])
        
        if _validate_partition(final, _edges):
            return final
    
    # Fallback: star decomposition
    fallback = []
    for u in range(_n):
        R = [v for v in range(_n) if v != u and (adj_bits[u] >> v) & 1]
        if R:
            fallback.append([[u], [int(x) for x in sorted(R)]])
    
    return fallback
# EVOLVE-BLOCK-END
