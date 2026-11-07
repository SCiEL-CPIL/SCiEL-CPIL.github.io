"""Cochromatic partition solver using branch-and-bound over maximal cliques/independent sets."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _is_clique(S, adj):
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if S[j] not in adj[S[i]]:
                return False
    return True

def _is_independent(S, adj):
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if S[j] in adj[S[i]]:
                return False
    return True

def _find_maximal_sets(adj, is_type):
    """Find all maximal sets that are either cliques (is_type=True) or independent sets (is_type=False)."""
    n = P['n']
    all_sets = []
    all_vertices = set(range(n))
    
    # Enumerate all subsets and check maximality
    # For n=14, 2^14 = 16384 subsets, manageable
    for mask in range(1, 1 << n):
        S = tuple(v for v in range(n) if mask & (1 << v))
        if len(S) < 1:
            continue
        if is_type:
            if not _is_clique(S, adj):
                continue
        else:
            if not _is_independent(S, adj):
                continue
        # Check maximality: no vertex outside S can be added
        S_set = set(S)
        maximal = True
        for v in range(n):
            if v in S_set:
                continue
            trial = S_set | {v}
            if is_type:
                if _is_clique(tuple(trial), adj):
                    maximal = False
                    break
            else:
                if _is_independent(tuple(trial), adj):
                    maximal = False
                    break
        if maximal:
            all_sets.append(S)
    return all_sets

def _solve_cochromatic(adj):
    """Solve cochromatic partition using branch-and-bound with stronger bounds."""
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Find all maximal cliques and maximal independent sets
    max_cliques = _find_maximal_sets(adj, True)
    max_indep = _find_maximal_sets(adj, False)
    
    # All candidate sets (deduplicated)
    seen = set()
    candidates = []
    for S in max_cliques:
        fs = frozenset(S)
        if fs not in seen:
            seen.add(fs)
            candidates.append(S)
    for S in max_indep:
        fs = frozenset(S)
        if fs not in seen:
            seen.add(fs)
            candidates.append(S)
    
    # Sort candidates by size descending, then by "density" (edges within for cliques, non-edges for indep)
    def candidate_key(S):
        size = len(S)
        # For larger sets, prefer those with more internal structure
        return (-size, -sum(1 for i in range(len(S)) for j in range(i+1, len(S)) if S[j] in adj[S[i]]))
    
    candidates.sort(key=candidate_key)
    
    # Convert to bitmask representation
    candidate_masks = []
    for S in candidates:
        mask = 0
        for v in S:
            mask |= (1 << v)
        candidate_masks.append(mask)
    
    num_candidates = len(candidate_masks)
    
    # Precompute: for each vertex, which candidate indices contain it
    vertex_candidates = [[] for _ in range(n)]
    for i, cm in enumerate(candidate_masks):
        for v in range(n):
            if cm & (1 << v):
                vertex_candidates[v].append(i)
    
    # Precompute conflict matrix: can two candidates be in the same partition?
    # Actually, we need: for a given remaining mask, what's the max clique/indep we can form?
    
    # Greedy initial solution - try multiple strategies
    best_count = n
    best_solution = None
    
    # Strategy 1: largest first
    remaining = full_mask
    greedy_sol = []
    while remaining:
        best_idx = -1
        best_covered = 0
        for i in range(num_candidates):
            covered = candidate_masks[i] & remaining
            cnt = bin(covered).count('1')
            if cnt > best_covered:
                best_covered = cnt
                best_idx = i
        if best_idx >= 0:
            greedy_sol.append(candidates[best_idx])
            remaining &= ~candidate_masks[best_idx]
    if len(greedy_sol) < best_count:
        best_count = len(greedy_sol)
        best_solution = greedy_sol[:]
    
    # Strategy 2: most constrained first (vertex with fewest options)
    remaining = full_mask
    greedy_sol2 = []
    while remaining:
        # Find vertex with fewest candidate options
        hardest_v = -1
        min_opts = float('inf')
        for v in range(n):
            if remaining & (1 << v):
                cnt = sum(1 for i in vertex_candidates[v] if candidate_masks[i] & remaining)
                if cnt < min_opts:
                    min_opts = cnt
                    hardest_v = v
        if hardest_v < 0:
            break
        # Pick best candidate for this vertex
        best_idx = -1
        best_score = -1
        for i in vertex_candidates[hardest_v]:
            if candidate_masks[i] & remaining:
                covered = candidate_masks[i] & remaining
                cnt = bin(covered).count('1')
                if cnt > best_score:
                    best_score = cnt
                    best_idx = i
        if best_idx >= 0:
            greedy_sol2.append(candidates[best_idx])
            remaining &= ~candidate_masks[best_idx]
    if len(greedy_sol2) < best_count:
        best_count = len(greedy_sol2)
        best_solution = greedy_sol2[:]
    
    # Memoization: best (minimum) count seen for a remaining mask
    memo = {}
    
    # Time limit with safety margin
    deadline = time.time() + 55
    
    def lower_bound(remaining_mask):
        """Compute a stronger lower bound on the number of additional sets needed."""
        if remaining_mask == 0:
            return 0
        vertices = [v for v in range(n) if remaining_mask & (1 << v)]
        uncovered = len(vertices)
        if uncovered == 0:
            return 0
        
        # Basic bound: max coverage per set
        max_cover = 0
        for i in range(num_candidates):
            covered = candidate_masks[i] & remaining_mask
            cnt = bin(covered).count('1')
            if cnt > max_cover:
                max_cover = cnt
        if max_cover == 0:
            return uncovered
        lb1 = (uncovered + max_cover - 1) // max_cover
        
        # Stronger bound: consider that each part must be clique or independent set
        # Count edges within remaining vertices
        edge_count = 0
        non_edge_count = 0
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if vertices[j] in adj[vertices[i]]:
                    edge_count += 1
                else:
                    non_edge_count += 1
        
        # In a clique of size k, all C(k,2) pairs are edges
        # In an independent set of size k, all C(k,2) pairs are non-edges
        # So if we have e edges and ne non-edges in remaining,
        # each clique part can account for at most C(size,2) edges
        # each independent set part can account for at most C(size,2) non-edges
        
        # Upper bound on edges that can be covered by t cliques:
        # If we use t sets, max edges covered is when we make them as large as possible
        # This is complex, so use a simpler bound:
        # Each set of size s can cover at most C(s,2) edges OR C(s,2) non-edges
        # Total pairs = C(uncovered, 2) = edge_count + non_edge_count
        
        # Simple bound: we need at least enough sets to "absorb" all pairs
        # If all sets are cliques, max edges per set of size s is C(s,2)
        # If all sets are independent, max non-edges per set of size s is C(s,2)
        
        # Use a heuristic: if edge_count is high, we need more cliques
        # if non_edge_count is high, we need more independent sets
        
        # A set of size s contributes at most C(s,2) to either edges or non-edges
        # Total capacity for t sets of average size s_avg: t * C(s_avg, 2)
        # But this is loose. Let's use: we need t such that t * max_possible_pairs_per_set >= total_pairs
        # max_possible_pairs_per_set for a set covering 'covered' vertices is C(covered, 2)
        
        # Better: iterate over possible number of sets and check feasibility
        # For t sets, what's the max number of pairs we can "explain"?
        # This is still complex. Use a simpler approach:
        
        # Each vertex must be in some set. If a set has k vertices, it can be clique (all edges) or independent (no edges).
        # The constraint is that within each set, either ALL pairs are edges or NO pairs are edges.
        
        # Lower bound: consider the maximum independent set in the remaining graph
        # and the maximum clique in the remaining graph
        max_indep_size = 1
        max_clique_size = 1
        for i in range(num_candidates):
            covered = candidate_masks[i] & remaining_mask
            if covered:
                size = bin(covered).count('1')
                if size > max_indep_size:
                    max_indep_size = size
                if size > max_clique_size:
                    max_clique_size = size
        
        # We need at least ceil(uncovered / max_size) sets where max_size is max of max clique and max independent
        max_size = max(max_indep_size, max_clique_size)
        lb2 = (uncovered + max_size - 1) // max_size
        
        return max(lb1, lb2)
    
    def dfs(remaining_mask, current_count, current_sol):
        nonlocal best_count, best_solution
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if current_count < best_count:
                best_count = current_count
                best_solution = current_sol[:]
            return
        
        # Check memo: if we've seen this mask with <= current_count, prune
        if remaining_mask in memo:
            if memo[remaining_mask] <= current_count:
                return
        else:
            memo[remaining_mask] = current_count
        
        # Lower bound pruning
        if current_count + lower_bound(remaining_mask) >= best_count:
            return
        
        # Pick the hardest vertex (fewest usable candidate options)
        hardest_v = -1
        min_cnt = float('inf')
        for v in range(n):
            if remaining_mask & (1 << v):
                cnt = 0
                for i in vertex_candidates[v]:
                    if candidate_masks[i] & remaining_mask:
                        cnt += 1
                if cnt < min_cnt:
                    min_cnt = cnt
                    hardest_v = v
                    if cnt <= 1:
                        break
        
        if hardest_v < 0:
            return
        
        # Try all candidates that contain hardest_v and cover at least one remaining vertex
        # Sort by coverage descending for better pruning
        options = []
        for i in vertex_candidates[hardest_v]:
            if candidate_masks[i] & remaining_mask:
                covered = candidate_masks[i] & remaining_mask
                cnt = bin(covered).count('1')
                options.append((cnt, i))
        options.sort(reverse=True)
        
        for cnt, i in options:
            new_remaining = remaining_mask & ~candidate_masks[i]
            current_sol.append(candidates[i])
            dfs(new_remaining, current_count + 1, current_sol)
            current_sol.pop()
            if current_count + 1 >= best_count:
                break
    
    dfs(full_mask, 0, [])
    
    if best_solution is None:
        return [[v] for v in range(n)]
    return best_solution

def construct():
    adj = _build_adj()
    solution = _solve_cochromatic(adj)
    
    # Validate: every part must be a clique or independent set
    edge_set = set()
    for a, b in P['edges']:
        edge_set.add((a, b))
        edge_set.add((b, a))
    
    for part in solution:
        part_set = set(part)
        is_clique = True
        is_indep = True
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                if (part[i], part[j]) in edge_set:
                    is_indep = False
                else:
                    is_clique = False
        if not is_clique and not is_indep:
            # Fallback: use greedy
            parts = []
            for v in range(P['n']):
                for part in parts:
                    trial = part + [v]
                    hits = [(u, w) in edge_set or (w, u) in edge_set 
                           for i, u in enumerate(trial) for w in trial[i+1:]]
                    if all(hits) or not any(hits):
                        part.append(v)
                        break
                else:
                    parts.append([v])
            return [list(p) for p in parts]
    
    # Ensure all vertices are covered exactly once
    all_vertices = set()
    for part in solution:
        for v in part:
            if v in all_vertices:
                # Duplicate, fallback
                parts = []
                for v2 in range(P['n']):
                    for part in parts:
                        trial = part + [v2]
                        hits = [(u, w) in edge_set or (w, u) in edge_set 
                               for i, u in enumerate(trial) for w in trial[i+1:]]
                        if all(hits) or not any(hits):
                            part.append(v2)
                            break
                    else:
                        parts.append([v2])
                return [list(p) for p in parts]
            all_vertices.add(v)
    
    if all_vertices != set(range(P['n'])):
        # Missing vertices, fallback
        parts = []
        for v in range(P['n']):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in edge_set or (w, u) in edge_set 
                       for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return [list(p) for p in parts]
    
    return [list(p) for p in solution]
# EVOLVE-BLOCK-END
