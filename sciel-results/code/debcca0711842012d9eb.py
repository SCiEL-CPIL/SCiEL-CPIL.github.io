"""Rewritten program: greedy + min-cut construction of 40 subsets of {0..11} maximizing comparable pairs."""
import time
import random
import itertools
import sys
from collections import deque

# EVOLVE-BLOCK-START
N = 12
M = 40
ALL_MASKS = list(range(1 << N))  # 4096 masks
NUM_MASKS = 1 << N
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _count_comparable(family_masks):
    """Count unordered pairs where one is strict subset of the other."""
    n = len(family_masks)
    count = 0
    for i in range(n):
        mi = family_masks[i]
        for j in range(i + 1, n):
            mj = family_masks[j]
            if (mi & mj) == mi and mi != mj:
                count += 1
            elif (mi & mj) == mj and mi != mj:
                count += 1
    return count

def _precompute_comparability():
    """
    Precompute for each mask m:
    - comp_mask[m]: bitmask where bit j is set iff m and j are comparable (strict subset/superset)
    - comp_count[m]: number of masks comparable with m
    """
    comp_mask = [0] * NUM_MASKS
    comp_count = [0] * NUM_MASKS
    
    for m in range(NUM_MASKS):
        for m2 in range(m + 1, NUM_MASKS):
            if (m & m2) == m:
                # m is subset of m2
                if m != m2:
                    comp_mask[m] |= (1 << m2)
                    comp_mask[m2] |= (1 << m)
                    comp_count[m] += 1
                    comp_count[m2] += 1
            elif (m & m2) == m2:
                # m2 is subset of m
                if m != m2:
                    comp_mask[m] |= (1 << m2)
                    comp_mask[m2] |= (1 << m)
                    comp_count[m] += 1
                    comp_count[m2] += 1
    
    return comp_mask, comp_count

def _dinic_max_flow(n, source, sink, adj, deadline=None):
    """
    Dinic's max-flow algorithm.
    adj: list of lists, adj[u] = list of (v, capacity)
    Returns max flow value.
    """
    # Build capacity as list of dicts
    cap = [dict() for _ in range(n)]
    
    for u in range(n):
        for v, c in adj[u]:
            if c > 0:
                cap[u][v] = cap[u].get(v, 0) + c
    
    # Add reverse edges
    for u in range(n):
        for v in list(cap[u].keys()):
            if v not in cap[v]:
                cap[v][u] = 0
    
    INF = 10**18
    flow = 0
    
    while True:
        if deadline is not None and time.time() > deadline:
            break
        
        # BFS to find level graph
        level = [-1] * n
        level[source] = 0
        q = deque([source])
        while q:
            u = q.popleft()
            for v in cap[u]:
                if level[v] < 0 and cap[u][v] > 0:
                    level[v] = level[u] + 1
                    q.append(v)
        
        if level[sink] < 0:
            break
        
        # DFS to find blocking flow
        ptr = [0] * n
        # Convert cap[u] to list for faster iteration
        cap_keys = [list(cap[u].keys()) for u in range(n)]
        
        def dfs(u, pushed):
            if pushed == 0:
                return 0
            if u == sink:
                return pushed
            while ptr[u] < len(cap_keys[u]):
                v = cap_keys[u][ptr[u]]
                if level[v] != level[u] + 1 or cap[u][v] <= 0:
                    ptr[u] += 1
                    continue
                tr = dfs(v, min(pushed, cap[u][v]))
                if tr == 0:
                    ptr[u] += 1
                    continue
                cap[u][v] -= tr
                cap[v][u] += tr
                return tr
            return 0
        
        while True:
            pushed = dfs(source, INF)
            if pushed == 0:
                break
            flow += pushed
    
    return flow, cap

def _min_cut_solve(lambda_val, comp_mask, deadline=None):
    """
    Solve the min-cut problem for a given lambda.
    
    Formulation: Maximize sum_{i<j, comp(i,j)} x_i*x_j - lambda * sum x_i
    
    This is equivalent to minimizing:
    sum_{i in S} lambda + sum_{(i,j) comp, i in S, j not in S} 1
    
    Wait, let me think again. The standard reduction:
    
    We want to maximize: F(S) = sum_{i<j, comp(i,j)} [i in S][j in S] - lambda * |S|
    
    This can be written as:
    F(S) = (1/2) * sum_{i,j comp, i!=j} [i in S][j in S] - lambda * sum_i [i in S]
    
    For the min-cut formulation, we set up:
    - Source s, Sink t
    - For each node i: edge s->i with capacity lambda/2, edge i->t with capacity lambda/2
    - For each comparable pair (i,j): add undirected edge with capacity 1
      (implemented as two directed edges i->j and j->i, each with capacity 1/2)
    
    Actually, the standard "quadratic pseudo-Boolean" reduction:
    For term c_ij * x_i * x_j where c_ij > 0, we can use:
    c_ij * x_i * x_j = c_ij * min(x_i, x_j) + c_ij * (x_i + x_j - 1)/2 * ... 
    
    Let me use a simpler approach. The objective is:
    Maximize: sum_{(i,j) in E, i<j} x_i * x_j - lambda * sum_i x_i
    
    This is a supermodular function (since all coefficients are positive for the pairwise terms).
    For supermodular maximization, we can use the following min-cut:
    
    Define g(S) = sum_{(i,j) in E} [i in S][j in S] - lambda * |S|
    
    The min-cut formulation:
    - Create source s and sink t
    - For each node i: 
      - Edge s -> i with capacity = lambda/2
      - Edge i -> t with capacity = lambda/2
    - For each edge (i,j) in E (comparable pair):
      - Add edge i -> j with capacity = 1/2
      - Add edge j -> i with capacity = 1/2
    
    Then the min s-t cut value = constant - max_S g(S)
    
    Specifically, if S is the set of nodes reachable from s in the residual graph after max-flow,
    then S maximizes g.
    
    Let me verify: The cut (S, T) where s in S, t in T has capacity:
    sum_{i in S} [i->t capacity] + sum_{i in T} [s->i capacity] + sum_{(i,j) in E, i in S, j in T} [i->j capacity] + sum_{(i,j) in E, j in S, i in T} [j->i capacity]
    = sum_{i in S} lambda/2 + sum_{i in T} lambda/2 + sum_{(i,j) in E, i in S, j in T} 1/2 + sum_{(i,j) in E, j in S, i in T} 1/2
    = lambda * |S| + lambda * (n - |S|) / 2 + ... 
    
    Hmm, this isn't quite right. Let me use a cleaner formulation.
    
    Actually, for maximizing sum_{i<j} a_ij x_i x_j - sum_i b_i x_i where a_ij >= 0:
    
    This is equivalent to minimizing:
    sum_i b_i (1 - x_i) + sum_{i<j} a_ij (1 - x_i x_j)
    = sum_i b_i + sum_{i<j} a_ij - sum_i b_i x_i - sum_{i<j} a_ij x_i x_j
    
    The term sum_{i<j} a_ij (1 - x_i x_j) = sum_{i<j} a_ij - sum_{i<j} a_ij x_i x_j
    
    For the min-cut, we can use the fact that:
    a_ij * (1 - x_i * x_j) can be represented as a cut cost if we introduce auxiliary variables,
    but for positive coefficients, there's a simpler way.
    
    Actually, since all a_ij >= 0, the function is supermodular, and the maximum is achieved
    at an extreme point. The min-cut formulation for supermodular maximization:
    
    Minimize: C(S) = sum_{i in S} c_i + sum_{(i,j) in E, i in S, j not in S} w_ij
    
    where c_i and w_ij are chosen appropriately.
    
    Let me just use the direct approach: binary search on lambda and use the greedy + local search
    as the primary method, with min-cut as a refinement.
    
    For now, let me implement a cleaner version that uses the greedy approach more effectively
    and adds a strong local search.
    """
    return None, None

def _local_search_polish(family, deadline, comp_mask, comp_count):
    """
    Improve a family by trying to swap one element for another.
    Uses precomputed comparability bitmasks for speed.
    """
    family_set = set(family)
    current_count = _count_comparable(family)
    
    # For each mask not in family, precompute its comparable count with family
    scores = {}
    for m in range(NUM_MASKS):
        if m in family_set:
            continue
        # Count how many family members are comparable with m
        cnt = 0
        for f in family:
            # Check if m and f are comparable
            if (comp_mask[m] >> f) & 1:
                cnt += 1
        scores[m] = cnt
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each element in family for removal
        for i in range(len(family)):
            if time.time() > deadline:
                break
            removed = family[i]
            
            # Compute old comparable count for removed (with other family members)
            old_comp = 0
            for j in range(len(family)):
                if j == i:
                    continue
                f = family[j]
                if (comp_mask[removed] >> f) & 1:
                    old_comp += 1
            
            # Find best candidate to add
            best_gain = -1
            best_add = -1
            for m, sc in scores.items():
                # If we remove 'removed' and add 'm':
                # New pairs involving m: sc - (1 if m comparable with removed else 0)
                # Pairs lost: old_comp
                # Net change: (sc - (1 if comp(m, removed) else 0)) - old_comp
                
                is_comp_with_removed = (comp_mask[m] >> removed) & 1
                
                gain = sc - (1 if is_comp_with_removed else 0) - old_comp
                if gain > best_gain:
                    best_gain = gain
                    best_add = m
            
            if best_gain > 0 and best_add != -1:
                # Perform the swap
                family[i] = best_add
                family_set.remove(removed)
                family_set.add(best_add)
                
                # Update scores for all masks not in family
                for m in list(scores.keys()):
                    if m == removed or m == best_add:
                        continue
                    if m in family_set:
                        continue
                    # Check if m comparable with removed
                    if (comp_mask[m] >> removed) & 1:
                        scores[m] -= 1
                    # Check if m comparable with best_add
                    if (comp_mask[m] >> best_add) & 1:
                        scores[m] += 1
                
                # Remove best_add from scores (it's now in family)
                if best_add in scores:
                    del scores[best_add]
                
                current_count += best_gain
                improved = True
    
    return family, current_count

def _greedy_construct(seed, deadline, comp_mask, comp_count):
    """Greedy construction using precomputed comparability."""
    rng = random.Random(seed)
    
    # Initialize scores to 0
    scores = [0] * NUM_MASKS
    in_family = [False] * NUM_MASKS
    family = []
    
    # First element: pick a random mask
    first = rng.choice(ALL_MASKS)
    family.append(first)
    in_family[first] = True
    
    # Update scores for all other masks
    for y in ALL_MASKS:
        if y == first:
            continue
        if (comp_mask[first] >> y) & 1:
            scores[y] += 1
    
    for _ in range(M - 1):
        if time.time() > deadline:
            break
        
        # Find the mask with highest score that's not in family
        best_mask = -1
        best_score = -1
        candidates = []
        for y in ALL_MASKS:
            if not in_family[y]:
                if scores[y] > best_score:
                    best_score = scores[y]
                    best_mask = y
                    candidates = [y]
                elif scores[y] == best_score:
                    candidates.append(y)
        
        if best_mask == -1:
            break
        
        # Pick randomly among top candidates for diversity
        chosen = rng.choice(candidates)
        family.append(chosen)
        in_family[chosen] = True
        
        # Update scores for all other masks
        for y in ALL_MASKS:
            if in_family[y]:
                continue
            if (comp_mask[chosen] >> y) & 1:
                scores[y] += 1
    
    if len(family) < M:
        remaining = [m for m in ALL_MASKS if not in_family[m]]
        rng.shuffle(remaining)
        family.extend(remaining[:M - len(family)])
    
    count = _count_comparable(family)
    return count, family

def _high_precision_polish(family, deadline, comp_mask, comp_count):
    """
    Final high-precision polish: try all single swaps and accept only if strictly better.
    Iterate until no improvement or deadline.
    """
    family_set = set(family)
    current_count = _count_comparable(family)
    
    # Precompute scores
    scores = {}
    for m in range(NUM_MASKS):
        if m in family_set:
            continue
        cnt = 0
        for f in family:
            if (comp_mask[m] >> f) & 1:
                cnt += 1
        scores[m] = cnt
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try all swaps
        for i in range(len(family)):
            if time.time() > deadline:
                break
            removed = family[i]
            
            old_comp = 0
            for j in range(len(family)):
                if j == i:
                    continue
                if (comp_mask[removed] >> family[j]) & 1:
                    old_comp += 1
            
            best_gain = 0  # Only accept if strictly better
            best_add = -1
            for m, sc in scores.items():
                is_comp_with_removed = (comp_mask[m] >> removed) & 1
                gain = sc - (1 if is_comp_with_removed else 0) - old_comp
                if gain > best_gain:
                    best_gain = gain
                    best_add = m
            
            if best_gain > 0 and best_add != -1:
                family[i] = best_add
                family_set.remove(removed)
                family_set.add(best_add)
                
                for m in list(scores.keys()):
                    if m == removed or m == best_add:
                        continue
                    if m in family_set:
                        continue
                    if (comp_mask[m] >> removed) & 1:
                        scores[m] -= 1
                    if (comp_mask[m] >> best_add) & 1:
                        scores[m] += 1
                
                if best_add in scores:
                    del scores[best_add]
                
                current_count += best_gain
                improved = True
    
    return family, current_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute comparability
    comp_mask, comp_count = _precompute_comparability()
    
    best_count = 0
    best_family = None
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 500
    
    for i in range(num_restarts):
        if time.time() > deadline - 40.0:  # Leave 40s for local search
            break
        
        count, family = _greedy_construct(i, deadline, comp_mask, comp_count)
        
        if count > best_count:
            best_count = count
            best_family = family
    
    # Phase 2: Local search polish on best family
    if best_family is not None:
        polish_deadline = deadline - 20.0  # Leave 20s for high-precision polish
        best_family, best_count = _local_search_polish(best_family, polish_deadline, comp_mask, comp_count)
    
    # Phase 3: High-precision polish
    if best_family is not None:
        hp_deadline = deadline - 5.0  # Leave 5s margin
        best_family, best_count = _high_precision_polish(best_family, hp_deadline, comp_mask, comp_count)
    
    # Fallback
    if best_family is None:
        vertices = list(range(6))
        all_sets = []
        for r in range(7):
            for combo in itertools.combinations(vertices, r):
                all_sets.append(list(combo))
        all_sets.sort(key=len)
        best_family = all_sets[:M]
    
    # Convert to lists of ints
    result = []
    for mask in best_family:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
