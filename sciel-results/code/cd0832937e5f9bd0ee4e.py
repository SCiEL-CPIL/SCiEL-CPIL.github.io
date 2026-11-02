import itertools
import numpy as np
import time
import sys

D = 11

# EVOLVE-BLOCK-START
def _generate_candidates(weights):
    """Generate all integer vectors with the specified weights."""
    candidates = []
    for w in weights:
        if w > D:
            continue
        for indices in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def _color_subgraph(adj, U):
    """
    Greedy coloring of the subgraph induced by U.
    Returns a list of color classes (each a list of nodes) and the maximum color number.
    This is the key component of the BB-MCS algorithm for upper bounding.
    """
    n = len(U)
    if n == 0:
        return [], 0
    
    # Map original node indices to 0..n-1 for the subgraph
    U_list = list(U)
    idx_map = {node: i for i, node in enumerate(U_list)}
    
    # Build adjacency for the subgraph (only edges between nodes in U)
    # We store colors as an array of length n
    colors = [0] * n
    color_classes = []
    
    # Greedy coloring: process nodes in U in some order (e.g., descending degree in U)
    # For simplicity and speed, we use the order they appear in U, but sorting by degree
    # can improve the bound. Let's try a simple first-fit coloring.
    
    # To improve the bound, we can order vertices by degree in the induced subgraph
    degrees = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if adj[U_list[i]][U_list[j]]:
                degrees[i] += 1
                degrees[j] += 1
    
    # Sort vertices by degree descending
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Greedy coloring
    assigned_color = [-1] * n
    max_color = 0
    
    for idx in order:
        # Find the smallest color not used by neighbors
        used_colors = set()
        for j in range(n):
            if j != idx and assigned_color[j] != -1 and adj[U_list[idx]][U_list[j]]:
                used_colors.add(assigned_color[j])
        
        c = 0
        while c in used_colors:
            c += 1
        
        assigned_color[idx] = c
        if c > max_color:
            max_color = c
    
    # Build color classes for the return (list of lists of original node indices)
    # The BB-MCS algorithm typically returns the colors to use in the branching step
    # specifically to order the vertices in the next recursion level.
    
    # Return the color assignment for the vertices in U (in original index order)
    # and the max color
    color_assignment = [0] * n
    for i in range(n):
        color_assignment[i] = assigned_color[i]
        
    return color_assignment, max_color + 1

def _bb_mcs(adj, U, best_set, deadline):
    """
    Recursive branch and bound for Maximum Independent Set.
    adj: adjacency matrix (boolean)
    U: set of candidate vertices (list of indices)
    best_set: current best independent set (list of indices)
    deadline: time limit
    """
    if time.time() > deadline:
        return
    
    if len(U) == 0:
        return
        
    # Upper bound: |U|
    if len(U) <= len(best_set):
        return
        
    # Coloring bound
    color_assignment, max_color = _color_subgraph(adj, U)
    
    # If the coloring bound is not better than the current best, prune
    if max_color <= len(best_set):
        return
        
    # Order U by color (descending) to improve pruning
    # We want to process vertices with the highest color first
    U_sorted = sorted(U, key=lambda x: -color_assignment[U.index(x)])
    
    # Branch on each vertex in U_sorted
    for v in U_sorted:
        if time.time() > deadline:
            return
            
        # Include v in the independent set
        new_set = best_set + [v]
        
        # The next candidates must be neighbors of v in the original graph? No.
        # The next candidates must be non-neighbors of v in the original graph, 
        # AND they must be in U.
        # Also, we can restrict U to only vertices that have a color > color(v)
        # to maintain the bound properties.
        
        # Find valid next candidates:
        # 1. In U
        # 2. Not adjacent to v
        # 3. Have color > color(v) (from the coloring of the current U)
        
        # Note: The coloring was done on the induced subgraph of U.
        # The property is that any independent set in U has size at most max_color.
        # If we pick v, the remaining independent set must be in U \ {v} \ N(v).
        # The coloring of the subgraph induced by U \ {v} \ N(v) is not directly
        # given by the previous coloring. However, the BB-MCS algorithm uses the
        # coloring to order the vertices. The recursive call will re-color the
        # new subgraph.
        
        # For efficiency, we just pass the valid candidates to the next level.
        # The next level will re-color them.
        
        # Valid candidates are those in U that are not adjacent to v and not v itself.
        # We can filter based on the color to prune early?
        # The standard BB-MCS branches on vertices in order of decreasing color.
        # When we branch on v, we remove v and its neighbors.
        # The upper bound for the subproblem is the max color of the remaining vertices?
        # Actually, the bound is simply the size of the remaining U if we don't re-color,
        # but re-coloring gives a tighter bound.
        
        # Let's construct the new U
        new_U = []
        for u in U:
            if u != v and not adj[v][u]:
                new_U.append(u)
                
        # Optimization: If len(new_U) + len(new_set) <= len(best_set), prune
        if len(new_U) + len(new_set) <= len(best_set):
            continue
            
        # Recursive call
        _bb_mcs(adj, new_U, new_set, deadline)
        
        # Update best set if found a better one
        # (The recursive call modifies best_set in place if we pass a reference,
        # but here we pass a list. We need to track the global best.)
        # To make this work with Python lists, we should pass a mutable container
        # or return the best found. Let's use a global-like variable or return.
        
        # For simplicity and speed, let's use a class or nonlocal variable.
        # Since we are in a function, let's use a list to hold the best size.
        
    # This implementation above is slightly inefficient because it doesn't
    # update the global best correctly within the recursion depth without
    # passing a mutable state. Let's refactor to use a class or closure.

class MISSolver:
    def __init__(self, adj, deadline):
        self.adj = adj
        self.deadline = deadline
        self.best_set = []
        self.best_size = 0
        
    def _color_subgraph(self, U):
        n = len(U)
        if n == 0:
            return [], 0
        
        U_list = list(U)
        # Calculate degrees in induced subgraph
        degrees = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if self.adj[U_list[i]][U_list[j]]:
                    degrees[i] += 1
                    degrees[j] += 1
        
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        assigned_color = [-1] * n
        max_color = 0
        
        for idx in order:
            used_colors = set()
            for j in range(n):
                if j != idx and assigned_color[j] != -1 and self.adj[U_list[idx]][U_list[j]]:
                    used_colors.add(assigned_color[j])
            
            c = 0
            while c in used_colors:
                c += 1
            
            assigned_color[idx] = c
            if c > max_color:
                max_color = c
                
        return assigned_color, max_color + 1

    def _solve(self, U, current_set):
        if time.time() > self.deadline:
            return
        
        if len(U) == 0:
            if len(current_set) > self.best_size:
                self.best_size = len(current_set)
                self.best_set = current_set[:]
            return
            
        if len(current_set) + len(U) <= self.best_size:
            return
            
        color_assignment, max_color = self._color_subgraph(U)
        
        if max_color + len(current_set) <= self.best_size:
            return
            
        # Order U by color descending
        # We need to map original indices to their colors
        # U is a list of original indices
        U_with_colors = [(u, color_assignment[U.index(u)]) for u in U]
        U_with_colors.sort(key=lambda x: -x[1])
        
        for v, c in U_with_colors:
            if time.time() > self.deadline:
                return
                
            # If the color of v plus current set size is not better than best, 
            # and since colors are processed in descending order, we can break?
            # No, because a lower color vertex might lead to a larger set later?
            # Actually, the bound is max_color. If we process in descending color,
            # and c + len(current_set) <= self.best_size, then any subsequent
            # vertex with color <= c will also have c' + len(current_set) <= self.best_size.
            # So we can break.
            if c + len(current_set) <= self.best_size:
                break
                
            new_set = current_set + [v]
            
            new_U = []
            for u in U:
                if u != v and not self.adj[v][u]:
                    new_U.append(u)
                    
            self._solve(new_U, new_set)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Generate candidates
    # Weight 2: 220 vectors. Norm^2 = 2.
    # Weight 3: 1320 vectors. Norm^2 = 3.
    # Weight 4: 3300 vectors. Norm^2 = 4.
    
    # A mixed pool is complex. Let's try a pool of Weight 2 and Weight 3 first.
    # Or just Weight 2? 
    # Kissing number in 11 is 506? No, it's 506 for dim 11? 
    # Wait, the known kissing number for dim 11 is 506? 
    # Actually, the kissing number for dim 11 is not exactly known, but bounds are 506 <= k(11) <= 1656.
    # Wait, let's check the seed. The seed found 130 (1.2455 * 104.4? No, seed score is 1.0? 
    # The task says "score is k". The seed program in the prompt returns 110 vectors? 
    # "The seed program provides 110 vectors...". 
    # Inspiration 1 says score 1.9909. 
    # If seed score is 1.0, then k_seed = 110? 
    # If score is k, and seed score is 1.0, then k=1? No.
    # "The evaluator measures the raw value (higher is better). The search score is value/seed".
    # So if score is 1.2455, and seed is 1.0, then value = 1.2455 * seed_value.
    # If seed_value (k) is 110, then current value is ~137.
    # Let's assume the goal is to maximize k.
    
    # Let's use a candidate pool of Weight 2 and Weight 3.
    # Weight 2: 220 vectors.
    # Weight 3: 1320 vectors.
    # Total 1540. This might be too large for BB-MCS in 120s.
    # Let's try Weight 2 only first? 220 nodes.
    # Or a subset of Weight 3?
    
    # Let's try a pool of Weight 2 and a random subset of Weight 3?
    # Or just Weight 2?
    # Max independent set in the conflict graph of Weight 2 vectors.
    # Two weight 2 vectors conflict if their distance squared < max(norm1^2, norm2^2) = 2.
    # dist^2 = n1 + n2 - 2 dot = 2 + 2 - 2 dot = 4 - 2 dot.
    # Condition: 4 - 2 dot >= 2 => 2 dot <= 2 => dot <= 1.
    # Dot product of two weight 2 vectors:
    # - Disjoint support: dot = 0. OK.
    # - 1 common index: dot = +/-1. If +1, OK. If -1, OK.
    # - 2 common indices: dot = +/-2. If +2, dist^2 = 0 < 2. Conflict. If -2, dist^2 = 8 >= 2. OK.
    # So conflict only if they are identical or negatives?
    # If v1 = (1, 1, 0...), v2 = (1, 1, 0...), dot=2, dist=0. Conflict.
    # If v1 = (1, 1, 0...), v2 = (-1, -1, 0...), dot=-2, dist^2=8. OK.
    # So for Weight 2, the conflict graph is just pairs of identical vectors?
    # No, wait. 
    # v1 = (1, 1, 0), v2 = (1, 0, 1). dot = 1. dist^2 = 2. OK.
    # v1 = (1, 1, 0), v2 = (1, 1, 0). dot = 2. dist^2 = 0. Conflict.
    # So we can pick at most one from each {v, -v} pair?
    # No, we can pick v and -v.
    # Can we pick v1=(1,1,0) and v2=(1,0,1)? Yes.
    # Can we pick v1=(1,1,0), v2=(1,0,1), v3=(0,1,1)? 
    # v1.v2=1, v1.v3=1, v2.v3=1. All OK.
    # This is related to the maximum size of a code with min distance 2.
    
    # Let's build the graph for Weight 2 vectors.
    # There are 220 vectors.
    # We can solve MIS on this graph exactly.
    
    candidates = _generate_candidates([2])
    n_cand = len(candidates)
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Build conflict graph
    # adj[i][j] is True if i and j conflict.
    # Conflict if dist^2 < max(norm_i^2, norm_j^2)
    # Since all norms are 2, conflict if dist^2 < 2.
    # dist^2 = 4 - 2 dot.
    # 4 - 2 dot < 2 => 2 dot > 2 => dot > 1.
    # Since dot is integer, dot >= 2.
    # Max dot is 2 (if identical).
    # So conflict only if dot == 2, i.e., v_i == v_j.
    # This means the conflict graph has no edges between distinct vectors?
    # Wait, if v_i != v_j, dot <= 1. So dist^2 >= 2.
    # So ALL weight 2 vectors are mutually compatible?
    # Let's check: v1=(1,1,0), v2=(1,1,0). Same. Conflict.
    # v1=(1,1,0), v2=(-1,-1,0). Dot=-2. Dist^2=8. OK.
    # v1=(1,1,0), v2=(1,0,1). Dot=1. Dist^2=2. OK.
    # So we can take ALL 220 weight 2 vectors?
    # If so, k=220.
    # Score = 220 / 110 = 2.0.
    # This matches Inspiration 1 score 1.9909 (approx 2.0).
    # So the Weight 2 set is already very good.
    
    # Can we add Weight 3 vectors?
    # Weight 3 norm^2 = 3.
    # Condition with Weight 2: dist^2 >= max(2, 3) = 3.
    # dist^2 = 2 + 3 - 2 dot = 5 - 2 dot.
    # 5 - 2 dot >= 3 => 2 dot <= 2 => dot <= 1.
    # Weight 3 and Weight 2 dot product:
    # If they share 2 indices, dot can be 2 or 0 or -2.
    # If dot = 2, conflict.
    # If they share 1 index, dot can be 1 or -1. OK.
    # If they share 0 indices, dot = 0. OK.
    
    # So we want to find a large set of Weight 3 vectors such that for each Weight 3 vector,
    # it does not have dot=2 with any Weight 2 vector in the set.
    # But we are taking ALL Weight 2 vectors.
    # So a Weight 3 vector v3 is compatible with the full Weight 2 set if
    # for all v2 in W2, dot(v3, v2) <= 1.
    # Let v3 have support {i, j, k} with signs s1, s2, s3.
    # Consider v2 = (s1, s2, 0, ...). Support {i, j}.
    # dot(v3, v2) = s1^2 + s2^2 = 2.
    # This is a conflict!
    # So NO Weight 3 vector is compatible with the full set of Weight 2 vectors.
    # Because for any v3, we can find a v2 that shares 2 indices with the same signs.
    
    # So we must drop some Weight 2 vectors to add Weight 3 vectors.
    # This is a trade-off.
    # Let's solve MIS on the combined graph of W2 and a subset of W3.
    # To keep the graph small, let's pick a random subset of W3.
    # Or just use W2?
    # If we just use W2, we get 220.
    # If we use W2 + W3, we might get more.
    
    # Let's try a pool of W2 and a small number of W3s.
    # Or just run MIS on W2? It's trivial (220 nodes, no edges between distinct nodes).
    # Wait, is the graph really edgeless?
    # Yes, if all norms are 2, and dot < 2 for distinct vectors.
    # So MIS size is 220.
    
    # To improve, we need to include higher weight vectors.
    # Let's include W2 and W3.
    # N = 220 + 1320 = 1540. Too large for BB-MCS?
    # Let's try a subset.
    
    # Alternative: Use a pool of W2 and W4?
    # Or just W3?
    # W3 norm^2 = 3.
    # Conflict between two W3 vectors:
    # dist^2 = 3 + 3 - 2 dot = 6 - 2 dot.
    # Condition: 6 - 2 dot >= 3 => 2 dot <= 3 => dot <= 1.
    # So conflict if dot >= 2.
    
    # Let's run MIS on W3 only.
    # 1320 nodes.
    
    # Let's try a hybrid: Start with W2 (220), then try to add W3s one by one,
    # removing conflicting W2s if it's beneficial.
    # This is complex.
    
    # Let's just run BB-MCS on a smaller pool.
    # Pool: W2 (220) + Random 100 W3s.
    
    rng = np.random.default_rng(42)
    w2 = _generate_candidates([2])
    w3_all = _generate_candidates([3])
    w3_sample = w3_all[rng.choice(len(w3_all), 100, replace=False)]
    
    candidates = np.vstack([w2, w3_sample])
    n_cand = len(candidates)
    
    # Build adjacency matrix
    norms_sq = np.sum(candidates * candidates, axis=1)
    adj = np.zeros((n_cand, n_cand), dtype=bool)
    
    for i in range(n_cand):
        for j in range(i+1, n_cand):
            dot = np.dot(candidates[i], candidates[j])
            dist_sq = norms_sq[i] + norms_sq[j] - 2 * dot
            max_norm = max(norms_sq[i], norms_sq[j])
            if dist_sq < max_norm:
                adj[i][j] = True
                adj[j][i] = True
                
    # Solve MIS
    solver = MISSolver(adj, deadline)
    solver._solve(list(range(n_cand)), [])
    
    if solver.best_size > 0:
        best_vectors = candidates[solver.best_set]
    else:
        # Fallback to W2
        best_vectors = w2
        
    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
