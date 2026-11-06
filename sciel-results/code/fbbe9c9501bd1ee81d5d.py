import math

def construct():
    # 1. Define Moduli and Weights
    moduli = list(range(20, 201))
    n = len(moduli)
    weights = [1.0 / m for m in moduli]
    
    # 2. Build Non-Coprime Graph (Adjacency Matrix)
    # adj[i][j] is True if gcd(moduli[i], moduli[j]) > 1
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(moduli[i], moduli[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
                
    # 3. Max Weight Clique B&B
    # Sort nodes by weight descending
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    best_cliques = [] # List of (weight, [indices_in_sorted_order])
    
    # To speed up coloring, we might need to remap adjacency to sorted order
    # But n is small (181), so we can just use original indices and check adj[i][j]
    
    def get_color_bound(candidates):
        # Greedy coloring of the subgraph induced by candidates
        # Returns the upper bound on the weight of a clique in candidates
        if not candidates:
            return 0.0
        
        # colors[node] = color_id
        # We assign colors in the order of `candidates` (which is already sorted by weight desc)
        colors = {}
        max_color = -1
        color_weights = {} # color_id -> max weight in that color
        
        for node in candidates:
            # Find available colors
            used_colors = set()
            for other in candidates:
                if other == node:
                    continue
                if adj[node][other] and other in colors:
                    used_colors.add(colors[other])
            
            # Assign the smallest available color
            c = 0
            while c in used_colors:
                c += 1
            
            colors[node] = c
            if c > max_color:
                max_color = c
                color_weights[c] = 0.0
            
            if weights[node] > color_weights[c]:
                color_weights[c] = weights[node]
        
        # The bound is the sum of max weights in each color class
        # Because a clique can contain at most one node from each color class
        bound = sum(color_weights.values())
        return bound

    def expand(clique, candidates):
        # clique: list of node indices (original)
        # candidates: list of node indices (original), sorted by weight desc
        if not candidates:
            if len(best_cliques) < 5:
                w = sum(weights[i] for i in clique)
                best_cliques.append((w, clique[:]))
            return
        
        # Calculate bound
        current_w = sum(weights[i] for i in clique)
        bound = get_color_bound(candidates)
        
        # Prune if bound is too low
        if len(best_cliques) >= 5:
            min_best_w = min(w for w, _ in best_cliques)
            if current_w + bound < min_best_w:
                return
        elif best_cliques:
            # If we have some solutions, we can still prune if we are sure we can't beat the worst in top 5?
            # Actually, if we have < 5, we don't have a 5th best to compare against yet.
            # We just explore.
            pass

        # Iterate candidates
        # We iterate from the end (lowest weight in candidates? No, candidates is sorted desc)
        # Tomita iterates from the last added (lowest priority) to first?
        # Standard: Iterate i from len(candidates)-1 down to 0.
        # This allows us to remove candidates[i] from the set for subsequent branches?
        # In our implementation, we pass a new list `new_candidates` which excludes `candidates[i]` and its non-neighbors.
        
        for i in range(len(candidates) - 1, -1, -1):
            node = candidates[i]
            
            # New clique includes node
            new_clique = clique + [node]
            
            # New candidates must be neighbors of node and have index < i in sorted list
            # (to avoid duplicates and maintain order)
            new_candidates = [c for c in candidates[:i] if adj[node][c]]
            
            expand(new_clique, new_candidates)
            
            # Optional: Update bound for next iteration?
            # The bound for the next iteration (i-1) will be calculated in the recursive call?
            # No, the bound is calculated at the start of expand.
            # We don't need to update here.

    expand([], sorted_indices)
    
    # Sort best_cliques by weight descending
    best_cliques.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate?
    # The B&B might find the same clique multiple times if not careful?
    # With the "index < i" constraint, each clique is found exactly once.
    
    # 4. Try to assign residues for each top clique
    for w, clique_indices in best_cliques:
        # Map back to moduli
        m_set = [moduli[i] for i in clique_indices]
        
        # Solve CSP for residues
        result = solve_residues(m_set)
        if result is not None:
            return result
            
    # Fallback: Return a single valid pair
    return [[20, 0]]

def solve_residues(m_set):
    # Backtracking to find residues
    m_set = sorted(m_set)
    n = len(m_set)
    
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcds[i][j] = math.gcd(m_set[i], m_set[j])
            
    assignment = [None] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        
        m = m_set[idx]
        # Try residues
        # Optimization: If we fail, we can try to skip some? No, we need a valid one.
        for r in range(m):
            conflict = False
            for j in range(idx):
                if assignment[j] is not None:
                    g = gcds[idx][j]
                    if r % g == assignment[j] % g:
                        conflict = True
                        break
            if not conflict:
                assignment[idx] = r
                if backtrack(idx + 1):
                    return True
                assignment[idx] = None
        return False
        
    if backtrack(0):
        return [[m_set[i], assignment[i]] for i in range(n)]
    return None

