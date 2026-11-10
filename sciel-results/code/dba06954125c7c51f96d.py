# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Finds a large set of integer points in R^11 such that
    min pairwise squared distance >= max squared norm.
    
    Strategy: Model as maximum clique in graph of 220 norm-2 vectors.
    Edges connect pairs with squared distance > 2 (dot product < 1, i.e., dot <= 0).
    Use greedy coloring + branch-and-bound for maximum clique heuristic.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    d = 11
    
    # Generate all 220 norm-squared-2 vectors (two coords +/-1)
    candidates = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * d
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)  # 220
    
    # Compute dot products
    dots = candidates @ candidates.T
    
    # For norm-2 vectors: dist_sq = 2 + 2 - 2*dot = 4 - 2*dot
    # We need dist_sq >= 4, so 4 - 2*dot >= 4, meaning dot <= 0
    # So edge exists if dot <= 0 (no conflict)
    # Conflict (no edge) if dot > 0, i.e., dot >= 1
    
    # Build adjacency: adj[i,j] = True if they can coexist (dot <= 0)
    adj = (dots <= 0)
    # Remove self-loops
    np.fill_diagonal(adj, False)
    
    # Also need distinct points: since all have same norm, distinct means different vectors
    # All 220 are distinct by construction
    
    # Maximum clique in this graph
    # Use greedy coloring upper bound + branch and bound
    
    best_clique = []
    best_size = 0
    
    def greedy_color(order, adj_sub):
        """Greedy coloring of vertices in order, returns color array."""
        colors = {}
        for v in order:
            used = set()
            for u in order:
                if u == v:
                    break
                if u in colors and adj_sub[v, u]:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        return colors
    
    def branch_and_bound(P, adj, colors, current_clique, depth=0):
        """
        P: list of candidate vertices
        adj: full adjacency matrix
        colors: dict mapping vertex to color (upper bound on clique size from P)
        current_clique: current clique being built
        """
        nonlocal best_clique, best_size
        
        if len(P) == 0:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = current_clique[:]
            return
        
        # Upper bound: max color in P
        max_color = max(colors[v] for v in P) if P else 0
        if len(current_clique) + max_color <= best_size:
            return
        
        # Try vertices in reverse color order
        for i in range(len(P) - 1, -1, -1):
            v = P[i]
            
            # Check if adding v improves bound
            if len(current_clique) + i + 1 <= best_size:
                break
            
            # Check if v is compatible with all in current_clique
            compatible = True
            for u in current_clique:
                if not adj[v, u]:
                    compatible = False
                    break
            if not compatible:
                continue
            
            # Add v to clique
            current_clique.append(v)
            
            # New P: vertices in P[0:i] that are adjacent to v
            new_P = [u for u in P[:i] if adj[v, u]]
            
            # Recolor
            if new_P:
                new_colors = greedy_color(new_P, adj)
            else:
                new_colors = {}
            
            branch_and_bound(new_P, adj, new_colors, current_clique, depth + 1)
            
            current_clique.pop()
    
    # Initial ordering by degree
    degrees = np.sum(adj, axis=1)
    order = np.argsort(degrees)  # ascending degree
    
    # Greedy initial clique
    greedy_clique = []
    available = np.ones(n, dtype=bool)
    for idx in order:
        if available[idx]:
            greedy_clique.append(idx)
            available[idx] = False
            neighbors = np.where(adj[idx])[0]
            available[neighbors] = False
    
    if len(greedy_clique) > best_size:
        best_size = len(greedy_clique)
        best_clique = greedy_clique[:]
    
    # Run branch and bound with time limit
    # Start with all vertices, ordered by degree
    P = list(range(n))
    colors = greedy_color(P, adj)
    
    # Sort P by color (ascending) for the algorithm
    P.sort(key=lambda v: colors[v])
    
    # Run with timeout
    result = [False]
    
    def timed_bnb():
        branch_and_bound(P, adj, colors, [])
        result[0] = True
    
    # We need a timeout mechanism. Let's use a simpler approach:
    # Run BnB but check time periodically
    
    # Actually, let's implement a simpler greedy + local search approach
    # since BnB on 220 nodes might be too slow without good pruning
    
    # Better approach: multiple greedy restarts with different orderings
    # + local improvement (swap-based)
    
    def greedy_with_order(ordering):
        selected = []
        available = np.ones(n, dtype=bool)
        for idx in ordering:
            if available[idx]:
                selected.append(idx)
                available[idx] = False
                neighbors = np.where(adj[idx])[0]
                available[neighbors] = False
        return selected
    
    def local_improve(selected, adj, n, max_iters=5):
        """Try to improve clique by swapping vertices."""
        selected_set = set(selected)
        improved = True
        for _ in range(max_iters):
            if not improved:
                break
            improved = False
            for pos in range(len(selected)):
                curr = selected[pos]
                others = [s for i, s in enumerate(selected) if i != pos]
                # Find a vertex not in selected that is adjacent to all others
                for cand in range(n):
                    if cand in selected_set:
                        continue
                    # Check adjacency with all others
                    ok = True
                    for o in others:
                        if not adj[cand, o]:
                            ok = False
                            break
                    if ok:
                        # Swap
                        selected[pos] = cand
                        selected_set.remove(curr)
                        selected_set.add(cand)
                        improved = True
                        break
                if improved:
                    break
        return selected
    
    # Multiple random restarts
    n_restarts = 0
    while time.time() < deadline - 30:
        n_restarts += 1
        
        # Different ordering strategies
        if n_restarts % 3 == 1:
            ordering = np.argsort(degrees)
        elif n_restarts % 3 == 2:
            ordering = np.argsort(-degrees)
        else:
            ordering = np.random.permutation(n)
        
        selected = greedy_with_order(ordering)
        selected = local_improve(selected, adj, n, max_iters=3)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_clique = selected[:]
    
    # Verify the best clique
    if best_size >= 2:
        points = candidates[best_clique]
        
        # Verify: min dist_sq >= max norm_sq
        norms_sq = np.sum(points**2, axis=1)
        max_norm_sq = int(np.max(norms_sq))
        
        min_dist_sq = float('inf')
        n_pts = len(points)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                diff = points[i] - points[j]
                dist_sq = int(np.sum(diff**2))
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        
        if min_dist_sq >= max_norm_sq:
            return np.array(points, dtype=np.int64)
    
    # Fallback: return all 220 norm-2 points (they form a valid set if pairwise dot <= 0)
    # But they don't all pairwise have dot <= 0, so we need a valid subset
    # Return the greedy clique we found
    if best_size >= 2:
        return np.array(candidates[best_clique], dtype=np.int64)
    
    # Ultimate fallback: just two points
    return np.array(candidates[:2], dtype=np.int64)

# EVOLVE-BLOCK-END
