"""Find the largest pairwise non-coprime subset of [1,255] containing 255."""

import math
import time

_N = 255
_TARGET = 255  # 255 = 3 * 5 * 17

def _is_valid_clique(S):
    """Check that S is pairwise non-coprime and contains 255."""
    if _TARGET not in S:
        return False
    S = list(S)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if math.gcd(S[i], S[j]) == 1:
                return False
    return True

def _build_graph():
    """Build adjacency: two numbers are adjacent if gcd > 1, and both share a factor with 255."""
    # Candidates must share a factor with 255
    candidates = [x for x in range(1, _N + 1) if math.gcd(x, _TARGET) > 1]
    n = len(candidates)
    idx = {x: i for i, x in enumerate(candidates)}
    
    # Build adjacency as bitmasks for speed
    adj = [0] * n
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            if math.gcd(a, b) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    return candidates, idx, adj

def _max_clique_tomita(candidates, idx, adj, deadline):
    """
    Maximum clique using Tomita's algorithm with greedy coloring bound.
    Returns the maximum clique found before deadline.
    """
    n = len(candidates)
    target_idx = idx[_TARGET]
    
    best_clique = []
    best_size = 0
    
    # We need to include 255, so start with it
    # The clique must contain target_idx
    
    # Greedy coloring for upper bound
    def greedy_color(P, color):
        """
        Greedy coloring of the subgraph induced by P.
        P is a list of vertex indices.
        Returns (num_colors, color_list) where color_list[i] is the color of P[i].
        """
        color_count = 0
        color_list = []
        # Process vertices in order of decreasing degree within P
        P_set = set(P)
        P_sorted = sorted(P, key=lambda v: sum(1 for w in P_set if (adj[v] >> w) & 1), reverse=True)
        
        for v in P_sorted:
            # Find smallest color not used by neighbors already colored
            used_colors = set()
            for w in P_sorted:
                if w != v and (adj[v] >> w) & 1:
                    # w is a neighbor of v in P, check if already colored
                    # We need to track which vertices have been colored
                    pass
            
            # Simpler: just assign colors greedily
            # For each vertex, find smallest color not used by its colored neighbors
            pass
        
        # Let's use a simpler approach: order by degree, assign smallest available color
        colored = {}
        for v in P_sorted:
            used = set()
            for w in colored:
                if (adj[v] >> w) & 1:
                    used.add(colored[w])
            c = 0
            while c in used:
                c += 1
            colored[v] = c
            color_count = max(color_count, c + 1)
        
        # Build color_list in original P order
        color_list = [colored[v] for v in P]
        return color_count, color_list
    
    def expand(R, P):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        # Prune: if |R| + |P| <= best_size, no improvement possible
        if len(R) + len(P) <= best_size:
            return
        
        # If P is empty, update best
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Greedy coloring for upper bound
        color_count, color_list = greedy_color(P, None)
        
        # If |R| + color_count <= best_size, prune
        if len(R) + color_count <= best_size:
            return
        
        # Process vertices in order of increasing color (low color first for pruning)
        # Actually, process in order of decreasing color for better pruning
        P_with_color = [(P[i], color_list[i]) for i in range(len(P))]
        P_with_color.sort(key=lambda x: x[1])  # Sort by color
        
        for i, (v, c) in enumerate(P_with_color):
            if time.time() > deadline:
                return
            
            # Prune: if |R| + c + 1 <= best_size, skip
            if len(R) + c + 1 <= best_size:
                continue
            
            R.append(v)
            # New P: vertices in P after position i that are adjacent to v
            P_new = [w for j, (w, cw) in enumerate(P_with_color) if j > i and (adj[v] >> w) & 1]
            expand(R, P_new)
            R.pop()
            
            # If even with all remaining vertices we can't beat best, break
            if len(R) + (len(P_with_color) - i - 1) <= best_size:
                break
    
    # Start with 255 in the clique
    # P = all other candidates adjacent to 255
    P_initial = [i for i in range(n) if i != target_idx and (adj[target_idx] >> i) & 1]
    expand([target_idx], P_initial)
    
    # Convert indices back to values
    if best_clique:
        return [candidates[i] for i in best_clique]
    return [_TARGET]

def _greedy_clique(candidates, idx, adj):
    """Simple greedy: start with 255, add vertices that connect to all current members."""
    target_idx = idx[_TARGET]
    clique = [target_idx]
    
    # Order by degree descending
    other_indices = [i for i in range(len(candidates)) if i != target_idx]
    other_indices.sort(key=lambda i: sum(1 for j in other_indices if j != i and (adj[i] >> j) & 1), reverse=True)
    
    for i in other_indices:
        if (adj[target_idx] >> i) & 1:  # Must connect to 255
            # Check if i connects to all in clique
            compatible = True
            for j in clique:
                if not ((adj[i] >> j) & 1):
                    compatible = False
                    break
            if compatible:
                clique.append(i)
    
    return [candidates[i] for i in clique]

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # Safety margin
    
    # Build graph
    candidates, idx, adj = _build_graph()
    n = len(candidates)
    
    # Quick baseline: multiples of 3
    mult3 = [x for x in candidates if x % 3 == 0]
    best = list(mult3)
    best_size = len(best)
    
    # Try greedy
    greedy_result = _greedy_clique(candidates, idx, adj)
    if len(greedy_result) > best_size:
        best = greedy_result
        best_size = len(greedy_result)
    
    # Run Tomita's algorithm with deadline
    if time.time() < deadline - 5:
        result = _max_clique_tomita(candidates, idx, adj, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Verify
    if _is_valid_clique(best):
        return best
    
    # Fallback
    return mult3
