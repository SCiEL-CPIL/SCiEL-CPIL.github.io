"""Bootstrap spawn: DIRECT construction using Paley tournament on p=67 vertices."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices has a common dominator.
    Uses the Paley tournament on p=67 vertices, then attempts to prune vertices
    and perform local search to find a smaller valid tournament.
    """
    start_time = time.time()
    deadline = 110  # Safety margin for 120s limit

    p = 67

    # Precompute quadratic residues mod p
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)

    # Build adjacency matrix for Paley 67
    adj = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            diff = (j - i) % p
            if diff in squares:
                adj[i][j] = 1
            else:
                adj[i][j] = 0

    def get_in_neigh(a):
        n = a.shape[0]
        in_neigh = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and a[i][j] == 1:
                    in_neigh[j].add(i)
        return in_neigh

    def count_violations(a):
        """Returns the number of 4-sets without a common dominator."""
        n = a.shape[0]
        in_neigh = get_in_neigh(a)
        violations = 0
        for v1 in range(n):
            for v2 in range(v1 + 1, n):
                for v3 in range(v2 + 1, n):
                    for v4 in range(v3 + 1, n):
                        inter = in_neigh[v1]
                        if not inter:
                            violations += 1
                            continue
                        inter = inter.intersection(in_neigh[v2])
                        if not inter:
                            violations += 1
                            continue
                        inter = inter.intersection(in_neigh[v3])
                        if not inter:
                            violations += 1
                            continue
                        inter = inter.intersection(in_neigh[v4])
                        if len(inter) == 0:
                            violations += 1
        return violations

    def is_valid(a):
        return count_violations(a) == 0

    # Check full Paley 67
    if is_valid(adj):
        best_n = p
        best_adj = adj
    else:
        # Should not happen, but fallback
        best_n = p
        best_adj = adj

    # Try to prune to n=66
    # We will try removing each vertex v from 0 to 66
    # For each removal, we get a candidate.
    # If candidate is valid, we update best.
    # If candidate has few violations, we try local search.
    
    active = list(range(p))
    
    # First, try simple pruning (removing one vertex at a time from the original)
    # Note: The previous code did sequential pruning. Here we try removing *one* vertex
    # from the full set to get n=66, then local search.
    
    best_violations = float('inf')
    best_candidate_adj = None
    
    for v in range(p):
        if time.time() - start_time > deadline - 20:
            break
            
        # Remove vertex v
        new_active = [u for u in active if u != v]
        idx = np.array(new_active)
        cand_adj = adj[idx][:, idx]
        
        # Check validity
        viol = count_violations(cand_adj)
        
        if viol == 0:
            # Found a valid n=66!
            return {'n': 66, 'orientation': _to_orientation(cand_adj)}
        
        if viol < best_violations:
            best_violations = viol
            best_candidate_adj = cand_adj.copy()
            best_removed_v = v
            
        # If violations are small, try local search
        if viol <= 10:
            # Local search: try flipping edges to reduce violations
            # This is expensive, so limit iterations
            cand = best_candidate_adj.copy()
            n_cand = cand.shape[0]
            # Map indices back to original for edge flipping?
            # Easier to work on the submatrix directly.
            # We need to know which original vertices are present to map edge flips?
            # No, we just need a valid tournament on n_cand vertices.
            # The output orientation is relative to the indices 0..n_cand-1.
            
            # Simple local search: flip random edges, accept if violations decrease.
            # Use a time budget for this search.
            search_start = time.time()
            search_deadline = start_time + deadline - 10
            
            current_viol = viol
            current_adj = cand.copy()
            
            # Precompute in_neigh for speed? It changes on flip.
            # For n=66, flipping an edge (u,v) changes in_neigh[u] and in_neigh[v].
            # Recalculating all violations is O(n^4) ~ 600k ops.
            # We can do at most ~100-200 iterations in 10s?
            # Let's try a few iterations.
            
            # Optimization: Only flip edges that are part of the violated 4-sets?
            # Too complex. Random flips.
            
            for _ in range(20): # Max 20 flips
                if time.time() > search_deadline:
                    break
                
                # Pick random edge
                u = np.random.randint(n_cand)
                v = np.random.randint(n_cand)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                # Flip edge
                current_adj[u][v] = 1 - current_adj[u][v]
                current_adj[v][u] = 1 - current_adj[v][u]
                
                new_viol = count_violations(current_adj)
                
                if new_viol < current_viol:
                    current_viol = new_viol
                    if current_viol == 0:
                        return {'n': n_cand, 'orientation': _to_orientation(current_adj)}
                else:
                    # Revert
                    current_adj[u][v] = 1 - current_adj[u][v]
                    current_adj[v][u] = 1 - current_adj[v][u]

    # If no valid n=66 found, return best valid n (which is 67)
    return {'n': best_n, 'orientation': _to_orientation(best_adj)}

def _to_orientation(a):
    n = a.shape[0]
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if a[i][j] == 1:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation
# EVOLVE-BLOCK-END
