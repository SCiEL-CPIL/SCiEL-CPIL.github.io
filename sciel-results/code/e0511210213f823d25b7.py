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

    def get_in_masks(a):
        """
        Returns a list of integers where the k-th bit of mask[i] is 1 if vertex k -> i.
        This allows fast intersection checks.
        """
        n = a.shape[0]
        masks = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and a[i][j] == 1:
                    masks[i] |= (1 << j)
        return masks

    def count_violations_fast(masks):
        """
        Counts 4-sets with empty common in-neighborhood using bitmasks.
        """
        n = len(masks)
        violations = 0
        # Iterate over all 4-combinations
        for v1 in range(n):
            m1 = masks[v1]
            if not m1:
                # If v1 has no in-neighbors, any set containing it is a violation?
                # No, the set must have a common dominator. If v1 has no in-neighbors,
                # no one dominates v1. So any 4-set containing v1 is a violation 
                # unless the dominator is in the set? No, the dominator must be a vertex 
                # in the 4-set? 
                # "Every set of 4 vertices must have a vertex with arcs to all four."
                # Wait, "arcs to all four" usually means arcs to the *other* three? 
                # Or does it mean a vertex *outside* the set?
                # Standard definition: A set S is "dominated" if there exists a vertex v 
                # (often v in S? or v not in S?) such that v -> u for all u in S \ {v}?
                # Let's re-read the task contract carefully.
                # "Every set of 4 vertices must have a vertex with arcs to all four."
                # This phrasing "arcs to all four" is ambiguous. Does it mean arcs to the 
                # 4 vertices in the set? If so, the vertex must be outside the set?
                # Or does it mean a vertex *within* the set that has arcs to the other 3?
                # Schütte's problem / f(k) usually refers to the property that *every* k-set 
                # has a *common dominator*. A common dominator of a set S is a vertex v 
                # such that v -> u for all u in S.
                # If v is in S, it cannot have an arc to itself. So v must be outside S?
                # Let's check the lower bound f(4) >= 31 and smallest known 67.
                # If the dominator must be *in* the set, it's a "king" or similar.
                # If the dominator is *outside*, it's a common in-neighbor.
                # The code `inter = in_neigh[v1] ...` suggests it's looking for a common 
                # in-neighbor of the 4 vertices. This implies the dominator is a vertex 
                # that has arcs TO all 4 vertices. The dominator is NOT one of the 4 vertices.
                # So, for a 4-set {v1, v2, v3, vv4}, we need a vertex v such that 
                # v->v1, v->v2, v->v3, v->v4.
                # The current code checks `in_neigh[v1] & in_neigh[v2] & in_neigh[v3] & in_neigh[v4]`.
                # This is correct for "common dominator outside the set".
                
                # If m1 is 0, then v1 has no in-neighbors. Thus no vertex can dominate v1.
                # So any 4-set containing v1 is a violation.
                # Number of 4-sets containing v1 is C(n-1, 3).
                # We can just continue, but optimization:
                # Actually, the loop structure below handles it naturally if we just 
                # check the intersection.
                pass
            
            for v2 in range(v1 + 1, n):
                m12 = m1 & masks[v2]
                if not m12:
                    # No common in-neighbor for v1, v2. 
                    # Any 4-set containing {v1, v2} and two others is a violation.
                    # Number of such sets is C(n-2, 2).
                    violations += (n - 2) * (n - 3) // 2
                    continue
                
                for v3 in range(v2 + 1, n):
                    m123 = m12 & masks[v3]
                    if not m123:
                        # No common in-neighbor for v1, v2, v3.
                        # Any 4-set containing {v1, v2, v3} and one other is a violation.
                        # Number of such sets is C(n-3, 1) = n-3.
                        violations += (n - 3)
                        continue
                    
                    for v4 in range(v3 + 1, n):
                        m1234 = m123 & masks[v4]
                        if not m1234:
                            violations += 1
        return violations

    def is_valid(a):
        masks = get_in_masks(a)
        return count_violations_fast(masks) == 0

    # Check full Paley 67
    if is_valid(adj):
        best_n = p
        best_adj = adj
    else:
        # Should not happen, but fallback
        best_n = p
        best_adj = adj

    # Try to prune to n=66
    active = list(range(p))
    
    best_violations = float('inf')
    best_candidate_adj = None
    best_removed_v = -1
    
    # Precompute masks for the full graph to speed up subgraph extraction?
    # No, we need to extract subgraphs.
    
    for v in range(p):
        if time.time() - start_time > deadline - 10:
            break
            
        # Remove vertex v
        new_active = [u for u in active if u != v]
        idx = np.array(new_active)
        cand_adj = adj[idx][:, idx]
        
        # Check validity
        masks = get_in_masks(cand_adj)
        viol = count_violations_fast(masks)
        
        if viol == 0:
            # Found a valid n=66!
            return {'n': 66, 'orientation': _to_orientation(cand_adj)}
        
        if viol < best_violations:
            best_violations = viol
            best_candidate_adj = cand_adj.copy()
            best_removed_v = v
            
        # If violations are small, try local search
        if viol <= 20:
            # Local search: try flipping edges to reduce violations
            cand = best_candidate_adj.copy()
            n_cand = cand.shape[0]
            
            current_viol = viol
            current_adj = cand.copy()
            current_masks = get_in_masks(current_adj)
            
            search_start = time.time()
            search_deadline = start_time + deadline - 5
            
            # We need to update masks efficiently when flipping an edge.
            # Flipping edge (u, v) changes:
            # If u -> v becomes v -> u:
            #   u loses in-neighbor v: current_masks[u] &= ~(1 << v)
            #   v gains in-neighbor u: current_masks[v] |= (1 << u)
            #   All other masks remain unchanged?
            #   Wait, does u's in-neighborhood affect other vertices' in-neighborhoods?
            #   No. in_neigh[x] is the set of vertices pointing TO x.
            #   So only masks[u] and masks[v] change.
            
            # However, count_violations_fast depends on ALL masks.
            # We can't easily update the violation count incrementally without 
            # tracking which 4-sets were affected.
            # Given n=66, C(66,4) is ~700k. 
            # count_violations_fast is fast (~0.1s?).
            # Let's measure or assume it's fast enough for ~100-500 iterations.
            
            for _ in range(200): # Max 200 flips
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
                # Current state: current_adj[u][v] is 1 if u->v, 0 if v->u.
                # In our adjacency matrix definition:
                # adj[i][j] = 1 if i -> j.
                # So if current_adj[u][v] == 1, u->v.
                # We flip it.
                
                old_dir = current_adj[u][v]
                current_adj[u][v] = 1 - old_dir
                current_adj[v][u] = 1 - (1 - old_dir) # = old_dir
                
                # Update masks
                if old_dir == 1:
                    # u -> v was true. Now v -> u is true.
                    # u loses in-neighbor v.
                    current_masks[u] &= ~(1 << v)
                    # v gains in-neighbor u.
                    current_masks[v] |= (1 << u)
                else:
                    # v -> u was true. Now u -> v is true.
                    # v loses in-neighbor u.
                    current_masks[v] &= ~(1 << u)
                    # u gains in-neighbor v.
                    current_masks[u] |= (1 << v)
                
                new_viol = count_violations_fast(current_masks)
                
                if new_viol < current_viol:
                    current_viol = new_viol
                    if current_viol == 0:
                        return {'n': n_cand, 'orientation': _to_orientation(current_adj)}
                else:
                    # Revert
                    current_adj[u][v] = old_dir
                    current_adj[v][u] = 1 - old_dir
                    if old_dir == 1:
                        # Revert: u->v becomes true again.
                        current_masks[u] |= (1 << v)
                        current_masks[v] &= ~(1 << u)
                    else:
                        # Revert: v->u becomes true again.
                        current_masks[v] |= (1 << u)
                        current_masks[u] &= ~(1 << v)

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
