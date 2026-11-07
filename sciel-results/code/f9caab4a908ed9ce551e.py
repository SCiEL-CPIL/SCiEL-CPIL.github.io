"""Cochromatic partition solver: Branch-and-bound with strong lower bounds."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    deadline = time.time() + 55.0
    n = P['n']
    full_mask = (1 << n) - 1

    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute complement adjacency
    comp_adj = [0] * n
    for u in range(n):
        comp_adj[u] = full_mask ^ (1 << u) ^ adj[u]

    # Precompute is_clique and is_ind_set for all masks
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True

    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb

        if is_clique[rest]:
            if (adj[v] & rest) == rest:
                is_clique[mask] = True

        if is_ind[rest]:
            if (adj[v] & rest) == 0:
                is_ind[mask] = True

    # Precompute omega[mask] (max clique size in induced subgraph of mask)
    # and alpha[mask] (max independent set size in induced subgraph of mask)
    omega = [0] * (1 << n)
    alpha = [0] * (1 << n)
    
    # omega[0] = 0, alpha[0] = 0
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Option 1: v is not in the max clique/ind set
        # Option 2: v is in the max clique/ind set
        # If v is in max clique, the rest of the clique must be in N(v) intersect rest
        # So size is 1 + omega[N(v) & rest]
        # If v is not in max clique, size is omega[rest]
        
        n_v_rest = adj[v] & rest
        omega[mask] = max(omega[rest], 1 + omega[n_v_rest])
        
        cn_v_rest = comp_adj[v] & rest
        alpha[mask] = max(alpha[rest], 1 + alpha[cn_v_rest])

    # Lower bound function
    def get_lower_bound(rem_mask):
        if rem_mask == 0:
            return 0
        count = rem_mask.bit_count()
        max_part_size = max(omega[rem_mask], alpha[rem_mask])
        if max_part_size == 0:
            return count
        return math.ceil(count / max_part_size)

    # Vertex ordering: Sort by degree descending to improve pruning
    # We need to map original indices to new indices or just permute the vertices
    # Permuting vertices is cleaner for the DP tables if we rebuild them, but we already built them.
    # Instead, we can just pick vertices dynamically in the B&B.
    # Let's use a fixed order based on degree for the B&B sequence.
    deg = [adj[i].bit_count() for i in range(n)]
    order = list(range(n))
    order.sort(key=lambda x: -deg[x])
    
    # In B&B, we will process vertices in 'order'.
    # rem_mask for lower bound needs to correspond to the vertices *not yet processed*.
    # Since we process in a fixed order, the set of remaining vertices is determined by the step index.
    # Let's precompute the remaining mask for each step.
    # step i means we have processed order[0]...order[i-1].
    # Remaining vertices are those in order[i]...order[n-1].
    
    rem_masks = [0] * (n + 1)
    rem_masks[n] = 0
    # Calculate rem_mask for each step
    # Actually, it's easier to just compute it on the fly or precompute a list of masks.
    # mask_at_step[i] = mask of vertices order[i:]
    mask_at_step = [0] * (n + 1)
    mask_at_step[n] = 0
    for i in range(n - 1, -1, -1):
        mask_at_step[i] = mask_at_step[i + 1] | (1 << order[i])
        
    # B&B Search
    # parts: list of masks
    # step: index in order
    # best: current best number of parts
    
    best = n + 1
    best_parts = []
    
    # Initial upper bound: Greedy
    # Simple greedy: assign each vertex to first compatible part, else new part
    greedy_parts = []
    for v in order:
        placed = False
        for i, p_mask in enumerate(greedy_parts):
            new_mask = p_mask | (1 << v)
            if is_clique[new_mask] or is_ind[new_mask]:
                greedy_parts[i] = new_mask
                placed = True
                break
        if not placed:
            greedy_parts.append(1 << v)
    best = len(greedy_parts)
    best_parts = list(greedy_parts)
    
    # If best is 1, return immediately
    if best == 1:
        part_verts = [v for v in range(n) if (best_parts[0] >> v) & 1]
        return [part_verts]

    # Recursive search
    def search(step, parts):
        nonlocal best, best_parts, deadline
        
        if time.time() > deadline:
            return False # Signal timeout
            
        if step == n:
            if len(parts) < best:
                best = len(parts)
                best_parts = list(parts)
            return True
            
        # Lower bound pruning
        rem_mask = mask_at_step[step]
        lb = get_lower_bound(rem_mask)
        if len(parts) + lb >= best:
            return True # Pruned, but search can continue with other branches
            
        v = order[step]
        v_bit = 1 << v
        
        # Option 1: Add v to existing parts
        for i in range(len(parts)):
            new_mask = parts[i] | v_bit
            if is_clique[new_mask] or is_ind[new_mask]:
                parts[i] = new_mask
                search(step + 1, parts)
                parts[i] ^= v_bit # Backtrack
                
        # Option 2: Create new part
        # Symmetry breaking: We only create a new part if the previous part (if any) is not "equivalent"?
        # Simple symmetry: If we have parts, and we are creating a new one, we don't need to check if it's identical to an existing one?
        # Actually, creating a new part with just {v} is always a valid move.
        # To reduce symmetry, we can say: if len(parts) > 0, we only create a new part if v is not compatible with the *last* part?
        # No, that's not correct.
        # Standard symmetry breaking for coloring: only assign to color i if color i-1 was used or i=0.
        # Here, "colors" are parts.
        # We can enforce that if we create a new part, it must be the first part? No.
        # Let's just try it. The bound should prune enough.
        
        parts.append(v_bit)
        search(step + 1, parts)
        parts.pop()
        
        return True

    search(0, [])
    
    # Reconstruct partition from best_parts
    if best_parts:
        parts_list = []
        for p_mask in best_parts:
            part_verts = [v for v in range(n) if (p_mask >> v) & 1]
            if part_verts:
                parts_list.append(part_verts)
        return parts_list
    else:
        # Fallback
        return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
