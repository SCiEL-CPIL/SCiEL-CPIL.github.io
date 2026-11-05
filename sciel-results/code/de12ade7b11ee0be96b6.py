"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Deadline guard: fall back to nearest-neighbor if time is running out
    if time.time() > _DEADLINE:
        return int(unv[np.argmin(distance_matrix[current_node, unv])])
    
    dm = distance_matrix
    best_score = np.inf
    best_node = int(unv[0])
    
    # 3-step lookahead with geometric discount
    # score = d1 + 0.5*d2 + 0.25*d3
    # This balances immediate cost with future connectivity
    
    # Precompute distances from current to all unvisited
    dists_from_cur = dm[current_node, unv]
    
    # Sort by distance for early pruning potential
    # But we need the actual node indices, so work with the array directly
    
    for i in range(n):
        c = unv[i]
        d1 = dists_from_cur[i]
        
        # Pruning: if immediate cost is already worse than best total, skip
        if d1 >= best_score:
            continue
        
        if n > 2:
            # Find next best node x from c among remaining
            # Create mask for remaining nodes (all except c)
            remaining = np.delete(unv, i)
            dists_c = dm[c, remaining]
            idx_x = np.argmin(dists_c)
            x = remaining[idx_x]
            d2 = dists_c[idx_x]
            
            # Early pruning: if d1 + 0.5*d2 >= best_score, skip deeper lookahead
            partial_score = d1 + 0.5 * d2
            if partial_score >= best_score:
                continue
            
            score = partial_score
            
            if n > 3:
                # Find next best node y from x among remaining (excluding c and x)
                # Remove x from remaining to get remaining2
                x_idx_in_remaining = idx_x
                remaining2 = np.delete(remaining, x_idx_in_remaining)
                dists_x = dm[x, remaining2]
                idx_y = np.argmin(dists_x)
                d3 = dists_x[idx_y]
                score += 0.25 * d3
            
            if score < best_score:
                best_score = score
                best_node = int(c)
        else:
            # n == 2: just pick the one that minimizes total tour cost
            # Tour: current -> c -> other -> destination
            # We need to consider both orderings
            other = int(unv[1 - i])
            # Cost if we pick c first: d1 + dm[c, other] + dm[other, destination_node]
            # But we're just picking next, so the "other" will be picked next
            # Actually for n=2, we should pick the one that leads to better overall
            # cost: d1 + dm[c, other]
            # But the final return to destination is handled by the caller
            # For consistency, use the same scoring: d1 + 0.5 * dm[c, other]
            d2 = dm[c, other]
            score = d1 + 0.5 * d2
            
            if score < best_score:
                best_score = score
                best_node = int(c)
                
    return best_node
# EVOLVE-BLOCK-END
