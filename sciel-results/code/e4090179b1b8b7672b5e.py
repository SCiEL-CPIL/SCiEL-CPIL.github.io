"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_bitmasks(n, edges):
    """Build adjacency bitmask for each vertex."""
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj_mask

def _can_add_clique(v, part_mask, adj_mask):
    """Check if vertex v can be added to a clique represented by part_mask."""
    if part_mask == 0:
        return True
    # v must be adjacent to all vertices in the part
    return (adj_mask[v] & part_mask) == part_mask

def _can_add_independent(v, part_mask, adj_mask):
    """Check if vertex v can be added to an independent set represented by part_mask."""
    if part_mask == 0:
        return True
    # v must NOT be adjacent to any vertex in the part
    return (adj_mask[v] & part_mask) == 0

def _greedy_partition(n, adj_mask):
    """Greedy partition using bitmask checks. Process high-degree vertices first."""
    degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    parts_masks = []  # list of (mask, is_clique) tuples
    
    for v in order:
        best_idx = -1
        best_size = -1
        for i, (pmask, is_cli) in enumerate(parts_masks):
            if is_cli:
                if _can_add_clique(v, pmask, adj_mask):
                    size = bin(pmask).count('1') + 1
                    if size > best_size:
                        best_size = size
                        best_idx = i
            else:
                if _can_add_independent(v, pmask, adj_mask):
                    size = bin(pmask).count('1') + 1
                    if size > best_size:
                        best_size = size
                        best_idx = i
        
        if best_idx >= 0:
            pmask, is_cli = parts_masks[best_idx]
            parts_masks[best_idx] = (pmask | (1 << v), is_cli)
        else:
            # New part: try clique first (single vertex is both)
            parts_masks.append((1 << v, True))
    
    # Convert masks back to lists
    result = []
    for pmask, _ in parts_masks:
        verts = [i for i in range(n) if (pmask >> i) & 1]
        if verts:
            result.append(verts)
    return result

def _search_with_target(n, adj_mask, target, deadline, order):
    """Find a cochromatic partition with at most 'target' parts using backtracking."""
    parts_masks = [0] * target
    parts_type = [0] * target  # 0=unassigned, 1=clique, 2=independent
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            # All parts must be non-empty
            for i in range(target):
                if parts_masks[i] == 0:
                    return False
            return True
        
        v = order[idx]
        v_bit = 1 << v
        
        # Try adding to each existing part (with symmetry breaking: skip identical empty parts)
        for i in range(target):
            if parts_masks[i] == 0:
                # Only allow first empty part to be used (symmetry breaking)
                if i > 0 and parts_masks[i-1] == 0:
                    continue
                # Try as clique
                parts_masks[i] = v_bit
                parts_type[i] = 1
                if backtrack(idx + 1):
                    return True
                # Try as independent
                parts_type[i] = 2
                if backtrack(idx + 1):
                    return True
                parts_masks[i] = 0
                parts_type[i] = 0
                break  # No need to try other empty parts due to symmetry
            else:
                t = parts_type[i]
                if t == 1:  # clique
                    if _can_add_clique(v, parts_masks[i], adj_mask):
                        parts_masks[i] |= v_bit
                        if backtrack(idx + 1):
                            return True
                        parts_masks[i] ^= v_bit
                elif t == 2:  # independent
                    if _can_add_independent(v, parts_masks[i], adj_mask):
                        parts_masks[i] |= v_bit
                        if backtrack(idx + 1):
                            return True
                        parts_masks[i] ^= v_bit
        
        return False
    
    if backtrack(0):
        result = []
        for i in range(target):
            if parts_masks[i] != 0:
                verts = [j for j in range(n) if (parts_masks[i] >> j) & 1]
                result.append(verts)
        return result
    return None

def construct():
    n = P['n']
    adj_mask = _build_bitmasks(n, P['edges'])
    
    # Deadline: 85 seconds (5s safety margin from 90s limit)
    deadline = time.time() + 85.0
    
    # Get greedy solution as upper bound
    greedy = _greedy_partition(n, adj_mask)
    best_parts = greedy
    best_count = len(greedy)
    
    # Order vertices by degree (high first) for better pruning
    degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    # Try to find a solution with fewer parts
    for target in range(1, best_count):
        if time.time() > deadline:
            break
        found = _search_with_target(n, adj_mask, target, deadline, order)
        if found is not None:
            best_parts = found
            best_count = target
            break
    
    # Ensure proper types: list of lists of Python ints
    final_result = []
    for part in best_parts:
        final_result.append([int(v) for v in part])
    
    return final_result
# EVOLVE-BLOCK-END
