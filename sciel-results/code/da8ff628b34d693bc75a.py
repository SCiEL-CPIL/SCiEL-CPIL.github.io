import time
import sys

N = 9
MAX_VERTS = 1 << N

# Precompute neighbors masks
neighbors_mask = [0] * MAX_VERTS
for v in range(MAX_VERTS):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    neighbors_mask[v] = mask

best_path = [0]
start_time = 0.0
deadline = 0.0

def construct():
    global best_path, start_time, deadline
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    sys.setrecursionlimit(10000)
    
    # Symmetry breaking: Fix start at 0, first move to 1 (bit 0)
    # This is valid because the hypercube is vertex-transitive and edge-transitive.
    # Any snake can be mapped to one starting 0->1 via XOR and bit permutation.
    
    path = [0, 1]
    used_mask = (1 << 0) | (1 << 1)
    # Conflict mask for the next step:
    # We are at 1. The previous vertex was 0.
    # Neighbors of 0 are now forbidden for the next vertex (distance 2 from 0 is ok, distance 1 is not).
    # Wait, the next vertex $v_2$ must be distance >= 2 from $v_0$.
    # So $v_2$ cannot be a neighbor of $v_0$.
    # So conflict_mask = neighbors_mask[0].
    conflict_mask = neighbors_mask[0]
    
    if len(path) > len(best_path):
        best_path = path[:]
        
    dfs(path, used_mask, conflict_mask)
    
    return best_path

def dfs(path, used_mask, conflict_mask):
    global best_path, deadline
    
    if time.time() > deadline:
        return
        
    if len(path) > len(best_path):
        best_path = path[:]
        
    cur = path[-1]
    # Candidates are neighbors of cur that are not used and not in conflict
    candidates_mask = neighbors_mask[cur] & ~used_mask & ~conflict_mask
    
    if candidates_mask == 0:
        return
        
    # Extract candidates
    candidates = []
    m = candidates_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        candidates.append(v)
        m ^= lsb
        
    # Heuristic: Sort candidates by the number of available next moves
    # If we move to v, the new conflict mask will be conflict_mask | neighbors_mask[cur]
    # The available moves from v will be neighbors of v that are not used and not in the new conflict mask.
    new_conflict_base = conflict_mask | neighbors_mask[cur]
    
    def score(v):
        # Count neighbors of v that are not in used_mask and not in new_conflict_base
        # Note: used_mask does not contain v yet.
        # new_conflict_base does not contain v (v is not a neighbor of cur? No, v IS a neighbor of cur).
        # Wait, v is a neighbor of cur. So v is in neighbors_mask[cur].
        # So v is in new_conflict_base.
        # But v is the vertex we are moving TO. We are counting moves FROM v.
        # The moves FROM v are neighbors of v.
        # We want neighbors of v that are not used and not in new_conflict_base.
        # Is v in new_conflict_base? Yes. But v is not a neighbor of v.
        # So the mask `neighbors_mask[v] & ~(used_mask | new_conflict_base)` correctly excludes
        # any neighbor of v that is also a neighbor of cur (which is forbidden).
        return (neighbors_mask[v] & ~(used_mask | new_conflict_base)).bit_count()
        
    # Sort by score descending
    candidates.sort(key=score, reverse=True)
    
    for v in candidates:
        if time.time() > deadline:
            return
            
        path.append(v)
        new_used = used_mask | (1 << v)
        # new_conflict is new_conflict_base
        dfs(path, new_used, new_conflict_base)
        
        # Backtrack
        path.pop()
        # used_mask and conflict_mask are integers, passed by value, so no restoration needed in caller scope
        # (they are local variables in the caller, and we passed copies of the values)

# EVOLVE-BLOCK-END
