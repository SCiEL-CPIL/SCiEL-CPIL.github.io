"""Snake-in-the-box in Q_9: improved construction via heuristic search."""
import random
import sys

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START

import time

def _precompute_neighbors():
    """Precompute neighbors and neighbor masks for all vertices."""
    neighbors = [[] for _ in range(TOTAL_VERTICES)]
    neighbor_masks = [0] * TOTAL_VERTICES
    for v in range(TOTAL_VERTICES):
        for b in range(N):
            nb = v ^ (1 << b)
            neighbors[v].append(nb)
            neighbor_masks[v] |= (1 << nb)
    return neighbors, neighbor_masks

# Global precomputed data
_NEIGHBORS, _NEIGHBOR_MASKS = _precompute_neighbors()

def construct():
    """
    Construct a snake in Q_9 using randomized DFS with symmetry breaking.
    """
    best_path = [0, 1, 3]  # Initialize with a valid short path
    time_limit = 115.0
    start_time = time.time()
    
    # Symmetry Breaking:
    # Fix the first three steps: 0 -> 1 -> 3
    # 0: 000000000
    # 1: 000000001 (flip bit 0)
    # 3: 000000011 (flip bit 1)
    
    start_v = 0
    first_v = 1
    second_v = 3
    
    initial_path = [start_v, first_v, second_v]
    initial_used = (1 << start_v) | (1 << first_v) | (1 << second_v)
    
    # Banned vertices:
    # Neighbors of 0 (excluding 1, which is used)
    # Neighbors of 1 (excluding 0 and 3, which are used/prev)
    # Note: 'banned' in DFS means vertices that cannot be visited next because they are adjacent to a non-immediate predecessor.
    # When at vertex 'cur', we ban all neighbors of 'prev_prev' (which is 'cur's predecessor's predecessor) that are not 'prev'.
    # Actually, the standard logic is:
    # When moving from prev to cur, we ban neighbors of prev (except cur) to prevent chords from cur to prev's neighbors.
    # So, initial state:
    # We are at 3. Previous was 1. Prev-prev was 0.
    # We must ban neighbors of 0 (except 1).
    # We must ban neighbors of 1 (except 0 and 3).
    
    initial_banned = 0
    # Ban neighbors of 0, except 1
    for nb in _NEIGHBORS[start_v]:
        if nb != first_v:
            initial_banned |= (1 << nb)
            
    # Ban neighbors of 1, except 0 and 3
    for nb in _NEIGHBORS[first_v]:
        if nb != start_v and nb != second_v:
            initial_banned |= (1 << nb)

    # Optimization: Store global masks in local variables for faster access in recursion
    masks = _NEIGHBOR_MASKS
    
    def dfs_randomized(cur, used, banned, depth, rng, current_path):
        nonlocal best_path
        
        if time.time() - start_time > time_limit:
            return
        
        if depth > len(best_path):
            best_path = list(current_path)
        
        cand_mask = masks[cur] & ~used & ~banned
        if cand_mask == 0:
            return
        
        candidates = []
        cm = cand_mask
        while cm:
            lsb = cm & -cm
            v = lsb.bit_length() - 1
            candidates.append(v)
            cm ^= lsb
            
        # Enhanced Heuristic: 2-step lookahead
        # Score = (next options) * 10 + (next-next options)
        def heuristic_score(v):
            new_used = used | (1 << v)
            new_banned = banned | masks[cur] # Ban neighbors of cur (except v) for the next step?
            # Wait, when we move cur -> v, the new 'banned' set for the *next* move (v -> w)
            # will include neighbors of cur (except v).
            # The candidates for w are neighbors of v, not in new_used, and not in (banned | masks[cur]).
            
            # Let's calculate options for v
            next_cand_mask = masks[v] & ~new_used & ~(banned | masks[cur])
            score1 = next_cand_mask.bit_count()
            
            # If score1 is 0, it's a dead end.
            # If score1 > 0, check the best option for w?
            # To keep it fast, just use score1. 
            # Or, if score1 is small, check the minimum score of its neighbors.
            
            # Let's stick to a simple weighted score to avoid overhead
            # But we can add a penalty if v is adjacent to many banned vertices?
            # The mask `next_cand_mask` already filters those out.
            
            return score1
            
        # Randomize to break symmetry in ties, then sort by heuristic
        rng.shuffle(candidates)
        candidates.sort(key=heuristic_score)
        
        for v in candidates:
            if time.time() - start_time > time_limit:
                return
            
            current_path.append(v)
            new_used = used | (1 << v)
            new_banned = banned | masks[cur]
            
            dfs_randomized(v, new_used, new_banned, depth + 1, rng, current_path)
            
            current_path.pop()
            
            if time.time() - start_time > time_limit:
                return

    # Run multiple independent searches with different random seeds
    # This explores different branches of the search space
    num_searches = 200
    for i in range(num_searches):
        if time.time() - start_time > time_limit:
            break
        
        rng = random.Random(i)
        # Start from the symmetry-broken state
        current_path = [0, 1, 3]
        # The current vertex is 3 (the last element in path)
        dfs_randomized(3, initial_used, initial_banned, 3, rng, current_path)

    return best_path

# EVOLVE-BLOCK-END
