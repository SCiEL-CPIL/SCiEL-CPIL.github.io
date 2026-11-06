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
    best_path = [0, 1]  # Initialize with a valid short path
    time_limit = 115.0
    start_time = time.time()
    
    # Symmetry Breaking:
    # Fix the first step to flip bit 0.
    # Path: 0 -> 1
    # Used: {0, 1}
    # Banned: Neighbors of 0 excluding 1.
    
    start_v = 0
    first_v = 1
    initial_path = [start_v, first_v]
    initial_used = (1 << start_v) | (1 << first_v)
    
    initial_banned = 0
    for b in range(1, N):
        initial_banned |= (1 << (1 << b))
        
    def dfs_randomized(cur, used, banned, depth, rng, current_path):
        nonlocal best_path
        
        if time.time() - start_time > time_limit:
            return
        
        if depth > len(best_path):
            best_path = list(current_path)
        
        cand_mask = _NEIGHBOR_MASKS[cur] & ~used & ~banned
        if cand_mask == 0:
            return
        
        candidates = []
        cm = cand_mask
        while cm:
            lsb = cm & -cm
            v = lsb.bit_length() - 1
            candidates.append(v)
            cm ^= lsb
            
        # Heuristic: Greedy Expansion (Maximize Remaining Options)
        # Prefer moves that leave more options for subsequent steps.
        def heuristic_score(v):
            new_used = used | (1 << v)
            new_banned = banned | _NEIGHBOR_MASKS[cur]
            next_cand = _NEIGHBOR_MASKS[v] & ~new_used & ~new_banned
            n1 = next_cand.bit_count()
            if n1 == 0:
                return 0
            # Lookahead 2: Sum of options for neighbors of v
            total_depth = 0
            cm = next_cand
            while cm:
                lsb = cm & -cm
                w = lsb.bit_length() - 1
                cm ^= lsb
                # Neighbors of w, excluding used, banned, and v (which is now used)
                # Note: v is in new_used, so ~new_used handles it.
                # banned is the same for w as for v? 
                # banned contains neighbors of cur. 
                # When we move to v, the new banned set for the *next* step (from v to w) 
                # would be banned | neighbors(v). 
                # But here we are evaluating the state *after* moving to v.
                # The "banned" for w's neighbors should be:
                # 1. Used vertices (including v)
                # 2. Neighbors of v (because they are adjacent to the previous vertex in the path? No, 
                #    in Snake, non-consecutive vertices cannot be adjacent.
                #    So if we are at v, the next vertex w cannot be adjacent to any vertex in the path 
                #    except v.
                #    The current `banned` mask represents vertices that are adjacent to the *current* vertex `cur` 
                #    and thus cannot be visited next (except cur itself which is used).
                #    Wait, the definition of `banned` in the code:
                #    `initial_banned`: Neighbors of 0 excluding 1.
                #    In recursion: `new_banned = banned | _NEIGHBOR_MASKS[cur]`.
                #    So `banned` at `v` contains neighbors of `prev(v)` (which is `cur`).
                #    So `banned` prevents us from going back to `cur`'s neighbors.
                #    This is correct for Snake: w cannot be adjacent to `cur` (unless w==cur, which is used).
                #    So for calculating options for `w`'s neighbors, the banned set would be `new_banned` (neighbors of v).
                #    Let's call this `next_banned`.
                next_banned = new_banned | _NEIGHBOR_MASKS[v]
                w_cand = _NEIGHBOR_MASKS[w] & ~new_used & ~next_banned
                total_depth += w_cand.bit_count()
            return total_depth
            
        # Randomize to break symmetry in ties, then sort by heuristic (Descending)
        rng.shuffle(candidates)
        candidates.sort(key=heuristic_score, reverse=True)
        
        for v in candidates:
            if time.time() - start_time > time_limit:
                return
            
            current_path.append(v)
            new_used = used | (1 << v)
            new_banned = banned | _NEIGHBOR_MASKS[cur]
            
            dfs_randomized(v, new_used, new_banned, depth + 1, rng, current_path)
            
            current_path.pop()
            
            if time.time() - start_time > time_limit:
                return

    # Run multiple independent searches with different random seeds
    # This explores different branches of the search space
    num_searches = 100
    for i in range(num_searches):
        if time.time() - start_time > time_limit:
            break
        
        rng = random.Random(i)
        # Start from the symmetry-broken state
        current_path = [0, 1]
        # The current vertex is 1 (the last element in path)
        dfs_randomized(1, initial_used, initial_banned, 2, rng, current_path)

    return best_path

# EVOLVE-BLOCK-END
