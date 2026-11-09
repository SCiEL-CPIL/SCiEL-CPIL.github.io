import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 110.0  # Safety margin for 120s limit

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Multi-start randomized greedy with lookahead heuristic.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Precompute neighbors for all vertices? 
    # 2^9 = 512 vertices. 
    # neighbors[v] = list of v ^ (1<<b)
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    def is_valid_extension(path, used_set, candidate):
        """
        Check if candidate can be appended to path.
        1. candidate not in used_set
        2. candidate not adjacent to any vertex in path[:-1]
           (adjacent means Hamming distance 1)
        """
        if candidate in used_set:
            return False
        # Check against all previous vertices except the last one (which is the current end)
        # We need to ensure candidate is not adjacent to any p in path[:-1]
        # Note: path[-1] is the current end. candidate is adjacent to path[-1] by definition of extension.
        # We must ensure candidate is NOT adjacent to any p in path[:-1].
        for p in path[:-1]:
            if (candidate ^ p).bit_count() == 1:
                return False
        return True

    def greedy_with_lookahead(start_vertex=0, time_limit=None):
        """
        Perform a greedy construction starting from start_vertex.
        Uses lookahead: picks the neighbor with the most available extensions.
        """
        path = [start_vertex]
        used = {start_vertex}
        
        while True:
            cur = path[-1]
            valid_nexts = []
            
            for v in neighbors[cur]:
                if is_valid_extension(path, used, v):
                    # Lookahead: count how many valid neighbors v has (excluding cur)
                    # This is an approximation of "openness"
                    count = 0
                    for w in neighbors[v]:
                        if w == cur:
                            continue
                        # Quick check: is w used?
                        if w in used:
                            continue
                        # Is w adjacent to any p in path[:-1]?
                        # This is O(L) per w, so O(L * 9) per v.
                        # Since L is small (< 200) and 9 is small, this is fast.
                        # But we can optimize: if w is adjacent to path[-2], it's invalid for v's extension?
                        # Actually, we just need to know if w is a valid extension for v.
                        # A valid extension for v means w is not used and not adjacent to path[:-1].
                        # Wait, if we pick v, the new path is path + [v].
                        # Then we look for w. w must be valid for path + [v].
                        # This means w is not in used ∪ {v} and not adjacent to (path + [v])[:-1] = path.
                        # So w must not be adjacent to any p in path.
                        
                        # Optimization: 
                        # We know v is not adjacent to path[:-1] (checked above).
                        # So we just need to check if w is adjacent to path[-1] (which is cur)?
                        # No, w is a neighbor of v. 
                        # The condition for w to be valid after adding v is:
                        # 1. w not in used
                        # 2. w not adjacent to any p in path.
                        #    - w adjacent to cur? If so, w is a neighbor of cur.
                        #    - w adjacent to p in path[:-1]?
                        
                        # Let's just do the full check for simplicity, it's fast enough.
                        is_adj_to_prev = False
                        for p in path:
                            if (w ^ p).bit_count() == 1:
                                is_adj_to_prev = True
                                break
                        if not is_adj_to_prev:
                            count += 1
                    
                    valid_nexts.append((count, v))
            
            if not valid_nexts:
                break
            
            # Sort by count descending, then randomize ties
            valid_nexts.sort(key=lambda x: x[0], reverse=True)
            
            # Pick from the top 3 to add randomness
            k = min(3, len(valid_nexts))
            top_candidates = [v for _, v in valid_nexts[:k]]
            next_v = random.choice(top_candidates)
            
            path.append(next_v)
            used.add(next_v)
            
            # Time check inside the loop for very long paths (unlikely to be an issue for 100s steps, but safe)
            if time_limit and (time.time() - start_time) > time_limit:
                break
        
        return path

    # Multi-start
    # We can start from 0, or from random vertices.
    # Starting from 0 is symmetric, so randomizing the bit order or the start vertex is key.
    # Actually, since the graph is vertex-transitive, starting from 0 is fine if we randomize the heuristic.
    # But different random seeds in the choice will lead to different paths.
    
    attempts = 0
    while time.time() - start_time < DEADLINE:
        # Randomize the start vertex slightly? No, symmetry means start=0 is fine.
        # The randomness comes from the tie-breaking in greedy_with_lookahead.
        # To increase diversity, we can also randomize the neighbor iteration order in is_valid_extension?
        # No, the choice is already randomized at the end.
        
        # Occasionally, try a "greedy with random bit order" without lookahead for variety?
        # Let's stick to lookahead greedy with random tie-breaking.
        
        p = greedy_with_lookahead(start_vertex=0, time_limit=DEADLINE - (time.time() - start_time))
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            # If we hit a high score, we might want to try to refine it?
            # For now, just keep the best.
            
        attempts += 1
        
        # If we've done enough attempts and time is running out, break
        if attempts > 1000: # Safety cap, shouldn't happen with time check
            break

    if best_path is None:
        # Fallback to seed
        path, used = [0], {0}
        while True:
            cur, nxt = path[-1], None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                return path
            path.append(nxt)
            used.add(nxt)
        return path

    return best_path

# EVOLVE-BLOCK-END
