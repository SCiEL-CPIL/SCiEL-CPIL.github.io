"""Improved snake-in-the-box for Q_9 using backtracking with heuristic ordering."""
import random
import sys

N = 9
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a long induced path (snake) in Q_9 using randomized backtracking
    with heuristic bit-flip ordering and periodic restarts.
    """
    best_path = None
    best_len = 0
    
    # Try multiple random seeds for diversity
    rng = random.Random(42)
    
    max_attempts = 20  # Number of restarts within time budget
    import time
    start_time = time.time()
    time_limit = 110  # Leave buffer for evaluation overhead
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize path with a random starting vertex
        start = rng.randint(0, (1 << N) - 1)
        path = [start]
        used = {start}
        
        # For each step, try to extend the path
        while True:
            cur = path[-1]
            
            # Generate candidate next vertices (all single-bit flips)
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # Check if adding v would violate the induced path property
                # i.e., v must not be at Hamming distance 1 from any non-adjacent vertex
                valid = True
                for i in range(len(path) - 1):
                    if (v ^ path[i]).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    candidates.append(v)
            
            if not candidates:
                break
            
            # Heuristic: prefer vertices that keep more options open
            # Score each candidate by how many of its neighbors are still unused
            # and not too close to the path
            scored_candidates = []
            for v in candidates:
                # Count how many neighbors of v are unused and not adjacent to path[:-1]
                score = 0
                for b in range(N):
                    nb = v ^ (1 << b)
                    if nb in used:
                        continue
                    # Check if nb is at distance >= 2 from all non-adjacent path vertices
                    # (we only care about immediate adjacency for the snake property)
                    score += 1
                # Also prefer vertices with lower degree in the remaining graph
                # Use a simple heuristic: prefer vertices with fewer unused neighbors
                # (to avoid getting stuck in dense regions)
                scored_candidates.append((-score, rng.random(), v))
            
            # Sort by score (prefer higher score, i.e., more options)
            scored_candidates.sort()
            
            # With some probability, pick the best; otherwise pick randomly from top few
            if rng.random() < 0.7:
                next_v = scored_candidates[0][2]
            else:
                # Pick from top 3 candidates
                top_k = min(3, len(scored_candidates))
                idx = rng.randint(0, top_k - 1)
                next_v = scored_candidates[idx][2]
            
            path.append(next_v)
            used.add(next_v)
        
        # Update best
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    if best_path is None:
        # Fallback: return the greedy seed
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                if any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        return path
    
    return best_path
# EVOLVE-BLOCK-END
