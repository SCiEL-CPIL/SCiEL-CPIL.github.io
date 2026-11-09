"""
Snake-in-the-box in Q_9 using randomized greedy with forbidden set pruning.
"""
import time
import random
from collections import deque

N = 9
MASK = (1 << N) - 1
SIZE = 1 << N

# Precompute popcount
_POP = [0] * SIZE
for i in range(1, SIZE):
    _POP[i] = _POP[i >> 1] + (i & 1)

# Precompute neighbors
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 5s safety margin
    
    best_path = []
    best_len = 0
    
    rng = random.Random(42)
    
    # To improve results, we can try different random seeds or strategies
    # We will run multiple iterations of a randomized greedy algorithm
    
    for iter_id in range(10000):
        if time.time() > DEADLINE:
            break
        
        # Random start
        start = rng.randint(0, SIZE - 1)
        
        # Path construction
        path = [start]
        used = [False] * SIZE
        used[start] = True
        
        # Forbidden set: vertices that cannot be chosen next because they are 
        # at distance 1 from a non-consecutive vertex in the path.
        # Specifically, if we are at 'cur', and we choose 'next', 
        # then all neighbors of 'next' (except 'cur') become forbidden.
        # Also, any vertex already in 'used' is obviously not allowed.
        # We can just check 'used' and 'forbidden' for the next step.
        
        forbidden = set()
        # Initially, no vertices are forbidden for the first step?
        # Wait, the first step has no non-consecutive vertices.
        # So forbidden is empty.
        
        cur = start
        
        while True:
            # Find valid next vertices
            valid_nexts = []
            for nb in NEIGHBORS[cur]:
                if used[nb]:
                    continue
                if nb in forbidden:
                    continue
                valid_nexts.append(nb)
                
            if not valid_nexts:
                break
                
            # Heuristic: Warnsdorff's rule
            # Choose the vertex with the fewest valid next-next steps
            # This helps avoid dead ends.
            
            # To add randomness, we can pick randomly from the top few
            # or add noise to the score.
            
            scored_nexts = []
            for nb in valid_nexts:
                # Count how many options nb would have
                # A vertex 'w' is a valid next-next if:
                # 1. w is neighbor of nb
                # 2. w is not used
                # 3. w is not forbidden (note: forbidden set will change if we pick nb)
                
                # Actually, we need to simulate the forbidden set update for nb.
                # New forbidden = forbidden - {nb} + {neighbors of nb except cur}
                # Let's approximate or compute exactly.
                
                # Exact computation is cheap since degree is 9.
                new_forbidden_count = 0
                # We don't need the full set, just the count of valid options for the next step
                # But the forbidden set affects which neighbors of nb are valid.
                
                # Let's just count neighbors of nb that are not used and not in the *current* forbidden set
                # This is an approximation because 'nb' itself is removed from forbidden,
                # and its neighbors are added.
                
                # Let's do a quick exact check for the top candidates only to save time?
                # Or just use a simpler heuristic: degree in the remaining graph.
                
                # Let's stick to a simpler heuristic first:
                # Score = number of unused neighbors of nb
                score = 0
                for nn in NEIGHBORS[nb]:
                    if not used[nn]:
                        score += 1
                scored_nexts.append((score, nb))
                
            # Sort by score (ascending)
            scored_nexts.sort()
            
            # Randomly pick from the best few to allow exploration
            # If we always pick the best, we might get stuck in a specific local minimum.
            # Let's pick from the top 3 if available, otherwise the best.
            if len(scored_nexts) > 3:
                # Pick randomly from top 3
                # But ensure we don't pick a terrible one.
                # Let's pick from top min(3, len) with some probability distribution
                # Or just pick the best one 80% of the time, random from top 3 20%
                if rng.random() < 0.8:
                    next_v = scored_nexts[0][1]
                else:
                    k = rng.randint(0, 2)
                    next_v = scored_nexts[k][1]
            else:
                next_v = scored_nexts[0][1]
                
            # Update state
            path.append(next_v)
            used[next_v] = True
            
            # Update forbidden set
            # Remove next_v from forbidden (it was invalid, now it's used)
            # Actually, next_v was NOT in forbidden because we checked it.
            # But its neighbors (except cur) become forbidden.
            for nb in NEIGHBORS[next_v]:
                if nb != cur:
                    forbidden.add(nb)
                    
            cur = next_v
            
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
    if not best_path:
        return [0]
        
    return best_path
