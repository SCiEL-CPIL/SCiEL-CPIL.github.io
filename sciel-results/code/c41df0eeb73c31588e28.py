import time
import random
import numpy as np

N = 9
NUM_VERTICES = 1 << N

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_path = []
    best_len = 0
    
    # Precompute neighbors for each vertex
    # neighbors[v] is a list of 9 neighbors
    neighbors = [[] for _ in range(NUM_VERTICES)]
    for v in range(NUM_VERTICES):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Helper to run a single greedy search
    def greedy_search(seed):
        random.seed(seed)
        path = [0]
        used = {0}
        # F is the set of vertices forbidden for the NEXT step 
        # because they are adjacent to non-consecutive vertices in the current path.
        # Initially path is [0], no non-consecutive vertices, so F is empty.
        F = set()
        
        while True:
            cur = path[-1]
            candidates = []
            
            # Generate potential next vertices
            for v in neighbors[cur]:
                if v in used or v in F:
                    continue
                candidates.append(v)
                
            if not candidates:
                break
            
            # Heuristic: Pick the candidate that maximizes the number of valid options for the NEXT step.
            # F will become F | neighbors(cur) after we move to v.
            # Let F_next = F | set(neighbors(cur))
            # For each candidate v, count how many of its neighbors are not in used and not in F_next.
            
            # Optimization: Compute F_next once
            # Note: neighbors(cur) are the vertices adjacent to cur.
            # We need to add them to F.
            # But we must be careful: F is a set.
            # We can create a temporary set or just check membership against F and neighbors(cur).
            
            best_v = None
            best_score = -1
            
            # To speed up, we can precompute the set of neighbors of cur
            cur_neighbors = neighbors[cur]
            
            for v in candidates:
                # Score: number of neighbors of v that are not in used and not in (F U cur_neighbors)
                score = 0
                for u in neighbors[v]:
                    if u in used:
                        continue
                    if u in F:
                        continue
                    if u in cur_neighbors:
                        continue
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_v = v
                elif score == best_score and random.random() < 0.5:
                    # Random tie-breaking to explore diversity
                    best_v = v
            
            if best_v is None:
                break
                
            path.append(best_v)
            used.add(best_v)
            
            # Update F: F = F U neighbors(cur)
            # Since F is a set, we can update it in place
            F.update(cur_neighbors)
            
            # Check time
            if time.time() > deadline:
                return path
        
        return path

    # Run multiple searches
    seed = 0
    while time.time() < deadline:
        path = greedy_search(seed)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            # Optional: If we reach a very high score, we might want to stop early?
            # But we don't know the max. Keep searching.
        seed += 1
        
    return best_path
