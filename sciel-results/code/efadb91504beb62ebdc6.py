"""
Optimized Snake-in-the-Box solver for Q_9.
Uses bitmask representation for O(1) set operations and a Warnsdorff-like heuristic.
"""
import time
import random

N = 9
NUM_VERTICES = 1 << N
TIME_LIMIT = 115.0  # Safety margin below 120s evaluator limit
SEED_LENGTH = 104

# Precompute neighbors for all vertices to speed up candidate generation
NEIGHBORS = [[] for _ in range(NUM_VERTICES)]
for v in range(NUM_VERTICES):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = [0]
    best_len = 1
    
    # Bitmask for exclusion: vertices at Hamming distance 1 from any non-terminal vertex
    # Initially path=[0], no non-terminal vertices, so excl_mask = 0
    # used_mask tracks all vertices in the path
    
    def count_extensions(v, used_mask, excl_mask, cur):
        """
        Count how many valid moves are available from vertex v, 
        assuming v is the new terminal (so cur is the previous terminal).
        A move to w is valid if:
        1. w is not used (w not in used_mask | (1 << v))
        2. w is not excluded (w not in excl_mask | neighbors_of_cur)
           Note: When v becomes terminal, cur becomes non-terminal.
           So the new exclusion set will include neighbors of cur.
           The current excl_mask contains neighbors of all previous non-terminals.
           We need to check w against (excl_mask | neighbors_of_cur).
        3. w != cur (w is a neighbor of v, cur is a neighbor of v, but w could be cur? 
           No, in a hypercube, neighbors of v are distinct. cur is one of them.
           We must not go back to cur immediately? 
           Wait, the path is induced. 
           Consecutive vertices differ by 1. 
           Non-consecutive differ by >= 2.
           If we go v -> w, and w == cur, then dist(w, cur)=0? No, w and cur are distinct vertices.
           If w == cur, then we are revisiting a vertex, which is invalid (distinct vertices).
           So w != cur is required by 'distinct vertices'.
           
           Actually, the exclusion logic handles this.
           If we are at cur, and we move to v.
           Now v is terminal. cur is non-terminal.
           Neighbors of cur are added to exclusion.
           So cur itself is not excluded, but its neighbors are.
           Wait, if w is a neighbor of v, and w == cur, then w is in used_mask.
           So condition 1 (not used) handles it.
           
           Let's refine the check for a candidate next vertex w from v:
           - w not in used_mask (current used)
           - w != v (guaranteed by neighbor)
           - w not in (excl_mask | neighbors_of_cur_mask)
             Why neighbors_of_cur_mask? Because cur becomes non-terminal.
             Any vertex at distance 1 from cur is forbidden for future moves.
             Since w is a neighbor of v, is w at distance 1 from cur?
             dist(v, cur) = 1.
             dist(v, w) = 1.
             dist(cur, w) can be 0 (if w=cur), 2 (if w is another neighbor of v that is also neighbor of cur? No, in hypercube, if dist(v,cur)=1 and dist(v,w)=1, then dist(cur,w) is 0 or 2).
             If w == cur, dist=0. But w is used.
             If w != cur, dist(cur, w) = 2. So w is NOT a neighbor of cur.
             Therefore, w is not in neighbors_of_cur_mask (unless there's a triangle, which hypercube doesn't have).
             So checking against neighbors_of_cur_mask is actually redundant for immediate neighbors of v?
             
             Let's re-evaluate.
             Path: ... -> a -> b -> c (cur=c).
             We move to v.
             New path: ... -> a -> b -> c -> v.
             Non-terminals: ... -> a -> b -> c.
             Exclusion set E contains neighbors of a, b, c.
             We want to check if we can move from v to w.
             w must not be in E.
             w must not be in Used.
             
             In the current state (before moving to v), `excl_mask` contains neighbors of a, b.
             `cur` is c.
             When we move to v, c becomes non-terminal.
             So the *effective* exclusion set for the next step from v is `excl_mask | neighbors(c)`.
             
             So for a candidate w (neighbor of v):
             Valid if:
             1. (1 << w) not in used_mask
             2. (1 << w) not in (excl_mask | neighbors_mask_of_cur)
             
             Note: neighbors_mask_of_cur is a bitmask of all neighbors of cur.
        """
        # Precompute neighbors_mask_of_cur outside the loop if possible?
        # Here we are inside the candidate loop, so cur is fixed.
        
        valid_count = 0
        # used_mask already includes cur and previous. It does NOT include v yet.
        # So we check w against used_mask.
        # Note: v is not in used_mask. w is a neighbor of v.
        # If w == cur, w is in used_mask. So it will be rejected.
        
        for w in NEIGHBORS[v]:
            bit = 1 << w
            if bit & used_mask:
                continue
            if bit & (excl_mask | neighbors_mask_of_cur):
                continue
            valid_count += 1
        return valid_count

    def search(rng):
        nonlocal best_path, best_len
        
        path = [0]
        used_mask = 1  # 0 is used
        excl_mask = 0  # No non-terminals yet
        
        while True:
            cur = path[-1]
            # Precompute neighbors mask of cur for the exclusion check
            neighbors_mask_of_cur = 0
            for nb in NEIGHBORS[cur]:
                neighbors_mask_of_cur |= (1 << nb)
                
            candidates = []
            for v in NEIGHBORS[cur]:
                bit = 1 << v
                if bit & used_mask:
                    continue
                if bit & excl_mask:
                    continue
                
                # Heuristic: count extensions from v
                # Note: We pass the current state.
                # The "new" exclusion set after adding v will be excl_mask | neighbors_mask_of_cur.
                # The "new" used set will be used_mask | bit.
                # But for counting extensions from v, we just need to know which vertices are blocked.
                # Blocked = used_mask | bit (v is now used) | (excl_mask | neighbors_mask_of_cur)
                # However, we don't need to add `bit` to the exclusion check for w, 
                # because w != v (neighbors are distinct).
                # And w != cur is handled by used_mask (cur is in used_mask).
                
                # Optimization: We can inline the count loop for speed
                valid_count = 0
                blocked_mask = used_mask | excl_mask | neighbors_mask_of_cur
                for w in NEIGHBORS[v]:
                    if (1 << w) & blocked_mask:
                        continue
                    valid_count += 1
                
                # Warnsdorff: prefer lower valid_count
                # Add small random noise to break ties and allow exploration
                score = valid_count + rng.random() * 0.1
                candidates.append((score, v))
            
            if not candidates:
                break
            
            # Sort by score (ascending, as lower is better for Warnsdorff)
            candidates.sort(key=lambda x: x[0])
            
            # Pick the best candidate
            # Sometimes, to avoid getting stuck in a specific pattern, 
            # we could pick randomly from the top 3? 
            # But pure Warnsdorff with restarts is usually robust.
            # Let's stick to picking the best, but the random noise in score 
            # ensures that if counts are equal, we pick randomly.
            best_score, next_v = candidates[0]
            
            path.append(next_v)
            used_mask |= (1 << next_v)
            
            # Update exclusion: cur becomes non-terminal
            # Add neighbors of cur to excl_mask
            excl_mask |= neighbors_mask_of_cur
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path.copy()
                
            # Time check
            if time.time() > deadline:
                return

    # Main loop: run multiple restarts
    # We use a different random seed for each restart to explore different paths
    restart_count = 0
    while time.time() < deadline:
        rng = random.Random(restart_count * 12345 + 678)
        search(rng)
        restart_count += 1
        # If we've found a very long snake, we might stop? 
        # No, keep trying to improve.
        
    return best_path

if __name__ == "__main__":
    res = construct()
    print(len(res))
