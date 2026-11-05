import time
import random

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    # Precompute all 4-APs
    aps = []
    # d is the difference
    # a, a+d, a+2d, a+3d
    # max a+3d <= n => a <= n - 3d
    # a >= 1
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        if max_a < 1:
            break
        for a in range(1, max_a + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # Build index: for each x, list of indices of APs containing x
    # Or better: for each x, list of the other 3 elements
    # neighbors[x] = list of (y, z, w)
    neighbors = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(aps):
        a, b, c, d_val = ap
        neighbors[a].append((b, c, d_val))
        neighbors[b].append((a, c, d_val))
        neighbors[c].append((a, b, d_val))
        neighbors[d_val].append((a, b, c))

    best_size = 0
    best_set = set()
    
    def run_greedy():
        in_S = [False] * (n + 1)
        blocked = [False] * (n + 1)
        active = list(range(1, n + 1))
        random.shuffle(active)
        
        current_size = 0
        
        i = 0
        while i < len(active):
            x = active[i]
            if blocked[x]:
                # Skip, effectively remove
                # We don't swap, just increment i? 
                # If we don't remove, the list stays large.
                # Better to compact or swap with end.
                # Let's swap with last and pop.
                last = active.pop()
                if i < len(active):
                    active[i] = last
                continue
            
            # Try to add x
            # Check if adding x creates a 4-AP with 3 elements in S?
            # No, we maintain 'blocked' such that any x in 'active' and not blocked
            # is safe to add?
            # Wait.
            # If we add x, we must ensure no 4-AP is completed.
            # Our 'blocked' logic:
            # y is blocked if there exist z, w in S such that {x, y, z, w} is a 4-AP?
            # No, that's for future additions.
            # If we add x, we check if there are 3 elements in S forming a 4-AP with x.
            # If yes, x is NOT safe.
            
            # So 'blocked' should mean "x is part of a 4-AP with >= 2 elements in S"?
            # No.
            # If x is in a 4-AP with 2 elements in S, say z, w, and y is the 4th.
            # Then if we add x, we need y to be NOT in S.
            # But y might be added later.
            # So if we add x, we must ensure y is blocked.
            # But y is already in 'active'.
            
            # Correct Logic:
            # A candidate x is "safe" to add if it does not complete any 4-AP with elements in S.
            # i.e., for all 4-APs containing x, the other 3 elements are not ALL in S.
            # Since we only add elements, the count of elements in S for any AP only increases.
            # If an AP has 3 elements in S, the 4th is "forbidden".
            # If an AP has 2 elements in S, the other 2 are "restricted" (if one is added, the other is forbidden).
            
            # So, we can maintain a `conflict_count` for each candidate y.
            # conflict_count[y] = number of 4-APs containing y that have >= 2 other elements in S.
            # Actually, simpler:
            # y is blocked if there exists a 4-AP containing y with 3 elements in S.
            # When we add x:
            #   For each 4-AP containing x:
            #     Let other 3 be y, z, w.
            #     If z, w are in S, then y is blocked.
            #     If z, y are in S, then w is blocked.
            #     If w, y are in S, then z is blocked.
            
            # So we need to know which of the other 3 are in S.
            
            safe = True
            for y, z, w in neighbors[x]:
                if in_S[y] and in_S[z] and in_S[w]:
                    safe = False
                    break
                # Check if adding x blocks y, z, or w?
                # If in_S[y] and in_S[z]: block w
                # If in_S[y] and in_S[w]: block z
                # If in_S[z] and in_S[w]: block y
                
                # We can handle blocking after confirming safe?
                # Yes.
                
            if not safe:
                # x cannot be added. Mark as blocked?
                # It might become unblocked? No, S only grows.
                # So x is permanently blocked.
                blocked[x] = True
                last = active.pop()
                if i < len(active):
                    active[i] = last
                continue
                
            # Add x
            in_S[x] = True
            current_size += 1
            
            # Update blocked status for neighbors
            for y, z, w in neighbors[x]:
                # If 2 of {y,z,w} are in S, block the 3rd
                if in_S[y] and in_S[z]:
                    blocked[w] = True
                if in_S[y] and in_S[w]:
                    blocked[z] = True
                if in_S[z] and in_S[w]:
                    blocked[y] = True
            
            # Remove x from active
            last = active.pop()
            if i < len(active):
                active[i] = last
            # Do NOT increment i, because we replaced x with last
            
        return current_size, in_S

    # Run multiple restarts
    while time.time() < deadline:
        size, in_S = run_greedy()
        if size > best_size:
            best_size = size
            best_set = set(i for i in range(1, n + 1) if in_S[i])
            
    return sorted(best_set)
