import time
import random
import itertools
from collections import defaultdict

N = 16
GRID_SIZE = N * N

def _dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist_sq(a, b)
    d2 = _dist_sq(a, c)
    d3 = _dist_sq(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _precompute_conflicts():
    """
    Precompute conflict masks.
    conflict_mask[i][j] is an integer bitmask where bit k is set if {i, j, k} is isosceles.
    """
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n = len(all_points)
    
    # conflict_mask[i][j]
    conflict_mask = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if _is_isosceles(all_points[k], all_points[i], all_points[j]):
                    mask |= (1 << k)
            conflict_mask[i][j] = mask
            conflict_mask[j][i] = mask
            
    return all_points, conflict_mask

def construct():
    all_points, conflict_mask = _precompute_conflicts()
    n = len(all_points)
    all_points_mask = (1 << n) - 1
    
    start_time = time.time()
    deadline = start_time + 290 # Safety margin
    
    best_set = None
    best_size = float('inf')

    # Helper to compute blocked_mask for a given S (list of indices)
    def compute_blocked_mask(S):
        S_set = set(S)
        blocked = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                blocked |= conflict_mask[S[i]][S[j]]
        return blocked

    # Check if S is maximal
    def is_maximal(S, blocked_mask):
        S_mask = 0
        for i in S:
            S_mask |= (1 << i)
        # Points not in S and not blocked can be added
        addable = all_points_mask & ~S_mask & ~blocked_mask
        return addable == 0

    # Reduce to minimal maximal
    def reduce_to_minimal_maximal(S):
        S = list(S)
        if not S:
            return S
        
        blocked_mask = compute_blocked_mask(S)
        
        changed = True
        while changed:
            changed = False
            for i in range(len(S)):
                x = S[i]
                S_without_x = [s for s in S if s != x]
                
                if not S_without_x:
                    # If S was size 1, removing it makes S empty. 
                    # Empty set is maximal only if n=0 (not the case) or all points blocked? No, empty set blocks nothing.
                    # So S of size 1 is minimal maximal only if it blocks all other points.
                    # We check this in the general case.
                    pass

                blocked_new = compute_blocked_mask(S_without_x)
                
                # Check if x is blocked by S_without_x
                # x is blocked if (blocked_new >> x) & 1 is 1
                if not ((blocked_new >> x) & 1):
                    continue # Cannot remove x, it would not be maximal (x could be added back)
                
                # Check if S_without_x is maximal
                S_mask_new = 0
                for s in S_without_x:
                    S_mask_new |= (1 << s)
                
                addable = all_points_mask & ~S_mask_new & ~blocked_new
                if addable == 0:
                    # Valid removal
                    S = S_without_x
                    blocked_mask = blocked_new
                    changed = True
                    break
        
        return S

    # Greedy construction
    def greedy_build():
        S = []
        S_mask = 0
        blocked_mask = 0
        remaining = list(range(n))
        random.shuffle(remaining)
        
        for p in remaining:
            # Check if p is blocked
            if (blocked_mask >> p) & 1:
                continue
            if (S_mask >> p) & 1:
                continue
            
            # Check if adding p is valid (it is, if not blocked and not in S)
            # But we need to ensure that p doesn't form isosceles with existing pairs?
            # "No-isosceles" means no 3 points in S form isosceles.
            # If p is not in blocked_mask, it means no pair in S forms isosceles with p.
            # So S U {p} is valid.
            
            # Update blocked_mask
            # New pairs are (p, q) for q in S
            new_blocked = 0
            for q in S:
                new_blocked |= conflict_mask[p][q]
            
            S.append(p)
            S_mask |= (1 << p)
            blocked_mask |= new_blocked
            
            # Optimization: if S is maximal, we can stop? 
            # No, we want minimal maximal. Greedy doesn't guarantee minimal.
            # But if we stop early, we might miss a smaller set.
            # However, if we keep adding, size increases.
            # We should stop if no more points can be added (maximal).
            # Then we reduce.
            
            if is_maximal(S, blocked_mask):
                break
                
        return S

    # Local Search
    def local_search(S):
        S = list(S)
        if not S:
            return S
            
        blocked_mask = compute_blocked_mask(S)
        
        start_time_ls = time.time()
        limit = start_time_ls + 5 # 5 seconds per local search instance
        
        while time.time() < limit:
            improved = False
            # Try to remove a point
            # We already do this in reduce_to_minimal_maximal, 
            # but local search can try swaps or different removal orders.
            
            # Try swap: remove x, add y
            # This is complex to verify quickly.
            # Let's stick to random restarts + reduction + maybe some perturbation.
            
            # Perturbation: pick random x in S, remove it (even if not minimal), 
            # then greedily fill again?
            
            # Let's try a simple 1-move: remove x, if valid and maximal, accept.
            # This is essentially reduce_to_minimal_maximal but randomized.
            
            # Let's try to find a point x in S that can be removed.
            # If none, try a swap.
            
            # Randomly pick x in S
            x = random.choice(S)
            S_without_x = [s for s in S if s != x]
            blocked_new = compute_blocked_mask(S_without_x)
            
            # Check if x is blocked by S_without_x
            if (blocked_new >> x) & 1:
                # x is blocked, so S_without_x is not maximal? 
                # Wait. If x is blocked by S_without_x, then x CANNOT be added to S_without_x.
                # So S_without_x is a valid no-isosceles set.
                # Is it maximal? 
                # We need to check if any other point y not in S_without_x can be added.
                S_mask_new = 0
                for s in S_without_x:
                    S_mask_new |= (1 << s)
                addable = all_points_mask & ~S_mask_new & ~blocked_new
                if addable == 0:
                    # S_without_x is maximal and smaller.
                    S = S_without_x
                    blocked_mask = blocked_new
                    improved = True
                    continue # Restart loop with new S
                
            # If removal didn't work, try a swap?
            # Or just break and rely on restarts.
            
            if not improved:
                # Try a random restart of the local search phase?
                # Or break.
                break
                
        return S

    # Run multiple restarts
    num_restarts = 1000 # Increased restarts due to speedup
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        S = greedy_build()
        S = reduce_to_minimal_maximal(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S.copy()
            # If we find a very small set, we can break early?
            # Theoretical minimum is not known, but likely < 20.
            if best_size <= 10:
                break
        
        # Local search
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            S = local_search(S)
            S = reduce_to_minimal_maximal(S)
            
            if len(S) < best_size:
                best_size = len(S)
                best_set = S.copy()
                if best_size <= 10:
                    break

    if best_set is None:
        best_set = list(range(n))
    
    return [all_points[i] for i in best_set]
