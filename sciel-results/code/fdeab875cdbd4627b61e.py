"""
Strategy: Deterministic Iterative Deepening Removal.
Instead of random local search, we:
1. Generate a valid maximal set using a deterministic greedy approach (or a few diverse deterministic orders).
2. Attempt to reduce the set size by iteratively removing points.
3. A point can be removed if the resulting set is still "maximal" (i.e., every removed point and every other non-member is still blocked).
   Wait, the definition of maximal is: no point OUTSIDE can be added.
   If we remove a point p from S, S' = S \ {p}. 
   S' is maximal if:
   a) S' is valid (no isosceles). Since S was valid, S' is always valid.
   b) Every point u not in S' is blocked by S'.
      The points not in S' are: (All points) \ S'.
      This set includes p (the one we removed) and all points that were originally not in S.
      So we must check:
      - Is p blocked by S'? (If p is not blocked by S', we could add it back, so S' is not maximal with respect to p? 
        Actually, if p is not blocked by S', it means S' U {p} is valid. Since S' U {p} = S, and S is valid, 
        this doesn't contradict validity. But for maximality, we require that NO point outside S' can be added.
        So if p can be added to S', then S' is NOT maximal.
        Therefore, to remove p, we must ensure p is blocked by S'.
      - Are all other points u (u not in S, u != p) still blocked by S'?
        Originally, u was blocked by S. Since S' is a subset of S, removing a point from the "blockers" might unblock u.
        So we must re-verify that every u outside S is blocked by S'.

   This check is expensive if done naively for every removal.
   However, we can precompute which points block which.
   A point u is blocked by S if there exist a, b in S such that {u, a, b} is isosceles.
   
   Let's define "Blockers(u)" as the set of points v such that v is part of an isosceles triangle with u and some other point?
   No, the condition is: u is blocked by S iff there exist a, b in S such that triangle(u, a, b) is isosceles.
   
   This is a 3-way interaction.
   However, we can simplify the "removal" check.
   If we remove p from S, the only new point that *might* become addable is p itself (since all other points were already not in S, and their blocking status depends on S, which shrinks).
   Wait, if u was blocked by S, it was because of some a, b in S. If we remove p, and p was one of the blockers (a or b), u might become unblocked.
   So removing p can unblock:
   1. p itself (trivially, p is now outside).
   2. Any u that was blocked ONLY by triangles involving p.
   
   To make this efficient, we can precompute for each point u, the set of points v in S that are "essential" for blocking u?
   That's complex.
   
   Alternative Strategy:
   Use a different definition of search.
   The problem asks for the MINIMUM MAXIMAL no-isosceles set.
   This is equivalent to finding a small dominating set in a specific hypergraph, or a small maximal independent set in a 3-uniform hypergraph (where edges are isosceles triangles).
   
   Let's stick to the removal heuristic but optimize the check.
   For a given S, to check if we can remove p:
   1. Check if p is blocked by S \ {p}.
      p is blocked by S \ {p} iff there exist a, b in S \ {p} such that (p, a, b) is isosceles.
   2. Check if any u in (All \ S) is unblocked by S \ {p}.
      u was blocked by S. It is unblocked by S \ {p} iff NO pair a, b in S \ {p} forms an isosceles triangle with u.
      This is equivalent to: For all a, b in S \ {p}, (u, a, b) is NOT isosceles.
      Or: The set of pairs (a,b) in S that form isosceles with u must be a subset of pairs involving p.
      
   This is still O(|S|^2) per removal candidate.
   With |S| ~ 20-30, and V=256, and trying to remove each point once per "pass", it's manageable.
   
   Let's implement:
   1. Precompute `is_isosceles[u][a][b]`? Too much memory (256^3).
   2. Precompute `dist2[u][v]`.
   3. For a given S (list of indices), and a candidate removal p:
      - Create S_rem = S \ {p}.
      - Check if p is blocked by S_rem:
        - For all a, b in S_rem, check if dist2[p][a] == dist2[p][b] or dist2[p][a] == dist2[a][b] or dist2[p][b] == dist2[a][b].
        - If any such pair exists, p is blocked.
      - Check if any u in (All \ S) is unblocked by S_rem:
        - For each u in (All \ S):
          - We need to check if there exists a, b in S_rem such that (u, a, b) is isosceles.
          - If NO such pair exists, u is unblocked -> Removal fails.
          
   This is O(|S|^2 * (V - |S|)) per removal attempt.
   If |S| ~ 20, V=256, this is ~ 400 * 236 ~ 100,000 ops per attempt.
   We try to remove each of the |S| points. So ~ 20 * 100,000 = 2,000,000 ops per full pass.
   We do multiple passes until no removal is possible.
   This is very fast.
   
   To improve the initial set size, we can use multiple deterministic starting orders (e.g., sorted by x, by y, by x+y, by distance from center) and pick the smallest result after the removal phase.
"""

import itertools
import time
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
V = len(POINTS)

# Precompute squared distances
DIST2 = np.zeros((V, V), dtype=np.int64)
for i in range(V):
    xi, yi = POINTS[i]
    for j in range(V):
        xj, yj = POINTS[j]
        DIST2[i, j] = (xi - xj) ** 2 + (yi - yj) ** 2

def _is_isosceles(u, a, b):
    """Check if triangle (u, a, b) is isosceles."""
    du_a = DIST2[u, a]
    du_b = DIST2[u, b]
    da_b = DIST2[a, b]
    return du_a == du_b or du_a == da_b or du_b == da_b

def _is_blocked_by(u, S_list):
    """Check if point u is blocked by set S_list."""
    n = len(S_list)
    if n < 2:
        return False
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if _is_isosceles(u, a, b):
                return True
    return False

def _try_remove(S_set, p):
    """
    Check if point p can be removed from S_set while maintaining maximality.
    Returns True if S_set \ {p} is maximal.
    """
    S_list = list(S_set)
    if p not in S_set:
        return False
    
    S_rem_list = [x for x in S_list if x != p]
    S_rem_set = set(S_rem_list)
    
    # 1. Check if p is blocked by S_rem
    if not _is_blocked_by(p, S_rem_list):
        return False
        
    # 2. Check if any other point u (not in S_set) is unblocked by S_rem
    # Points not in S_set:
    # We only need to check points that were previously blocked by pairs involving p.
    # But for simplicity and correctness, check all u not in S_set.
    # Optimization: Only check u if u was blocked by S. 
    # Actually, if u was not blocked by S, it wasn't in S to begin with (since S is maximal).
    # So all u not in S were blocked by S.
    # We need to ensure they are STILL blocked by S_rem.
    
    # To speed up, we can precompute for each u, the set of pairs (a,b) in S that block it?
    # That's dynamic.
    
    # Let's just iterate u not in S_set.
    for u in range(V):
        if u in S_set:
            continue
        # u was blocked by S. Is it blocked by S_rem?
        if not _is_blocked_by(u, S_rem_list):
            return False
            
    return True

def _greedy_construct(order):
    """
    Construct a maximal valid set by iterating through 'order' and adding points if valid.
    """
    S = set()
    for p in order:
        # Check if p can be added to S
        # p can be added if no isosceles triangle is formed with any two points in S.
        # i.e., for all a, b in S, (p, a, b) is not isosceles.
        # And for all a in S, d(p,a) is distinct? 
        # Wait, the condition for S U {p} to be valid is:
        # 1. S is valid (already true).
        # 2. No isosceles triangle involving p.
        #    Triangles involving p are (p, a, b) for a, b in S.
        #    So we just need to check that for all a, b in S, (p, a, b) is NOT isosceles.
        
        S_list = list(S)
        n = len(S_list)
        can_add = True
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if _is_isosceles(p, a, b):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            S.add(p)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_size = V
    best_set = []
    
    # Define several deterministic orders to start with
    # 1. Standard row-major
    order1 = list(range(V))
    # 2. Reverse row-major
    order2 = list(reversed(range(V)))
    # 3. Sorted by x + y
    order3 = sorted(range(V), key=lambda i: POINTS[i][0] + POINTS[i][1])
    # 4. Sorted by distance from center (8, 8)
    order4 = sorted(range(V), key=lambda i: (POINTS[i][0]-8)**2 + (POINTS[i][1]-8)**2)
    # 5. Sorted by x - y
    order5 = sorted(range(V), key=lambda i: POINTS[i][0] - POINTS[i][1])
    
    orders = [order1, order2, order3, order4, order5]
    
    for order in orders:
        if time.time() > deadline:
            break
            
        # 1. Greedy construction
        S = _greedy_construct(order)
        current_size = len(S)
        
        if current_size < best_size:
            best_size = current_size
            best_set = list(S)
            
        # 2. Iterative removal
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove each point in S
            # Order of removal attempts: maybe sort by something? 
            # Let's try removing points in the order they were added? Or reverse?
            # Or just iterate through the set.
            S_list = list(S)
            # Shuffle for diversity? No, deterministic is better for reproducibility, 
            # but different orders might yield different local minima.
            # Let's iterate in the order of the initial 'order' list restricted to S.
            # Or just sorted by index.
            for p in S_list:
                if time.time() > deadline:
                    break
                if _try_remove(S, p):
                    S.remove(p)
                    improved = True
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_set = list(S)
                    # After a removal, the set changed, so we should restart the loop 
                    # to check if other points can now be removed, or continue?
                    # Restarting is safer to ensure we don't miss cascading removals.
                    break
                    
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
