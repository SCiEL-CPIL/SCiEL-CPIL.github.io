"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import sys
from collections import defaultdict

# Precompute isosceles conflicts
# iso_conflict[p][a] is a set of q such that {p, a, q} is isosceles
# Actually, we need to know for a given p and existing set S, which q are forbidden.
# Forbidden q if exists a in S such that {p, a, q} is isosceles.
# So we precompute for each pair (p, a), the set of q.

# Grid points
points = [(x, y) for x in range(N) for y in range(N)]
point_to_idx = {p: i for i, p in enumerate(points)}
num_points = len(points)

# Precompute dist2
def dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

# Precompute iso_conflict_pairs[p_idx][a_idx] = set of q_idx
# This is 256 * 256 * (set of q). Might be memory heavy if not careful, but 256*256=65k pairs.
# Each set is small.
# We can use bitsets (integers) for q sets.
iso_conflict = [[0] * num_points for _ in range(num_points)]

for p_idx, p in enumerate(points):
    for a_idx, a in enumerate(points):
        if p_idx == a_idx:
            continue
        d_pa = dist2(p, a)
        mask = 0
        for q_idx, q in enumerate(points):
            if q_idx == p_idx or q_idx == a_idx:
                continue
            d_pq = dist2(p, q)
            d_aq = dist2(a, q)
            if d_pa == d_pq or d_pa == d_aq or d_pq == d_aq:
                mask |= (1 << q_idx)
        iso_conflict[p_idx][a_idx] = mask

# Helper to check if a set S (list of indices) is maximal
def is_maximal(S):
    S_set = set(S)
    # For each q not in S, check if it is blocked by some pair in S
    # Blocked if exists a, b in S such that {q, a, b} is isosceles.
    # We can precompute for each q, the pairs (a,b) that conflict?
    # Or just iterate pairs in S.
    # Since S is small, iterating pairs is fine.
    for q_idx, q in enumerate(points):
        if q_idx in S_set:
            continue
        blocked = False
        for i in range(len(S)):
            a_idx = S[i]
            # Check if q is in iso_conflict[a_idx][b_idx] for any b in S
            # Actually, iso_conflict[a_idx][b_idx] contains q if {a,b,q} is isosceles.
            for j in range(i+1, len(S)):
                b_idx = S[j]
                if iso_conflict[a_idx][b_idx] & (1 << q_idx):
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            return False
    return True

def construct():
    n = N
    # Try increasing K
    # Lower bound for maximal set? 
    # A single point is not maximal (can add others).
    # Let's start from K=1
    for K in range(1, num_points + 1):
        # DFS to find a maximal set of size K
        found = _find_maximal_set_size_K(K)
        if found:
            return [points[i] for i in found]
    # Fallback to greedy if no small set found (should not happen)
    return _greedy_construct(n)

def _find_maximal_set_size_K(K):
    # State: current set S (list of indices), available mask (int)
    # We want to pick K points.
    # To reduce symmetry, we can force the first point to be the smallest index in the available set?
    # Or just iterate.
    
    # Optimization: If K is small, we can use recursion.
    # Pruning: if popcount(available) < K - len(S), fail.
    
    def popcount(x):
        return bin(x).count('1')
        
    def dfs(S, available_mask, depth):
        if len(S) == K:
            if is_maximal(S):
                return S
            return None
        
        if popcount(available_mask) < K - len(S):
            return None
            
        # Iterate through available points
        # To break symmetry, we can require that the next point is the smallest available?
        # No, that restricts to sets where elements are sorted, which is fine for finding ONE set.
        # But we need to ensure we don't miss valid sets.
        # If we always pick the smallest available index, we only find sets that are "greedy" in index order?
        # No, that would only find ONE specific set (the lexicographically smallest).
        # We need to search all possibilities.
        
        # However, for "Minimum Maximal", we just need ANY set of size K.
        # So we can break symmetry by fixing the first point?
        # If we fix the first point to be index 0, we might miss the optimal set if it doesn't contain 0.
        # So we should iterate over the first point.
        
        if depth == 0:
            # First point: iterate over all available
            # To reduce branching, we can just pick the first available? No.
            # Let's iterate.
            for p_idx in range(num_points):
                if available_mask & (1 << p_idx):
                    new_avail = available_mask & ~(1 << p_idx)
                    # Remove points that conflict with p and any future point?
                    # No, we just remove p for now.
                    # The conflict removal happens when we add the SECOND point.
                    # Wait, if we pick p, then pick a, then q is forbidden if {p,a,q} is isosceles.
                    # So we only remove q from available when we have at least 2 points?
                    # No, if we have S={p}, and we pick a, then we remove q if {p,a,q} is isosceles.
                    # So the update of `available` depends on the new point and the existing set.
                    
                    # Let's pass S and available.
                    # When adding a new point `p_idx` to `S`:
                    # The new available mask should exclude:
                    # 1. p_idx itself
                    # 2. Any q such that exists a in S (old S) where {p_idx, a, q} is isosceles.
                    
                    # In the recursive call, we will compute the new available mask.
                    
                    res = dfs(S + [p_idx], new_avail, depth + 1)
                    if res:
                        return res
            return None
        else:
            # Pick next point
            for p_idx in range(num_points):
                if not (available_mask & (1 << p_idx)):
                    continue
                
                # Calculate new available mask
                new_avail = available_mask & ~(1 << p_idx)
                
                # For each a in S, remove q such that {p_idx, a, q} is isosceles
                # iso_conflict[p_idx][a_idx] gives the mask of such q's
                for a_idx in S:
                    new_avail &= ~iso_conflict[p_idx][a_idx]
                
                # Check if we have enough points left
                if popcount(new_avail) < K - (len(S) + 1):
                    continue
                
                res = dfs(S + [p_idx], new_avail, depth + 1)
                if res:
                    return res
            return None

    # Start with all points available
    initial_mask = (1 << num_points) - 1
    return dfs([], initial_mask, 0)

def _greedy_construct(n):
    """Greedy construction in row-major order."""
    chosen = []
    for p in itertools.product(range(n), repeat=2):
        if all(not _is_isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
            chosen.append(p)
    return chosen

def _is_isosceles(a, b, c):
    d1 = dist2(a, b)
    d2 = dist2(a, c)
    d3 = dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

# EVOLVE-BLOCK-END
