"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import reduce
from collections import defaultdict

_divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_n = len(_divisors)
_max_val = 1000000
_start_time = time.time()
_deadline = _start_time + 82.0  # 8s safety margin before 90s kill

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_lcm_all = reduce(_lcm, _divisors)
# _lcm_all = 32730

# Precompute for each divisor d, the residues mod _lcm_all that are multiples of d.
# Actually, for a window [L, L+W], the multiples of d in the window are:
# k*d where L <= k*d <= L+W
# i.e., k in [ceil(L/d), floor((L+W)/d)]

# Key structural change: Instead of searching over L and W separately,
# model this as a constraint satisfaction problem over the residues.
#
# New approach: Use the fact that the problem is periodic with period _lcm_all.
# The window [L, L+W] is feasible iff for each divisor d_i, there exists a multiple
# of d_i in the window, and all chosen multiples are distinct.
#
# Since the pattern repeats every _lcm_all, we can think of the "phase" of the window
# as r = L mod _lcm_all. For a given phase r and width W, the set of available
# multiples for each divisor d_i is determined.
#
# Critical insight: The minimal W is achieved when the window is "tight" - i.e.,
# the chosen multiples are as close together as possible. This suggests we should
# search over the relative positions of the multiples within one LCM period.
#
# Alternative structural approach: Formulate as a shortest path / minimum span problem.
# For each divisor d_i, consider the set of all multiples {k*d_i : k >= 1, k*d_i <= _max_val}.
# We need to pick one from each set, all distinct, minimizing max-min.
#
# This is equivalent to: find the minimum W such that there exists an interval of length W
# containing at least one multiple of each d_i, with all chosen multiples distinct.
#
# Since the divisors are fixed and small (max 39), and _lcm_all = 32730, we can:
# 1. For each possible phase r in [0, _lcm_all), determine which multiples of each d_i
#    fall in a window of width W starting at phase r.
# 2. Use backtracking with strong pruning to find the minimal W.
#
# But 32730 phases is too many to brute force with backtracking.
#
# Better structural change: Use a SAT-like approach with bitsets.
# For a given W and phase r, the candidates for divisor d_i are:
#   {k*d_i : k*d_i mod _lcm_all is in [r, r+W] mod _lcm_all} intersect [1, _max_val]
#
# Actually, let's try a completely different structural approach:
#
# OBSERVATION: The divisors have many common factors. Let's compute the GCD structure.
# gcd of all divisors: gcd(6,10,14,15,21,22,26,33,35,39) = 1
# So the LCM approach is necessary.
#
# NEW STRUCTURAL APPROACH: Local search / simulated annealing on the assignment.
# Represent the solution as a tuple (v_1, ..., v_10) where v_i is a multiple of d_i.
# Objective: minimize max(v) - min(v).
# Start from a random valid assignment, then iteratively try to reduce the span.
#
# Actually, let's try a more systematic approach using the periodicity:
#
# For each residue r in [0, _lcm_all), we can compute the "best" window of minimal width
# starting at phase r. The key is that for a fixed phase r, the problem decomposes:
# each divisor d_i has a set of candidate multiples within [r, r+W] (mod _lcm_all),
# and we need to pick one from each set, all distinct.
#
# Since _lcm_all = 32730 is manageable, and for each phase we can use a fast
# feasibility check, let's try this:
#
# 1. Precompute for each phase r and each divisor d_i, the list of multiples
#    of d_i that fall in [r, r + _lcm_all - 1] (one full period).
# 2. For each phase r, binary search on W to find the minimal feasible width.
# 3. Track the global minimum.
#
# But 32730 * log(32730) * DFS is still too slow.
#
# Let's try yet another structural change: Use the fact that the optimal window
# will have its endpoints at multiples of some divisors. So we only need to consider
# L values that are multiples of at least one divisor.
#
# Number of such L values up to _max_val: roughly _max_val / 2 * 10 = 5 million. Too many.
#
# FINAL STRUCTURAL APPROACH: Use a priority queue / Dijkstra-like search over
# "states" where a state is (min_val, max_val, assigned_set). Start from the
# smallest possible multiples and expand outward.
#
# Actually, the simplest effective structural change: Use the periodicity to
# reduce the search space, and use a very fast feasibility check via bipartite
# matching or greedy with backtracking.
#
# Let's implement: For each phase r (0 to _lcm_all-1), compute the minimal W
# using a greedy + backtracking approach that is very fast because the candidate
# sets are small (at most W/d_i + 1 candidates per divisor).

def _solve_window_fast(L, W):
    """Fast feasibility check for window [L, L+W]."""
    U = L + W
    if U > _max_val or U < 1:
        return None
    
    candidates = []
    for d in _divisors:
        k_min = max(1, (L + d - 1) // d)
        k_max = U // d
        if k_max < k_min:
            return None
        cands = [k * d for k in range(k_min, k_max + 1)]
        candidates.append(cands)
    
    # Sort by candidate count (most constrained first)
    order = sorted(range(_n), key=lambda i: len(candidates[i]))
    used = set()
    assignment = [None] * _n
    
    def dfs(idx):
        if idx == _n:
            return True
        i = order[idx]
        for v in candidates[i]:
            if v not in used:
                used.add(v)
                assignment[i] = v
                if dfs(idx + 1):
                    return True
                used.discard(v)
                assignment[i] = None
        return False
    
    if dfs(0):
        return [int(assignment[i]) for i in range(_n)]
    return None

def construct():
    best_result = None
    best_raw = float('inf')
    
    # Phase 1: Direct search over small W values with smart L selection.
    # For each W, try L values that are "aligned" with the divisor structure.
    
    # Key insight: The optimal L will be near a multiple of one of the divisors.
    # Generate candidate L values efficiently.
    
    # For W from 9 upward, check if there's a feasible window.
    # To speed up, for each W, generate candidate L values that are multiples
    # of small divisors or near them.
    
    for W in range(9, 300):
        if time.time() > _deadline:
            break
        
        # Generate candidate L values for this W
        L_candidates = set()
        
        # L should be such that [L, L+W] contains multiples of all divisors.
        # A necessary condition: for each d_i, there exists k such that
        # L <= k*d_i <= L+W, i.e., k*d_i - W <= L <= k*d_i.
        # So L must be in the union over i of (k*d_i - W, k*d_i] for some k.
        
        # For efficiency, just try L values that are multiples of each divisor
        # minus offsets from 0 to W.
        for d in _divisors:
            # k*d - W <= L <= k*d, so L can be k*d - j for j in [0, W]
            # But L >= 1 and L + W <= _max_val
            k_min = max(1, (1 + W) // d)  # k*d - W >= 1 => k >= (1+W)/d
            k_max = (_max_val - W) // d   # k*d <= _max_val - W + W = _max_val... 
            # Actually L + W <= _max_val => L <= _max_val - W
            # And k*d - W <= L => L >= k*d - W
            # Also L >= 1
            # So for each k, L can range from max(1, k*d - W) to min(k*d, _max_val - W)
            # We'll sample a few L values per k.
            
            # Limit k range to keep it manageable
            k_start = max(1, (1 + W) // d)
            k_end = min((_max_val - W) // d + 1, k_start + 500)  # Limit to 500 k values
            
            for k in range(k_start, k_end + 1):
                base = k * d
                # Try L = base - j for j in a small range
                for j in range(0, min(W + 1, 20)):
                    L = base - j
                    if 1 <= L <= _max_val - W:
                        L_candidates.add(L)
        
        # Also add some random L values
        import random
        for _ in range(100):
            L = random.randint(1, _max_val - W)
            L_candidates.add(L)
        
        # Sort L candidates and try them
        L_list = sorted(L_candidates)
        
        for L in L_list:
            if time.time() > _deadline:
                break
            
            result = _solve_window_fast(L, W)
            if result is not None:
                raw = max(result) - min(result)
                if raw < best_raw:
                    best_raw = raw
                    best_result = result[:]
        
        # If we found a solution with raw == W, we can potentially stop
        # but only if we're confident no smaller W works. Since we're going
        # upward in W, the first W with a solution is the minimal W.
        if best_raw == W:
            # We found a solution with span exactly W. Check if we can do better
            # with the same W by trying more L values, but if best_raw == W,
            # we've found the minimum possible for this W.
            # Continue to see if a smaller W was missed (shouldn't happen since
            # we're iterating upward, but just in case).
            break
    
    # Phase 2: If phase 1 didn't find a good solution, use periodicity-based search
    if best_raw > 50 or best_result is None:
        # For each residue r in [0, _lcm_all), find the minimal W
        # Use binary search on W for each residue.
        
        for r in range(0, _lcm_all):
            if time.time() > _deadline:
                break
            
            # Earliest positive L with L ≡ r (mod _lcm_all)
            if r == 0:
                L_min = _lcm_all
            else:
                L_min = r
            
            if L_min > _max_val:
                continue
            
            # Binary search on W for this residue
            lo = 9
            hi = min(_max_val - L_min, 500)  # Cap hi to keep it fast
            
            if hi < lo:
                continue
            
            # Quick check: is hi feasible?
            result = _solve_window_fast(L_min, hi)
            if result is None:
                continue
            
            best_w_for_r = hi
            while lo < hi:
                if time.time() > _deadline:
                    break
                mid = (lo + hi) // 2
                result = _solve_window_fast(L_min, mid)
                if result is not None:
                    best_w_for_r = mid
                    hi = mid
                else:
                    lo = mid + 1
            
            if best_w_for_r < best_raw:
                result = _solve_window_fast(L_min, best_w_for_r)
                if result is not None:
                    raw = max(result) - min(result)
                    if raw < best_raw:
                        best_raw = raw
                        best_result = result[:]
    
    if best_result is None:
        # Fallback: return the divisors themselves
        return [int(d) for d in _divisors]
    
    return best_result
# EVOLVE-BLOCK-END
