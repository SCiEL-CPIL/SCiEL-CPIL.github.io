"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from collections import deque

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // gcd(a, b)

def _get_candidates(L, W, div_idx):
    """Get all multiples of DIVISORS[div_idx] in [L, L+W]."""
    d = DIVISORS[div_idx]
    k_start = (L + d - 1) // d
    k_end = (L + W) // d
    if k_start > k_end:
        return []
    return [k * d for k in range(k_start, k_end + 1)]

def _check_window(L, W):
    """Check if there exist distinct multiples for all divisors in [L, L+W] using bipartite matching."""
    if L < 1 or L + W > MAX_VAL:
        return None
    
    # Get candidates for each divisor
    candidates = []
    for i in range(N):
        cands = _get_candidates(L, W, i)
        if not cands:
            return None
        candidates.append(cands)
    
    # Build bipartite graph: left side = divisors (0..N-1), right side = values
    # Use Hopcroft-Karp for efficiency, but for small N, simple DFS augmenting paths work well
    
    # Map values to indices for quick lookup
    all_vals = []
    val_to_idx = {}
    for cands in candidates:
        for v in cands:
            if v not in val_to_idx:
                val_to_idx[v] = len(all_vals)
                all_vals.append(v)
    
    n_vals = len(all_vals)
    if n_vals < N:
        return None
    
    # adjacency: adj[i] = list of value indices that divisor i can take
    adj = []
    for i in range(N):
        adj.append([val_to_idx[v] for v in candidates[i]])
    
    # DFS-based augmenting path matching (Kuhn's algorithm)
    match_val = [-1] * n_vals  # which divisor is matched to each value
    
    def try_augment(div_idx, visited):
        for v_idx in adj[div_idx]:
            if visited[v_idx]:
                continue
            visited[v_idx] = True
            if match_val[v_idx] == -1 or try_augment(match_val[v_idx], visited):
                match_val[v_idx] = div_idx
                return True
        return False
    
    match_div = [-1] * N
    for i in range(N):
        visited = [False] * n_vals
        if try_augment(i, visited):
            # Find which value i got matched to
            for v_idx in range(n_vals):
                if match_val[v_idx] == i:
                    match_div[i] = v_idx
                    break
        else:
            return None
    
    # Reconstruct assignment
    result = [all_vals[match_div[i]] for i in range(N)]
    return result

def _check_window_greedy(L, W):
    """Faster greedy check: sort divisors by number of candidates (ascending) and assign greedily with backtracking only on conflicts."""
    if L < 1 or L + W > MAX_VAL:
        return None
    
    candidates = []
    for i in range(N):
        cands = _get_candidates(L, W, i)
        if not cands:
            return None
        candidates.append(cands)
    
    # Order divisors by number of candidates (most constrained first)
    order = sorted(range(N), key=lambda i: len(candidates[i]))
    
    assigned = [None] * N
    used = set()
    
    def backtrack(pos):
        if pos == N:
            return True
        i = order[pos]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if backtrack(pos + 1):
                    return True
                used.remove(val)
                assigned[i] = None
        return False
    
    if backtrack(0):
        return assigned[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s
    
    # Strategy: 
    # 1. Try small W values from 9 upward, checking feasibility at strategic L positions
    # 2. For each W, check L at positions that are "aligned" with the divisors
    # 3. Use the faster greedy+backtracking check
    
    best_result = None
    best_W = None
    
    # Precompute LCM of all divisors for alignment
    lcm_all = 1
    for d in DIVISORS:
        lcm_all = _lcm(lcm_all, d)
    
    # For a given W, good L positions are near multiples of small LCMs
    # Try L = 1, and L near multiples of each divisor and their pairwise LCMs
    
    def get_L_candidates(W):
        """Generate strategic L positions to check for a given window width W."""
        positions = set()
        # Start from 1
        positions.add(1)
        # Try L such that L mod d is favorable for each divisor
        for d in DIVISORS:
            # L where L is a multiple of d (so first candidate is L itself)
            positions.add(1)
            # L just before a multiple of d
            positions.add(d)
        # Try small range of L values
        for L in range(1, min(200, MAX_VAL - W + 1)):
            positions.add(L)
        # Try L at multiples of 2, 3, 5, 7 (small primes)
        for p in [2, 3, 5, 7, 11, 13]:
            for k in range(1, min(MAX_VAL // p - W + 1, 500) + 1):
                L = k * p
                if 1 <= L <= MAX_VAL - W:
                    positions.add(L)
        # Try L near LCM multiples
        for k in range(1, min(MAX_VAL // lcm_all + 1, 100) + 1):
            L = k * lcm_all
            if 1 <= L <= MAX_VAL - W:
                positions.add(L)
            # Also L = k*lcm_all - offset for small offsets
            for off in range(1, 10):
                L2 = k * lcm_all - off
                if 1 <= L2 <= MAX_VAL - W:
                    positions.add(L2)
        return sorted(positions)
    
    # First pass: try W from 9 upward
    for W in range(9, 200):
        if time.time() > deadline - 5:
            break
        
        L_positions = get_L_candidates(W)
        # Limit number of L positions to check
        if len(L_positions) > 500:
            # Sample strategically
            step = len(L_positions) // 500
            L_positions = L_positions[::step]
            L_positions = L_positions[:500]
        
        for L in L_positions:
            if time.time() > deadline - 5:
                break
            result = _check_window_greedy(L, W)
            if result is not None:
                best_result = result
                best_W = W
                break
        
        if best_result is not None:
            # Found a solution with this W; try to see if smaller W works
            # But since we're going upward, this is the minimum W found so far
            # Let's do a few more checks with this W at different L to confirm
            break
    
    # If we found a solution, try to improve by checking W-1, W-2, etc.
    if best_result is not None and best_W > 10:
        for W in range(best_W - 1, 8, -1):
            if time.time() > deadline - 5:
                break
            L_positions = get_L_candidates(W)
            if len(L_positions) > 300:
                step = len(L_positions) // 300
                L_positions = L_positions[::step]
                L_positions = L_positions[:300]
            
            found = False
            for L in L_positions:
                if time.time() > deadline - 5:
                    break
                result = _check_window_greedy(L, W)
                if result is not None:
                    best_result = result
                    best_W = W
                    found = True
                    break
            if found:
                continue
            else:
                # Can't go smaller
                break
    
    # If no solution found yet, expand W
    if best_result is None:
        for W in range(200, 2000, 50):
            if time.time() > deadline - 5:
                break
            L_positions = get_L_candidates(W)
            if len(L_positions) > 200:
                step = len(L_positions) // 200
                L_positions = L_positions[::step]
                L_positions = L_positions[:200]
            
            for L in L_positions:
                if time.time() > deadline - 5:
                    break
                result = _check_window_greedy(L, W)
                if result is not None:
                    best_result = result
                    best_W = W
                    break
            if best_result is not None:
                break
    
    # Final fallback
    if best_result is None:
        # Use the divisors themselves
        divs = DIVISORS[:]
        if len(set(divs)) == N and max(divs) <= MAX_VAL:
            best_result = divs
        else:
            best_result = [d * 1 for d in DIVISORS]
    
    # Verify result
    assert len(best_result) == N
    assert all(isinstance(x, int) for x in best_result)
    assert len(set(best_result)) == N
    assert all(x >= 1 and x <= MAX_VAL for x in best_result)
    for i in range(N):
        assert best_result[i] % DIVISORS[i] == 0
    
    return list(best_result)

# EVOLVE-BLOCK-END
