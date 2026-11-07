"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from functools import lru_cache

def _has_matching(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i."""
    n = 12
    # For each i in 1..12, find candidates in (m, m+L] divisible by i
    # Use DFS with bitmasks for the 12 indices
    # Candidates: for each i, multiples of i in (m, m+L]
    # Build adjacency: index i -> list of candidate numbers
    # Then check perfect matching via DFS
    
    # Collect all candidate numbers in (m, m+L]
    # A number x is usable if it's divisible by at least one i in 1..12
    # For matching, each i needs a distinct multiple
    
    # For each i, the multiples in (m, m+L]:
    # first multiple > m: floor(m/i)*i + i
    # last multiple <= m+L: floor((m+L)/i)*i
    
    # Use DFS matching: try to assign each index i to a distinct number
    # Numbers are shared across indices, so we need bipartite matching
    
    # Build list of candidate numbers
    # Each candidate number x can serve any i where i|x
    # We need to assign distinct numbers to indices 1..12
    
    # Approach: DFS on indices, tracking used numbers
    # Since L <= 144, there are at most 144 numbers, but we only care about
    # numbers divisible by some i in 1..12
    
    # For speed: precompute for each i the list of multiples in (m, m+L]
    # Then do backtracking matching
    
    # Actually, since the interval is small, let's just enumerate candidates
    # A number x in (m, m+L] is a candidate if any i in 1..12 divides x
    
    # Build: for each i, list of x in (m, m+L] with i|x
    # Then check if perfect matching exists
    
    # Use a simple DFS with memoization on (index, used_mask)
    # But used_mask over up to 144 numbers is too big.
    # Instead, use the classic bipartite matching (Hopcroft-Karp or simple augmenting paths)
    
    # Simpler: since n=12 is small, use DFS trying indices in order
    # Track which numbers are used via a set
    
    # For efficiency, order indices by fewest options first
    
    # Get candidates for each i
    candidates = {}
    for i in range(1, 13):
        # multiples of i in (m, m+L]
        first = (m // i + 1) * i  # smallest multiple of i > m
        if first <= m:
            first += i
        cands = []
        x = first
        while x <= m + L:
            cands.append(x)
            x += i
        candidates[i] = cands
    
    # Check if any i has no candidates
    for i in range(1, 13):
        if not candidates[i]:
            return False
    
    # Order indices by number of candidates (fewest first) for pruning
    order = sorted(range(1, 13), key=lambda i: len(candidates[i]))
    
    # DFS: assign numbers to indices in 'order'
    # State: which numbers are used
    # Since numbers can be up to m+144, use a set
    
    # For better pruning, note that a number x can be used by any i where i|x
    # We need to track used numbers
    
    # Given small n=12, simple backtracking should work
    # But worst case could be slow. Let's use a more structured approach.
    
    # Alternative: model as bipartite graph and use augmenting path matching
    # Left side: indices 1..12
    # Right side: candidate numbers (union of all candidates[i])
    # Edge: i -> x if i|x
    
    # Collect all candidate numbers
    all_cands = set()
    for i in range(1, 13):
        all_cands.update(candidates[i])
    all_cands = sorted(all_cands)
    
    # Map number to index
    num_to_idx = {x: j for j, x in enumerate(all_cands)}
    num_count = len(all_cands)
    
    if num_count < 12:
        return False
    
    # Build adjacency: for each i (0-indexed: i-1), list of right-side indices
    adj = [set() for _ in range(12)]
    for i in range(1, 13):
        for x in candidates[i]:
            adj[i-1].add(num_to_idx[x])
    
    # Bipartite matching using augmenting paths
    # match_r[j] = left node matched to right node j, or -1
    match_r = [-1] * num_count
    
    def try_augment(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or try_augment(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    match_count = 0
    for u in range(12):
        visited = [False] * num_count
        if try_augment(u, visited):
            match_count += 1
    
    return match_count == 12

def _compute_f(m):
    """Compute f(12, m): least L > 0 such that matching exists in (m, m+L]."""
    for L in range(1, 145):
        if _has_matching(m, L):
            return L
    return 145  # shouldn't happen per contract

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_m = 12
    best_f = _compute_f(12)
    
    # Search strategy: sample m values
    # Key insight: f(12,m) is large when the interval (m, m+L] is "sparse"
    # in terms of having multiples of 1..12.
    # Numbers that are NOT divisible by any i in 1..12 are "wasted" space.
    # Actually, every number IS divisible by 1, so that's not right.
    # The constraint is that we need DISTINCT numbers, one for each i.
    # So we need at least 12 distinct numbers, each serving its respective i.
    
    # A number x can serve i if i|x. To make f large, we want the interval
    # to be filled with numbers that are "expensive" to use - i.e., numbers
    # that are only divisible by small i's, forcing conflicts.
    
    # Let's just do a broad search
    # Phase 1: coarse scan
    step = 1
    max_m = 1000000
    
    # Try a range of m values
    # First, try some structured candidates
    candidates_to_try = []
    
    # m values where the interval boundary aligns with multiples
    # Try m = k * lcm(1..12) - 1 for various k, so the interval starts just after a common multiple
    # lcm(1..12) = 27720
    lcm_12 = 27720
    for k in range(1, max_m // lcm_12 + 1):
        m_val = k * lcm_12 - 1
        if 0 <= m_val <= max_m:
            candidates_to_try.append(m_val)
    
    # Also try random-ish spread
    for m_val in range(0, max_m + 1, 1000):
        candidates_to_try.append(m_val)
    
    # Deduplicate and sort
    candidates_to_try = sorted(set(candidates_to_try))
    
    for m_val in candidates_to_try:
        if time.time() > deadline:
            break
        if not (0 <= m_val <= max_m):
            continue
        f_val = _compute_f(m_val)
        if f_val > best_f:
            best_f = f_val
            best_m = m_val
    
    # Phase 2: local search around best_m
    if time.time() < deadline - 5:
        for dm in range(-50, 51):
            m_val = best_m + dm
            if time.time() > deadline:
                break
            if not (0 <= m_val <= max_m):
                continue
            f_val = _compute_f(m_val)
            if f_val > best_f:
                best_f = f_val
                best_m = m_val
    
    # Phase 3: broader local search if time permits
    if time.time() < deadline - 2:
        for dm in range(-200, 201):
            m_val = best_m + dm
            if time.time() > deadline:
                break
            if not (0 <= m_val <= max_m):
                continue
            f_val = _compute_f(m_val)
            if f_val > best_f:
                best_f = f_val
                best_m = m_val
    
    return int(best_m)
# EVOLVE-BLOCK-END
