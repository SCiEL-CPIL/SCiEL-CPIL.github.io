"""Find worst interval start for distinct-multiple matching with n=12."""
import time
from math import gcd
from itertools import product

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_lcm_table(n):
    L = 1
    for i in range(2, n + 1):
        L = _lcm(L, i)
    return L

def _can_match(m, L, n):
    # Check if (m, m+L] contains distinct b_i with i | b_i
    # For each i in 1..n, candidate multiples of i in (m, m+L]
    # We need a system of distinct representatives (SDR)
    # Use bipartite matching (Hopcroft-Karp or simple DFS for small n)
    
    # Build candidate lists for each i
    # For i=1: any integer in (m, m+L]
    # For i>1: multiples of i in (m, m+L]
    
    # Collect all candidate values
    # For each i, candidates are: ceil((m+1)/i)*i, ..., floor((m+L)/i)*i
    # that are > m
    
    # Use DFS-based bipartite matching (n=12 is small)
    # Left side: indices 1..n, Right side: candidate values
    
    # Build adjacency: for each i, list of candidate values
    adj = {}
    for i in range(1, n + 1):
        # Multiples of i in (m, m+L]
        # First multiple > m: ((m // i) + 1) * i
        # Last multiple <= m+L: ((m + L) // i) * i
        first = (m // i + 1) * i
        last = ((m + L) // i) * i
        if first > last:
            return False
        adj[i] = list(range(first, last + 1, i))
    
    # Now check if there's a perfect matching using DFS (Kuhn's algorithm)
    # Since n is small (12), and candidate sets may be large, we need to be careful
    # But for feasibility checking with binary search, L is at most ~144, so candidate sets are small
    
    # Use simple DFS matching
    match_to_val = {}  # val -> i
    
    def dfs(i, visited):
        for v in adj[i]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_to_val or dfs(match_to_val[v], visited):
                match_to_val[v] = i
                return True
        return False
    
    for i in range(1, n + 1):
        visited = set()
        if not dfs(i, visited):
            return False
    return True

def _compute_f(m, n, max_L=200):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    for L in range(1, max_L + 1):
        if _can_match(m, L, n):
            return L
    return max_L  # shouldn't happen

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = 12
    LCM = _build_lcm_table(n)  # 27720
    
    # Precompute f(m) for all residue classes mod LCM
    # f(m) depends only on m mod LCM
    # But m ranges 0..1000000, and residues 0..27719
    # For each residue r, f(r) = f(r + k*LCM) for any valid k
    
    # Actually, f(m) depends on m mod LCM because the divisibility conditions
    # repeat with period LCM. Let's verify: if we shift m by LCM, the set of
    # multiples of i in (m+LCM, m+LCM+L] is just (multiples in (m, m+L]) + LCM,
    # so the structure is identical. Thus f(m) = f(m mod LCM) (with 0 meaning LCM).
    
    best_m = 0
    best_f = 0
    
    for r in range(LCM):
        if time.time() > deadline:
            break
        f_val = _compute_f(r, n)
        if f_val > best_f:
            best_f = f_val
            best_m = r
    
    # Now find the actual m in [0, 1000000] with residue best_m mod LCM
    # that gives the best f value. Since f depends only on residue, any m with
    # m mod LCM == best_m works. Pick the smallest such m.
    # But we should pick m in [0, 1000000].
    # m = best_m (if best_m <= 1000000) or best_m + k*LCM
    
    # Actually, we want to maximize f, and f only depends on residue.
    # So any m with the best residue is equally good. Pick the smallest valid m.
    
    if best_m <= 1000000:
        result = best_m
    else:
        # best_m is a residue in 0..27719, so it's always <= 1000000
        result = best_m
    
    return int(result)

construct()
