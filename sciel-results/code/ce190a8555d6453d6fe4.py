import time
import math

P = {'n': 12, 'maximum_start': 1000000}

def _has_matching(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i.
    m is the start. The numbers are m+1, ..., m+L.
    """
    # Candidates for each i
    # We represent the right side as 0..L-1 corresponding to m+1..m+L
    # adj[i] is a bitmask of which offsets j (0..L-1) are valid for index i
    # i ranges 1..n. Let's use 0..n-1 for indices in code.
    
    adj = [0] * n
    
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, L + 1):
            val = m + j
            if val % i == 0:
                mask |= (1 << (j - 1))
        adj[i - 1] = mask
        
        # Quick fail: if no candidates for i
        if mask == 0:
            return False
            
    # Check Hall's condition is hard to do quickly without iterating subsets.
    # But N=12 is small. Let's use DFS with bitmask of used right-nodes.
    
    # Order indices by popcount of adj mask (most constrained first)
    order = sorted(range(n), key=lambda i: bin(adj[i]).count('1'))
    
    # If any has 0, already returned False.
    
    used_mask = 0
    # Recursive DFS
    # idx in order
    def dfs(k):
        if k == n:
            return True
        i = order[k]
        mask = adj[i]
        # Iterate through set bits in mask that are not in used_mask
        available = mask & ~used_mask
        while available:
            # Pick lowest set bit
            lsb = available & (-available)
            available ^= lsb
            if dfs(k + 1) is False: # Wait, need to update used_mask
                pass
            # This structure is hard in Python without nonlocal or class.
            # Let's use a standard recursive function with a mutable state or return properly.
            
    # Let's rewrite DFS properly
    def solve(k, used):
        if k == n:
            return True
        i = order[k]
        mask = adj[i] & ~used
        while mask:
            lsb = mask & (-mask)
            mask ^= lsb
            if solve(k + 1, used | lsb):
                return True
        return False

    return solve(0, 0)

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m)."""
    for L in range(1, max_L + 1):
        if _has_matching(m, L, n):
            return L
    return max_L

def construct():
    deadline = time.time() + 85
    n = 12
    period = 27720 # lcm(1..12)
    
    best_r = 0
    best_f = 0
    
    # We only need to check r in [0, period-1]
    # Because f(12, m) depends only on m % period
    
    for r in range(period):
        if time.time() > deadline - 5:
            break
        f_val = _compute_f(r, n)
        if f_val > best_f:
            best_f = f_val
            best_r = r
            
    # Return the smallest m in [0, 1000000] with m % period == best_r
    # Since best_r < period <= 1000000, best_r is a valid m.
    return best_r
