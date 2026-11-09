import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _dfs_backtrack(start_val, used_sums, max_val=256, deadline=None):
    """
    DFS with backtracking using the forward-difference representation.
    
    Key insight: if we have sequence a[0], a[1], ..., a[n-1], the consecutive
    block sums are all sums a[i]+a[i+1]+...+a[j]. These are exactly the
    differences of prefix sums. Let P[k] = a[0]+...+a[k-1] (P[0]=0).
    Then block sum a[i]..a[j] = P[j+1] - P[i].
    
    So we need all P[j+1] - P[i] for 0<=i<j+1<=n to be distinct.
    Equivalently, all P values must have distinct pairwise differences.
    
    This is a "distinct differences" constraint on the set {P[0], P[1], ..., P[n]}.
    Since a[k] = P[k+1]-P[k] > 0 (strictly increasing sequence), we need
    P to be strictly increasing.
    
    We build P incrementally: P[0]=0, P[1]=a[0], P[2]=a[0]+a[1], etc.
    At each step, P[k] must be > P[k-1], and all pairwise differences among
    {P[0],...,P[k]} must be distinct.
    
    The maximum P[n] is bounded by sum of max possible values, but more tightly:
    a[k] >= a[k-1]+1, so P grows at least quadratically.
    Actually a[k] can be any value > a[k-1], up to 256.
    So P[k] <= P[k-1] + 256.
    
    For n elements, P[n] <= n*256. But also a[k] >= k+1 (minimum strictly increasing
    starting from 1), so P[n] >= n(n+1)/2.
    
    The constraint that all pairwise differences are distinct is the key.
    For m points, there are m*(m-1)/2 differences, all must be distinct.
    The maximum difference is P[n]-P[0] = P[n].
    So we need m*(m-1)/2 <= P[n] (roughly, since differences range from 1 to P[n]).
    
    This gives us a strong upper bound: if P[n] < m*(m-1)/2, we can't have m points.
    """
    if deadline is None:
        deadline = time.time() + 45
    
    # We'll build the prefix sum array P.
    # P[0] = 0 always.
    # a[0] = P[1] - P[0] = P[1], so a[0] = P[1].
    # a[k] = P[k+1] - P[k].
    # Constraint: a[k] in [1, 256] and strictly increasing: a[k] > a[k-1].
    # So P[k+1] - P[k] > P[k] - P[k-1], i.e., differences are strictly increasing.
    # Also P[k+1] - P[k] <= 256.
    
    # Let's work with the differences d[k] = a[k] = P[k+1]-P[k].
    # d[0] < d[1] < ... < d[n-1], each in [1, 256].
    # P[k] = d[0] + d[1] + ... + d[k-1].
    # All pairwise differences of P must be distinct.
    
    # State: current list of d values, current P values, current set of all pairwise diffs.
    
    best_seq = []
    best_len = 0
    
    # Start with empty sequence
    d = []       # the a-values (differences)
    P = [0]      # prefix sums, P[0]=0
    diffs = set() # all pairwise differences P[j]-P[i] for i<j
    
    def _upper_bound(n_current, P_current_max):
        """
        Upper bound on total length given current state.
        Current length = n_current.
        Remaining elements: each new d must be > d[-1] and <= 256.
        So max remaining = 256 - d[-1] (if d is non-empty).
        But also the distinct-differences constraint limits us.
        
        Total points in P will be n_current + 1 (current) + remaining.
        Let total_points = n_current + 1 + rem.
        Need total_points*(total_points-1)/2 <= P_total_max.
        P_total_max <= P_current_max + rem * 256 (loose) 
        Actually P_total_max = P_current_max + sum of remaining d's.
        Each remaining d <= 256, and there are at most (256 - d[-1]) of them.
        But more tightly: remaining d's are d[-1]+1, d[-1]+2, ..., 256.
        Sum of remaining d's = sum_{k=d[-1]+1}^{256} k = (256*(256+1)/2 - d[-1]*(d[-1]+1)/2)
        So P_total_max = P_current_max + that sum.
        Then total_points*(total_points-1)/2 <= P_total_max gives max total_points.
        """
        if not d:
            max_rem = 256
            sum_rem = 256 * 257 // 2
        else:
            last_d = d[-1]
            if last_d >= 256:
                return n_current
            max_rem = 256 - last_d
            # Sum of (last_d+1) + (last_d+2) + ... + 256
            sum_rem = (256 * 257 // 2) - (last_d * (last_d + 1) // 2)
        
        P_total_max = P[-1] + sum_rem
        # Need m*(m-1)//2 <= P_total_max where m = total number of P points
        # Current P points = len(P) = n_current + 1
        # Additional P points = max_rem (one per new d)
        # So max m = n_current + 1 + max_rem
        # But also m*(m-1)//2 <= P_total_max
        m_max = n_current + 1 + max_rem
        # Solve m*(m-1)//2 <= P_total_max for m
        # m^2 - m - 2*P_total_max <= 0
        # m <= (1 + sqrt(1 + 8*P_total_max)) / 2
        import math
        m_bound = (1 + math.isqrt(1 + 8 * P_total_max)) // 2
        if m_bound < m_max:
            m_max = m_bound
        # m is total P points, sequence length = m - 1
        return m_max - 1
    
    def dfs(d_list, P_list, diffs_set, depth):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return
        
        n_cur = len(d_list)
        
        if n_cur > best_len:
            best_len = n_cur
            best_seq = list(d_list)
        
        # Upper bound check
        ub = _upper_bound(n_cur, P_list[-1])
        if ub <= best_len:
            return
        
        # Try next d value
        if not d_list:
            start = 1
        else:
            start = d_list[-1] + 1
        
        for new_d in range(start, 257):
            if time.time() > deadline:
                return
            
            # Compute new P value
            new_P = P_list[-1] + new_d
            
            # Check: new_P - P[i] for all i must not be in diffs_set
            valid = True
            new_diffs = []
            for i in range(len(P_list)):
                diff = new_P - P_list[i]
                if diff in diffs_set:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
            
            # Also need new_d > previous d (ensured by start)
            # And new_d <= 256 (ensured by range)
            
            # Recurse
            dfs(d_list + [new_d], P_list + [new_P], diffs_set | set(new_diffs), depth + 1)
            
            # Backtrack: no need to explicitly undo since we pass copies
            # But for efficiency, we could mutate. For now, copies are fine
            # since depth is limited.
            
            # Pruning: if we've already found a sequence of length >= ub, stop
            if best_len >= ub:
                break
    
    dfs(d, P, diffs, 0)
    
    return best_seq

def _dfs_with_heuristics(max_val=256, deadline=None):
    """
    DFS with smarter ordering: try larger d values first (they leave more
    "room" in terms of the distinct-differences constraint being easier to satisfy
    for small differences, but actually larger differences create larger gaps
    which might help avoid collisions).
    
    Actually, let's think about this differently. The constraint is that all
    pairwise differences of P must be distinct. If we make d values large,
    P grows fast, and the range of differences is large, making collisions
    less likely. But we're limited to d <= 256.
    
    Alternative: try d values in decreasing order (largest first) to quickly
    get long sequences with large P values, then the bound is tighter.
    """
    if deadline is None:
        deadline = time.time() + 45
    
    best_seq = []
    best_len = 0
    
    import math
    
    def _upper_bound(n_current, P_last, last_d):
        if last_d >= max_val:
            return n_current
        max_rem = max_val - last_d
        sum_rem = (max_val * (max_val + 1) // 2) - (last_d * (last_d + 1) // 2)
        P_total_max = P_last + sum_rem
        m_max = n_current + 1 + max_rem
        m_bound = (1 + math.isqrt(1 + 8 * P_total_max)) // 2
        if m_bound < m_max:
            m_max = m_bound
        return m_max - 1
    
    def dfs(d_list, P_list, diffs_set):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return
        
        n_cur = len(d_list)
        
        if n_cur > best_len:
            best_len = n_cur
            best_seq = list(d_list)
        
        ub = _upper_bound(n_cur, P_list[-1], d_list[-1] if d_list else 0)
        if ub <= best_len:
            return
        
        if not d_list:
            start = 1
        else:
            start = d_list[-1] + 1
        
        # Try in increasing order (standard) - we could try different orders
        for new_d in range(start, max_val + 1):
            if time.time() > deadline:
                return
            
            new_P = P_list[-1] + new_d
            
            valid = True
            new_diffs = []
            for i in range(len(P_list)):
                diff = new_P - P_list[i]
                if diff in diffs_set:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
            
            dfs(d_list + [new_d], P_list + [new_P], diffs_set | set(new_diffs))
            
            if best_len >= ub:
                break
    
    dfs([], [0], set())
    
    return best_seq

def _randomized_dfs(max_val=256, deadline=None, seed=0):
    """
    Randomized DFS: at each step, try a random subset of candidates in random order.
    This helps escape local optima that pure DFS might get stuck in.
    """
    if deadline is None:
        deadline = time.time() + 45
    
    rng = random.Random(seed)
    best_seq = []
    best_len = 0
    
    import math
    
    def _upper_bound(n_current, P_last, last_d):
        if last_d >= max_val:
            return n_current
        max_rem = max_val - last_d
        sum_rem = (max_val * (max_val + 1) // 2) - (last_d * (last_d + 1) // 2)
        P_total_max = P_last + sum_rem
        m_max = n_current + 1 + max_rem
        m_bound = (1 + math.isqrt(1 + 8 * P_total_max)) // 2
        if m_bound < m_max:
            m_max = m_bound
        return m_max - 1
    
    def dfs(d_list, P_list, diffs_set, remaining_budget):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return
        
        n_cur = len(d_list)
        
        if n_cur > best_len:
            best_len = n_cur
            best_seq = list(d_list)
        
        ub = _upper_bound(n_cur, P_list[-1], d_list[-1] if d_list else 0)
        if ub <= best_len:
            return
        
        if not d_list:
            start = 1
        else:
            start = d_list[-1] + 1
        
        candidates = list(range(start, max_val + 1))
        
        # Shuffle for randomization
        rng.shuffle(candidates)
        
        for new_d in candidates:
            if time.time() > deadline:
                return
            
            new_P = P_list[-1] + new_d
            
            valid = True
            new_diffs = []
            for i in range(len(P_list)):
                diff = new_P - P_list[i]
                if diff in diffs_set:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
            
            dfs(d_list + [new_d], P_list + [new_P], diffs_set | set(new_diffs), remaining_budget - 1)
            
            if best_len >= ub:
                break
    
    dfs([], [0], set(), 100)
    
    return best_seq

def _iterative_deepening(max_val=256, deadline=None):
    """
    Iterative deepening: try to find a sequence of length L, then L+1, etc.
    For each target length, use DFS with strong pruning.
    """
    if deadline is None:
        deadline = time.time() + 45
    
    best_seq = []
    best_len = 0
    
    import math
    
    def _can_extend(d_list, P_list, diffs_set, target_len):
        """Check if we can reach target_len from current state."""
        n_cur = len(d_list)
        if n_cur >= target_len:
            return True
        
        # Upper bound
        last_d = d_list[-1] if d_list else 0
        if last_d >= max_val:
            return n_cur >= target_len
        
        max_rem = max_val - last_d
        if n_cur + max_rem < target_len:
            return False
        
        sum_rem = (max_val * (max_val + 1) // 2) - (last_d * (last_d + 1) // 2)
        P_total_max = P_list[-1] + sum_rem
        m_needed = target_len + 1  # P points for target_len elements
        if m_needed * (m_needed - 1) // 2 > P_total_max:
            return False
        
        return True
    
    def dfs_fixed_depth(d_list, P_list, diffs_set, depth, target_len):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return False
        
        n_cur = len(d_list)
        
        if n_cur == target_len:
            if n_cur > best_len:
                best_len = n_cur
                best_seq = list(d_list)
            return True
        
        # Pruning
        if not _can_extend(d_list, P_list, diffs_set, target_len):
            return False
        
        if not d_list:
            start = 1
        else:
            start = d_list[-1] + 1
        
        for new_d in range(start, max_val + 1):
            if time.time() > deadline:
                return False
            
            new_P = P_list[-1] + new_d
            
            valid = True
            new_diffs = []
            for i in range(len(P_list)):
                diff = new_P - P_list[i]
                if diff in diffs_set:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if not valid:
                continue
            
            if dfs_fixed_depth(d_list + [new_d], P_list + [new_P], diffs_set | set(new_diffs), depth + 1, target_len):
                return True
        
        return False
    
    # Start from a reasonable lower bound
    greedy = _greedy_forward([])
    best_seq = greedy
    best_len = len(greedy)
    
    # Try to extend beyond greedy
    for target in range(best_len + 1, best_len + 20):
        if time.time() > deadline - 2:
            break
        if dfs_fixed_depth([], [0], set(), 0, target):
            best_len = target
            # Try next
        else:
            # Can't achieve this length, try next? Or stop?
            # With iterative deepening, if we can't do target, we might still
            # be able to do target+1 with different choices. But that's unlikely.
            # Let's try one more level then stop.
            if target > best_len + 2:
                break
    
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit (actually 90s, but be safe)
    
    # 1. Greedy for quick lower bound
    greedy_seq = _greedy_forward([])
    best = greedy_seq
    best_len = len(best)
    
    # 2. Standard DFS (increasing order)
    if time.time() < deadline - 5:
        dfs_result = _dfs_backtrack(0, set(), max_val=256, deadline=deadline)
        if len(dfs_result) > best_len:
            best = dfs_result
            best_len = len(best)
    
    # 3. Iterative deepening from greedy bound
    if time.time() < deadline - 5:
        id_result = _iterative_deepening(max_val=256, deadline=deadline)
        if len(id_result) > best_len:
            best = id_result
            best_len = len(best)
    
    # 4. Randomized DFS with different seeds
    if time.time() < deadline - 3:
        for seed in [1, 42, 123, 999]:
            if time.time() > deadline - 2:
                break
            rand_result = _randomized_dfs(max_val=256, deadline=deadline, seed=seed)
            if len(rand_result) > best_len:
                best = rand_result
                best_len = len(best)
    
    # Final validation
    if best and not _is_valid_full(best):
        best = _greedy_forward([])
    
    if not best:
        best = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return best

# EVOLVE-BLOCK-END
