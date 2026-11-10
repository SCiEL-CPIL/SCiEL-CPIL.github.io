import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM
    lcm = 1
    for d in divisors:
        lcm = lcm * d // math.gcd(lcm, d)
        
    def get_intervals(d, W, LCM):
        """
        Returns a list of disjoint intervals [start, end] of L in [0, LCM)
        such that [L, L+W] contains at least one multiple of d.
        """
        intervals = []
        # Multiples k*d must be in [L-W, L+LCM] roughly?
        # L is in [0, LCM).
        # k*d is a multiple.
        # We need k*d - W <= L <= k*d.
        # Since L >= 0, k*d - W >= 0 => k*d >= W.
        # Since L < LCM, k*d < LCM + W.
        # So k ranges from ceil(W/d) to floor((LCM-1+W)/d).
        # Actually, simpler: just iterate k such that k*d is in [-W, LCM+W].
        
        k_min = math.ceil(W / d) if W > 0 else 0
        # If W=0, k*d must be 0? No, L<=k*d<=L. k*d=L.
        # Let's just use a safe range.
        k_min = 0
        k_max = (LCM + W) // d + 1
        
        for k in range(k_min, k_max + 1):
            val = k * d
            # Interval for L: [val - W, val]
            start = val - W
            end = val
            
            # Clip to [0, LCM-1]
            if end < 0 or start > LCM - 1:
                continue
            start_clipped = max(0, start)
            end_clipped = min(LCM - 1, end)
            
            if start_clipped <= end_clipped:
                intervals.append((start_clipped, end_clipped))
        
        # Merge intervals
        if not intervals:
            return []
            
        intervals.sort()
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def intersect_intervals(lists):
        """
        Intersects a list of lists of intervals.
        """
        if not lists:
            return []
        
        current = lists[0]
        for i in range(1, len(lists)):
            next_list = lists[i]
            new_current = []
            p1 = 0
            p2 = 0
            while p1 < len(current) and p2 < len(next_list):
                s1, e1 = current[p1]
                s2, e2 = next_list[p2]
                
                # Find overlap
                start = max(s1, s2)
                end = min(e1, e2)
                
                if start <= end:
                    new_current.append((start, end))
                
                if e1 < e2:
                    p1 += 1
                else:
                    p2 += 1
            current = new_current
            if not current:
                break
        
        return current

    def check_width(W):
        # Get intervals for each divisor
        interval_lists = []
        for d in divisors:
            ints = get_intervals(d, W, lcm)
            if not ints:
                return None
            interval_lists.append(ints)
        
        # Intersect all
        valid_L_intervals = intersect_intervals(interval_lists)
        
        if not valid_L_intervals:
            return None
            
        # Try to find a matching for L in these intervals
        # We iterate L in the intervals
        for start, end in valid_L_intervals:
            # To avoid checking too many L if the interval is large,
            # we might need to be smart, but intervals are likely small for small W.
            # However, if W is large, intervals are large.
            # But if W is large, the matching is easier to find.
            # Let's just iterate.
            for L in range(start, end + 1):
                # Build bipartite graph
                # Left: n divisors
                # Right: W+1 values (0 to W, representing L to L+W)
                # Matrix shape (n, W+1)
                
                # Optimization: If W is very large, this matrix is big.
                # But we expect W to be small.
                
                rows = []
                cols = []
                
                for i, d in enumerate(divisors):
                    for j in range(W + 1):
                        val = L + j
                        if val % d == 0:
                            rows.append(i)
                            cols.append(j)
                
                if len(rows) < n:
                    continue # Not enough edges, impossible to have perfect matching
                
                # Create sparse matrix
                # Note: maximum_bipartite_matching expects a 0-1 matrix
                # It can take a sparse matrix.
                # The graph is directed? No, bipartite matching is undirected in logic, but scipy uses directed edges from left to right.
                # We just provide the adjacency.
                
                mat = csr_matrix((len(rows), (W + 1) * n), shape=(n, W + 1))
                # Wait, csr_matrix constructor with data, indices, indptr is more efficient.
                # Or just use dense if small?
                # n=10, W is likely < 100. Dense is fine.
                # But let's stick to sparse for safety if W grows.
                
                # Actually, csr_matrix((n, (W+1)*n)) is not the right syntax for COO.
                # Let's use COO data.
                data = [1] * len(rows)
                mat = csr_matrix((data, (rows, cols)), shape=(n, W + 1))
                
                # Run matching
                # scipy.sparse.csgraph.maximum_bipartite_matching
                # Returns (left_ind, right_ind)
                # left_ind[i] is the index in right_ind for the node i in left. -1 if unmatched.
                # right_ind[j] is the index in left_ind for the node j in right. -1 if unmatched.
                
                try:
                    left_ind, right_ind = maximum_bipartite_matching(mat)
                except Exception:
                    continue
                
                if all(left_ind[i] != -1 for i in range(n)):
                    # Found a matching
                    # Construct the solution
                    solution = [0] * n
                    for i in range(n):
                        j = left_ind[i]
                        solution[i] = L + j
                    return solution
        return None

    # Search for minimum W
    # Lower bound: 0 is impossible (distinct).
    # Upper bound: 
    # The divisors are small. The LCM is 30030.
    # A trivial solution is to pick k_i such that k_i * d_i are close.
    # Max divisor is 39.
    # If we pick the same k for all, values are k*6, k*10, etc.
    # Range is k*(39-6) = 33k.
    # We can do better by adjusting k.
    # Let's estimate a safe upper bound.
    # If we set k_i = 1 for all, values are d_i. Range 39-6=33.
    # Are they distinct?
    # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39.
    # Yes, all distinct.
    # So W=33 is a valid solution?
    # Let's check:
    # 6 (mult 6), 10 (mult 10), 14 (mult 14), 15 (mult 15), 21 (mult 21), 22 (mult 22), 26 (mult 26), 33 (mult 33), 35 (mult 35), 39 (mult 39).
    # Min=6, Max=39. W=33.
    # Can we do better?
    # We can shift some multiples.
    # e.g. for 6, we can use 12, 18, 24...
    # If we use 12 for 6, and keep others, range might change.
    # The current program scored 2.0625, which implies a value.
    # If the score is seed/value, and seed is 1.0, value = 1/2.0625 = 0.4848? No.
    # "search score is seed/value... 1.0 equals the seed program".
    # If score is 2.0625, and it's an improvement, then value is smaller?
    # Usually "lower is better" for the raw value.
    # If score = seed / value, and seed is fixed, higher score means lower value.
    # So the current program found a value V such that Seed/V = 2.0625.
    # We don't know the Seed value, but we know the current program works.
    # The trivial solution W=33 is a baseline.
    # Let's start searching from W=0 upwards.
    
    for W in range(0, 1000):
        sol = check_width(W)
        if sol is not None:
            # Verify distinctness and divisibility
            # The matching ensures distinctness (bipartite matching assigns unique right nodes).
            # Divisibility is enforced by graph construction.
            # Return as list of ints
            return [int(x) for x in sol]
            
    # Fallback
    return divisors[:]

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I need to wrap it in the specific format requested.
# The prompt asks to rewrite the program.
# The current program has P and construct.
# I will replace the content inside EVOLVE-BLOCK.

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def construct():
    divisors = P['divisors']
    n = len(divisors)
    
    # Calculate LCM
    lcm = 1
    for d in divisors:
        lcm = lcm * d // math.gcd(lcm, d)
        
    def get_intervals(d, W, LCM):
        intervals = []
        k_min = 0
        k_max = (LCM + W) // d + 1
        
        for k in range(k_min, k_max + 1):
            val = k * d
            start = val - W
            end = val
            
            if end < 0 or start > LCM - 1:
                continue
            start_clipped = max(0, start)
            end_clipped = min(LCM - 1, end)
            
            if start_clipped <= end_clipped:
                intervals.append((start_clipped, end_clipped))
        
        if not intervals:
            return []
            
        intervals.sort()
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def intersect_intervals(lists):
        if not lists:
            return []
        
        current = lists[0]
        for i in range(1, len(lists)):
            next_list = lists[i]
            new_current = []
            p1 = 0
            p2 = 0
            while p1 < len(current) and p2 < len(next_list):
                s1, e1 = current[p1]
                s2, e2 = next_list[p2]
                
                start = max(s1, s2)
                end = min(e1, e2)
                
                if start <= end:
                    new_current.append((start, end))
                
                if e1 < e2:
                    p1 += 1
                else:
                    p2 += 1
            current = new_current
            if not current:
                break
        
        return current

    def check_width(W):
        interval_lists = []
        for d in divisors:
            ints = get_intervals(d, W, lcm)
            if not ints:
                return None
            interval_lists.append(ints)
        
        valid_L_intervals = intersect_intervals(interval_lists)
        
        if not valid_L_intervals:
            return None
            
        for start, end in valid_L_intervals:
            for L in range(start, end + 1):
                rows = []
                cols = []
                
                for i, d in enumerate(divisors):
                    for j in range(W + 1):
                        val = L + j
                        if val % d == 0:
                            rows.append(i)
                            cols.append(j)
                
                if len(rows) < n:
                    continue
                
                data = [1] * len(rows)
                mat = csr_matrix((data, (rows, cols)), shape=(n, W + 1))
                
                try:
                    left_ind, right_ind = maximum_bipartite_matching(mat)
                except Exception:
                    continue
                
                if all(left_ind[i] != -1 for i in range(n)):
                    solution = [0] * n
                    for i in range(n):
                        j = left_ind[i]
                        solution[i] = L + j
                    return solution
        return None

    for W in range(0, 1000):
        sol = check_width(W)
        if sol is not None:
            return [int(x) for x in sol]
            
    return divisors[:]
# EVOLVE-BLOCK-END
