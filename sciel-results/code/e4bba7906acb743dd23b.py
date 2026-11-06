"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit

    best_set = []
    best_len = 0

    # Helper to check if x is a mean of a subset of size k
    # dp[k] is a bitmask of achievable sums with exactly k elements
    def is_mean_of_subset(dp, x, k):
        target = k * x
        if target < 0 or target > 24576:
            return False
        return (dp[k] >> target) & 1

    def try_greedy_v2(order):
        selected = []
        n = 0
        dp_all = [1] # dp_all[0] = 1
        dp_without = {}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= 48:
                break
            
            valid = True
            
            # Check 1
            for k in range(2, n + 1):
                target = k * x
                if target > 24576:
                    break
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2
            for a in selected:
                dp_w = dp_without[a]
                for k in range(1, n):
                    target = (k + 1) * a - x
                    if target < 0 or target > 24576:
                        continue
                    if (dp_w[k] >> target) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Valid, add x
            selected.append(x)
            
            # Save old state for x's exclusion set
            old_dp_all = dp_all
            
            # Update dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = old_dp_all[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (old_dp_all[k-1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without for existing elements
            new_dp_without = {}
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k-1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For x, the set of others is the old selected set.
            # So dp_without[x] is old_dp_all.
            # We need to pad old_dp_all to size n+2 (new n is old_n + 1)
            # old_dp_all had size n+1 (indices 0..n).
            # new_dp_without[x] needs size (n+1)+1 = n+2?
            # The list size should be consistent with max k.
            # Max k for dp_without[x] is n (since there are n other elements).
            # So we need indices 0..n.
            # old_dp_all has indices 0..n.
            # So we can just use a copy.
            new_dp_without[x] = old_dp_all + [0] * max(0, (n + 1) - len(old_dp_all))
            
            dp_without = new_dp_without
            n += 1
            
        return selected

    # Conflict graph approach: precompute 3-term AP conflicts
    # Two elements a, b conflict if there exists a third element c in [1,512]
    # such that {a,b,c} forms a 3-term AP (i.e., one is the mean of the other two)
    # This is a necessary condition for the full constraint, so avoiding these
    # gives a larger independent set
    
    def build_conflict_graph(candidates):
        """Build adjacency list for conflict graph based on 3-term APs."""
        cand_set = set(candidates)
        n = len(candidates)
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(candidates)}
        adj = [set() for _ in range(n)]
        
        # For each pair (i, j), check if they form a 3-term AP with some k
        # a, b, c form AP if 2*middle = a + c
        # So for pair (i, j) with values vi, vj:
        #   If vi < vj: check if 2*vj - vi is in candidates (vi is middle of vi, vj, 2*vj-vi? No)
        #   Actually: three values x, y, z form AP if x+z = 2y
        #   For pair (vi, vj), they could be:
        #     - Endpoints: need middle = (vi+vj)/2 to be in candidates
        #     - One is endpoint, one is middle: need the other endpoint
        #       If vi is middle: need 2*vi - vj in candidates
        #       If vj is middle: need 2*vj - vi in candidates
        
        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = candidates[i], candidates[j]
                # Check if vi and vj are endpoints: middle = (vi+vj)/2
                s = vi + vj
                if s % 2 == 0:
                    mid = s // 2
                    if mid in cand_set and mid != vi and mid != vj:
                        adj[i].add(j)
                        adj[j].add(i)
                # Check if vi is middle: other endpoint = 2*vi - vj
                other = 2 * vi - vj
                if 1 <= other <= 512 and other in cand_set and other != vi and other != vj:
                    adj[i].add(j)
                    adj[j].add(i)
                # Check if vj is middle: other endpoint = 2*vj - vi
                other = 2 * vj - vi
                if 1 <= other <= 512 and other in cand_set and other != vi and other != vj:
                    adj[i].add(j)
                    adj[j].add(i)
        
        return adj

    def greedy_mis(adj, n, order):
        """Greedy maximum independent set using vertex ordering."""
        selected = set()
        available = set(range(n))
        
        for v in order:
            if v not in available:
                continue
            # Check if v is adjacent to any selected vertex
            conflict = False
            for u in selected:
                if u in adj[v]:
                    conflict = True
                    break
            if not conflict:
                selected.add(v)
                available.discard(v)
                # Remove neighbors from available
                for u in adj[v]:
                    available.discard(u)
        
        return selected

    def local_search_mis(adj, n, initial_set, max_iters=1000):
        """Local search to improve MIS: try swapping vertices to increase size."""
        selected = set(initial_set)
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try to add any vertex not in selected that has no conflicts
            for v in range(n):
                if v in selected:
                    continue
                conflict = False
                for u in selected:
                    if u in adj[v]:
                        conflict = True
                        break
                if not conflict:
                    selected.add(v)
                    improved = True
                    break
            
            # Try swapping: remove one selected vertex, add one non-selected
            if not improved:
                for v in list(selected):
                    for w in range(n):
                        if w in selected:
                            continue
                        # Check if w conflicts with selected - {v}
                        conflict = False
                        for u in selected:
                            if u == v:
                                continue
                            if u in adj[w]:
                                conflict = True
                                break
                        if not conflict:
                            selected.discard(v)
                            selected.add(w)
                            improved = True
                            break
                    if improved:
                        break
        
        return selected

    def validate_set(s):
        """Full bitset verification that no element is mean of subset of >=2 others."""
        n = len(s)
        if n < 3:
            return True
        
        # Build dp for full set
        dp_all = [1]
        for x in s:
            new_dp = [0] * (len(dp_all) + 1)
            new_dp[0] = 1
            for k in range(1, len(dp_all) + 1):
                val = dp_all[k-1] if k-1 < len(dp_all) else 0
                if k-1 < len(dp_all):
                    val |= (dp_all[k-1] << x)
                new_dp[k] = val
            dp_all = new_dp
        
        # For each element a, check if a is mean of any subset of size >= 2 from others
        for i, a in enumerate(s):
            others = s[:i] + s[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            # Build dp for others
            dp_others = [1]
            for x in others:
                new_dp = [0] * (len(dp_others) + 1)
                new_dp[0] = 1
                for k in range(1, len(dp_others) + 1):
                    val = dp_others[k-1] if k-1 < len(dp_others) else 0
                    if k-1 < len(dp_others):
                        val |= (dp_others[k-1] << x)
                    new_dp[k] = val
                dp_others = new_dp
            
            # Check all subset sizes k >= 2
            for k in range(2, m + 1):
                target = k * a
                if target > 24576:
                    break
                if (dp_others[k] >> target) & 1:
                    return False
        
        return True

    # Strategy 1: Conflict graph + MIS
    candidates = list(range(1, 513))
    
    # Try multiple orderings for greedy MIS
    mis_orders = []
    
    # Natural order
    mis_orders.append(list(range(512)))
    
    # Reverse
    mis_orders.append(list(range(511, -1, -1)))
    
    # Random shuffles
    for seed in range(10):
        random.seed(seed)
        order = list(range(512))
        random.shuffle(order)
        mis_orders.append(order)
    
    # Degree-based ordering (greedy coloring style)
    # We'll compute degrees after building the graph
    adj = build_conflict_graph(candidates)
    degrees = [len(adj[i]) for i in range(512)]
    
    # Sort by degree ascending (pick low-degree vertices first)
    mis_orders.append(sorted(range(512), key=lambda i: degrees[i]))
    # Sort by degree descending
    mis_orders.append(sorted(range(512), key=lambda i: -degrees[i]))
    
    best_mis_set = set()
    best_mis_len = 0
    
    for order in mis_orders:
        if time.time() - start_time > deadline:
            break
        
        mis = greedy_mis(adj, 512, order)
        # Local search to improve
        mis = local_search_mis(adj, 512, mis, max_iters=200)
        
        if len(mis) > best_mis_len:
            best_mis_len = len(mis)
            best_mis_set = mis
    
    # Convert MIS to sorted list and validate
    if best_mis_len > 0:
        mis_list = sorted([candidates[i] for i in best_mis_set])
        if validate_set(mis_list):
            if len(mis_list) > best_len:
                best_set = mis_list
                best_len = len(mis_list)
    
    # Strategy 2: Original greedy with various orders
    orders = []
    
    # Spread
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(lo)
        if lo != hi:
            spread.append(hi)
        lo += 1
        hi -= 1
    orders.append(spread)
    
    # Random shuffles
    for i in range(20):
        random.seed(i)
        cand = list(range(1, 513))
        random.shuffle(cand)
        orders.append(cand)
        
    # Ascending/Descending
    orders.append(list(range(1, 513)))
    orders.append(list(range(512, 0, -1)))
    
    for order in orders:
        if time.time() - start_time > deadline:
            break
        res = try_greedy_v2(order)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break
                
    return best_set

# EVOLVE-BLOCK-END
