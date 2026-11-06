"""Rewritten program: Maximum weight independent set on conflict graph with branch and bound."""
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    N = 256
    vertices = list(range(2, 257))
    n = len(vertices)
    
    # Precompute conflict graph: adj[i] is set of j where lcm(vertices[i], vertices[j]) <= 256
    adj = [set() for _ in range(n)]
    for i in range(n):
        vi = vertices[i]
        for j in range(i + 1, n):
            vj = vertices[j]
            g = gcd(vi, vj)
            if (vi // g) * vj <= 256:
                adj[i].add(j)
                adj[j].add(i)
    
    # Weights as Fractions for exact arithmetic
    weights = [Fraction(1, v) for v in vertices]
    
    # Convert adj to bitmasks for fast operations
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    # Branch and bound for maximum weight independent set
    # State: available mask (vertices that can still be added), current set mask, current sum
    best_sum = Fraction(0)
    best_mask = 0
    
    # Sort vertices by weight descending for better pruning
    order = sorted(range(n), key=lambda i: -weights[i])
    
    # Upper bound: sum of weights of all available vertices
    # We can tighten: for each available vertex, it can only be included if we don't include its neighbors
    # Simple bound: sum of all available weights
    
    # Use a recursive branch and bound
    # To avoid deep recursion, use iterative approach with a stack
    
    # Precompute weight sum for any subset? Too expensive.
    # Instead, use greedy upper bound: sort available by weight, greedily pick non-conflicting
    
    def greedy_bound(available_mask):
        """Upper bound: greedy independent set sum on available_mask"""
        total = Fraction(0)
        remaining = available_mask
        # Process in order of decreasing weight
        for i in order:
            if not (remaining & (1 << i)):
                continue
            total += weights[i]
            # Remove i and its neighbors from remaining
            remaining &= ~(adj_mask[i] | (1 << i))
        return total
    
    # Branch and bound using stack-based DFS
    # Each state: (available_mask, current_mask, current_sum)
    # We branch on the highest-weight available vertex: include it or exclude it
    
    # To limit memory and time, we'll use a priority queue approach (best-first)
    # or iterative deepening. Given n=255, this could be large.
    
    # Alternative: Use the fact that the conflict graph has structure.
    # Numbers > 128: there are 128 of them (129..256). 
    # For two numbers a, b > 128, lcm(a,b) > 128. When is lcm(a,b) <= 256?
    # lcm(a,b) = a*b/gcd(a,b) <= 256
    # Since a, b > 128, a*b > 128*128 = 16384, so gcd(a,b) > 16384/256 = 64.
    # So gcd(a,b) >= 65. But gcd(a,b) divides both a and b.
    # If a and b both > 128 and gcd(a,b) >= 65, then a and b share a large factor.
    
    # Let's just run branch and bound with a time limit.
    
    # Better approach: Use a maximum weight independent set via integer programming
    # or use the fact that we can enumerate subsets of "large" numbers and then add "small" numbers.
    
    # Key observation: If we include a number a, we cannot include any b where lcm(a,b) <= 256.
    # For small numbers (2..128), they conflict with many others.
    # For large numbers (129..256), they conflict with fewer.
    
    # Strategy: Enumerate which "small" numbers (2..128) to include, then for each choice,
    # determine which "large" numbers (129..256) can be added, and pick the best subset of large numbers.
    # But even small numbers can conflict with each other.
    
    # Let's try: separate into small (2..128) and large (129..256).
    small = [i for i in range(n) if vertices[i] <= 128]
    large = [i for i in range(n) if vertices[i] > 128]
    
    # For large numbers, check conflicts among themselves and with small.
    # Number of large: 128. Number of small: 127.
    
    # For a fixed subset of small, the large numbers that can be added are those not conflicting with any small in the subset AND forming an independent set among themselves.
    
    # This is still complex. Let me just do branch and bound on the full graph with good pruning.
    
    # Use a different strategy: local search with strong moves, starting from good initial solutions.
    
    def greedy_init(exclude_mask=0):
        """Greedy: add vertices in order of decreasing weight if they don't conflict."""
        selected = set()
        selected_mask = 0
        current_sum = Fraction(0)
        available = ((1 << n) - 1) & ~exclude_mask
        
        for i in order:
            if not (available & (1 << i)):
                continue
            # Check if i conflicts with any selected
            if adj_mask[i] & selected_mask:
                continue
            selected.add(i)
            selected_mask |= (1 << i)
            current_sum += weights[i]
            available &= ~adj_mask[i]
            available &= ~(1 << i)
        
        return selected_mask, current_sum, selected
    
    # Try multiple greedy initializations
    best_mask, best_sum, best_set = greedy_init()
    
    # Local search with 1-swap, 2-swap, and add moves
    def local_search(start_mask, start_sum, start_set, max_iters=2000):
        current_mask = start_mask
        current_sum = start_sum
        current_set = set(start_set)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Compute conflict mask: union of adj[i] for i in current_set, plus current_set itself
            conflict = 0
            for i in current_set:
                conflict |= (adj_mask[i] | (1 << i))
            
            # Phase 1: Try to add any vertex
            added = -1
            best_w = Fraction(0)
            for i in order:
                if not (conflict & (1 << i)):
                    if weights[i] > best_w:
                        best_w = weights[i]
                        added = i
            
            if added != -1:
                current_set.add(added)
                current_mask |= (1 << added)
                current_sum += weights[added]
                improved = True
                continue
            
            # Phase 2: Try 1-swap moves
            best_swap_gain = Fraction(0)
            best_i = -1
            best_j = -1
            
            for i in list(current_set):
                # Remove i
                mask_without_i = current_mask & ~(1 << i)
                # Recompute conflict for mask_without_i
                cm_i = 0
                for k in current_set:
                    if k != i:
                        cm_i |= (adj_mask[k] | (1 << k))
                
                # Find best j to add
                for j in order:
                    if not (cm_i & (1 << j)):
                        gain = weights[j] - weights[i]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_i = i
                            best_j = j
            
            if best_i != -1 and best_swap_gain > 0:
                current_set.discard(best_i)
                current_set.add(best_j)
                current_mask &= ~(1 << best_i)
                current_mask |= (1 << best_j)
                current_sum += best_swap_gain
                improved = True
                continue
            
            # Phase 3: Try remove-and-regreedy for a random vertex
            if current_set:
                # Pick the vertex with smallest weight to remove
                i_remove = min(current_set, key=lambda i: weights[i])
                trial_set = current_set - {i_remove}
                trial_mask = current_mask & ~(1 << i_remove)
                trial_sum = current_sum - weights[i_remove]
                
                # Greedy add from trial
                conflict_trial = 0
                for k in trial_set:
                    conflict_trial |= (adj_mask[k] | (1 << k))
                
                available = ((1 << n) - 1) & ~conflict_trial
                for j in order:
                    if not (available & (1 << j)):
                        continue
                    trial_set.add(j)
                    trial_mask |= (1 << j)
                    trial_sum += weights[j]
                    conflict_trial |= (adj_mask[j] | (1 << j))
                    available &= ~conflict_trial
                
                if trial_sum > current_sum:
                    current_set = trial_set
                    current_mask = trial_mask
                    current_sum = trial_sum
                    improved = True
                    continue
            
            if not improved:
                break
        
        return current_mask, current_sum, current_set
    
    best_mask, best_sum, best_set = local_search(best_mask, best_sum, best_set)
    
    # Try more initializations if time permits
    if time.time() < deadline - 15.0:
        # Try excluding different small sets
        for exclude_count in [1, 2, 3]:
            # Exclude the exclude_count smallest-weight vertices
            small_excludes = sorted(range(n), key=lambda i: weights[i])[:exclude_count]
            exclude_mask = 0
            for i in small_excludes:
                exclude_mask |= (1 << i)
            
            m, s, st = greedy_init(exclude_mask=exclude_mask)
            m, s, st = local_search(m, s, st, max_iters=1000)
            if s > best_sum:
                best_sum = s
                best_mask = m
                best_set = st
            
            if time.time() > deadline - 10.0:
                break
    
    # Convert mask back to list of values
    result = [vertices[i] for i in range(n) if (best_mask >> i) & 1]
    
    return result
