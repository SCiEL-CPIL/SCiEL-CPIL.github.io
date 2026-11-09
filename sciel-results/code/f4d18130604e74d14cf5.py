import random
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s for overhead
    
    # We will try to find a valid tournament for n starting from 31 up to 82.
    # If we find one, we return it.
    # If we run out of time, we return the Paley 83 solution (known valid).
    
    def get_paley_83():
        p = 83
        squares = {x * x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                # 0 means i -> j, 1 means j -> i
                # i -> j iff (j - i) % p is a square
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}

    # Helper to check if a 4-set has a dominator
    # adj[i] is a bitmask where bit j is 1 if i -> j
    def has_dominator(a, b, c, d, adj):
        # Check a
        if (adj[a] & (1 << b)) and (adj[a] & (1 << c)) and (adj[a] & (1 << d)):
            return True
        # Check b
        if (adj[b] & (1 << a)) and (adj[b] & (1 << c)) and (adj[b] & (1 << d)):
            return True
        # Check c
        if (adj[c] & (1 << a)) and (adj[c] & (1 << b)) and (adj[c] & (1 << d)):
            return True
        # Check d
        if (adj[d] & (1 << a)) and (adj[d] & (1 << b)) and (adj[d] & (1 << c)):
            return True
        return False

    def count_bad_sets(adj, n):
        count = 0
        # Iterate all 4-combinations
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        if not has_dominator(a, b, c, d, adj):
                            count += 1
        return count

    def local_search(n, time_budget):
        # Initialize random tournament
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                else:
                    adj[j] |= (1 << i)
        
        current_cost = count_bad_sets(adj, n)
        if current_cost == 0:
            return adj
        
        best_adj = adj[:]
        best_cost = current_cost
        
        end_time = time.time() + time_budget
        no_improve_count = 0
        max_no_improve = 10000 # Restart if stuck
        
        while time.time() < end_time:
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Ensure u < v for consistent flip logic? No, just flip the bit.
            # If u -> v currently, flipping makes v -> u.
            
            # Determine current state
            u_to_v = (adj[u] & (1 << v)) != 0
            
            # Calculate delta cost
            # Only 4-sets containing u and v are affected.
            delta = 0
            # Iterate other two vertices w, x
            # To avoid duplicates, we can just iterate all pairs w, x != u, v
            # But we must be careful not to double count or miss.
            # The set is {u, v, w, x}.
            
            # Optimization: Precompute list of other vertices?
            # For n=40, n-2 = 38. Pairs = 703.
            # This is fast.
            
            others = [k for k in range(n) if k != u and k != v]
            
            for i in range(len(others)):
                w = others[i]
                for j in range(i + 1, len(others)):
                    x = others[j]
                    
                    # Check old state
                    old_good = has_dominator(u, v, w, x, adj)
                    
                    # Flip edge in adj
                    if u_to_v:
                        adj[u] &= ~(1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] &= ~(1 << u)
                    
                    # Check new state
                    new_good = has_dominator(u, v, w, x, adj)
                    
                    # Undo flip
                    if u_to_v:
                        adj[u] |= (1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] |= (1 << u)
                    
                    if old_good and not new_good:
                        delta += 1
                    elif not old_good and new_good:
                        delta -= 1
            
            if delta < 0:
                # Accept flip
                if u_to_v:
                    adj[u] &= ~(1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] &= ~(1 << u)
                
                current_cost += delta
                no_improve_count = 0
                
                if current_cost == 0:
                    return adj
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
                    
            elif delta == 0:
                # Accept with some probability to escape plateaus
                if random.random() < 0.1:
                    if u_to_v:
                        adj[u] &= ~(1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] &= ~(1 << u)
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Restart with new random init
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                        else:
                            adj[j] |= (1 << i)
                current_cost = count_bad_sets(adj, n)
                if current_cost == 0:
                    return adj
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
                no_improve_count = 0

        return best_adj if best_cost == 0 else None

    # Try n from 31 to 82
    # Allocate time budget per n
    # If we find a solution, return it immediately.
    
    # To be safe, let's try a few promising n values.
    # The lower bound is 31. Let's try 31, 32, ... up to maybe 40 first with high effort,
    # then jump to higher n if time permits.
    
    # Actually, just iterating 31 to 82 with a small time slice might be too slow if 31-39 are impossible.
    # Let's try 31 to 40 with 2s each. If fail, try 41 to 50 with 2s each.
    # If all fail, return Paley 83.
    
    for n in range(31, 83):
        if time.time() > start_time + time_limit - 10:
            break
        
        # Allocate remaining time proportionally? Or fixed slice?
        # Let's give each n a chance.
        # If n is small, search space is smaller? No, constraint is tighter.
        
        # Let's try to find *any* solution.
        # We'll try a few restarts for each n.
        
        # Time per n: let's say 1 second for small n, more for larger?
        # Actually, if 31 is impossible, we waste time.
        # But we don't know it's impossible.
        
        # Let's try 31-40 with 1.5s each.
        # 41-50 with 1.5s each.
        # ...
        
        time_slice = 1.0 if n < 50 else 0.5
        
        if time.time() + time_slice > start_time + time_limit - 5:
            break
            
        res = local_search(n, time_slice)
        if res is not None:
            # Convert adj to orientation list
            orientation = []
            for i in range(n):
                for j in range(i + 1, n):
                    # 0 if i -> j, 1 if j -> i
                    if (res[i] & (1 << j)):
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': n, 'orientation': orientation}

    # Fallback to known valid solution
    return get_paley_83()
