import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def combos_gosper(n, k):
        if k == 0:
            yield 0
            return
        x = (1 << k) - 1
        limit = 1 << n
        while x < limit:
            yield x
            u = x & -x
            v = u + x
            x = v + (((v ^ x) // u) >> 2)

    def is_valid(n, adj):
        """
        adj: list of int bitmasks.
        """
        for k in range(6, n + 1):
            for mask in combos_gosper(n, k):
                # Get list of vertices in mask?
                # We need to check degrees.
                # We can iterate through set bits.
                # Or, we can compute degrees for all vertices and check if those in mask are equal.
                # Faster:
                # d = [0]*n
                # for u in range(n):
                #   if mask & (1<<u):
                #     d[u] = (adj[u] & mask).bit_count()
                #     if u != first: check equality
                # 
                # To avoid list allocation, we can just track the first degree.
                
                # Find first set bit
                # u0 = (mask & -mask).bit_length() - 1
                u0_bit = mask & -mask
                u0 = u0_bit.bit_length() - 1
                d0 = (adj[u0] & mask).bit_count()
                
                # Check others
                m = mask ^ u0_bit
                while m:
                    u_bit = m & -m
                    u = u_bit.bit_length() - 1
                    if (adj[u] & mask).bit_count() != d0:
                        break
                    m ^= u_bit
                else:
                    # All degrees equal
                    return False
        
        return True

    def find_violation(n, adj):
        for k in range(6, n + 1):
            for mask in combos_gosper(n, k):
                u0_bit = mask & -mask
                u0 = u0_bit.bit_length() - 1
                d0 = (adj[u0] & mask).bit_count()
                
                m = mask ^ u0_bit
                is_reg = True
                while m:
                    u_bit = m & -m
                    u = u_bit.bit_length() - 1
                    if (adj[u] & mask).bit_count() != d0:
                        is_reg = False
                        break
                    m ^= u_bit
                
                if is_reg:
                    # Convert mask to list of vertices for repair
                    combo = []
                    mm = mask
                    while mm:
                        u_bit = mm & -mm
                        u = u_bit.bit_length() - 1
                        combo.append(u)
                        mm ^= u_bit
                    return combo
        return None

    def repair(n, adj, combo):
        u, v = random.sample(combo, 2)
        bit = 1 << v
        if adj[u] & bit:
            adj[u] ^= bit
            adj[v] ^= (1 << u)
        else:
            adj[u] |= bit
            adj[v] |= (1 << u)
        return adj

    def search_for_n(n, time_limit):
        end_time = time.time() + time_limit
        while time.time() < end_time:
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            for _ in range(5000): # Max steps
                if time.time() > end_time:
                    break
                
                combo = find_violation(n, adj)
                if combo is None:
                    return adj
                
                adj = repair(n, adj, combo)
        
        return None

    for n_try, t_limit in [(22, 50), (21, 50), (20, 40)]:
        if time.time() > deadline - 5:
            break
        res = search_for_n(n_try, t_limit)
        if res is not None:
            edges = []
            for i in range(n_try):
                for j in range(i + 1, n_try):
                    if res[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n_try, 'edges': edges}
    
    return {'n': 5, 'edges': []}
