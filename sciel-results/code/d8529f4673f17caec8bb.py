import time
import random
import numpy as np

N = 40
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]

def _init_adj():
    return [0] * N

def _add(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _edge_count(adj):
    return sum(bin(x).count('1') for x in adj) // 2

def _has_c6(adj, u, v):
    nu = adj[u]
    nv = adj[v]
    
    a_bits = nu
    while a_bits:
        a_lsb = a_bits & -a_bits
        a = a_lsb.bit_length() - 1
        a_bits ^= a_lsb
        
        if a == v:
            continue
            
        b_bits = nv
        while b_bits:
            b_lsb = b_bits & -b_bits
            b = b_lsb.bit_length() - 1
            b_bits ^= b_lsb
            
            if b == a or b == u:
                continue
            
            # x in N(a) \ {u, v, b}
            x_bits = adj[a]
            x_bits &= ~(1 << u)
            x_bits &= ~(1 << v)
            x_bits &= ~(1 << b)
            
            # y in N(b) \ {u, v, a}
            y_bits = adj[b]
            y_bits &= ~(1 << u)
            y_bits &= ~(1 << v)
            y_bits &= ~(1 << a)
            
            x_tmp = x_bits
            while x_tmp:
                x_lsb = x_tmp & -x_tmp
                x = x_lsb.bit_length() - 1
                x_tmp ^= x_lsb
                
                if adj[x] & y_bits:
                    return True
    return False

def _greedy_build(rng):
    adj = _init_adj()
    order = list(ALL_PAIRS)
    rng.shuffle(order)
    for u, v in order:
        if not (adj[u] & (1 << v)):
            if not _has_c6(adj, u, v):
                _add(adj, u, v)
    return adj

def _local_search_sa(adj, rng, time_limit):
    cur = list(adj)
    cur_count = _edge_count(cur)
    best = list(cur)
    best_count = cur_count
    start = time.time()
    temp = 1.0
    
    while time.time() - start < time_limit:
        # Try to add a random non-edge
        success = False
        for _ in range(10): # Retry a few times to find a non-edge
            u = rng.randrange(N)
            v = rng.randrange(N)
            if u == v: continue
            if u > v: u, v = v, u
            if adj[u] & (1 << v): continue
            
            if not _has_c6(cur, u, v):
                _add(cur, u, v)
                cur_count += 1
                if cur_count > best_count:
                    best = list(cur)
                    best_count = cur_count
                success = True
                break
            else:
                # If blocked, try a swap
                # Pick a random edge to remove
                e_u = rng.randrange(N)
                # Pick a random neighbor
                nb = cur[e_u]
                if nb == 0: continue
                # Get a random set bit
                # Convert to list? Or use random index?
                # For N=40, we can just iterate
                nb_list = [i for i in range(N) if (nb >> i) & 1]
                if not nb_list: continue
                e_v = rng.choice(nb_list)
                
                _remove(cur, e_u, e_v)
                cur_count -= 1
                
                if not _has_c6(cur, u, v):
                    _add(cur, u, v)
                    cur_count += 1
                    # Count is same as before swap
                    # But structure changed
                    if cur_count > best_count:
                        best = list(cur)
                        best_count = cur_count
                    success = True
                    break
                else:
                    # Revert
                    _add(cur, e_u, e_v)
                    cur_count += 1
        
        if not success:
            # Just do a random swap to explore
            e_u = rng.randrange(N)
            nb = cur[e_u]
            if nb == 0: continue
            nb_list = [i for i in range(N) if (nb >> i) & 1]
            if not nb_list: continue
            e_v = rng.choice(nb_list)
            
            # Find a non-edge
            for _ in range(10):
                n_u = rng.randrange(N)
                n_v = rng.randrange(N)
                if n_u == n_v: continue
                if n_u > n_v: n_u, n_v = n_v, n_u
                if cur[n_u] & (1 << n_v): continue
                
                _remove(cur, e_u, e_v)
                cur_count -= 1
                if not _has_c6(cur, n_u, n_v):
                    _add(cur, n_u, n_v)
                    cur_count += 1
                    # Accept with SA prob?
                    # Delta is 0
                    # Let's just accept to explore
                    break
                else:
                    _add(cur, e_u, e_v)
                    cur_count += 1
                    break

        temp = max(0.1, temp * 0.99)
        
    return best, best_count

def construct():
    deadline = 145.0 # 150s limit, 5s margin
    start = time.time()
    rng = random.Random(42)
    
    best_adj = None
    best_count = -1
    
    def update_best(adj, cnt):
        nonlocal best_adj, best_count
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj)
            
    # Phase 1: Greedy Builds
    for i in range(20):
        if time.time() > deadline - 60:
            break
        adj = _greedy_build(rng)
        cnt = _edge_count(adj)
        update_best(adj, cnt)
        
    # Phase 2: SA on best
    if best_adj:
        remaining = deadline - 10 - (time.time() - start)
        if remaining > 10:
            adj2, cnt2 = _local_search_sa(best_adj, rng, remaining)
            update_best(adj2, cnt2)
            
    # Phase 3: More Greedy + Short SA
    for i in range(10):
        if time.time() > deadline - 20:
            break
        adj = _greedy_build(rng)
        cnt = _edge_count(adj)
        update_best(adj, cnt)
        remaining = deadline - 10 - (time.time() - start)
        if remaining > 5:
            adj2, cnt2 = _local_search_sa(adj, rng, min(remaining * 0.2, 5))
            update_best(adj2, cnt2)
            
    if best_adj is None:
        # Fallback
        adj = _init_adj()
        for i in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    _add(adj, 5*i+a, 5*i+b)
        best_adj = adj
        best_count = _edge_count(adj)
        
    edges = []
    for u in range(N):
        for v in range(N):
            if u < v and (best_adj[u] & (1 << v)):
                edges.append([u, v])
                
    return edges
