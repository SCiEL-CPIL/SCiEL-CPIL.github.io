import time
import random

def construct():
    deadline = time.time() + 85
    n = 256
    
    # Precompute conflict mask for all pairs
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
                
    # Precompute non-conflict (compatibility) mask
    adj = [0] * (n + 1)
    all_bits = (1 << (n + 1)) - 2
    for i in range(1, n + 1):
        adj[i] = all_bits & ~conflict_mask[i]
        
    def count_bits(mask):
        return bin(mask).count('1')
        
    def is_compatible(x, current_mask):
        return (conflict_mask[x] & current_mask) == 0
        
    conflict_counts = [0] * (n + 1)
    for i in range(1, n + 1):
        conflict_counts[i] = count_bits(conflict_mask[i])
    
    order_by_conflict = list(range(1, n + 1))
    order_by_conflict.sort(key=lambda x: conflict_counts[x])
    
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy
    current_mask = 0
    for x in order_by_conflict:
        if is_compatible(x, current_mask):
            current_mask |= (1 << x)
    sz = count_bits(current_mask)
    if sz > best_size:
        best_size = sz
        best_mask = current_mask
        
    # Strategy 2: Greedy from primes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    initial_mask = 0
    for i in range(1, n + 1):
        if is_prime[i] or i == 1:
            initial_mask |= (1 << i)
            
    current_mask = initial_mask
    for x in order_by_conflict:
        if not (current_mask >> x) & 1:
            if is_compatible(x, current_mask):
                current_mask |= (1 << x)
    sz = count_bits(current_mask)
    if sz > best_size:
        best_size = sz
        best_mask = current_mask

    # Randomized restarts
    rng = random.Random(42)
    remaining_time = deadline - time.time()
    num_restarts = max(1, int(remaining_time / 0.3))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        order = list(range(1, n + 1))
        rng.shuffle(order)
        order.sort(key=lambda x: conflict_counts[x] + rng.random() * 5)
        
        current_mask = 0
        for x in order:
            if is_compatible(x, current_mask):
                current_mask |= (1 << x)
        sz = count_bits(current_mask)
        if sz > best_size:
            best_size = sz
            best_mask = current_mask

    # Local search
    if time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 18:
            improved = False
            for x in range(1, n + 1):
                if not (best_mask >> x) & 1:
                    if is_compatible(x, best_mask):
                        best_mask |= (1 << x)
                        best_size += 1
                        improved = True

    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 12:
            improved = False
            for x in range(1, n + 1):
                if (best_mask >> x) & 1:
                    continue
                conflicts_with_x = conflict_mask[x] & best_mask
                if conflicts_with_x == 0:
                    best_mask |= (1 << x)
                    best_size += 1
                    improved = True
                    break
                c = conflicts_with_x
                conflicting = []
                while c:
                    lb = c & (-c)
                    y = lb.bit_length() - 1
                    conflicting.append(y)
                    c ^= lb
                
                if len(conflicting) == 1:
                    y = conflicting[0]
                    new_mask = (best_mask & ~(1 << y)) | (1 << x)
                    if conflict_counts[x] < conflict_counts[y]:
                        best_mask = new_mask
                        improved = True
                        break

    # Maximum Clique (Tomita's Algorithm)
    best_size_exact = best_size
    best_mask_exact = best_mask
    
    def solve_max_clique(P, X, current_mask, size):
        nonlocal best_size_exact, best_mask_exact
        
        if time.time() > deadline - 1:
            return
            
        if P == 0:
            if size > best_size_exact:
                best_size_exact = size
                best_mask_exact = current_mask
            return
            
        verts = []
        p = P
        while p:
            lb = p & (-p)
            v = lb.bit_length() - 1
            verts.append(v)
            p ^= lb
            
        colors = {}
        for v in verts:
            neigh = adj[v] & P
            used_colors = set()
            nn = neigh
            while nn:
                lb = nn & (-nn)
                u = lb.bit_length() - 1
                if u in colors:
                    used_colors.add(colors[u])
                nn ^= lb
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            
        verts_sorted = sorted(verts, key=lambda v: colors[v], reverse=True)
        
        for v in verts_sorted:
            c = colors[v]
            if size + c < best_size_exact:
                break
                
            # Exclude v from new_P to avoid self-loop issues
            new_P = (P & adj[v]) & ~(1 << v)
            new_X = X & adj[v]
            new_mask = current_mask | (1 << v)
            
            P &= ~(1 << v)
            X |= (1 << v)
            
            solve_max_clique(new_P, new_X, new_mask, size + 1)
            
    if time.time() < deadline - 5:
        initial_P = (1 << (n + 1)) - 2
        initial_X = 0
        solve_max_clique(initial_P, initial_X, 0, 0)
        
    if best_size_exact > best_size:
        best_size = best_size_exact
        best_mask = best_mask_exact

    if time.time() < deadline - 0.5:
        for x in range(1, n + 1):
            if not (best_mask >> x) & 1:
                if is_compatible(x, best_mask):
                    best_mask |= (1 << x)
                    best_size += 1

    best = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    return best
