import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Helper to get edges from adj
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def is_valid_full(adj):
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    if (cij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def can_add(adj, u, v):
        # Check if adding (u,v) creates a violation
        # Violation in triple {u, x, y} if v in N(x) & N(y) and |N(u) & N(x) & N(y)| >= 2
        # Violation in triple {v, x, y} if u in N(x) & N(y) and |N(v) & N(x) & N(y)| >= 2
        
        nu = adj[u]
        nv = adj[v]
        
        # Check triples containing u
        # {u, x, y}. Common neighbors: N(u) & N(x) & N(y).
        # New common neighbor v is added if v in N(x) and v in N(y).
        # So if v in N(x) & N(y), then new count = old count + 1.
        # Old count = (N(u) & N(x) & N(y)).bit_count()
        # We need new count < 3 => old count + 1 < 3 => old count <= 1.
        # So if old count >= 2 and v in N(x) & N(y), it's invalid.
        
        for x in range(n):
            if x == u or x == v: continue
            nx = adj[x]
            if not (nv & (1 << x)): # v not connected to x, so v not in N(x)
                continue
            for y in range(x+1, n):
                if y == u or y == v: continue
                ny = adj[y]
                if not (nv & (1 << y)): # v not connected to y
                    continue
                # v is in N(x) and N(y)
                # Check old common neighbors of u, x, y
                old_cn = nu & nx & ny
                if old_cn.bit_count() >= 2:
                    return False
        
        # Check triples containing v
        for x in range(n):
            if x == u or x == v: continue
            nx = adj[x]
            if not (nu & (1 << x)): # u not connected to x
                continue
            for y in range(x+1, n):
                if y == u or y == v: continue
                ny = adj[y]
                if not (nu & (1 << y)): # u not connected to y
                    continue
                # u is in N(x) and N(y)
                old_cn = nv & nx & ny
                if old_cn.bit_count() >= 2:
                    return False
            
        return True

    def is_valid_swap(adj, a, b, c, d):
        # Remove (a,b), Add (c,d)
        # Check validity of resulting graph
        # Only triples involving a,b,c,d need checking
        # Create a temporary adj? No, just calculate on the fly or use a fast full check on affected parts.
        # Given n=20, a full check is 1140 triples * ~10 ops = 11400 ops.
        # This is fast enough to just do a full check if we modify a copy?
        # Modifying a copy of 20 ints is cheap.
        
        adj_temp = adj[:]
        adj_temp[a] &= ~(1 << b)
        adj_temp[b] &= ~(1 << a)
        adj_temp[c] |= (1 << d)
        adj_temp[d] |= (1 << c)
        
        return is_valid_full(adj_temp)

    def make_valid(adj):
        # Remove edges to fix violations
        while True:
            if time.time() > deadline:
                break
            # Find violation
            viol = None
            for i in range(n):
                ai = adj[i]
                for j in range(i+1, n):
                    cij = ai & adj[j]
                    if cij.bit_count() < 3: continue
                    for k in range(j+1, n):
                        cijk = cij & adj[k]
                        if cijk.bit_count() >= 3:
                            viol = (i, j, k, cijk)
                            break
                    if viol: break
                if viol: break
            
            if not viol:
                break
            
            i, j, k, common = viol
            # Remove an edge incident to one of i,j,k and a common neighbor
            # Pick a common neighbor x
            x = (common & -common).bit_length() - 1
            # Remove edge (i, x)
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
            
        return adj

    best_adj = None
    best_edges = 0
    
    def update_best(adj):
        nonlocal best_adj, best_edges
        c = count_edges(adj)
        if c > best_edges:
            best_edges = c
            best_adj = adj[:]

    # Strategy: Multiple random starts
    for trial in range(100): # Use time to decide how many
        if time.time() > deadline - 10:
            break
        
        random.seed(trial * 123 + 456)
        
        # Init: Random dense graph
        p = 0.6 + 0.3 * (trial % 5) / 5.0
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        adj = make_valid(adj)
        
        # Local Search
        stuck_count = 0
        while time.time() < deadline - 5:
            added = False
            # Try to add a random edge
            # Shuffling all pairs every time is O(n^2 log n). 
            # Better: keep a list of missing edges? Or just pick random pair.
            # n=20, 190 pairs. Picking random is fine.
            for _ in range(50): # Try 50 random adds
                u, v = random.choice(all_pairs)
                if adj[u] & (1 << v):
                    continue
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    update_best(adj)
                    break
            
            if added:
                stuck_count = 0
                continue
            
            # If no add, try swap
            # Pick random existing edge to remove, random missing to add
            existing = [(i,j) for i,j in all_pairs if adj[i] & (1 << j)]
            missing = [(i,j) for i,j in all_pairs if not (adj[i] & (1 << j))]
            
            if not missing:
                break # Complete graph (impossible if invalid, but just in case)
            
            swapped = False
            for _ in range(20): # Try 20 random swaps
                if not existing:
                    break
                a, b = random.choice(existing)
                c, d = random.choice(missing)
                if (a,b) == (c,d): continue
                
                if is_valid_swap(adj, a, b, c, d):
                    # Perform swap
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    swapped = True
                    break
            
            if not swapped:
                # Stuck even with swaps?
                stuck_count += 1
                if stuck_count > 100:
                    break # Restart this trial
            else:
                stuck_count = 0
        
        update_best(adj)

    if best_adj is None:
        return []
        
    return adj_to_edges(best_adj)
