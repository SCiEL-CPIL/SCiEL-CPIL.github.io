import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 145.0  # 5s buffer
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # adj[i] is bitset of neighbors of i
    adj = [0] * n
    
    def can_add(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        C6: u-v-x-y-z-w-u
        x in N(v), w in N(u)
        y in N(x), z in N(w)
        y~z
        """
        Nv = adj[v]
        Nu = adj[u]
        
        # x in N(v) \ {u}
        x_mask = Nv & ~(1 << u)
        while x_mask:
            lsb_x = x_mask & -x_mask
            x = lsb_x.bit_length() - 1
            x_mask ^= lsb_x
            
            # w in N(u) \ {v, x}
            w_mask = Nu & ~(1 << v) & ~(1 << x)
            while w_mask:
                lsb_w = w_mask & -w_mask
                w = lsb_w.bit_length() - 1
                w_mask ^= lsb_w
                
                # y in N(x) \ {u, v, w}
                Sy = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                if not Sy:
                    continue
                
                # z in N(w) \ {u, v, x}
                Sz = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                if not Sz:
                    continue
                
                # Check if any y in Sy is connected to any z in Sz
                y_mask = Sy
                while y_mask:
                    lsb_y = y_mask & -y_mask
                    y = lsb_y.bit_length() - 1
                    y_mask ^= lsb_y
                    if adj[y] & Sz:
                        return False
        return True

    def get_edges():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges

    random.seed(42)
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = 0
    best_adj = [0] * n
    
    # Phase 1: Random Greedy
    # We want to find a good starting point quickly.
    # 200 trials, each trial takes ~50ms? 200 * 50ms = 10s.
    # Let's run until time_left < 10s.
    
    trial = 0
    while time_left() > 10.0:
        adj_try = [0] * n
        ec = 0
        
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time_left() < 10.0:
                break
            if can_add(u, v):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                ec += 1
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]
            
        trial += 1
        # Occasionally check if we can break early if we hit 105?
        # But we don't know if 105 is the max.
        # 105 is the known best. If we hit 105, we can stop greedy and go to local search.
        if best_edges >= 105:
            break

    # Phase 2: Local Search
    # Start from best_adj
    adj = best_adj[:]
    current_edges = best_edges
    
    no_improve = 0
    last_improve_time = time.time()
    
    # We will try to add edges. If stuck, remove edges.
    # To improve, we can also try "swaps" but add/remove is simpler.
    
    while time_left() > 2.0:
        # Try to add a random non-edge
        added = False
        
        # Try 100 random non-edges
        for _ in range(100):
            if time_left() < 2.0:
                break
            
            u, v = random.choice(all_pairs)
            
            # Check if edge exists
            if adj[u] & (1 << v):
                continue
            
            if can_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                added = True
                no_improve = 0
                last_improve_time = time.time()
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                break
        
        if not added:
            no_improve += 1
            
            # Escape mechanism
            # If stuck for a while, remove a random edge
            # Probability of removing increases with no_improve
            if no_improve > 10:
                # Remove a random edge
                # Pick random u, then random neighbor v
                u = random.randint(0, n-1)
                mask = adj[u]
                if mask:
                    # Pick a random bit
                    # Convert to list of indices? No, just pick random bit.
                    # To pick a random bit from mask:
                    # 1. count bits? No, just pick random index and check?
                    # Better: pick random k in [0, popcount(mask)-1]?
                    # Or just pick random v and check if in mask?
                    # Since n=40, picking random v and checking is fast.
                    v = random.randint(0, n-1)
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        current_edges -= 1
                        no_improve = 0
                    else:
                        # Try a few more
                        for _ in range(10):
                            v = random.randint(0, n-1)
                            if adj[u] & (1 << v):
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                current_edges -= 1
                                no_improve = 0
                                break
                        else:
                            # If u has no neighbors, pick another u
                            pass
                else:
                    # u has no neighbors, pick another u
                    for i in range(n):
                        if adj[i]:
                            u = i
                            v = random.randint(0, n-1)
                            # Find a neighbor
                            mask = adj[u]
                            # Pick random bit from mask
                            # Simple method:
                            bits = []
                            m = mask
                            while m:
                                lsb = m & -m
                                bits.append(lsb.bit_length() - 1)
                                m ^= lsb
                            if bits:
                                v = random.choice(bits)
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                current_edges -= 1
                                no_improve = 0
                            break

    # Return the best found
    return get_edges()

