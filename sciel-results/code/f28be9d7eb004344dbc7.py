import random
import time
import math
import numpy as np

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # --- Data Structures ---
    # Use bitmasks for adjacency for fast intersection checks
    adj = [0] * n
    degs = [0] * n
    
    def get_edges_from_adj():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def calc_score_max():
        """Calculate the maximum triangle degree sum."""
        max_sum = 0
        for u in range(n):
            if adj[u] == 0:
                continue
            # Iterate over neighbors v > u
            mask_v = adj[u] & ~((1 << (u + 1)) - 1)
            while mask_v:
                lsb = mask_v & -mask_v
                v = lsb.bit_length() - 1
                mask_v ^= lsb
                
                common = adj[u] & adj[v]
                while common:
                    lsb_w = common & -common
                    w = lsb_w.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                    common ^= lsb_w
        return max_sum

    def calc_score_sq():
        """Calculate the sum of squared triangle degree sums (smooth surrogate)."""
        sum_sq = 0
        for u in range(n):
            if adj[u] == 0:
                continue
            mask_v = adj[u] & ~((1 << (u + 1)) - 1)
            while mask_v:
                lsb = mask_v & -mask_v
                v = lsb.bit_length() - 1
                mask_v ^= lsb
                
                common = adj[u] & adj[v]
                while common:
                    lsb_w = common & -common
                    w = lsb_w.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    sum_sq += s * s
                    common ^= lsb_w
        return sum_sq

    def make_swap(u1, v1, u2, v2):
        """Remove edge (u1,v1), add edge (u2,v2)."""
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        degs[u1] -= 1
        degs[v1] -= 1
        
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        degs[u2] += 1
        degs[v2] += 1

    def unmake_swap(u1, v1, u2, v2):
        """Revert the swap."""
        adj[u2] &= ~(1 << v2)
        adj[v2] &= ~(1 << u2)
        degs[u2] -= 1
        degs[v2] -= 1
        
        adj[u1] |= (1 << v1)
        adj[v1] |= (1 << u1)
        degs[u1] += 1
        degs[v1] += 1

    def random_graph_init():
        """Initialize a random graph with exactly target_edges edges."""
        for i in range(n):
            adj[i] = 0
            degs[i] = 0
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs[:target_edges]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1

    def seed_graph_init():
        """Initialize with K_8,8 + (0,1)."""
        for i in range(n):
            adj[i] = 0
            degs[i] = 0
        # K_8,8
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
        # Add (0,1)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        degs[0] += 1
        degs[1] += 1

    # --- Search Strategy ---
    best_score_max = float('inf')
    best_adj = [0] * n
    best_degs = [0] * n
    best_edges = None
    
    all_edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # We will run several restarts.
    # 1. Seed based
    # 2. Random based
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Initialization
        if r == 0:
            seed_graph_init()
        else:
            random_graph_init()
            
        current_score_max = calc_score_max()
        current_score_sq = calc_score_sq()
        
        if current_score_max < best_score_max:
            best_score_max = current_score_max
            best_adj = adj[:]
            best_degs = degs[:]
            best_edges = get_edges_from_adj()
        
        # 2. Local Search (Simulated Annealing)
        # Phase 1: Explore with smooth surrogate (sum of squares)
        T = 5000.0  # Scale for sum of squares
        cooling = 0.995
        no_improve = 0
        max_no_improve = 5000
        
        # Use sum of squares for acceptance during exploration
        use_sq_objective = True
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Pick move
            # 80% Single Swap, 20% Double Swap (if stuck)
            
            if no_improve > 1000 and random.random() < 0.5:
                # Try Double Swap
                # Pick 2 edges to remove
                e1 = None
                e2 = None
                while e1 is None:
                    u1 = random.randint(0, n - 1)
                    v1 = random.randint(0, n - 1)
                    if u1 < v1 and (adj[u1] & (1 << v1)):
                        e1 = (u1, v1)
                        
                while e2 is None:
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 < v2 and (adj[u2] & (1 << v2)):
                        if (u2, v2) != e1:
                            e2 = (u2, v2)
                            
                if e1 and e2:
                    # Pick 2 non-edges to add
                    f1 = None
                    f2 = None
                    while f1 is None:
                        u3 = random.randint(0, n - 1)
                        v3 = random.randint(0, n - 1)
                        if u3 < v3 and not (adj[u3] & (1 << v3)):
                            if (u3, v3) != e1 and (u3, v3) != e2:
                                f1 = (u3, v3)
                                
                    while f2 is None:
                        u4 = random.randint(0, n - 1)
                        v4 = random.randint(0, n - 1)
                        if u4 < v4 and not (adj[u4] & (1 << v4)):
                            if (u4, v4) != e1 and (u4, v4) != e2 and (u4, v4) != f1:
                                f2 = (u4, v4)
                    
                    if f1 and f2:
                        # Apply double swap
                        # Remove e1, e2
                        make_swap(e1[0], e1[1], e1[0], e1[1]) # No, make_swap adds too.
                        # Need explicit remove/add for double
                        # Remove e1
                        adj[e1[0]] &= ~(1 << e1[1])
                        adj[e1[1]] &= ~(1 << e1[0])
                        degs[e1[0]] -= 1
                        degs[e1[1]] -= 1
                        # Remove e2
                        adj[e2[0]] &= ~(1 << e2[1])
                        adj[e2[1]] &= ~(1 << e2[0])
                        degs[e2[0]] -= 1
                        degs[e2[1]] -= 1
                        # Add f1
                        adj[f1[0]] |= (1 << f1[1])
                        adj[f1[1]] |= (1 << f1[0])
                        degs[f1[0]] += 1
                        degs[f1[1]] += 1
                        # Add f2
                        adj[f2[0]] |= (1 << f2[1])
                        adj[f2[1]] |= (1 << f2[0])
                        degs[f2[0]] += 1
                        degs[f2[1]] += 1
                        
                        new_score_max = calc_score_max()
                        new_score_sq = calc_score_sq()
                        
                        # Acceptance based on current objective
                        if use_sq_objective:
                            delta = new_score_sq - current_score_sq
                            T_eff = T
                        else:
                            delta = new_score_max - current_score_max
                            T_eff = T / 100.0 # Scale T for max sum
                        
                        if delta <= 0:
                            current_score_max = new_score_max
                            current_score_sq = new_score_sq
                            no_improve = 0
                            if current_score_max < best_score_max:
                                best_score_max = current_score_max
                                best_adj = adj[:]
                                best_degs = degs[:]
                                best_edges = get_edges_from_adj()
                        else:
                            prob = math.exp(-delta / T_eff)
                            if random.random() < prob:
                                current_score_max = new_score_max
                                current_score_sq = new_score_sq
                                no_improve = 0
                            else:
                                # Revert
                                # Remove f1, f2
                                adj[f1[0]] &= ~(1 << f1[1])
                                adj[f1[1]] &= ~(1 << f1[0])
                                degs[f1[0]] -= 1
                                degs[f1[1]] -= 1
                                adj[f2[0]] &= ~(1 << f2[1])
                                adj[f2[1]] &= ~(1 << f2[0])
                                degs[f2[0]] -= 1
                                degs[f2[1]] -= 1
                                # Add e1, e2
                                adj[e1[0]] |= (1 << e1[1])
                                adj[e1[1]] |= (1 << e1[0])
                                degs[e1[0]] += 1
                                degs[e1[1]] += 1
                                adj[e2[0]] |= (1 << e2[1])
                                adj[e2[1]] |= (1 << e2[0])
                                degs[e2[0]] += 1
                                degs[e2[1]] += 1
                                no_improve += 1
                        T *= cooling
                        if T < 1.0: T = 1.0
                        continue # Skip single swap logic below
            
            # Single Swap
            u1 = random.randint(0, n - 1)
            v1 = random.randint(0, n - 1)
            if u1 > v1: u1, v1 = v1, u1
            if u1 == v1 or not (adj[u1] & (1 << v1)):
                continue
                
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 > v2: u2, v2 = v2, u2
            if u2 == v2 or (adj[u2] & (1 << v2)):
                continue
            if (u2, v2) == (u1, v1):
                continue

            make_swap(u1, v1, u2, v2)
            new_score_max = calc_score_max()
            new_score_sq = calc_score_sq()
            
            if use_sq_objective:
                delta = new_score_sq - current_score_sq
                T_eff = T
            else:
                delta = new_score_max - current_score_max
                T_eff = T / 100.0

            if delta <= 0:
                current_score_max = new_score_max
                current_score_sq = new_score_sq
                no_improve = 0
                if current_score_max < best_score_max:
                    best_score_max = current_score_max
                    best_adj = adj[:]
                    best_degs = degs[:]
                    best_edges = get_edges_from_adj()
            else:
                prob = math.exp(-delta / T_eff)
                if random.random() < prob:
                    current_score_max = new_score_max
                    current_score_sq = new_score_sq
                    no_improve = 0
                else:
                    unmake_swap(u1, v1, u2, v2)
                    no_improve += 1
            
            T *= cooling
            if T < 1.0: T = 1.0
            
            # Switch to hard objective if T is low
            if T < 10.0 and use_sq_objective:
                use_sq_objective = False
                T = 5.0 # Reset T for max sum scale
            
        # 3. Polish Phase: Greedy Hill Climbing on Max Sum
        # Try to improve current_score_max with strict greedy moves
        polish_no_improve = 0
        max_polish_iters = 5000
        while polish_no_improve < max_polish_iters and time.time() < deadline:
            improved = False
            # Try random single swaps
            for _ in range(10):
                u1 = random.randint(0, n - 1)
                v1 = random.randint(0, n - 1)
                if u1 > v1: u1, v1 = v1, u1
                if u1 == v1 or not (adj[u1] & (1 << v1)): continue
                
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 > v2: u2, v2 = v2, u2
                if u2 == v2 or (adj[u2] & (1 << v2)): continue
                if (u2, v2) == (u1, v1): continue

                make_swap(u1, v1, u2, v2)
                new_score_max = calc_score_max()
                
                if new_score_max < current_score_max:
                    current_score_max = new_score_max
                    improved = True
                    if current_score_max < best_score_max:
                        best_score_max = current_score_max
                        best_adj = adj[:]
                        best_degs = degs[:]
                        best_edges = get_edges_from_adj()
                else:
                    unmake_swap(u1, v1, u2, v2)
            
            if improved:
                polish_no_improve = 0
            else:
                polish_no_improve += 1

    if best_edges is None:
        # Fallback
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return {'n': n, 'edges': edges}
        
    return {'n': n, 'edges': best_edges}
