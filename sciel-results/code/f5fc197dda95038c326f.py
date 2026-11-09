import random
import time
import math
import numpy as np

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # --- Data Structures ---
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

    def calc_metrics():
        """Calculate max triangle degree sum and sum of squares of triangle degree sums."""
        max_sum = 0
        sum_sq = 0
        # Iterate over all triples u < v < w
        # But faster: iterate over edges (u,v) and common neighbors w
        for u in range(n):
            if adj[u] == 0:
                continue
            mask_v = adj[u]
            # Only consider v > u
            mask_v &= ~((1 << (u + 1)) - 1)
            while mask_v:
                lsb = mask_v & -mask_v
                v = lsb.bit_length() - 1
                mask_v ^= lsb
                
                common = adj[u] & adj[v]
                # Iterate over common neighbors w
                while common:
                    lsb_w = common & -common
                    w = lsb_w.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                    sum_sq += s * s
                    common ^= lsb_w
        return max_sum, sum_sq

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

    def init_k88_plus_one():
        """Initialize K_{8,8} plus one edge (0,1)."""
        for i in range(n):
            adj[i] = 0
            degs[i] = 0
        # K_{8,8}: A={0..7}, B={8..15}
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
        # Add edge (0,1)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        degs[0] += 1
        degs[1] += 1

    def init_random():
        """Initialize a random graph with 65 edges."""
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

    best_score = float('inf')
    best_adj = [0] * n
    best_degs = [0] * n
    best_edges = None

    # Pre-compute list of all possible edges
    all_edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_edges_list)

    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Alternate between K88+1 and Random
        if r % 3 == 0:
            init_k88_plus_one()
        elif r % 3 == 1:
            # Perturb K88+1
            init_k88_plus_one()
            # Do 5 random swaps
            for _ in range(5):
                while True:
                    u1, v1 = random.choice(all_edges_list)
                    if adj[u1] & (1 << v1): break
                while True:
                    u2, v2 = random.choice(all_edges_list)
                    if not (adj[u2] & (1 << v2)) and (u2, v2) != (u1, v1): break
                make_swap(u1, v1, u2, v2)
        else:
            init_random()
            
        current_max, current_penalty = calc_metrics()
        
        if current_max < best_score:
            best_score = current_max
            best_adj = adj[:]
            best_degs = degs[:]
            best_edges = get_edges_from_adj()
            
        # Simulated Annealing
        T = 10.0
        cooling = 0.995
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Pick edge (u1, v1) to remove
            while True:
                u1 = random.randint(0, n - 1)
                v1 = random.randint(0, n - 1)
                if u1 < v1 and (adj[u1] & (1 << v1)):
                    break
            
            # Pick non-edge (u2, v2) to add
            while True:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 < v2 and not (adj[u2] & (1 << v2)):
                    if (u2, v2) == (u1, v1):
                        continue
                    break
            
            make_swap(u1, v1, u2, v2)
            new_max, new_penalty = calc_metrics()
            
            delta = new_penalty - current_penalty
            
            if delta <= 0:
                current_penalty = new_penalty
                no_improve = 0
                
                if new_max < best_score:
                    best_score = new_max
                    best_adj = adj[:]
                    best_degs = degs[:]
                    best_edges = get_edges_from_adj()
            else:
                if T > 0.01:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_penalty = new_penalty
                        no_improve = 0
                        # Check max score even if penalty worsened, as it might be lower
                        if new_max < best_score:
                            best_score = new_max
                            best_adj = adj[:]
                            best_degs = degs[:]
                            best_edges = get_edges_from_adj()
                    else:
                        unmake_swap(u1, v1, u2, v2)
                        no_improve += 1
                else:
                    unmake_swap(u1, v1, u2, v2)
                    no_improve += 1
            
            T *= cooling

    if best_edges is None:
        # Fallback
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return {'n': n, 'edges': edges}
        
    return {'n': n, 'edges': best_edges}
