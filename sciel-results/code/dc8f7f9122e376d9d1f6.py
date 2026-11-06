import numpy as np
import random
import time
from collections import defaultdict

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    # 1. Precompute distances
    pts = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sum(diff.astype(np.int64) ** 2, axis=2)
    del diff
    
    # 2. Precompute circles: circles[i][d] = list of j such that dist(i,j)==d
    # This is memory heavy if we store lists, but M=4096.
    # Max dist ~ 8000.
    # We can use a dictionary of lists.
    circles = [defaultdict(list) for _ in range(M)]
    for i in range(M):
        for j in range(i + 1, M):
            d = dist_mat[i, j]
            circles[i][d].append(j)
            circles[j][d].append(i)
            
    # Helper to count active points in a circle
    # active_set is a set of indices
    def count_in_circle(active_set, i, d):
        if d not in circles[i]:
            return 0
        # Intersection of active_set and circles[i][d]
        # Since circles[i][d] is a list, we can check membership
        # To speed up, we can keep active_set as a set.
        return sum(1 for x in circles[i][d] if x in active_set)

    def calc_delta_add(p, active_set):
        """Calculate number of new isosceles triangles if p is added to active_set."""
        if not active_set:
            return 0
        
        delta = 0
        
        # Case 1: p is the apex
        # Count pairs (a, b) in active_set such that dist(p, a) == dist(p, b)
        # Get distances from p to all in active_set
        dists = []
        for a in active_set:
            dists.append(dist_mat[p, a])
        
        # Count duplicates
        d_counts = defaultdict(int)
        for d in dists:
            d_counts[d] += 1
        for d, count in d_counts.items():
            delta += count * (count - 1) // 2
            
        # Case 2: a in active_set is the apex
        # Pairs (p, b) with b in active_set such that dist(a, p) == dist(a, b)
        for a in active_set:
            d_ap = dist_mat[a, p]
            # Count b in active_set such that dist(a, b) == d_ap
            # Note: b cannot be a.
            # circles[a][d_ap] contains a? No, dist(a,a)=0. d_ap > 0.
            # So circles[a][d_ap] does not contain a.
            # It contains p? dist(a,p) = d_ap. So p is in circles[a][d_ap].
            # But p is NOT in active_set yet.
            # So we just count how many points in active_set are in circles[a][d_ap].
            delta += count_in_circle(active_set, a, d_ap)
            
        # Case 3: b in active_set is the apex
        # Symmetric to Case 2.
        # Pairs (p, a) with a in active_set such that dist(b, p) == dist(b, a)
        for b in active_set:
            d_bp = dist_mat[b, p]
            delta += count_in_circle(active_set, b, d_bp)
            
        return delta

    def greedy_init(rng):
        active_set = set()
        order = list(range(M))
        rng.shuffle(order)
        for p in order:
            if calc_delta_add(p, active_set) == 0:
                active_set.add(p)
        return active_set

    def local_improve(active_set, rng, deadline):
        # Try to add points
        added = True
        while added and time.time() < deadline - 5.0:
            added = False
            candidates = [p for p in range(M) if p not in active_set]
            rng.shuffle(candidates)
            for p in candidates:
                if calc_delta_add(p, active_set) == 0:
                    active_set.add(p)
                    added = True
                    break
        
        # Try swaps
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try random swaps to escape local minima
            # Or systematic?
            # Let's try a few random swaps
            for _ in range(100):
                if time.time() > deadline - 5.0:
                    break
                if len(active_set) < 2:
                    break
                u = rng.choice(list(active_set))
                v = rng.choice([p for p in range(M) if p not in active_set])
                
                # Check if swap is valid
                # New set: S' = (S \ {u}) U {v}
                # We need to check if v conflicts with S \ {u}
                # This is equivalent to calc_delta_add(v, S \ {u}) == 0
                
                # Temporarily remove u
                active_set.remove(u)
                delta = calc_delta_add(v, active_set)
                if delta == 0:
                    active_set.add(v)
                    improved = True
                    break
                else:
                    # Revert
                    active_set.add(u)
                    
        return active_set

    best_set = set()
    best_len = 0
    rng = random.Random(42)
    
    for restart in range(100):
        if time.time() > deadline - 10.0:
            break
            
        current_set = greedy_init(rng)
        current_len = len(current_set)
        
        if current_len > best_len:
            best_set = current_set.copy()
            best_len = current_len
            
        # Improve
        if current_len >= best_len - 5:
            improved_set = local_improve(current_set, rng, deadline)
            if len(improved_set) > best_len:
                best_set = improved_set
                best_len = len(improved_set)
                
    return [pts[i] for i in best_set]
