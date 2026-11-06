import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290s deadline, 40s margin
    
    best_points = []
    best_count = 0
    
    def is_valid_addition(S, dist_sets, p):
        """Check if point p can be added to set S without creating isosceles triangles."""
        if not S:
            return True
        
        px, py = p
        seen_dists = set()
        
        for q in S:
            qx, qy = q
            d_sq = (px - qx)**2 + (py - qy)**2
            
            # Check if this distance already exists from q to another point
            if d_sq in dist_sets[q]:
                return False
            
            # Check if this distance already exists from p to another point
            if d_sq in seen_dists:
                return False
            seen_dists.add(d_sq)
        
        return True

    def add_point(S, dist_sets, p):
        """Add point p to set S, updating dist_sets."""
        px, py = p
        S.append(p)
        dist_sets[p] = set()
        
        for q in S[:-1]:
            qx, qy = q
            d_sq = (px - qx)**2 + (py - qy)**2
            dist_sets[p].add(d_sq)
            dist_sets[q].add(d_sq)
    
    def greedy_construct_deterministic():
        """Construct using a deterministic pattern that avoids isosceles triangles."""
        S = []
        dist_sets = {}
        
        # Use a spiral pattern with specific spacing
        # Start from center and spiral outward
        cx, cy = N // 2, N // 2
        
        # Add center first
        S.append((cx, cy))
        dist_sets[(cx, cy)] = set()
        
        # Spiral outward
        dx, dy = 0, 1
        for radius in range(1, N):
            for _ in range(4):
                for _ in range(radius):
                    cx += dx
                    cy += dy
                    if 0 <= cx < N and 0 <= cy < N:
                        p = (cx, cy)
                        if is_valid_addition(S, dist_sets, p):
                            add_point(S, dist_sets, p)
                # Rotate direction
                dx, dy = -dy, dx
        
        return S, dist_sets

    def greedy_construct_random(seed_val):
        """Construct using random greedy with incremental distance sets."""
        random.seed(seed_val)
        S = []
        dist_sets = {}
        
        candidates = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(candidates)
        
        for p in candidates:
            if is_valid_addition(S, dist_sets, p):
                add_point(S, dist_sets, p)
        
        return S, dist_sets

    def simulated_annealing(S, dist_sets, time_limit):
        """Simulated annealing to improve the set."""
        start_sa = time.time()
        current_S = list(S)
        current_dist_sets = {k: set(v) for k, v in dist_sets.items()}
        best_S = list(current_S)
        best_dist_sets = {k: set(v) for k, v in current_dist_sets.items()}
        best_count = len(current_S)
        
        # Temperature schedule
        T = 1.0
        cooling_rate = 0.9999
        
        all_points = set((x, y) for x in range(N) for y in range(N))
        current_set = set(current_S)
        
        while time.time() - start_sa < time_limit:
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random point
                p = (random.randint(0, N-1), random.randint(0, N-1))
                if p not in current_set:
                    if is_valid_addition(current_S, current_dist_sets, p):
                        add_point(current_S, current_dist_sets, p)
                        current_set.add(p)
                        if len(current_S) > best_count:
                            best_S = list(current_S)
                            best_dist_sets = {k: set(v) for k, v in current_dist_sets.items()}
                            best_count = len(current_S)
                    else:
                        # Reject with probability exp(-delta/T) where delta = 1 (penalty)
                        if random.random() < np.exp(-1.0 / T):
                            # Force add (this will create violations, but we'll clean up)
                            add_point(current_S, current_dist_sets, p)
                            current_set.add(p)
                            # Clean up: remove points that cause violations
                            cleanup_violations(current_S, current_dist_sets)
                            if len(current_S) > best_count:
                                best_S = list(current_S)
                                best_dist_sets = {k: set(v) for k, v in current_dist_sets.items()}
                                best_count = len(current_S)
                        
            elif move_type < 0.7:
                # Try to remove a random point
                if current_S:
                    idx = random.randint(0, len(current_S) - 1)
                    p = current_S[idx]
                    remove_point(current_S, current_dist_sets, p)
                    current_set.discard(p)
                    if len(current_S) > best_count:
                        best_S = list(current_S)
                        best_dist_sets = {k: set(v) for k, v in current_dist_sets.items()}
                        best_count = len(current_S)
                        
            else:
                # Try a swap: remove one point, add another
                if current_S:
                    idx = random.randint(0, len(current_S) - 1)
                    p_remove = current_S[idx]
                    p_add = (random.randint(0, N-1), random.randint(0, N-1))
                    
                    if p_add not in current_set:
                        # Try the swap
                        temp_S = list(current_S)
                        temp_dist_sets = {k: set(v) for k, v in current_dist_sets.items()}
                        
                        remove_point(temp_S, temp_dist_sets, p_remove)
                        if is_valid_addition(temp_S, temp_dist_sets, p_add):
                            add_point(temp_S, temp_dist_sets, p_add)
                            # Accept the swap
                            current_S = temp_S
                            current_dist_sets = temp_dist_sets
                            current_set.discard(p_remove)
                            current_set.add(p_add)
                            if len(current_S) > best_count:
                                best_S = list(current_S)
                                best_dist_sets = {k: set(v) for k, v in current_dist_sets.items()}
                                best_count = len(current_S)
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
        
        return best_S, best_dist_sets

    def remove_point(S, dist_sets, p):
        """Remove point p from set S, updating dist_sets."""
        px, py = p
        S.remove(p)
        del dist_sets[p]
        
        for q in S:
            qx, qy = q
            d_sq = (px - qx)**2 + (py - qy)**2
            dist_sets[q].discard(d_sq)

    def cleanup_violations(S, dist_sets):
        """Remove points that create isosceles triangles."""
        changed = True
        while changed:
            changed = False
            for i, p in enumerate(list(S)):
                # Check if p is involved in any violation
                if is_point_in_violation(p, S, dist_sets):
                    remove_point(S, dist_sets, p)
                    changed = True

    def is_point_in_violation(p, S, dist_sets):
        """Check if point p is part of any isosceles triangle."""
        px, py = p
        dists_from_p = dist_sets[p]
        
        # Check if any distance from p appears twice
        if len(dists_from_p) != len(S) - 1:
            return True
        
        # Check if p creates isosceles with any pair of other points
        for q in S:
            if q == p:
                continue
            qx, qy = q
            d_pq_sq = (px - qx)**2 + (py - qy)**2
            
            # Check if there's another point r such that dist(p,r) == dist(p,q) or dist(p,r) == dist(q,r) or dist(q,r) == dist(p,q)
            for r in S:
                if r == p or r == q:
                    continue
                rx, ry = r
                d_pr_sq = (px - rx)**2 + (py - ry)**2
                d_qr_sq = (qx - rx)**2 + (qy - ry)**2
                
                if d_pq_sq == d_pr_sq or d_pq_sq == d_qr_sq or d_pr_sq == d_qr_sq:
                    return True
        
        return False

    # Run multiple trials
    trial_count = 0
    while time.time() < deadline - 10:
        trial_count += 1
        seed_val = trial_count * 7919 + int(time.time() * 1000) % 7919
        
        # Alternate between deterministic and random construction
        if trial_count % 3 == 1:
            S, dist_sets = greedy_construct_deterministic()
        else:
            S, dist_sets = greedy_construct_random(seed_val)
        
        current_count = len(S)
        
        if current_count > best_count:
            best_points = S
            best_count = current_count
        
        # Apply simulated annealing if we have time
        if time.time() < deadline - 30 and current_count > 0:
            S_sa, dist_sets_sa = simulated_annealing(S, dist_sets, min(15.0, deadline - time.time() - 30))
            if len(S_sa) > best_count:
                best_points = S_sa
                best_count = len(S_sa)

    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
