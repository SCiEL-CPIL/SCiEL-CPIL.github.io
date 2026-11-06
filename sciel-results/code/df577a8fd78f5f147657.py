# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill

    # Triangular lattice: point = a + b*omega, omega = exp(i*pi/3)
    # Squared distance between (a1,b1) and (a2,b2):
    #   da^2 + da*db + db^2
    # Unit-distance (edge) offsets (dist_sq == 1):
    EDGES = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + da * db + db * db

    def count_edges(pts):
        e = 0
        n = len(pts)
        for i in range(n):
            pi = pts[i]
            for j in range(i + 1, n):
                if dist_sq(pi, pts[j]) == 1:
                    e += 1
        return e

    def is_valid(pts):
        if len(pts) != 32:
            return False
        seen = set()
        for p in pts:
            if p in seen:
                return False
            seen.add(p)
            if abs(p[0]) > 64 or abs(p[1]) > 64:
                return False
        for i in range(32):
            pi = pts[i]
            for j in range(i + 1, 32):
                if dist_sq(pi, pts[j]) < 1:
                    return False
        return True

    # ---- Build a hexagonal/spiral seed ----
    def build_hex_seed():
        """Build a compact hexagonal cluster of 32 points."""
        pts = [(0, 0)]
        # Add rings: ring k has 6*k points
        # Ring 1: 6 pts (total 7)
        # Ring 2: 12 pts (total 19)
        # Ring 3: 18 pts (total 37) -> we need only 32, so take 13 from ring 3
        ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0),
                 (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2), (1,-2)]
        # Wait, let me compute ring 2 properly.
        # Ring k consists of points at hex distance k from origin.
        # For ring 2: points (a,b) where max(|a|,|b|,|a+b|) == 2... 
        # Actually in this coordinate system, hex distance is different.
        # Let me just use a spiral approach.
        
        # Better: use a known good hexagonal packing
        # Center at (0,0), add points in expanding hexagonal rings
        pts = [(0, 0)]
        
        # Ring 1: 6 points
        ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        pts.extend(ring1)
        
        # Ring 2: 12 points
        # Going around: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0),
        #                 (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2), (1,-2)
        # Let me verify: hex distance in (a,b) coords with omega = exp(i*pi/3)
        # The 6 neighbors of (0,0) are the EDGES list.
        # Ring 2 points are at distance 2 from origin.
        ring2 = []
        for a in range(-2, 3):
            for b in range(-2, 3):
                if a == 0 and b == 0:
                    continue
                if dist_sq((a,b), (0,0)) == 4:  # distance 2
                    ring2.append((a,b))
        # Filter to only those that are valid (not too close to existing)
        valid_ring2 = []
        for p in ring2:
            valid = True
            for q in pts:
                if dist_sq(p, q) < 1:
                    valid = False
                    break
            if valid:
                valid_ring2.append(p)
        pts.extend(valid_ring2[:12])  # Take up to 12 from ring 2
        
        # Ring 3: need 32 - len(pts) more points
        needed = 32 - len(pts)
        ring3 = []
        for a in range(-3, 4):
            for b in range(-3, 4):
                if dist_sq((a,b), (0,0)) == 9:  # distance 3
                    ring3.append((a,b))
        valid_ring3 = []
        for p in ring3:
            valid = True
            for q in pts:
                if dist_sq(p, q) < 1:
                    valid = False
                    break
            if valid:
                valid_ring3.append(p)
        # Sort by distance to center for compactness
        valid_ring3.sort(key=lambda p: p[0]*p[0] + p[0]*p[1] + p[1]*p[1])
        pts.extend(valid_ring3[:needed])
        
        return pts[:32]

    def build_greedy_seed():
        pts = [(0, 0)]
        
        for _ in range(31):
            best_cand = None
            best_score = -1
            best_dist_center = 999
            
            min_a = min(p[0] for p in pts)
            max_a = max(p[0] for p in pts)
            min_b = min(p[1] for p in pts)
            max_b = max(p[1] for p in pts)
            
            for a in range(min_a - 2, max_a + 3):
                for b in range(min_b - 2, max_b + 3):
                    if abs(a) > 64 or abs(b) > 64:
                        continue
                    cand = (a, b)
                    if cand in pts:
                        continue
                    valid = True
                    for p in pts:
                        if dist_sq(cand, p) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    edges = 0
                    for p in pts:
                        if dist_sq(cand, p) == 1:
                            edges += 1
                    
                    d_center = a*a + a*b + b*b
                    
                    if edges > best_score or (edges == best_score and d_center < best_dist_center):
                        best_score = edges
                        best_dist_center = d_center
                        best_cand = cand
            
            if best_cand is None:
                for _ in range(100):
                    da = random.randint(-5, 5)
                    db = random.randint(-5, 5)
                    cand = (pts[-1][0] + da, pts[-1][1] + db)
                    if abs(cand[0]) <= 64 and abs(cand[1]) <= 64 and cand not in pts:
                        valid = True
                        for p in pts:
                            if dist_sq(cand, p) < 1:
                                valid = False
                                break
                        if valid:
                            best_cand = cand
                            break
            
            if best_cand:
                pts.append(best_cand)
            else:
                break
                
        return pts[:32]

    # ---- Local search with 2-opt style swaps ----
    def local_search(pts, max_time):
        pts = list(pts)
        best = count_edges(pts)
        improved = True
        no_improve_count = 0
        while improved and time.time() < max_time:
            improved = False
            no_improve_count = 0
            for i in range(32):
                if time.time() > max_time:
                    break
                old = pts[i]
                # Try moving point i to various positions
                candidates = []
                # 6 neighbors
                for da, db in EDGES:
                    c = (old[0] + da, old[1] + db)
                    if abs(c[0]) <= 64 and abs(c[1]) <= 64:
                        candidates.append(c)
                # Distance 2 moves
                for da in range(-2, 3):
                    for db in range(-2, 3):
                        if da == 0 and db == 0:
                            continue
                        c = (old[0] + da, old[1] + db)
                        if abs(c[0]) <= 64 and abs(c[1]) <= 64:
                            candidates.append(c)
                # Random nearby
                for _ in range(3):
                    da = random.randint(-4, 4)
                    db = random.randint(-4, 4)
                    c = (old[0] + da, old[1] + db)
                    if abs(c[0]) <= 64 and abs(c[1]) <= 64:
                        candidates.append(c)

                for cand in candidates:
                    if cand in pts:
                        continue
                    valid = True
                    for j in range(32):
                        if j == i:
                            continue
                        if dist_sq(cand, pts[j]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    old_e = sum(1 for j in range(32) if j != i and dist_sq(old, pts[j]) == 1)
                    new_e = sum(1 for j in range(32) if j != i and dist_sq(cand, pts[j]) == 1)
                    delta = new_e - old_e
                    if delta > 0:
                        pts[i] = cand
                        best += delta
                        improved = True
                        break
                if not improved:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
            # If no improvement in a full pass, break
            if not improved:
                break
        return pts, best

    # ---- Simulated annealing with better move generation ----
    def anneal(pts, max_time):
        pts = list(pts)
        cur_e = count_edges(pts)
        best_pts = list(pts)
        best_e = cur_e
        T = 2.0
        T_min = 1e-5
        cooldown = 0.999
        while time.time() < max_time:
            i = random.randrange(32)
            old = pts[i]
            # Pick a random candidate
            for _ in range(10):
                # Mix of local and slightly larger moves
                if random.random() < 0.7:
                    da = random.randint(-2, 2)
                    db = random.randint(-2, 2)
                else:
                    da = random.randint(-4, 4)
                    db = random.randint(-4, 4)
                cand = (old[0] + da, old[1] + db)
                if abs(cand[0]) > 64 or abs(cand[1]) > 64:
                    continue
                if cand in pts:
                    continue
                valid = True
                for j in range(32):
                    if j == i:
                        continue
                    if dist_sq(cand, pts[j]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                old_e = sum(1 for j in range(32) if j != i and dist_sq(old, pts[j]) == 1)
                new_e = sum(1 for j in range(32) if j != i and dist_sq(cand, pts[j]) == 1)
                delta = new_e - old_e
                if delta >= 0 or random.random() < math.exp(delta / T):
                    pts[i] = cand
                    cur_e += delta
                    if cur_e > best_e:
                        best_e = cur_e
                        best_pts = list(pts)
                break
            T = max(T_min, T * cooldown)
        return best_pts, best_e

    # ---- Main ----
    # Try both seeds and use the better one
    seed1 = build_hex_seed()
    seed2 = build_greedy_seed()
    
    if not is_valid(seed1):
        seed1 = seed2
    if not is_valid(seed2):
        seed2 = seed1
    
    e1 = count_edges(seed1)
    e2 = count_edges(seed2)
    
    if e1 >= e2:
        seed = seed1
    else:
        seed = seed2
    
    best_pts, best_e = local_search(seed, deadline - 15.0)
    
    if best_e == 76:
        return [[a, b] for a, b in best_pts]

    # Multiple annealing passes
    while time.time() < deadline - 10.0:
        # Animate from current best
        cand_pts, cand_e = anneal(best_pts, deadline - 10.0)
        if cand_e > best_e:
            best_pts, best_e = cand_pts, cand_e
            if best_e == 76:
                break
            # Polish
            polished, pe = local_search(best_pts, deadline - 10.0)
            if pe > best_e:
                best_pts, best_e = polished, pe
                if best_e == 76:
                    break
        else:
            # Try fresh seed occasionally
            fresh = build_greedy_seed()
            if is_valid(fresh):
                fp, fe = local_search(fresh, deadline - 10.0)
                if fe > best_e:
                    best_pts, best_e = fp, fe
                    if best_e == 76:
                        break

    if not is_valid(best_pts):
        return [[a, b] for a in range(16) for b in range(2)]

    return [[a, b] for a, b in best_pts]
# EVOLVE-BLOCK-END
