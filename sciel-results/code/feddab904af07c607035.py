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

    def count_edges_fast(pts):
        """Count edges using set lookup for O(n) per point instead of O(n^2)"""
        pt_set = set(pts)
        e = 0
        for p in pts:
            for da, db in EDGES:
                q = (p[0] + da, p[1] + db)
                if q in pt_set:
                    e += 1
        return e // 2  # Each edge counted twice

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
        
        # Ring 1: 6 points
        ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        pts.extend(ring1)
        
        # Ring 2: 12 points at distance 2 from origin
        ring2 = []
        for a in range(-2, 3):
            for b in range(-2, 3):
                if a == 0 and b == 0:
                    continue
                if dist_sq((a,b), (0,0)) == 4:  # distance 2
                    ring2.append((a,b))
        
        valid_ring2 = []
        for p in ring2:
            valid = True
            for q in pts:
                if dist_sq(p, q) < 1:
                    valid = False
                    break
            if valid:
                valid_ring2.append(p)
        pts.extend(valid_ring2[:12])
        
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
        """Build seed by greedily adding points that maximize edges."""
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

    def build_spiral_seed():
        """Build a triangular lattice spiral that's known to be near-optimal."""
        pts = [(0, 0)]
        
        # Generate spiral points in order of distance from origin
        candidates = []
        for r in range(0, 8):
            for a in range(-r, r + 1):
                for b in range(-r, r + 1):
                    if abs(a) > r or abs(b) > r:
                        continue
                    candidates.append((a, b))
        
        # Sort by distance from origin, then by angle for spiral ordering
        def spiral_key(p):
            d = p[0]*p[0] + p[0]*p[1] + p[1]*p[1]
            # Add a small angular component for spiral ordering
            angle = math.atan2(p[1] + 0.5*p[0], p[0])
            return (d, angle)
        
        candidates.sort(key=spiral_key)
        
        seen = set([(0, 0)])
        for p in candidates:
            if len(pts) >= 32:
                break
            if p in seen:
                continue
            valid = True
            for q in pts:
                if dist_sq(p, q) < 1:
                    valid = False
                    break
            if valid:
                pts.append(p)
                seen.add(p)
        
        return pts[:32]

    # ---- Fast local search ----
    def local_search_fast(pts, max_time):
        """Local search with O(1) edge delta computation."""
        pts = list(pts)
        pt_set = set(pts)
        best = count_edges_fast(pts)
        
        # Precompute edge count per point
        def get_point_edges(i):
            p = pts[i]
            e = 0
            for da, db in EDGES:
                q = (p[0] + da, p[1] + db)
                if q in pt_set:
                    e += 1
            return e
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(32):
                if time.time() > max_time:
                    break
                old = pts[i]
                old_edges = get_point_edges(i)
                
                # Try all valid positions within distance 3
                best_cand = None
                best_delta = 0
                best_new_edges = 0
                
                for da in range(-3, 4):
                    for db in range(-3, 4):
                        if da == 0 and db == 0:
                            continue
                        cand = (old[0] + da, old[1] + db)
                        if abs(cand[0]) > 64 or abs(cand[1]) > 64:
                            continue
                        if cand in pt_set:
                            continue
                        
                        # Check validity
                        valid = True
                        for da2, db2 in EDGES:
                            neighbor = (cand[0] + da2, cand[1] + db2)
                            if neighbor in pt_set:
                                # Check if it's the point we're moving
                                idx = pts.index(neighbor) if neighbor in pts else -1
                                if idx != i:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        
                        # Count new edges
                        new_edges = 0
                        for da2, db2 in EDGES:
                            q = (cand[0] + da2, cand[1] + db2)
                            if q in pt_set and q != old:
                                new_edges += 1
                        
                        delta = new_edges - old_edges
                        if delta > best_delta:
                            best_delta = delta
                            best_cand = cand
                            best_new_edges = new_edges
                
                if best_cand is not None and best_delta > 0:
                    # Remove old from set
                    pt_set.remove(old)
                    # Add new
                    pts[i] = best_cand
                    pt_set.add(best_cand)
                    best += best_delta
                    improved = True
        
        return pts, best

    # ---- Simulated annealing with fast evaluation ----
    def anneal_fast(pts, max_time):
        pts = list(pts)
        pt_set = set(pts)
        cur_e = count_edges_fast(pts)
        best_pts = list(pts)
        best_e = cur_e
        T = 3.0
        T_min = 0.01
        cooldown = 0.9995
        
        def get_point_edges(p, exclude_idx):
            e = 0
            for da, db in EDGES:
                q = (p[0] + da, p[1] + db)
                if q in pt_set:
                    # Check if this is the point we're evaluating
                    idx = pts.index(q) if q in pts else -1
                    if idx != exclude_idx:
                        e += 1
            return e
        
        while time.time() < max_time:
            i = random.randrange(32)
            old = pts[i]
            old_edges = get_point_edges(old, i)
            
            # Generate candidate
            if random.random() < 0.8:
                da = random.randint(-2, 2)
                db = random.randint(-2, 2)
            else:
                da = random.randint(-4, 4)
                db = random.randint(-4, 4)
            cand = (old[0] + da, old[1] + db)
            
            if abs(cand[0]) > 64 or abs(cand[1]) > 64 or cand in pt_set:
                T = max(T_min, T * cooldown)
                continue
            
            # Check validity
            valid = True
            for da2, db2 in EDGES:
                neighbor = (cand[0] + da2, cand[1] + db2)
                if neighbor in pt_set:
                    idx = pts.index(neighbor) if neighbor in pts else -1
                    if idx != i:
                        valid = False
                        break
            
            if not valid:
                T = max(T_min, T * cooldown)
                continue
            
            # Count new edges
            new_edges = 0
            for da2, db2 in EDGES:
                q = (cand[0] + da2, cand[1] + db2)
                if q in pt_set and q != old:
                    new_edges += 1
            
            delta = new_edges - old_edges
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                pt_set.remove(old)
                pts[i] = cand
                pt_set.add(cand)
                cur_e += delta
                if cur_e > best_e:
                    best_e = cur_e
                    best_pts = list(pts)
            
            T = max(T_min, T * cooldown)
        
        return best_pts, best_e

    # ---- Main ----
    # Try multiple seeds and use the best one
    seeds = [build_hex_seed(), build_greedy_seed(), build_spiral_seed()]
    
    valid_seeds = [s for s in seeds if is_valid(s)]
    if not valid_seeds:
        valid_seeds = [build_greedy_seed()]
    
    # Pick the best seed by edge count
    best_seed = max(valid_seeds, key=lambda s: count_edges_fast(s))
    
    # Initial local search
    best_pts, best_e = local_search_fast(best_seed, deadline - 30.0)
    
    if best_e == 76:
        return [[a, b] for a, b in best_pts]

    # Multiple annealing passes with restarts
    while time.time() < deadline - 15.0:
        # Animate from current best
        cand_pts, cand_e = anneal_fast(best_pts, deadline - 15.0)
        if cand_e > best_e:
            best_pts, best_e = cand_pts, cand_e
            if best_e == 76:
                break
            # Polish with local search
            polished, pe = local_search_fast(best_pts, deadline - 15.0)
            if pe > best_e:
                best_pts, best_e = polished, pe
                if best_e == 76:
                    break
        else:
            # Try a fresh seed occasionally
            fresh = random.choice([build_hex_seed(), build_greedy_seed(), build_spiral_seed()])
            if is_valid(fresh):
                fp, fe = local_search_fast(fresh, deadline - 15.0)
                if fe > best_e:
                    best_pts, best_e = fp, fe
                    if best_e == 76:
                        break

    if not is_valid(best_pts):
        return [[a, b] for a in range(16) for b in range(2)]

    return [[a, b] for a, b in best_pts]
# EVOLVE-BLOCK-END
