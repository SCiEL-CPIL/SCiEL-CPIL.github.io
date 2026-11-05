import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    def line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, p1[0], p1[1])
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)

    def count_triples(points):
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1
        triples = 0
        for key, count in line_counts.items():
            if count >= 3:
                triples += count * (count - 1) * (count - 2) // 6
            if count >= 4:
                return -1000000 # Invalid
        return triples

    def is_valid_move(points, idx, new_pos):
        # Check if placing new_pos at idx creates 4 collinear
        # Temporarily replace
        # Check all lines through new_pos and any other point j
        # If a line through new_pos and j contains 2 other points k, l (k,l != idx, j), then invalid.
        # Actually, if line contains new_pos, j, k, l, then 4 points.
        # So for each j != idx:
        #   count points on line(new_pos, j) excluding idx.
        #   If count >= 3, then new_pos + j + 2 others = 4 points. Invalid.
        
        n = len(points)
        for j in range(n):
            if j == idx: continue
            key = line_key(new_pos, points[j])
            # Count how many points (excluding idx) are on this line
            cnt = 1 # for points[j]
            for k in range(n):
                if k == idx or k == j: continue
                if line_key(new_pos, points[k]) == key:
                    cnt += 1
            if cnt >= 3:
                return False
        return True

    def greedy_line_packing(seed):
        rng = random.Random(seed)
        points = []
        used = set()
        
        # Start with a few random points
        for _ in range(5):
            p = (rng.randint(-100, 100), rng.randint(-100, 100))
            if p not in used:
                points.append(p)
                used.add(p)
                
        for step in range(15):
            if time.time() > deadline - 5: break
            
            # Find best position to add
            best_gain = -1
            best_pos = None
            best_cands = []
            
            # Candidate generation:
            # 1. Extensions of existing pairs
            cands = set()
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    p1, p2 = points[i], points[j]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    g = gcd(abs(dx), abs(dy))
                    if g == 0: continue
                    dx_n = dx // g
                    dy_n = dy // g
                    for t in range(1, 10):
                        for sign in [1, -1]:
                            np = (p2[0] + sign*t*dx_n, p2[1] + sign*t*dy_n)
                            if -100 <= np[0] <= 100 and -100 <= np[1] <= 100:
                                cands.add(np)
                            np = (p1[0] - sign*t*dx_n, p1[1] - sign*t*dy_n)
                            if -100 <= np[0] <= 100 and -100 <= np[1] <= 100:
                                cands.add(np)
            
            # 2. Random
            for _ in range(100):
                cands.add((rng.randint(-100, 100), rng.randint(-100, 100)))
                
            for pos in cands:
                if pos in used: continue
                # Check validity (no 4 collinear)
                # Append pos to points, check, remove
                test_pts = points + [pos]
                # Quick check: count triples and check 4
                # We can just check if any line has 4
                line_counts = {}
                for i in range(len(test_pts)):
                    for j in range(i+1, len(test_pts)):
                        key = line_key(test_pts[i], test_pts[j])
                        line_counts[key] = line_counts.get(key, 0) + 1
                        if line_counts[key] > 3:
                            break
                    if any(v > 3 for v in line_counts.values()):
                        break
                if any(v > 3 for v in line_counts.values()):
                    continue
                
                # Calculate gain in triples
                # Triples involving pos
                gain = 0
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        key = line_key(points[i], points[j])
                        # Check if pos is on this line
                        dx = points[j][0] - points[i][0]
                        dy = points[j][1] - points[i][1]
                        if (pos[0] - points[i][0]) * dy == (pos[1] - points[i][1]) * dx:
                            # pos is on line(i,j)
                            # This line now has 3 points. It contributes 1 new triple (i,j,pos)
                            # But we must ensure it didn't already have 3 (which would make 4, already checked)
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [pos]
                elif gain == best_gain:
                    best_cands.append(pos)
                    
            if not best_cands:
                # Pick random valid
                while True:
                    pos = (rng.randint(-100, 100), rng.randint(-100, 100))
                    if pos in used: continue
                    test_pts = points + [pos]
                    line_counts = {}
                    valid = True
                    for i in range(len(test_pts)):
                        for j in range(i+1, len(test_pts)):
                            key = line_key(test_pts[i], test_pts[j])
                            line_counts[key] = line_counts.get(key, 0) + 1
                            if line_counts[key] > 3:
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        break
                points.append(pos)
                used.add(pos)
            else:
                pos = rng.choice(best_cands)
                points.append(pos)
                used.add(pos)
                
        return points

    def simulated_annealing(init_pts, seed, time_limit):
        rng = random.Random(seed)
        current_pts = [p[:] for p in init_pts]
        current_score = count_triples(current_pts)
        if current_score < 0: return None
        
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            # Move
            idx = rng.randint(0, 19)
            new_pos = [rng.randint(-100, 100), rng.randint(-100, 100)]
            
            if not is_valid_move(current_pts, idx, new_pos):
                continue
                
            # Calculate new score
            # We can optimize this, but n=20 is small
            test_pts = [p[:] for p in current_pts]
            test_pts[idx] = new_pos
            new_score = count_triples(test_pts)
            
            if new_score < 0:
                continue
                
            delta = new_score - current_score
            if delta > 0 or rng.random() < pow(2.718, delta / T):
                current_pts = test_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = [p[:] for p in current_pts]
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        return best_pts

    best_overall = None
    best_overall_score = -1
    seed = 0
    
    while time.time() < deadline - 5:
        init = greedy_line_packing(seed)
        if init is None:
            seed += 1
            continue
            
        score = count_triples(init)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = init
            
        # Refine
        if time.time() < deadline - 2:
            t_left = deadline - 2 - time.time()
            sa_time = min(2.0, t_left)
            if sa_time > 0.5:
                refined = simulated_annealing(init, seed, sa_time)
                if refined:
                    ref_score = count_triples(refined)
                    if ref_score > best_overall_score:
                        best_overall_score = ref_score
                        best_overall = refined
        seed += 1
        
    return best_overall
