import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_pts = None
    best_score = -1
    
    def is_valid(pts):
        n = len(pts)
        if len(set(pts)) != n:
            return False
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) - dy * (x3 - x1) == 0:
                        count += 1
                if count >= 4:
                    return False
        return True

    def count_triples(pts):
        n = len(pts)
        count = 0
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) - dy * (x3 - x1) == 0:
                        count += 1
        return count

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def find_initial_valid():
        for _ in range(500):
            pts = [random_point() for _ in range(20)]
            if is_valid(pts):
                return pts
        return None

    # Phase 1: Find initial valid config
    init_pts = find_initial_valid()
    if init_pts:
        best_pts = [p[:] for p in init_pts]
        best_score = count_triples(best_pts)

    if best_pts is None:
        return [[x, x * x] for x in range(-10, 10)]

    # Phase 2: Local Search with Restarts
    current_pts = [p[:] for p in best_pts]
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        i = random.randint(0, 19)
        old_p = current_pts[i][:]
        new_p = random_point()
        
        if new_p == old_p:
            continue
            
        current_pts[i] = new_p
        
        if is_valid(current_pts):
            new_score = count_triples(current_pts)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_pts = [p[:] for p in current_pts]
                no_improve_count = 0
            else:
                current_pts[i] = old_p
                no_improve_count += 1
        else:
            current_pts[i] = old_p
            no_improve_count += 1

        if no_improve_count > max_no_improve:
            # Restart from best or new random
            if random.random() < 0.5 and best_pts:
                current_pts = [p[:] for p in best_pts]
                current_score = best_score
            else:
                new_init = find_initial_valid()
                if new_init:
                    current_pts = [p[:] for p in new_init]
                    current_score = count_triples(current_pts)
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = [p[:] for p in current_pts]
            no_improve_count = 0

    return best_pts
