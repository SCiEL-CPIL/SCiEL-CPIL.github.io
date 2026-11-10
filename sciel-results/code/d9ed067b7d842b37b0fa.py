import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s kill

    def collinear3(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def has_4_collinear(points):
        n = len(points)
        # For each pair, count how many other points are collinear
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if collinear3(points[i], points[j], points[k]):
                            count += 1
                if count >= 4:
                    return True
        return False

    def compute_independence_number(points):
        """Compute the largest subset with no three collinear using greedy + local search."""
        n = len(points)
        
        # Build conflict graph: edge between i,j if they're part of some triple
        # Actually, we need: a set S is valid if no three points in S are collinear.
        # This is a 3-uniform hypergraph independent set problem.
        
        # Greedy: repeatedly add a point that doesn't create a collinear triple
        # Try multiple random orderings
        best_size = 0
        
        for trial in range(10):
            random_order = list(range(n))
            random.shuffle(random_order)
            subset = []
            for idx in random_order:
                valid = True
                for a in range(len(subset)):
                    for b in range(a + 1, len(subset)):
                        if collinear3(points[idx], points[subset[a]], points[subset[b]]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    subset.append(idx)
            best_size = max(best_size, len(subset))
        
        # Local search on the best subset found
        # Try to add points not in subset, remove points to allow additions
        # This is complex; use a simple approach: try all single swaps
        
        return best_size

    def fast_score(points):
        """Fast approximate score."""
        n = len(points)
        # Greedy with fixed ordering
        subset = []
        for i in range(n):
            valid = True
            for a in range(len(subset)):
                for b in range(a + 1, len(subset)):
                    if collinear3(points[i], points[subset[a]], points[subset[b]]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                subset.append(i)
        return len(subset)

    # Initial configuration: try structured points
    random.seed(42)
    
    def make_initial():
        # Use a perturbed grid
        pts = []
        # 5x4 grid perturbed
        for i in range(5):
            for j in range(4):
                x = -40 + i * 20 + random.randint(-3, 3)
                y = -30 + j * 20 + random.randint(-3, 3)
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                pts.append([x, y])
        # Ensure distinct
        seen = set()
        unique_pts = []
        for p in pts:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                unique_pts.append(p)
        while len(unique_pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_pts.append([x, y])
        return unique_pts[:20]

    best_pts = None
    best_score = 20

    # Try several initial configurations
    for init_trial in range(3):
        if time.time() > deadline:
            break
        
        if init_trial == 0:
            # Structured: points near a circle
            pts = []
            for i in range(20):
                angle = 2 * math.pi * i / 20
                r = 50 + random.randint(-10, 10)
                x = int(round(r * math.cos(angle)))
                y = int(round(r * math.sin(angle)))
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                pts.append([x, y])
            # Ensure distinct
            seen = set()
            unique_pts = []
            for p in pts:
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    unique_pts.append(p)
            while len(unique_pts) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in seen:
                    seen.add((x, y))
                    unique_pts.append([x, y])
            pts = unique_pts[:20]
        else:
            pts = make_initial()

        if has_4_collinear(pts):
            continue

        score = fast_score(pts)
        if score < best_score:
            best_score = score
            best_pts = [p[:] for p in pts]

        # Simulated annealing
        current = [p[:] for p in pts]
        current_score = score
        T = 5.0
        cooling = 0.995

        for step in range(5000):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = random.randint(0, 19)
            old_x, old_y = current[idx]
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            if new_x == old_x and new_y == old_y:
                continue
            
            candidate = [p[:] for p in current]
            candidate[idx] = [new_x, new_y]
            
            # Check distinctness
            if len(set(map(tuple, candidate))) != 20:
                continue
            
            # Check no 4 collinear
            if has_4_collinear(candidate):
                continue
            
            new_score = fast_score(candidate)
            delta = new_score - current_score
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current = candidate
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_pts = [p[:] for p in current]
            
            T *= cooling
            if T < 0.1:
                T = 0.1

    if best_pts is None:
        # Fallback: simple valid configuration
        best_pts = [
            [0, 0], [1, 0], [2, 1], [3, 0], [4, 1],
            [0, 2], [1, 3], [2, 2], [3, 4], [4, 3],
            [5, 0], [6, 1], [7, 0], [8, 2], [9, 1],
            [10, 3], [11, 2], [12, 4], [13, 3], [14, 5]
        ]
        # Verify and fix if needed
        if has_4_collinear(best_pts):
            # Generate random valid
            while True:
                best_pts = []
                seen = set()
                for _ in range(20):
                    while True:
                        x = random.randint(-100, 100)
                        y = random.randint(-100, 100)
                        if (x, y) not in seen:
                            break
                    seen.add((x, y))
                    best_pts.append([x, y])
                if not has_4_collinear(best_pts):
                    break

    return best_pts
