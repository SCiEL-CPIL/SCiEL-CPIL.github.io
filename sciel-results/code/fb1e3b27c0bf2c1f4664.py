import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 15.0  # 15 seconds is plenty for N=32
    
    # Hexagonal spiral generation
    def generate_spiral(n):
        points = [(0, 0)]
        if n <= 1:
            return points
        
        dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        step_len = 1
        dir_idx = 0
        
        while len(points) < n:
            for _ in range(2):
                dx, dy = dirs[dir_idx]
                for _ in range(step_len):
                    if len(points) >= n:
                        break
                    last_a, last_b = points[-1]
                    points.append((last_a + dx, last_b + dy))
                dir_idx = (dir_idx + 1) % 6
            step_len += 1
        return points

    # Distance squared in hex lattice (axial coordinates a, b)
    # d^2 = a^2 + a*b + b^2
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db

    # Count edges (pairs with dist_sq == 1)
    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Validity check: distinct, |a|,|b| <= 64
    def is_valid(points):
        if len(points) != 32:
            return False
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
            if abs(p[0]) > 64 or abs(p[1]) > 64:
                return False
        return True

    # Local Search
    def local_search(initial_points, max_steps=100000):
        current_points = [tuple(p) for p in initial_points]
        current_score = count_edges(current_points)
        best_points = current_points[:]
        best_score = current_score
        
        # Neighbors for moving a point
        neighbors = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        
        steps = 0
        stuck_count = 0
        
        while steps < max_steps and (time.time() - start_time) < deadline:
            steps += 1
            if current_score == 76:
                break
                
            # Pick a random point to move
            idx = random.randint(0, 31)
            p = current_points[idx]
            
            # Try moving to a random neighbor
            # To be more efficient, we can try all 6 neighbors and pick the best one
            best_new_score = current_score
            best_new_pos = None
            
            point_set = set(current_points)
            
            for dx, dy in neighbors:
                new_p = (p[0] + dx, p[1] + dy)
                if new_p in point_set:
                    continue
                if abs(new_p[0]) > 64 or abs(new_p[1]) > 64:
                    continue
                
                # Calculate delta score
                # Remove edges from p
                old_edges = 0
                for j, q in enumerate(current_points):
                    if j == idx: continue
                    if dist_sq(p, q) == 1:
                        old_edges += 1
                        
                # Add edges to new_p
                new_edges = 0
                for j, q in enumerate(current_points):
                    if j == idx: continue
                    if dist_sq(new_p, q) == 1:
                        new_edges += 1
                        
                delta = new_edges - old_edges
                if delta > 0:
                    new_score = current_score + delta
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_pos = new_p

            if best_new_pos is not None:
                # Update current points
                current_points[idx] = best_new_pos
                current_score = best_new_score
                stuck_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points[:]
            else:
                stuck_count += 1
                # If stuck for too long, try a random jump or restart
                if stuck_count > 100:
                    # Random jump: move a random point to a random empty spot within radius 3
                    jump_idx = random.randint(0, 31)
                    p = current_points[jump_idx]
                    point_set = set(current_points)
                    candidates = []
                    for da in range(-3, 4):
                        for db in range(-3, 4):
                            if da == 0 and db == 0: continue
                            new_p = (p[0] + da, p[1] + db)
                            if new_p not in point_set and abs(new_p[0]) <= 64 and abs(new_p[1]) <= 64:
                                candidates.append(new_p)
                    if candidates:
                        new_p = random.choice(candidates)
                        # Calculate new score
                        old_edges = 0
                        for j, q in enumerate(current_points):
                            if j == jump_idx: continue
                            if dist_sq(p, q) == 1:
                                old_edges += 1
                        new_edges = 0
                        for j, q in enumerate(current_points):
                            if j == jump_idx: continue
                            if dist_sq(new_p, q) == 1:
                                new_edges += 1
                        new_score = current_score + (new_edges - old_edges)
                        
                        # Accept if it's not much worse (simulated annealing style) or just accept to escape local optima
                        # For simplicity, accept if it's better or equal, or with small prob if worse
                        if new_score >= current_score or random.random() < 0.05:
                            current_points[jump_idx] = new_p
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_points = current_points[:]
                    stuck_count = 0

        return best_points, best_score

    # Generate initial seed
    seed = generate_spiral(32)
    
    # Run local search
    best_points, best_score = local_search(seed)
    
    # If we didn't reach 76, try a few more restarts with perturbed seeds
    if best_score < 76 and (time.time() - start_time) < deadline - 2.0:
        for _ in range(5):
            # Perturb seed: swap two random points in the spiral
            perturbed = seed[:]
            i, j = random.sample(range(32), 2)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            # Or just move one point randomly
            # Let's try a different initial configuration: 
            # Random dense packing
            if random.random() < 0.5:
                # Generate a random set of 32 points in a hexagon of radius 5
                points = [(0,0)]
                while len(points) < 32:
                    a = random.randint(-5, 5)
                    b = random.randint(-5, 5)
                    if (a,b) not in points:
                        points.append((a,b))
                perturbed = points
            
            p_best, s_best = local_search(perturbed, max_steps=50000)
            if s_best > best_score:
                best_points = p_best
                best_score = s_best
            if best_score == 76:
                break

    # Final validation and formatting
    if not is_valid(best_points):
        # Fallback to spiral if search failed badly
        best_points = generate_spiral(32)
        
    return [[a, b] for a, b in best_points]
