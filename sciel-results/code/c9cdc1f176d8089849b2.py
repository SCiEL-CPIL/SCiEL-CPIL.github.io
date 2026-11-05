import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def count_triples(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def has_four_collinear(points):
        n = len(points)
        # Efficient check: for each pair, count points on that line
        # But n=20 is small, O(n^4) is 48450 checks, very fast.
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if is_collinear(points[i], points[j], points[k]) and is_collinear(points[i], points[j], points[l]):
                            return True
        return False

    def generate_initial_points():
        # Start with a 4x5 grid structure which has many collinearities
        pts = []
        for x in range(4):
            for y in range(5):
                pts.append([x * 10, y * 10])
        return pts

    # Strategy: 
    # 1. Generate a valid starting point (no 4 collinear) by perturbing a grid.
    # 2. Use local search to MINIMIZE the number of collinear triples.
    #    The objective is to minimize T = number of collinear triples.
    #    This is equivalent to maximizing the independence number (no 3 collinear subset).
    #    Wait, the score is the LARGEST subset with no three collinear.
    #    Minimizing the score means making it hard to find a large subset with no 3 collinear.
    #    This is equivalent to MAXIMIZING the density of collinear triples.
    #    
    #    Let's re-read the task carefully.
    #    "The score is the largest subset with no three collinear (exact branch and bound). Minimize."
    #    
    #    So we want to MINIMIZE the size of the largest 3-collinear-free subset.
    #    This means we want the points to be "as collinear as possible" while maintaining the constraint "no four collinear".
    #    
    #    Therefore, the objective function for local search should be to MAXIMIZE the number of collinear triples.
    #    
    #    Current program does exactly this: "We want to MAXIMIZE triples".
    #    
    #    Why is the current program stuck at 1.1765?
    #    Maybe the search space is rugged, or the initial grid is too constrained.
    #    
    #    Alternative Strategy: 
    #    Instead of random mutation of points, use a "line-based" construction.
    #    Or, use a different heuristic: 
    #    Start with a configuration known to have high triple density.
    #    
    #    Let's try a different approach: 
    #    Use a grid of size 5x4 but with slightly perturbed coordinates to break 4-collinearities.
    #    Then, use a local search that swaps points or moves points to "better" positions on existing lines.
    #    
    #    Actually, the most effective way to maximize triples without 4-collinear is to cluster points on lines.
    #    For example, have 5 lines with 4 points each. 5 * 4 = 20 points.
    #    If we can arrange 5 lines such that no 4 points are collinear (which is impossible if a line has 4 points), 
    #    wait, the constraint is "no four collinear". So each line can have at most 3 points.
    #    
    #    If each line has at most 3 points, the maximum number of triples is limited.
    #    Total points N=20.
    #    If we have k lines, each with 3 points, and they intersect in a way that minimizes overlap of points?
    #    No, we want to MAXIMIZE triples.
    #    A triple is defined by 3 points. If 3 points are collinear, they form a triple.
    #    If we have a line with 3 points, it contributes 1 triple.
    #    If we have a line with 4 points, it contributes 4 triples (choose 3 from 4), but it is INVALID (4 collinear).
    #    
    #    So, to maximize triples under the constraint "no 4 collinear", we should pack as many points as possible onto lines of size 3.
    #    Can we have disjoint lines of size 3?
    #    20 points / 3 = 6 lines with 2 points left over.
    #    If the lines are disjoint, we have 6 triples.
    #    But lines in a plane usually intersect.
    #    
    #    Consider a configuration where points are arranged on a grid.
    #    In a 4x5 grid:
    #    Horizontal lines: 4 lines of 5 points. Invalid.
    #    Vertical lines: 5 lines of 4 points. Invalid.
    #    
    #    We need to perturb the grid to break the 4-collinearities.
    #    If we perturb slightly, we might break the 4-collinearities but keep the 3-collinearities?
    #    No, if we break the 4-collinearity by moving one point, we break the line.
    #    
    #    Let's think about the structure of "no 4 collinear" sets.
    #    These are known as "3-general position" sets.
    #    The problem is to minimize the 3-free subset size.
    #    
    #    Let's try a different search method: 
    #    Instead of moving a random point to a random location, 
    #    try to "fix" a point to a position that creates the most new triples.
    #    
    #    New Strategy:
    #    1. Generate a random valid set (no 4 collinear).
    #    2. For each point p_i, try moving it to every possible integer coordinate in a local neighborhood.
    #    3. Calculate the change in the number of triples.
    #    4. Accept the move that maximizes the number of triples.
    #    5. Repeat until no improvement is found.
    #    6. Use simulated annealing to escape local optima.
    
    def get_score(points):
        if has_four_collinear(points):
            return -1000000
        return count_triples(points)

    # Generate initial points: A perturbed grid
    def generate_valid_initial():
        pts = []
        # 4x5 grid
        for x in range(4):
            for y in range(5):
                pts.append([x * 10, y * 10])
        
        # Perturb to remove 4-collinearities
        # We will use a greedy approach: 
        # Start with the grid. 
        # While there is a 4-collinear set, move one of the points slightly.
        
        current_pts = pts
        valid = False
        attempts = 0
        while not valid and attempts < 1000:
            if not has_four_collinear(current_pts):
                valid = True
            else:
                # Find a 4-collinear set
                found = False
                for i in range(20):
                    for j in range(i+1, 20):
                        for k in range(j+1, 20):
                            for l in range(k+1, 20):
                                if is_collinear(current_pts[i], current_pts[j], current_pts[k]) and is_collinear(current_pts[i], current_pts[j], current_pts[l]):
                                    # Move point i slightly
                                    dx = random.randint(-2, 2)
                                    dy = random.randint(-2, 2)
                                    if dx == 0 and dy == 0:
                                        dx = 1
                                    new_p = [current_pts[i][0] + dx, current_pts[i][1] + dy]
                                    if 0 <= new_p[0] <= 100 and 0 <= new_p[1] <= 100: # Keep in bounds, actually bounds are -100 to 100
                                        if new_p not in current_pts:
                                            current_pts[i] = new_p
                                            found = True
                                            break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                attempts += 1
        return current_pts

    # Local Search: Hill Climbing with restarts
    def local_search(points, time_limit):
        current_pts = [p[:] for p in points]
        current_score = get_score(current_pts)
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        while time.time() < time_limit:
            improved = False
            # Try moving each point to a nearby location
            for i in range(20):
                if time.time() > time_limit:
                    break
                old_p = current_pts[i]
                # Try a small set of moves
                # Random moves in a box of size 5
                moves = []
                for _ in range(20):
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    new_p = [old_p[0] + dx, old_p[1] + dy]
                    if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                        if new_p not in current_pts:
                            moves.append(new_p)
                
                for new_p in moves:
                    test_pts = current_pts[:]
                    test_pts[i] = new_p
                    new_score = get_score(test_pts)
                    if new_score > current_score:
                        current_pts = test_pts
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best_pts = [p[:] for p in current_pts]
                        break # Accept first improvement and restart point loop? Or continue?
                        # Restarting point loop is safer for hill climbing
            if not improved:
                # No improvement, break or do a random jump
                # Random jump to escape local optimum
                idx = random.randint(0, 19)
                old_p = current_pts[idx]
                new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                if new_p not in current_pts:
                    test_pts = current_pts[:]
                    test_pts[idx] = new_p
                    new_score = get_score(test_pts)
                    if new_score > -1000000: # Accept if valid, even if score drops? 
                        # Only accept if it's not too bad? 
                        # For now, accept if valid.
                        current_pts = test_pts
                        current_score = new_score
                # If stuck, break and restart from best
                break
        
        return best_pts

    # Run multiple restarts
    best_overall = None
    best_overall_score = -1
    
    # Initial point
    init_pts = generate_valid_initial()
    
    # Run local search
    t_end = deadline
    while time.time() < t_end - 5: # Leave 5s for return
        current_best = local_search(init_pts, min(time.time() + 5, t_end - 5))
        score = get_score(current_best)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = current_best
        
        # Generate a new initial point for next restart
        if time.time() < t_end - 10:
            init_pts = generate_valid_initial()
        else:
            break
            
    return best_overall
