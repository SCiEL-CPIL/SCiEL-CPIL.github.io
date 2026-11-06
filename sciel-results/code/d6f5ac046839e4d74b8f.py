import itertools
import random
import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for safety

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

    def get_triples(points):
        """Returns a list of collinear triples (indices)."""
        triples = []
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    def count_triples(points):
        """Returns the number of collinear triples."""
        return len(get_triples(points))

    def has_4_collinear(points):
        """Checks if any 4 points are collinear."""
        n = len(points)
        # For each pair, count how many other points are on the line
        # Optimization: We can break early if count >= 4
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
                        if count >= 4:
                            return True
        return False

    def min_vertex_cover_3uniform(triples, n_points):
        """
        Exact branch and bound for minimum vertex cover of 3-uniform hypergraph.
        Returns the size of the minimum vertex cover.
        """
        if not triples:
            return 0
        
        # Convert to set of tuples for fast lookup
        triples_set = set(triples)
        best_cover = n_points
        
        # Greedy heuristic for initial upper bound
        def greedy_cover(triples_list):
            if not triples_list:
                return 0
            remaining = set(triples_list)
            cover = 0
            while remaining:
                # Count frequency of each point in remaining triples
                counts = {i: 0 for i in range(n_points)}
                for t in remaining:
                    for p in t:
                        counts[p] += 1
                # Pick point with max count
                best_point = max(counts, key=counts.get)
                cover += 1
                # Remove all triples containing best_point
                remaining = {t for t in remaining if best_point not in t}
            return cover
        
        best_cover = min(best_cover, greedy_cover(triples_set))
        
        # Backtracking
        # State: remaining_triples, current_cover
        # Pruning: if current_cover >= best_cover, return
        # Branching: pick a triple, branch on its 3 vertices
        
        # To improve speed, we can order triples by "criticality" or just use the first one.
        # For n=20, the number of triples is usually manageable.
        
        def backtrack(remaining_triples, current_cover):
            nonlocal best_cover
            if not remaining_triples:
                if current_cover < best_cover:
                    best_cover = current_cover
                return
            
            if current_cover >= best_cover:
                return
                
            # Pick a triple to branch on
            # Heuristic: pick a triple with the most "constrained" points?
            # Or just the first one.
            t = next(iter(remaining_triples))
            
            # Branch 1: Pick t[0]
            new_remaining_1 = {rt for rt in remaining_triples if t[0] not in rt}
            backtrack(new_remaining_1, current_cover + 1)
            
            # Branch 2: Pick t[1]
            new_remaining_2 = {rt for rt in remaining_triples if t[1] not in rt}
            backtrack(new_remaining_2, current_cover + 1)
            
            # Branch 3: Pick t[2]
            new_remaining_3 = {rt for rt in remaining_triples if t[2] not in rt}
            backtrack(new_remaining_3, current_cover + 1)

        backtrack(triples_set, 0)
        return best_cover

    def calculate_exact_score(points):
        """
        Returns the size of the largest subset with no 3 collinear.
        This is n_points - min_vertex_cover_size.
        """
        if has_4_collinear(points):
            return None # Invalid configuration
        triples = get_triples(points)
        cover_size = min_vertex_cover_3uniform(triples, 20)
        return 20 - cover_size

    def generate_initial_config():
        # Start with a random set of points that is likely valid
        # A grid is bad. A random set is better.
        # We can also try a specific structured set known to be dense.
        # Let's start with a random set and fix it.
        pts = []
        seen = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        return pts

    def find_snap_moves(points, i, radius=10):
        """
        Find moves for point i that create new triples.
        For each pair (j, k) not collinear with i, find a point on line(j, k)
        close to i that is not occupied and doesn't create 4-collinear.
        """
        moves = []
        p_i = points[i]
        n = len(points)
        
        # Pre-compute occupied points for fast lookup
        occupied = set(tuple(p) for p in points)
        
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                
                p_j = points[j]
                p_k = points[k]
                
                # Check if i is already on line(j, k)
                if are_collinear(p_i, p_j, p_k):
                    continue
                
                # We want to move p_i to a point (nx, ny) on the line defined by p_j and p_k
                # Line equation: (y - y_j) * (x_k - x_j) = (x - x_j) * (y_k - y_j)
                # Let dx = x_k - x_j, dy = y_k - y_j
                # (ny - y_j) * dx = (nx - x_j) * dy
                
                # We search for integer solutions (nx, ny) near p_i
                # Let nx = p_i[0] + dx_off, ny = p_i[1] + dy_off
                # (p_i[1] + dy_off - y_j) * dx = (p_i[0] + dx_off - x_j) * dy
                
                # This is a linear Diophantine equation.
                # Alternatively, just iterate over a small grid around p_i and check collinearity.
                # This is O(radius^2) per pair. Pairs = ~190. Radius=10 -> 400 checks. 190*400 = 76000 ops. Fast.
                
                for dx_off in range(-radius, radius + 1):
                    nx = p_i[0] + dx_off
                    if nx < -100 or nx > 100:
                        continue
                    for dy_off in range(-radius, radius + 1):
                        ny = p_i[1] + dy_off
                        if ny < -100 or ny > 100:
                            continue
                        
                        new_p = [nx, ny]
                        if (nx, ny) in occupied:
                            continue
                        
                        # Check if new_p is collinear with p_j and p_k
                        if not are_collinear(new_p, p_j, p_k):
                            continue
                        
                        # Check if this move creates a 4-collinear situation
                        # We need to check if any other point l is on the line (new_p, p_j) or (new_p, p_k) or (p_j, p_k)?
                        # Actually, if new_p, p_j, p_k are collinear, we just need to ensure no 4th point is on that line.
                        # And we need to ensure new_p doesn't create 4-collinear with any other pair.
                        
                        # Efficient check:
                        # 1. Check line (new_p, p_j): count points on it.
                        # 2. Check line (new_p, p_k): count points on it.
                        # 3. Check line (p_j, p_k): count points on it. (Should be 3: new_p, p_j, p_k)
                        
                        # Actually, the only new lines involving new_p are (new_p, l) for all l.
                        # We only care if any line has >= 4 points.
                        
                        # Let's just do a full check for 4-collinear with the new point set.
                        # It's expensive but radius is small, so few candidates.
                        
                        # Optimization: 
                        # The line containing new_p, p_j, p_k must have exactly 3 points.
                        # So, for any other point l, are_collinear(new_p, p_j, l) must be False,
                        # and are_collinear(new_p, p_k, l) must be False.
                        # Also, new_p must not be collinear with any other pair (m, n) such that the line has >= 2 other points.
                        
                        # Let's just construct the new points list and check has_4_collinear.
                        # Since we only move one point, we can optimize has_4_collinear.
                        
                        new_points = points.copy()
                        new_points[i] = new_p
                        
                        if has_4_collinear(new_points):
                            continue
                        
                        # This move is valid.
                        # Calculate the change in triple count.
                        # Old triples involving i:
                        # New triples involving i:
                        # We can just count triples in new_points.
                        # But that's O(N^3) = 8000 ops.
                        # Since we have many candidates, we should be smarter.
                        
                        # However, for now, let's just collect valid moves and score them.
                        moves.append((new_p, new_points))
        
        return moves

    # Main Search Loop
    best_score = 20
    best_points = None
    best_surrogate = -1
    
    # Initialize
    current_points = generate_initial_config()
    
    # Ensure initial config is valid
    for _ in range(100):
        if not has_4_collinear(current_points):
            break
        current_points = generate_initial_config()
    
    if has_4_collinear(current_points):
        # Fallback: random until valid
        while True:
            current_points = generate_initial_config()
            if not has_4_collinear(current_points):
                break

    current_surrogate = count_triples(current_points)
    current_exact_score = calculate_exact_score(current_points)
    if current_exact_score is not None:
        best_score = current_exact_score
        best_points = current_points
        best_surrogate = current_surrogate

    while time.time() - start_time < time_limit:
        improved = False
        
        # Local Search: Try to improve surrogate score (triple count)
        for i in range(20):
            if time.time() - start_time > time_limit:
                break
                
            moves = find_snap_moves(current_points, i, radius=5)
            
            if not moves:
                continue
                
            # Find the move that maximizes triple count
            best_move = None
            best_move_score = -1
            
            for new_p, new_points in moves:
                # Calculate triple count
                # Optimization: We can calculate the delta.
                # But count_triples is fast enough for small number of moves.
                score = count_triples(new_points)
                if score > best_move_score:
                    best_move_score = score
                    best_move = (new_p, new_points)
            
            if best_move and best_move_score > current_surrogate:
                current_points = best_move[1]
                current_surrogate = best_move_score
                improved = True
                
                # Update best if exact score is better
                # We don't want to calculate exact score every step as it's slow.
                # But we should check periodically.
                if random.random() < 0.1: # Check exact score 10% of the time
                    exact_score = calculate_exact_score(current_points)
                    if exact_score is not None and exact_score < best_score:
                        best_score = exact_score
                        best_points = current_points
                        best_surrogate = current_surrogate
                else:
                    # Assume surrogate improvement correlates with exact score improvement
                    # But we must ensure validity.
                    pass
        
        if not improved:
            # Stuck in local optimum. Perturb.
            # Move 2-3 points randomly
            new_points = current_points.copy()
            for _ in range(random.randint(2, 4)):
                idx = random.randint(0, 19)
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                nx = new_points[idx][0] + dx
                ny = new_points[idx][1] + dy
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_points[idx] = [nx, ny]
            
            # Check validity
            if len(set(tuple(p) for p in new_points)) == 20 and not has_4_collinear(new_points):
                new_surrogate = count_triples(new_points)
                # Accept if it's not much worse, or if it's better
                if new_surrogate >= current_surrogate - 2:
                    current_points = new_points
                    current_surrogate = new_surrogate
                    
                    exact_score = calculate_exact_score(current_points)
                    if exact_score is not None and exact_score < best_score:
                        best_score = exact_score
                        best_points = current_points
                        best_surrogate = current_surrogate

    # Final check
    final_score = calculate_exact_score(best_points)
    if final_score is not None and final_score < best_score:
        best_score = final_score

    return best_points

# EVOLVE-BLOCK-END
