import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Deterministic Construction: Layered Method
    # Choose 20 distinct x-coordinates (primes for simplicity/spacing)
    # Choose y-coordinates such that no 3 points are collinear.
    # A known method: y_i = i^2 mod p for large prime p, or simply y_i = i^2 if we ensure x_i are distinct and we check.
    # Actually, the simplest "no 3 collinear" set is points on the parabola y=x^2.
    # Let's use x = 0..19, y = x^2.
    # Check bounds: x in [0,19], y in [0,361]. y > 100.
    # We need |y| <= 100.
    # Let's use smaller x.
    # If we use x in [-10, 9], y = x^2/4? No, integer points.
    # Let's use a known small "no 3 collinear" set or construct one.
    # Parabola y=x^2 with x in {-7, -6, ..., 6, 7} -> 15 points. y max 49.
    # We need 20 points.
    # We can use two parabolas? Or a different curve.
    # Or just use random search to find a valid "no 3 collinear" set first?
    # The task says "no four collinear" is the constraint. The score is the largest subset with NO THREE collinear.
    # Wait, the constraint is "no four collinear".
    # The score is "largest subset with no three collinear".
    # We want to MINIMIZE this score.
    # This means we want the points to be "collinear-heavy" but not 4-collinear.
    # i.e., we want many triples to be collinear, so that any large subset will likely contain a collinear triple.
    # If we use a parabola (no 3 collinear), the score is 20. This is BAD (we want low score).
    # We want to MAXIMIZE the number of collinear triples, subject to no 4 being collinear.
    # This is the opposite of the current program's stated goal in the text ("We want to MAXIMIZE triples") but the current program returns best_points based on maximizing count_triples.
    # Wait, the current program says: "The score is the largest subset with no three collinear... Minimize."
    # And "We want to MAXIMIZE triples".
    # If we maximize collinear triples, we make it harder to find a large subset with NO three collinear.
    # So yes, maximizing collinear triples (while avoiding 4-collinear) is the correct strategy to minimize the score.
    
    # Current program uses Simulated Annealing to maximize triples.
    # My new strategy: 
    # 1. Use a structured initial guess that is known to have many collinear triples but no 4-collinear.
    #    Example: Points on a grid, but perturbed to break 4-collinearities while keeping 3-collinearities.
    #    Or: Start with a set of lines, place 3 points on each line, ensure no point is on more than 1 line? 
    #    No, points can be on multiple lines.
    #    If we take a 4x5 grid, it has many 4-collinear rows and cols.
    #    We need to break the 4-collinearities.
    #    Strategy: Take a 4x5 grid. For each row of 4, move one point slightly off the row. 
    #    This breaks the 4-collinearity but might break other collinearities.
    #    
    #    Better: Use a "Pappus-like" or "Fermat-like" configuration?
    #    
    #    Let's try a different search method: 
    #    Local Search with a specific objective: 
    #    Objective = (Number of collinear triples) - 1000 * (Number of 4-collinear sets)
    #    But we need to minimize the "largest subset with no 3 collinear".
    #    This is a complex global property.
    #    However, maximizing triples is a good proxy.
    #    
    #    Structural Change: 
    #    Instead of random point moves, use "Line-based" moves.
    #    Identify lines with 3 points. Try to create more.
    #    Identify lines with 4 points. Try to break them.
    #    
    #    Let's implement a faster local search that focuses on fixing 4-collinearities and adding 3-collinearities.
    #    
    #    Also, the current program's `has_four_collinear` is O(n^4) = 48450 checks. This is fast.
    #    `count_triples` is O(n^3) = 1140 checks.
    #    Total per iteration ~ 50k ops. In 110s, we can do ~ 10M iterations? 
    #    Python is slow. Maybe 100k iterations.
    #    
    #    Let's optimize the inner loop.
    #    Precompute slopes? No, points move.
    
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def count_triples_and_check_four(points):
        """
        Returns (num_triples, has_four).
        Efficiently checks for 4-collinear while counting triples.
        """
        n = len(points)
        triple_count = 0
        has_four = False
        
        # For each pair, count how many other points are on the line.
        # If count >= 2, we have a 4-collinear set (the pair + 2 others).
        # If count >= 1, we have a triple (the pair + 1 other).
        # But this counts each triple 3 times (once for each pair in the triple).
        # And each 4-set is counted C(4,2)=6 times as pairs, but we just need to know if it exists.
        
        # To count unique triples:
        # A triple is defined by 3 points. 
        # If we iterate pairs (i,j), and find k such that collinear(i,j,k), we count it.
        # Each triple (i,j,k) is found 3 times: (i,j)->k, (i,k)->j, (j,k)->i.
        # So triple_count = sum(counts) / 3.
        
        # For 4-collinear:
        # If for any pair (i,j), there are >= 2 other points k1, k2 on the line, then {i,j,k1,k2} is 4-collinear.
        
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate line key? Or just check all k?
                # Checking all k is O(n). Total O(n^3).
                # n=20. 20*19/2 * 18 = 3420 checks.
                # This is faster than O(n^4) for 4-collinear.
                
                count_on_line = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if is_collinear(points[i], points[j], points[k]):
                        count_on_line += 1
                
                if count_on_line >= 2:
                    has_four = True
                    # We can break early if we only care about validity, but we need triple count too.
                    # So we continue to count.
                
                triple_count += count_on_line
        
        unique_triples = triple_count // 3
        return unique_triples, has_four

    def get_score(points):
        triples, has_four = count_triples_and_check_four(points)
        if has_four:
            return -1000000 + triples # Still maximize triples among invalid? No, invalid is -inf.
            # Actually, if invalid, score is very low.
            return -1000000
        return triples

    # Initialize with a grid that is "almost" valid?
    # 4x5 grid has 4-collinear rows and cols.
    # Let's start with random points and use SA.
    # But the "Structural Change" directive asks for a different design.
    # Let's use a "Greedy Line Packing" initialization.
    # 1. Pick a point.
    # 2. Pick a line through it. Place 2 other points on it. (3 collinear)
    # 3. Ensure no 4th point falls on this line.
    # 4. Repeat.
    # This is hard to do greedily without backtracking.
    
    # Alternative Structural Change: 
    # Use a different representation. 
    # Represent points as indices into a candidate pool? No.
    
    # Let's stick to Local Search but change the move strategy.
    # Current: Random point replacement.
    # New: "Swap" points? Or "Slide" points along lines?
    # Sliding a point along a line that already has 2 other points preserves the triple.
    # If we slide a point to create a new triple, we gain.
    
    # Let's implement "Directed Moves":
    # 1. Find a pair (i,j).
    # 2. Find a third point k that is NOT collinear with i,j.
    # 3. Move k to a position on line ij (if possible within bounds and not creating 4-collinear).
    # This directly creates a triple.
    
    def generate_initial_points():
        # Start with a random set
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        return pts

    current_points = generate_initial_points()
    current_score = get_score(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    temperature = 10.0
    cooling = 0.995
    min_temp = 0.1
    
    while time.time() < deadline:
        # Try a directed move: Create a triple
        # Pick 2 points i, j
        i, j = random.sample(range(20), 2)
        p_i = current_points[i]
        p_j = current_points[j]
        
        # Find a point k that is not collinear with i,j
        candidates_k = []
        for k in range(20):
            if k == i or k == j:
                continue
            if not is_collinear(p_i, p_j, current_points[k]):
                candidates_k.append(k)
        
        if not candidates_k:
            # All other points are collinear with i,j? Then we have 4-collinear already (if >=2 others).
            # If exactly 1 other, we have a triple.
            # If 0 others, no triple.
            # If this happens, it means the line ij has many points.
            # We should try to break a 4-collinear if it exists.
            pass
        else:
            k = random.choice(candidates_k)
            p_k = current_points[k]
            
            # We want to move p_k to a new position on the line ij.
            # Line equation: (y - y_i) = m (x - x_i)
            # dx = x_j - x_i, dy = y_j - y_i
            dx = p_j[0] - p_i[0]
            dy = p_j[1] - p_i[1]
            
            # Parametric: x = x_i + t*dx, y = y_i + t*dy
            # We need integer coordinates.
            # So t must be such that x_i + t*dx is integer.
            # Since dx is integer, t can be any integer? No, t*dx must be integer.
            # If we pick integer t, x and y are integers.
            # But we must avoid existing points and 4-collinearities.
            
            # Try a few small integer t values
            found_move = False
            for t in [-3, -2, -1, 1, 2, 3]:
                if t == 0: continue
                new_x = p_i[0] + t * dx
                new_y = p_i[1] + t * dy
                
                if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                    continue
                
                if [new_x, new_y] in current_points:
                    continue
                
                # Check if this move creates a 4-collinear
                # The line ij already has i and j.
                # We are moving k to this line.
                # So the line will have i, j, k(new).
                # Are there any OTHER points on this line?
                # We checked that k was NOT on the line.
                # But what about other points m?
                # We need to ensure no other point m is on line ij.
                # If there is an m on line ij, then {i,j,m,k} is 4-collinear.
                
                has_other_on_line = False
                for m in range(20):
                    if m == i or m == j or m == k:
                        continue
                    if is_collinear(p_i, p_j, current_points[m]):
                        has_other_on_line = True
                        break
                
                if has_other_on_line:
                    continue
                
                # If no other point is on the line, the move is valid regarding 4-collinearity on THIS line.
                # But moving k might create a 4-collinear on a DIFFERENT line?
                # No, k is moving TO line ij. It leaves its old position.
                # It might have been part of a 4-collinear on another line?
                # If k was part of a 4-collinear, the current state is invalid.
                # We assume current state is valid (or we handle invalid states).
                # If current state is valid, k is not in any 4-collinear.
                # So moving k to a new spot on line ij (which has only i,j) creates a triple (i,j,k) and no 4-collinear on that line.
                # Could it create a 4-collinear on a line passing through new_k and some other two points?
                # Yes. We must check the full validity.
                
                test_points = current_points.copy()
                test_points[k] = [new_x, new_y]
                
                new_triples, has_four = count_triples_and_check_four(test_points)
                if has_four:
                    continue
                
                new_score = new_triples
                old_score = current_score
                
                if new_score > old_score:
                    current_points = test_points
                    current_score = new_score
                    if new_score > best_score:
                        best_points = [p[:] for p in current_points]
                        best_score = new_score
                    found_move = True
                    break
                elif new_score == old_score and random.random() < 0.5:
                    current_points = test_points
                    current_score = new_score
                    found_move = True
                    break
            
            if not found_move:
                # Try a random move if directed move fails
                idx = random.randint(0, 19)
                new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                if new_p in current_points:
                    continue
                test_points = current_points.copy()
                test_points[idx] = new_p
                new_triples, has_four = count_triples_and_check_four(test_points)
                if has_four:
                    new_score = -1000000
                else:
                    new_score = new_triples
                
                delta = new_score - current_score
                if delta > 0:
                    current_points = test_points
                    current_score = new_score
                    if new_score > best_score:
                        best_points = [p[:] for p in current_points]
                        best_score = new_score
                else:
                    if delta == 0:
                        if random.random() < 0.5:
                            current_points = test_points
                    else:
                        prob = math.exp(delta / temperature)
                        if random.random() < prob:
                            current_points = test_points
                            current_score = new_score
        
        temperature *= cooling
        if temperature < min_temp:
            temperature = 10.0

    # Final check: if best_points is invalid, try to repair it quickly
    triples, has_four = count_triples_and_check_four(best_points)
    if has_four:
        # Fallback: return a known valid set? 
        # Or just return best_points and hope the evaluator handles it? 
        # "Invalid result ... scores 0". 
        # We must return a valid set.
        # Let's construct a simple valid set.
        # Points on parabola y=x^2, x in [-7, 7] is 15 pts.
        # Add 5 more points far away?
        # Let's just use the random generator until valid.
        while has_four:
            best_points = generate_initial_points()
            triples, has_four = count_triples_and_check_four(best_points)
    
    return best_points
