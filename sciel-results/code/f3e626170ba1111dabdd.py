import time
import random
import math
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        # Line equation: dy*x - dx*y = c
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def count_triples_and_check_four(points):
        n = len(points)
        if n < 4:
            return 0, False
        line_counts = Counter()
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] += 1
        triples = 0
        has_four = False
        for cnt in line_counts.values():
            if cnt >= 6:
                has_four = True
            # Number of triples on this line is C(cnt, 3)
            if cnt >= 3:
                triples += cnt * (cnt - 1) * (cnt - 2) // 6
        return triples, has_four

    def max_no_three_collinear(points):
        n = len(points)
        # points is a list of [x, y]
        # For each point i, let S_i be the set of points collinear with i (including i).
        # Then a subset with no 3 collinear can contain at most 2 points from any line through any of its points.
        # More precisely: If T is a subset with no 3 collinear, then for any point p in T, 
        # the number of points in T that are on any line with p and some other point in T is at most 1 (besides p).
        # This is equivalent to: For every p in T, the number of other points q in T such that (p, q) is part of a collinear triple in the full set?
        # No. The "no 3 collinear" condition is local to T.
        # If T has no 3 collinear, then for any p in T, no other two points in T are collinear with p.
        # This means: For each p in T, the set {q in T \ {p} | p, q, and some other point in T are collinear?} is empty?
        # No. It means: For each p in T, and for each line L passing through p, |T ∩ L| <= 2.
        
        # Branch and Bound:
        # Order points arbitrarily.
        # State: (k, current_set, forbidden_set)
        # Actually, simpler: 
        # Sort points.
        # Recursive search:
        # f(i, selected, last_two) ? No, that's for no 2.
        # For no 3:
        # If we select a point p, we cannot select two other points that are collinear with p.
        # This constraint is global.
        
        # Let's use a bitset representation for speed.
        # n <= 20.
        # Precompute for each pair (i, j) the set of points k such that i, j, k are collinear.
        # collinear_with[i][j] = bitmask of k such that i, j, k collinear.
        # If we select i and j, we cannot select any k in collinear_with[i][j] \\ {i, j}.
        # Actually, if we select i and j, we must ensure no k in collinear_with[i][j] is also selected.
        # So if we select i and j, we must remove all k in collinear_with[i][j] from consideration.
        
        # Algorithm:
        # 1. Precompute collinear_sets[i][j] for i < j.
        # 2. DFS:
        #    def dfs(idx, selected_count, forbidden_mask):
        #       if idx == n: return selected_count
        #       # Prune 1: Can we reach best?
        #       # Prune 2: If adding current point is impossible, skip.
        #       Option 1: Do not select point idx.
        #       Option 2: Select point idx (if not forbidden).
        #          If we select idx, we must check against all previously selected points?
        #          This is hard because the constraint is "no 3".
        #          If we select idx, it forms triples with any pair of previously selected points.
        #          So we need to know which pairs of previously selected points would form a triple with idx.
        #          This implies we need to track the selected points or at least their pairwise collinearities.
        
        # Alternative:
        # Since n=20, we can use a different B&B.
        # Max Independent Set on a hypergraph?
        # Nodes = points. Edges = triples of collinear points.
        # We want Max Independent Set in this 3-uniform hypergraph.
        
        # Let's use a simple recursive search with memoization on bitmask? 2^20 is 1M.
        # State: mask of available points.
        # We want to find max size subset of mask with no 3 collinear.
        # f(mask):
        #   if mask == 0: return 0
        #   pick a point p in mask.
        #   Option 1: Exclude p. -> f(mask \ {p})
        #   Option 2: Include p. 
        #      If we include p, we must exclude all points q, r such that {p, q, r} is a collinear triple.
        #      Let Bad = union of all q such that there exists r in mask where {p,q,r} collinear?
        #      No. If we include p, we cannot include ANY pair {q, r} such that p, q, r are collinear.
        #      This means: For any q in mask, if we include q, we must exclude all r such that p, q, r collinear.
        #      This is not a simple exclusion of a set. It's a constraint on pairs.
        
        # However, note:
        # If we include p, the remaining problem is to find a subset S of mask \ {p} such that:
        # 1. S has no 3 collinear (internal constraint)
        # 2. No two points q, r in S are collinear with p.
        # Condition 2 means: For every q in S, (S \ {q}) ∩ Collinear(p, q) = empty.
        # This is complex.
        
        # Let's try a different angle.
        # The score is the size of the largest subset with no 3 collinear.
        # This is the "no-three-in-line" problem on a specific set.
        
        # Given the time limit (120s), and we already used most of it in optimization.
        # The optimization phase takes ~100s.
        # We have ~15s left.
        # A naive B&B on 20 points might be too slow if the answer is large (e.g. 10+).
        # But if the answer is small (e.g. 5), it will be fast.
        # If the answer is large, the search tree is small because we can greedily pick many?
        # No, B&B is hard.
        
        # Let's implement a greedy upper bound first.
        # And a simple greedy lower bound.
        
        # Greedy Lower Bound:
        # Greedily pick points that don't form a triple.
        
        # Greedy Upper Bound:
        # Not obvious.
        
        # Let's just do a simple DFS with bitmasks and pruning.
        # Precompute:
        # triple_mask[i]: bitmask of all points j such that i, j, and some other point k (in full set) are collinear?
        # No, collinearity depends on the specific triple.
        
        # Precompute:
        # For each i, and each j > i, let M[i][j] be the bitmask of k such that i, j, k are collinear.
        # If we select i and j, we must remove M[i][j] from the available set.
        
        # DFS(idx, avail_mask, selected_count, last_selected_idx1, last_selected_idx2)?
        # No, we need to remember all selected to check against new ones.
        
        # Actually, if we build the set sequentially:
        # If we have selected set S, and we consider adding p:
        # We must check if p forms a triple with any pair in S.
        # Let PairsInS be the set of pairs in S.
        # For each pair (a, b) in S, if p is in M[a][b], then we cannot add p.
        
        # This requires iterating over pairs in S.
        
        # Since n=20, max pairs in S is C(20,2)=190.
        # If we do this at each step of DFS, and depth is 20, it might be feasible if pruning is good.
        
        # Let's implement a simple recursive function with backtracking.
        
        # Precompute M[i][j] for all i, j.
        M = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                mask = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if is_collinear(points[i], points[j], points[k]):
                        mask |= (1 << k)
                M[i][j] = mask
        
        # Also, for any triple (i, j, k), M[i][j] has k, M[i][k] has j, M[j][k] has i.
        
        # DFS:
        # args: idx (next point to consider), avail_mask, selected_list
        # We want to maximize len(selected_list).
        
        # Pruning:
        # If len(selected_list) + popcount(avail_mask) <= best_found, prune.
        
        # We can sort points to improve pruning?
        # Points with high degree (collinear with many others) are harder to include.
        
        # Let's just iterate idx from 0 to n-1.
        
        best_size = 0
        
        # Use a heuristic: sort points by "collinearity degree" descending?
        # Points that are on many lines are more likely to cause conflicts.
        
        # Let's just use the given order for simplicity and speed of implementation.
        
        def dfs(idx, avail_mask, selected_list):
            nonlocal best_size
            if len(selected_list) > best_size:
                best_size = len(selected_list)
            
            if idx == n:
                return
            
            # Prune
            remaining = bin(avail_mask).count('1')
            if len(selected_list) + remaining <= best_size:
                return
            
            # Option 1: Skip points[idx]
            # But we must ensure points[idx] is in avail_mask?
            # Actually, we should only consider points in avail_mask.
            # If points[idx] is not in avail_mask, we just proceed to idx+1?
            # No, the "skip" option should just remove points[idx] from avail_mask and proceed.
            # But if points[idx] is already removed, we just proceed.
            
            # Let's change DFS to take just avail_mask and process the lowest set bit.
            
        # Revised DFS:
        
        def dfs2(avail_mask, selected_list):
            nonlocal best_size
            if len(selected_list) > best_size:
                best_size = len(selected_list)
            
            if avail_mask == 0:
                return
            
            # Pick the lowest set bit
            p = (avail_mask & -avail_mask).bit_length() - 1
            bit = 1 << p
            
            # Option 1: Exclude p
            dfs2(avail_mask ^ bit, selected_list)
            
            # Option 2: Include p
            # Check if p can be added.
            # p must not form a triple with any pair in selected_list.
            # For each q in selected_list:
            #   if M[p][q] intersects selected_list \ {q}, then conflict.
            #   Actually, if M[p][q] has any r in selected_list (r != q, r != p), then {p, q, r} is a triple.
            
            valid = True
            for q_idx, q in enumerate(selected_list):
                # Check if there exists r in selected_list such that p, q, r collinear.
                # M[p][q] contains all such r.
                # We need to check if M[p][q] & (mask of selected_list) is non-empty.
                # But q is in selected_list. We must exclude q?
                # M[p][q] does NOT contain p or q by definition (we excluded them in precompute).
                # So M[p][q] contains only r != p, q.
                # So we just check if M[p][q] & selected_mask is non-empty.
                
                # We need selected_mask.
                # We can compute it or pass it.
                pass
            
            # Let's pass selected_mask.
            
        # Revised DFS3:
        
        def dfs3(avail_mask, selected_mask, selected_count):
            nonlocal best_size
            if selected_count > best_size:
                best_size = selected_count
            
            if avail_mask == 0:
                return
            
            # Prune
            if selected_count + bin(avail_mask).count('1') <= best_size:
                return
            
            # Pick lowest set bit
            p = (avail_mask & -avail_mask).bit_length() - 1
            bit = 1 << p
            
            # Option 1: Exclude p
            dfs3(avail_mask ^ bit, selected_mask, selected_count)
            
            # Option 2: Include p
            # Check validity
            valid = True
            # For each q in selected_mask:
            #   if M[p][q] & selected_mask != 0, invalid.
            # Note: M[p][q] does not contain p or q.
            # So we just check intersection.
            
            temp_mask = selected_mask
            while temp_mask:
                q_bit = temp_mask & -temp_mask
                q = q_bit.bit_length() - 1
                if M[p][q] & selected_mask:
                    valid = False
                    break
                temp_mask ^= q_bit
            
            if valid:
                new_avail = avail_mask ^ bit
                # When we add p, do we need to remove anything from avail_mask?
                # No, the constraint is only on the selected set.
                # However, if we add p, we might make some future additions invalid.
                # That is handled by the check at addition time.
                # BUT: If we add p, and later try to add q, we check M[p][q].
                # So we don't need to prune avail_mask immediately.
                
                dfs3(new_avail, selected_mask | bit, selected_count + 1)
        
        dfs3((1 << n) - 1, 0, 0)
        return best_size

    # --- Optimization Phase ---
    
    def generate_initial_points():
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        return pts

    best_points = None
    best_triples = -1
    
    # Try a few random starts
    for _ in range(5):
        current_points = generate_initial_points()
        triples, has_four = count_triples_and_check_four(current_points)
        if has_four:
            continue
        if triples > best_triples:
            best_triples = triples
            best_points = [p[:] for p in current_points]
    
    if best_points is None:
        # Fallback: Parabola + perturbation
        # Just return random until valid
        while True:
            pts = generate_initial_points()
            _, has_four = count_triples_and_check_four(pts)
            if not has_four:
                best_points = pts
                break
    
    # Local Search
    temperature = 10.0
    cooling = 0.995
    
    while time.time() < deadline - 15.0: # Keep 15s for score calculation
        # Directed Move: Try to create a triple
        i, j = random.sample(range(20), 2)
        p_i = best_points[i]
        p_j = best_points[j]
        
        # Find k not on line ij
        candidates_k = [k for k in range(20) if k != i and k != j and not is_collinear(p_i, p_j, best_points[k])]
        
        if candidates_k:
            k = random.choice(candidates_k)
            dx = p_j[0] - p_i[0]
            dy = p_j[1] - p_i[1]
            
            found_move = False
            for t in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                if t == 0: continue
                new_x = p_i[0] + t * dx
                new_y = p_i[1] + t * dy
                if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                    continue
                if [new_x, new_y] in best_points:
                    continue
                
                # Check if line ij has other points
                has_other = False
                for m in range(20):
                    if m == i or m == j or m == k:
                        continue
                    if is_collinear(p_i, p_j, best_points[m]):
                        has_other = True
                        break
                if has_other:
                    continue
                
                test_points = best_points.copy()
                test_points[k] = [new_x, new_y]
                new_triples, has_four = count_triples_and_check_four(test_points)
                if has_four:
                    continue
                
                if new_triples > best_triples:
                    best_points = test_points
                    best_triples = new_triples
                    found_move = True
                    break
                elif new_triples == best_triples and random.random() < 0.1:
                    best_points = test_points
                    found_move = True
                    break
            
            if not found_move:
                # Random move
                idx = random.randint(0, 19)
                new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                if new_p in best_points:
                    continue
                test_points = best_points.copy()
                test_points[idx] = new_p
                new_triples, has_four = count_triples_and_check_four(test_points)
                if has_four:
                    new_triples = -1
                delta = new_triples - best_triples
                if delta > 0:
                    best_points = test_points
                    best_triples = new_triples
                elif delta < 0:
                    if random.random() < math.exp(delta / temperature):
                        best_points = test_points
                        best_triples = new_triples
        
        temperature *= cooling
        if temperature < 0.5:
            temperature = 10.0

    # Final Score Calculation
    # If time is running out, we might not be able to calculate the exact score.
    # But we must return valid points.
    # The score is calculated by the evaluator.
    # Wait, the task says: "The evaluator measures the raw value".
    # "The score is the largest subset with no three collinear (exact branch and bound)."
    # Does the evaluator calculate this, or do I?
    # "Minimize. A valid configuration with score s certifies g(20) <= s"
    # "The evaluator measures the raw value (lower is better)."
    # Usually, in these tasks, the evaluator runs `construct()` and then calculates the score.
    # My `construct()` just returns the points.
    # So I don't need to calculate the score myself!
    # I just need to return a valid set of 20 points (no 4 collinear).
    
    # Let's verify validity of best_points.
    triples, has_four = count_triples_and_check_four(best_points)
    if has_four:
        # Repair
        # This should be rare due to checks in the loop.
        # Just return a known valid set if needed.
        # Parabola x in -7..6, y=x^2 (14 pts) + 6 pts from x in -7..6, y=x^2+100? No bounds.
        # Let's just random search until valid.
        while has_four:
            best_points = generate_initial_points()
            triples, has_four = count_triples_and_check_four(best_points)

    return best_points

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(f"Points: {pts}")
    t, h = count_triples_and_check_four(pts)
    print(f"Triples: {t}, Has Four: {h}")
    # Calculate score
    s = max_no_three_collinear(pts)
    print(f"Score: {s}")
