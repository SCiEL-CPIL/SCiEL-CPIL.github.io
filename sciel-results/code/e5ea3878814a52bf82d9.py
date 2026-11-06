import time
import random
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # We'll work on a 5x5 grid for simplicity, then scale if needed
    # Actually, let's use a larger grid to have more options, but keep it manageable
    # The constraint is |x|, |y| <= 100, so we have a lot of room.
    # But for the search to be tractable, let's stick to a moderate grid.
    # Let's use a 7x7 grid centered at (0,0) => x,y in {-3..3}
    # Or even better, let's use the 5x5 grid {-2..2} x {-2..2} first, and if that's not enough, expand.
    # Wait, 20 points out of 25 (5x5) is quite dense.
    # Let's try a 6x6 grid = 36 points. Choose 20.
    # Or 7x7 = 49 points. Choose 20.
    
    # Let's use a 7x7 grid: x, y in range(-3, 4)
    points = [(x, y) for x in range(-3, 4) for y in range(-3, 4)]
    n_points = len(points)  # 49
    idx_to_point = {i: points[i] for i in range(n_points)}
    point_to_idx = {p: i for i, p in enumerate(points)}

    # Precompute collinearity for all triples
    # collinear[i][j] = set of k such that i, j, k are collinear
    # Actually, for efficiency in branch and bound, we need:
    # For a given set of selected points S, and a candidate point p,
    # we need to know if adding p creates a collinear triple.
    # This happens if there exist i, j in S such that i, j, p are collinear.
    # So, for each pair (i, j), we need the set of k collinear with them.
    
    # Let's precompute: line_mask[i][j] = bitmask of all k collinear with i and j
    # Since n=49, we can use integers as bitmasks (Python handles big ints)
    # But 49 bits is fine.
    
    line_mask = [[0] * n_points for _ in range(n_points)]
    for i in range(n_points):
        for j in range(i + 1, n_points):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            mask = 0
            for k in range(n_points):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                # Check collinearity
                if (x3 - x1) * dy == (y3 - y1) * dx:
                    mask |= (1 << k)
            line_mask[i][j] = mask
            line_mask[j][i] = mask

    # Precompute: for each point i, and each other point j, line_mask[i][j]
    # We also need to check "no four collinear" for the full set of 20.
    # This is a separate constraint.
    # For a set S of 20 points, no 4 collinear means:
    # For any line, at most 3 points from S lie on it.
    
    # Let's precompute all lines (as frozensets of indices) that have >= 4 points in the full grid.
    # Actually, we need to check if any line has >= 4 points in S.
    # So we can precompute all maximal collinear sets in the grid.
    
    # A line is defined by two points. We can collect all points on that line.
    # To avoid duplicates, we can canonicalize.
    
    all_lines = set()
    for i in range(n_points):
        for j in range(i + 1, n_points):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g > 0:
                dx //= g
                dy //= g
            # Direction vector (dx, dy)
            # Canonicalize direction: ensure dx > 0, or dx == 0 and dy > 0
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
                x1, y1 = x2, y2
            
            # Collect all points on this line
            pts_on_line = []
            for k in range(n_points):
                x3, y3 = points[k]
                # Check if (x3, y3) is on the line through (x1, y1) with direction (dx, dy)
                # (x3 - x1) * dy - (y3 - y1) * dx == 0
                if (x3 - x1) * dy == (y3 - y1) * dx:
                    # Also check if it's in the same "ray" or just collinear?
                    # For collinearity, the cross product is sufficient.
                    pts_on_line.append(k)
            
            if len(pts_on_line) >= 4:
                all_lines.add(frozenset(pts_on_line))

    # Now, all_lines is a set of frozensets of indices.
    # A configuration S (set of 20 indices) is valid if for every L in all_lines, |L ∩ S| <= 3.

    def is_valid(S):
        """Check if set S (of indices) has no 4 collinear points."""
        for L in all_lines:
            if len(L & S) >= 4:
                return False
        return True

    def max_no_3_collinear(S):
        """
        Exact branch and bound for largest subset of S with no 3 collinear.
        S is a set of indices.
        """
        pts_list = list(S)
        n = len(pts_list)
        
        # Remap indices 0..n-1 for local bitmask operations
        # local_idx_to_global[i] = pts_list[i]
        # We need line_mask for local indices.
        
        # Precompute local line masks
        local_line_mask = [[0] * n for _ in range(n)]
        for i in range(n):
            gi = pts_list[i]
            for j in range(i + 1, n):
                gj = pts_list[j]
                # Get global mask
                mask = line_mask[gi][gj]
                # Convert to local mask
                local_mask = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if mask & (1 << pts_list[k]):
                        local_mask |= (1 << k)
                local_line_mask[i][j] = local_mask
                local_line_mask[j][i] = local_mask

        best = [0]
        
        def backtrack(A, C, depth):
            # A: selected set (bitmask, local indices)
            # C: candidate set (bitmask, local indices)
            # depth: current size of A
            remaining = bin(C).count('1')
            if depth + remaining <= best[0]:
                return
            
            if C == 0:
                if depth > best[0]:
                    best[0] = depth
                return
            
            # Heuristic: pick point with most conflicts
            # For speed, let's pick the first one, or use a simple heuristic.
            # Let's try to pick the point that, if added, removes the most candidates.
            # But that's expensive. Let's just pick the lowest index for now.
            # Actually, a good heuristic is to pick the point with the highest degree in the conflict graph induced by C.
            # Let's just pick the first available point to keep it simple and fast.
            
            p_bit = C & -C
            p = p_bit.bit_length() - 1
            
            # Option 1: Exclude p
            backtrack(A, C ^ p_bit, depth)
            
            # Option 2: Include p
            # Check if p is compatible with A
            compatible = True
            temp_A = A
            while temp_A:
                a_bit = temp_A & -temp_A
                a = a_bit.bit_length() - 1
                if local_line_mask[p][a] & A:
                    compatible = False
                    break
                temp_A ^= a_bit
            
            if compatible:
                new_C = C ^ p_bit
                # Remove from C any q that is collinear with p and some a in A
                temp_A = A
                while temp_A:
                    a_bit = temp_A & -temp_A
                    a = a_bit.bit_length() - 1
                    new_C &= ~local_line_mask[p][a]
                    temp_A ^= a_bit
                backtrack(A | p_bit, new_C, depth + 1)

        backtrack(0, (1 << n) - 1, 0)
        return best[0]

    # Local search
    # We need to find a set of 20 indices S such that is_valid(S) and max_no_3_collinear(S) is minimized.
    
    # Initial valid configuration
    # Start with a random set of 20, then fix violations.
    
    def find_valid_initial():
        for _ in range(1000):
            if time.time() > deadline - 15:
                break
            S = set(random.sample(range(n_points), 20))
            if is_valid(S):
                return S
        # Fallback: try to construct one greedily
        S = set()
        while len(S) < 20:
            candidates = list(set(range(n_points)) - S)
            random.shuffle(candidates)
            for c in candidates:
                new_S = S | {c}
                if is_valid(new_S):
                    S = new_S
                    break
            if len(S) == 20:
                break
        if len(S) == 20:
            return S
        return None

    current_S = find_valid_initial()
    if current_S is None:
        # Should not happen, but if it does, return a dummy valid config?
        # We must return a valid config. Let's try a grid pattern.
        # Remove 4 corners and center from 5x5? No, we're on 7x7.
        # Let's just return the first 20 points that form a valid set.
        # This is a last resort.
        S = set()
        for i in range(n_points):
            if len(S) == 20:
                break
            if is_valid(S | {i}):
                S.add(i)
        current_S = S

    current_score = max_no_3_collinear(current_S)
    
    # Local search: swap one point in S with one point not in S
    best_score = current_score
    best_S = current_S

    for step in range(2000):
        if time.time() > deadline - 10:
            break
        
        # Pick a random point to remove
        pts_list = list(current_S)
        i = random.randint(0, 19)
        old_idx = pts_list[i]
        
        # Pick a random point to add
        available = list(set(range(n_points)) - current_S)
        new_idx = random.choice(available)
        
        new_S = set(current_S)
        new_S.remove(old_idx)
        new_S.add(new_idx)
        
        if is_valid(new_S):
            new_score = max_no_3_collinear(new_S)
            if new_score < best_score:
                best_score = new_score
                best_S = new_S
                current_S = new_S
                current_score = new_score
            elif new_score == best_score:
                # Keep current, or randomly accept to escape plateaus?
                # Let's accept with small probability
                if random.random() < 0.1:
                    current_S = new_S
                    current_score = new_score

    # Return the best configuration found
    result = [[points[i][0], points[i][1]] for i in best_S]
    return result
