"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _are_collinear(p1, p2, p3):
    return (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) == 0

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i, j, k, l in combinations(range(n), 4):
        if _are_collinear(points[i], points[j], points[k]) and \
           _are_collinear(points[i], points[j], points[l]) and \
           _are_collinear(points[i], points[k], points[l]):
            return True
    return False

def _max_no_three_collinear(points):
    """
    Find the largest subset with no three collinear using an optimized branch and bound.
    """
    n = len(points)
    if n == 0:
        return 0
    
    # Precompute collinear triples and conflict pairs
    # conflict_pairs[i] = list of (j, k) such that i, j, k are collinear and j < k
    conflict_pairs = [[] for _ in range(n)]
    triples_count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    triples_count += 1
                    # Add to conflict pairs for each of the three points
                    # For i: pair (j, k)
                    conflict_pairs[i].append((j, k))
                    # For j: pair (i, k) -> need i < k? No, just the pair of other points
                    # We store pairs of indices. The check is if both are in S.
                    conflict_pairs[j].append((i, k))
                    conflict_pairs[k].append((i, j))
                    
    if triples_count == 0:
        return n

    best = [0]
    
    def can_add(p, S_mask):
        """Check if point p can be added to subset S_mask without creating a collinear triple."""
        for j, k in conflict_pairs[p]:
            # Check if both j and k are in S_mask
            if (S_mask & (1 << j)) and (S_mask & (1 << k)):
                return False
        return True

    def bnb(S_mask, R_mask, depth):
        """
        S_mask: bitmask of points in current subset
        R_mask: bitmask of remaining candidate points
        depth: number of points in S
        """
        # Upper bound: current size + remaining
        # A better bound: depth + max independent set of the conflict graph on R? Too expensive.
        # Simple bound: depth + popcount(R_mask)
        if depth + bin(R_mask).count('1') <= best[0]:
            return
        
        if R_mask == 0:
            if depth > best[0]:
                best[0] = depth
            return
        
        # Heuristic: Pick the point in R that is most constrained
        # i.e., has the most conflict pairs with points in S? Or just pick the first one?
        # Picking the most constrained helps prune earlier.
        # Let's pick the point with the highest "conflict score" with S.
        best_p = -1
        max_conflicts = -1
        temp = R_mask
        while temp:
            p = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            # Count how many conflict pairs of p have both elements in S
            conflicts = 0
            for j, k in conflict_pairs[p]:
                if (S_mask & (1 << j)) and (S_mask & (1 << k)):
                    conflicts += 1
                    # If conflicts > 0, it can't be added, but we still pick it to prune?
                    # Actually, if it can't be added, we should skip it immediately?
                    # No, bnb explores "add" and "don't add".
                    # If it can't be added, the "add" branch is pruned.
                    # It's good to pick such points to prune the tree.
            if conflicts > max_conflicts:
                max_conflicts = conflicts
                best_p = p
        
        if best_p == -1:
            # No points in R?
            return

        p = best_p
        bit = 1 << p
        new_R = R_mask & ~bit
        
        # Option 1: Add p
        if can_add(p, S_mask):
            bnb(S_mask | bit, new_R, depth + 1)
        
        # Option 2: Don't add p
        bnb(S_mask, new_R, depth)
    
    bnb(0, (1 << n) - 1, 0)
    return best[0]

def _find_valid_initial(deadline):
    """Find a valid initial 20-point set with no four collinear."""
    while time.time() < deadline:
        points = []
        for _ in range(20):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = [x, y]
                if p not in points:
                    break
            points.append(p)
        if not _has_four_collinear(points):
            return points
    return None

def _local_search(points, deadline):
    """
    Local search that tries to decrease the 3-free subset size.
    Moves points to random locations if it improves the score.
    """
    best_points = points[:]
    best_score = _max_no_three_collinear(points)
    current_points = points[:]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(20):
            if time.time() > deadline:
                break
            
            best_new_point = None
            best_new_score = best_score
            
            # Try a few random moves
            for _ in range(10):
                if time.time() > deadline:
                    break
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                new_p = [x, y]
                if new_p in current_points:
                    continue
                
                new_points = current_points[:]
                new_points[idx] = new_p
                if _has_four_collinear(new_points):
                    continue
                
                new_score = _max_no_three_collinear(new_points)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_point = new_p
            
            if best_new_point is not None:
                current_points[idx] = best_new_point
                if best_new_score < best_score:
                    best_score = best_new_score
                    best_points = current_points[:]
                improved = True
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_points = None
    best_score = 20  # Worst possible
    
    # Try multiple random restarts
    while time.time() < deadline - 20:
        points = _find_valid_initial(deadline - 20)
        if points is None:
            continue
        
        # Local search
        improved_points, score = _local_search(points, deadline - 20)
        
        if score < best_score:
            best_score = score
            best_points = improved_points[:]
            
        # If we find a very good score, we can stop early?
        # The theoretical minimum is unknown, but for n=20, it's likely around 8-10.
        # If we find 10, it's a strong candidate.
        if best_score <= 10:
            break

    if best_points is None:
        # Fallback
        best_points = _find_valid_initial(deadline)
    
    return best_points
# EVOLVE-BLOCK-END
