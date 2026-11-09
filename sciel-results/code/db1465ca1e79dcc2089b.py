import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def are_collinear(p1, p2, p3):
        return (p1[0] - p2[0]) * (p2[1] - p3[1]) == (p2[0] - p3[0]) * (p1[1] - p2[1])

    def has_four_collinear(points):
        n = len(points)
        # Check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(p1, p2, points[k]):
                        count += 1
                        if count >= 4:
                            return True
        return False

    def calculate_score(points):
        # Find largest subset with no 3 collinear
        # This is equivalent to Maximum Independent Set in 3-uniform hypergraph
        # Or simply: Maximize size of S such that no 3 points in S are collinear.
        
        n = len(points)
        # Precompute collinearity triples
        # triple[i][j][k] is True if i, j, k are collinear
        # Actually, we can just check on the fly in B&B or precompute a bitmask of conflicts.
        
        # For N=20, we can use a simple recursive search with pruning.
        # State: index of next point to consider, current set.
        # To speed up, we can use bitmasks.
        
        # Let's use a simple greedy + local search for the score calculation itself?
        # No, the task says "exact branch and bound". We should try to be exact or very close.
        # But for the *search* phase, we can use a heuristic estimate if exact is too slow.
        # However, N=20 is small enough that a decent B&B is fast.
        
        # Let's precompute for each pair (i,j) the list of k such that i,j,k are collinear.
        # If we pick i and j, we cannot pick any k in collinear[i][j].
        
        collinear_with = [set() for _ in range(n)]
        # collinear_with[i] will contain indices j such that i and j are "conflicting" in some way?
        # Actually, the constraint is "no 3 collinear".
        # If we pick a set S, it is valid if for any i, j in S, there is no k in S such that i,j,k are collinear.
        
        # Precompute: for each pair (i, j), list of k such that i, j, k are collinear.
        pair_conflicts = {}
        for i in range(n):
            for j in range(i + 1, n):
                conflicts = []
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(points[i], points[j], points[k]):
                        conflicts.append(k)
                pair_conflicts[(i, j)] = set(conflicts)
        
        # Now, find max subset S.
        # We can use a backtracking search.
        # Sort points? Maybe not necessary.
        
        best_score = 0
        current_set = []
        
        def backtrack(idx, current_size, forbidden):
            nonlocal best_score
            if idx == n:
                if current_size > best_score:
                    best_score = current_size
                return
            
            # Pruning: if current_size + (n - idx) <= best_score, return
            if current_size + (n - idx) <= best_score:
                return
            
            # Option 1: Do not include point idx
            backtrack(idx + 1, current_size, forbidden)
            
            # Option 2: Include point idx
            # Check if idx conflicts with any pair in current_set?
            # Actually, we need to check if adding idx creates a triple with any two existing points.
            # Let's maintain the set of points in current_set as a list.
            # For efficiency, let's pass the set of indices.
            
            # If we add idx, we must ensure that for any i, j in current_set, 
            # idx is not in pair_conflicts[(i, j)].
            # And also, for any i in current_set, the pair (i, idx) must not have any k in current_set?
            # Wait, the condition is: no 3 points in the *final set* are collinear.
            # If we have a valid set S, and we add x, the new set S U {x} is valid iff:
            # 1. For any i, j in S, x is not collinear with i and j.
            # This is sufficient because if x, i, k were collinear, then x, i, k is a triple.
            # But we only checked x with pairs from S.
            # Yes, any triple in S U {x} either is in S (already valid) or involves x.
            # If it involves x, it must be {x, i, j} with i, j in S.
            # So we just need to check if x is collinear with any pair from S.
            
            valid = True
            for i in range(len(current_set)):
                for j in range(i + 1, len(current_set)):
                    pi = current_set[i]
                    pj = current_set[j]
                    if idx in pair_conflicts[(pi, pj)]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set.append(idx)
                backtrack(idx + 1, current_size + 1, forbidden)
                current_set.pop()
        
        backtrack(0, 0, set())
        return best_score

    def generate_random_points():
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        return points

    best_points = None
    best_score = 21  # Max possible is 20
    
    # Initial random restarts
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        points = generate_random_points()
        if not has_four_collinear(points):
            score = calculate_score(points)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in points]
    
    # Local search
    if best_points:
        # Try to improve by moving one point at a time
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(20):
                if time.time() > deadline:
                    break
                old_point = best_points[i][:]
                # Try a few random new positions
                for _ in range(10):
                    new_point = [random.randint(-100, 100), random.randint(-100, 100)]
                    # Check if new point is distinct
                    if any(new_point == p for p in best_points if best_points.index(p) != i):
                        continue
                    # Check if new configuration has 4 collinear
                    test_points = best_points[:]
                    test_points[i] = new_point
                    if has_four_collinear(test_points):
                        continue
                    score = calculate_score(test_points)
                    if score < best_score:
                        best_score = score
                        best_points = test_points
                        improved = True
                        break # Move to next i
    else:
        # Fallback: use parabola if no valid random set found (should not happen)
        best_points = [[x, x * x] for x in range(-10, 10)]

    return best_points
