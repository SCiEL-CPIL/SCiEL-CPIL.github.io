import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety

    # Grid dimensions
    GRID = 31
    HALF = GRID // 2  # 15
    all_pts = []
    for x in range(-HALF, HALF + 1):
        for y in range(-HALF, HALF + 1):
            all_pts.append((x, y))

    # Helper: Check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    # Helper: Check if any 4 points are collinear in a list of points
    def has_4_collinear(points):
        n = len(points)
        if n < 4:
            return False
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1

                g = gcd(abs(A), abs(B))
                if g != 0:
                    g = gcd(g, abs(C))
                    if g != 0:
                        A //= g
                        B //= g
                        C //= g
                    else:
                        if A == 0 and B == 0:
                            continue
                        if A == 0:
                            if B < 0:
                                A, B, C = -A, -B, -C
                        elif B < 0:
                            A, B, C = -A, -B, -C
                        else:
                            if A < 0:
                                A, B, C = -A, -B, -C
                else:
                    pass

                if A < 0 or (A == 0 and B < 0):
                    A, B, C = -A, -B, -C

                key = (A, B, C)
                if key not in line_counts:
                    line_counts[key] = set()
                line_counts[key].add(i)
                line_counts[key].add(j)

        for pts_on_line in line_counts.values():
            if len(pts_on_line) >= 4:
                return True
        return False

    # Heuristic Score for Simulated Annealing
    def heuristic_score(points):
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1

                g = gcd(abs(A), abs(B))
                if g != 0:
                    g = gcd(g, abs(C))
                    if g != 0:
                        A //= g
                        B //= g
                        C //= g
                else:
                    pass

                if A < 0 or (A == 0 and B < 0):
                    A, B, C = -A, -B, -C

                key = (A, B, C)
                if key not in line_counts:
                    line_counts[key] = set()
                line_counts[key].add(i)
                line_counts[key].add(j)

        score = 0
        for pts_on_line in line_counts.values():
            if len(pts_on_line) >= 3:
                score += len(pts_on_line)

        return score

    # Exact branch and bound for largest subset with no 3 collinear
    # Using Maximum Clique formulation
    def max_no_3_collinear(points):
        n = len(points)

        # Build adjacency matrix for "no three collinear" graph
        # Edge (i, j) exists if adding i and j doesn't create a triple with any existing point
        # Actually, for Maximum Clique: vertices are points, edge (i,j) means i and j can coexist in a "no 3 collinear" set
        # Two points i, j can coexist if there's no point k such that i, j, k are collinear
        # Wait, that's not quite right. The condition is: no three points in the subset are collinear.
        # So for a subset S, for any three distinct points i, j, k in S, they must not be collinear.
        # This is equivalent to: for any pair (i, j) in S, there must be no k in S such that i, j, k are collinear.
        # So edge (i, j) in the graph means: i and j are NOT collinear with any other point in the full set?
        # No, that's not right either.
        #
        # Let me think again. We want the largest subset S such that no three points in S are collinear.
        # This is the independence number of the 3-uniform hypergraph where edges are collinear triples.
        #
        # For Maximum Clique formulation:
        # Create a graph G where vertices are the n points.
        # Edge (i, j) exists if i and j do NOT form a collinear triple with any other point in the full set.
        # No, that's not correct.
        #
        # Actually, the standard approach for this problem is:
        # The conflict graph has vertices = points, and edge (i, j) means i and j "conflict" (i.e., they are part of some collinear triple with a third point).
        # Then we want the largest independent set in this conflict graph? No, that's not right either.
        #
        # Let me reconsider. The condition "no three collinear" is a 3-uniform constraint.
        # A cleaner approach: 
        # For each point i, define conflict[i] = set of pairs (j, k) such that i, j, k are collinear.
        # When building a subset S, if we add point i, we must ensure that no pair (j, k) in S is in conflict[i].
        #
        # For Maximum Clique:
        # Create graph G where vertices are points.
        # Edge (i, j) exists if i and j are "compatible", meaning there is no point k such that i, j, k are collinear.
        # Then a clique in G is a set of points where no three are collinear? 
        # Let's check: If S is a clique in G, then for any i, j in S, edge (i, j) exists, meaning no k exists with i, j, k collinear.
        # But this only checks triples where two points are from S and the third is from the full set.
        # If i, j, k are all in S, and they are collinear, then edge (i, j) should NOT exist (since k exists and i, j, k are collinear).
        # So if S is a clique, then for any i, j in S, no k in the full set makes i, j, k collinear, which implies no k in S makes i, j, k collinear.
        # Conversely, if S has no three collinear, then for any i, j in S, no k in S makes i, j, k collinear.
        # But edge (i, j) in G requires that NO k in the full set makes i, j, k collinear, which is stronger.
        # So a clique in G is a valid "no 3 collinear" set, but not all valid sets are cliques in G.
        # So Maximum Clique in G gives a lower bound on the true answer, not the exact answer.
        #
        # So Maximum Clique is not the right formulation for the exact answer.
        #
        # Let's stick with the branch-and-bound on the 3-uniform hypergraph.
        # But we can optimize it using bitmasks.

        # Precompute collinearity
        # conflict[i] is a bitmask where bit (j, k) is set if i, j, k are collinear
        # Actually, let's store for each i, a list of pairs (j, k) that conflict with i
        conflict = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        conflict[i].add(frozenset((j, k)))
                        conflict[j].add(frozenset((i, k)))
                        conflict[k].add(frozenset((i, j)))

        best = 0

        # Backtracking with pruning
        # S_mask: bitmask of selected points
        # start_idx: next point to consider
        def backtrack(S_mask, start_idx):
            nonlocal best
            current_len = bin(S_mask).count('1')

            # Prune
            if current_len + (n - start_idx) <= best:
                return

            if start_idx == n:
                if current_len > best:
                    best = current_len
                return

            # Option 1: Skip point start_idx
            backtrack(S_mask, start_idx + 1)

            # Option 2: Include point start_idx
            # Check if adding start_idx creates a collinear triple with any two points in S_mask
            valid = True
            for j in range(start_idx):
                if not (S_mask & (1 << j)):
                    continue
                for k in range(j + 1, start_idx):
                    if not (S_mask & (1 << k)):
                        continue
                    if frozenset((j, k)) in conflict[start_idx]:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                backtrack(S_mask | (1 << start_idx), start_idx + 1)

        backtrack(0, 0)
        return best

    # Simulated Annealing
    def simulated_annealing():
        # Initial state: Random 20 points from grid
        current_pts = random.sample(all_pts, 20)

        # Check validity
        if has_4_collinear(current_pts):
            for _ in range(100):
                current_pts = random.sample(all_pts, 20)
                if not has_4_collinear(current_pts):
                    break

        current_score = heuristic_score(current_pts)
        best_pts = current_pts[:]
        best_score = current_score
        best_exact_score = max_no_3_collinear(best_pts)

        T = 10.0
        T_min = 0.01
        alpha = 0.999

        steps = 0
        while T > T_min:
            if time.time() > deadline - 5:
                break

            idx = random.randint(0, 19)
            old_p = current_pts[idx]
            new_p = random.choice(all_pts)

            if new_p == old_p:
                continue

            if new_p in current_pts:
                continue

            current_pts[idx] = new_p

            if has_4_collinear(current_pts):
                current_pts[idx] = old_p
                continue

            new_score = heuristic_score(current_pts)
            delta = new_score - current_score

            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score

                if new_score > best_score or steps % 50 == 0:
                    exact = max_no_3_collinear(current_pts)
                    if exact < best_exact_score:
                        best_exact_score = exact
                        best_pts = current_pts[:]
                        best_score = new_score
                else:
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = current_pts[:]
            else:
                current_pts[idx] = old_p

            T *= alpha
            steps += 1

        return best_pts, best_exact_score

    # Run SA multiple times
    best_global_pts = None
    best_global_score = 21

    for run in range(5):
        if time.time() > deadline - 10:
            break

        pts, score = simulated_annealing()

        if score < best_global_score:
            best_global_score = score
            best_global_pts = pts

        if best_global_score <= 9:
            break

    if best_global_pts is None:
        return [[x, x * x] for x in range(-10, 10)]

    return [list(p) for p in best_global_pts]

# Test harness
if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Valid (no 4 collinear):", not has_4_collinear(pts))
    print("Score:", max_no_3_collinear(pts))
