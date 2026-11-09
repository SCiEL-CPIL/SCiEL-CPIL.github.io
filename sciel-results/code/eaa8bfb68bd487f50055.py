# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 (F_49 = F_7[alpha], alpha^2 = 3).

    We want N such that for every point x in F_49^2, there exists a line through x
    (in one of the 50 allowed directions: (1, m) for m in F_49, or (0,1)) whose
    other 48 points all lie in N. We minimize |N|, equivalently maximize the
    complement S = F_49^2 \ N.

    Strategy:
      1. Precompute, for each point x, the set of directions d for which the
         punctured line through x in direction d is fully contained in N (i.e.,
         no point on that line other than x is in S).
      2. Start with an initial excluded set S (a random subset of points).
      3. Greedily remove points from S (i.e., add them to N) only when necessary
         to maintain the Nikodym property: a point x is "bad" if it has no valid
         direction. When a point becomes bad, we must add some point on one of its
         lines to N. Use a priority queue to choose the "cheapest" fix: the point
         whose removal from S causes the fewest new bad points.
      4. Multiple restarts with different random initial S to escape local minima.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill

    # Precompute all points as flat indices: idx = xa*q + xb*q + ya*q + yb
    # Actually, let's use a simpler encoding: idx = ((xa * p + xb) * p + ya) * p + yb
    # Or better: map each F_49 element to 0..48, then point to 0..2400
    # F_49 element (a, b) -> a + b*p (since a,b in 0..6, this gives 0..48)
    
    def elem_to_int(a, b):
        return a + b * p
    
    def int_to_elem(v):
        return (v % p, v // p)
    
    # Precompute all 50 directions
    # Directions: (1, m) for m in F_49, and (0, 1)
    # Represent direction as (dx_int, dy_int) where dx_int, dy_int are in 0..48
    directions = []
    for m in range(q):
        directions.append((1, m))  # (1, m)
    directions.append((0, 1))  # vertical direction
    
    # Precompute, for each point, the list of points on the punctured line
    # in each direction. Actually, we need to know: for point x and direction d,
    # which points are on the line {x + t*d : t in F_49, t != 0}?
    
    # Let's precompute for each point x and each direction d, the set of 48 other points.
    # This is 2401 * 50 * 48 = ~5.76 million entries, which is fine.
    
    # Actually, let's be smarter. For a fixed direction d, the lines partition the plane.
    # For point x, the line in direction d contains 49 points. The other 48 are x + t*d for t in F_49\{0}.
    
    # Let's precompute: for each point x (0..2400) and each direction d (0..49),
    # the list of 48 other point indices on the line.
    
    # Point index: idx = x_int * q + y_int, where x_int, y_int in 0..48
    # x_int = xa + xb*p, y_int = ya + yb*p
    
    # For direction (dx, dy) in F_49^2, the line through x is {x + t*(dx, dy) : t in F_49}
    # We need to compute t*(dx, dy) in F_49^2.
    
    # F_49 arithmetic: (a + b*alpha) * (c + d*alpha) = (ac + bd*3) + (ad + bc)*alpha
    # since alpha^2 = 3
    
    def mul_f49(a, b):
        """Multiply two F_49 elements represented as (a0, a1) and (b0, b1)."""
        a0, a1 = a
        b0, b1 = b
        r0 = (a0 * b0 + a1 * b1 * 3) % p
        r1 = (a0 * b1 + a1 * b0) % p
        return (r0, r1)
    
    def add_f49(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    # Precompute all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_int = xa + xb * p
                    y_int = ya + yb * p
                    idx = x_int * q + y_int
                    all_points.append((idx, (xa, xb), (ya, yb)))
    
    # Build a mapping from idx to (x_elem, y_elem) and vice versa
    idx_to_xy = {}
    xy_to_idx = {}
    for idx, (xa, xb), (ya, yb) in all_points:
        x_elem = (xa, xb)
        y_elem = (ya, yb)
        idx_to_xy[idx] = (x_elem, y_elem)
        xy_to_idx[(x_elem, y_elem)] = idx
    
    # Precompute line memberships: for each point and direction, the 48 other points
    # line_members[x_idx][d_idx] = list of 48 point indices
    line_members = [[[] for _ in range(len(directions))] for _ in range(total_points)]
    
    for d_idx, (dx_a, dx_b) in enumerate([(m // p, m % p) for m in range(q)] + [(0, 0)]):
        # Wait, I need to be careful. The directions are (1, m) for m in F_49, and (0,1).
        # Let me recompute.
        pass
    
    # Let me redo the direction representation.
    # Direction (1, m) where m is an F_49 element (ma, mb): dx = (1, 0), dy = (ma, mb)
    # Direction (0, 1): dx = (0, 0), dy = (1, 0)
    
    dir_list = []
    for m in range(q):
        ma = m % p
        mb = m // p
        dir_list.append(((1, 0), (ma, mb)))  # (dx, dy)
    dir_list.append(((0, 0), (1, 0)))  # (0, 1)
    
    num_dirs = len(dir_list)
    
    # Precompute line_members
    line_members = [[[] for _ in range(num_dirs)] for _ in range(total_points)]
    
    for x_idx in range(total_points):
        x_elem, y_elem = idx_to_xy[x_idx]
        for d_idx in range(num_dirs):
            dx, dy = dir_list[d_idx]
            other_points = []
            for t in range(q):
                if t == 0:
                    continue
                t_elem = (t % p, t // p)
                # t * dx
                td_x = mul_f49(t_elem, dx)
                td_y = mul_f49(t_elem, dy)
                # x + t*dx, y + t*dy
                nx = add_f49(x_elem, td_x)
                ny = add_f49(y_elem, td_y)
                other_idx = xy_to_idx[(nx, ny)]
                other_points.append(other_idx)
            line_members[x_idx][d_idx] = other_points
    
    # Now, for a given excluded set S (boolean array of size total_points),
    # a point x is "covered" if there exists a direction d such that none of
    # line_members[x][d] are in S.
    
    # We want to find a large S such that every point is covered.
    
    # Greedy approach:
    # Start with S = empty (all points in N). Then try to add points to S.
    # Actually, let's start with a random S and then fix violations.
    
    # Better: start with S = all points, and remove points until no violations.
    # But that's the opposite direction.
    
    # Let's think differently. We want to maximize |S| subject to: for every x,
    # there exists d such that line_members[x][d] ∩ S = empty.
    
    # This is equivalent to: for every x, at least one of the 50 lines through x
    # (punctured) is entirely outside S.
    
    # Greedy: start with S = all points. Then for each x, if all 50 punctured lines
    # through x intersect S, we must remove at least one point from S on each line.
    # Actually, we just need to remove points so that at least one line is clear.
    
    # Let's use the following greedy:
    # 1. Start with S = all points.
    # 2. For each point x, compute the set of "blocking" directions: those d for which
    #    line_members[x][d] ∩ S ≠ empty.
    # 3. If all 50 directions are blocking, x is uncovered. We need to unblock at least one.
    # 4. To unblock direction d for x, we need to remove all points in line_members[x][d] from S.
    # 5. Choose the direction d that minimizes the number of points to remove (i.e., the
    #    direction with the fewest points currently in S on that line).
    # 6. Remove those points from S.
    # 7. Repeat until all points are covered.
    
    # But this might be too aggressive. Let's try a different approach.
    
    # Actually, the known result is that the minimum Nikodym set size is around p^4 - O(p^3).
    # The seed program achieves size p^4 - p^2*(p+1)/2 = 2401 - 49*8/2 = 2401 - 196 = 2205.
    # So |S| = 196.
    
    # Let's try to find a larger S.
    
    # New idea: use a random initial S of size ~200-300, then fix violations by
    # removing points from S.
    
    def is_covered(x_idx, S):
        """Check if point x_idx is covered given excluded set S."""
        for d_idx in range(num_dirs):
            all_outside = True
            for other in line_members[x_idx][d_idx]:
                if S[other]:
                    all_outside = False
                    break
            if all_outside:
                return True
        return False
    
    def count_covered(S):
        """Count how many points are covered."""
        count = 0
        for x_idx in range(total_points):
            if is_covered(x_idx, S):
                count += 1
        return count
    
    # Precompute for faster coverage checking:
    # For each point x and direction d, count how many points on the punctured line are in S.
    # x is covered by direction d if this count is 0.
    
    def compute_line_counts(S):
        """For each (x, d), compute the number of points on the punctured line in S."""
        counts = np.zeros((total_points, num_dirs), dtype=np.int32)
        for x_idx in range(total_points):
            for d_idx in range(num_dirs):
                cnt = 0
                for other in line_members[x_idx][d_idx]:
                    if S[other]:
                        cnt += 1
                counts[x_idx][d_idx] = cnt
        return counts
    
    def is_all_covered(counts):
        """Check if all points are covered (i.e., each point has at least one direction with count 0)."""
        for x_idx in range(total_points):
            if np.all(counts[x_idx] > 0):
                return False
        return True
    
    # Greedy algorithm:
    # 1. Start with a random S.
    # 2. Compute line counts.
    # 3. While not all covered:
    #    a. Find an uncovered point x.
    #    b. For each direction d, compute the number of points in S on the punctured line.
    #    c. Choose the direction d with the minimum count (cheapest to clear).
    #    d. Remove all points in S on that line from S.
    #    e. Update counts.
    # 4. Return S.
    
    # But updating counts after each removal is expensive. Let's use a different approach.
    
    # Actually, let's just do a simple iterative repair:
    
    def greedy_repair(S, max_iters=100000):
        """Repair S to make it a valid excluded set."""
        S = S.copy()
        for it in range(max_iters):
            # Find uncovered points
            uncovered = []
            for x_idx in range(total_points):
                covered = False
                for d_idx in range(num_dirs):
                    all_outside = True
                    for other in line_members[x_idx][d_idx]:
                        if S[other]:
                            all_outside = False
                            break
                    if all_outside:
                        covered = True
                        break
                if not covered:
                    uncovered.append(x_idx)
            
            if not uncovered:
                break
            
            # For the first uncovered point, find the cheapest direction to clear
            x_idx = uncovered[0]
            best_d = -1
            best_count = q + 1
            for d_idx in range(num_dirs):
                cnt = sum(1 for other in line_members[x_idx][d_idx] if S[other])
                if cnt < best_count:
                    best_count = cnt
                    best_d = d_idx
            
            # Remove all points on the chosen line from S
            for other in line_members[x_idx][best_d]:
                S[other] = False
        
        return S
    
    # Let's try multiple random initializations and keep the best result.
    
    best_S = None
    best_size = -1
    
    num_restarts = 20
    rng = np.random.RandomState(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 60:
            break
        
        # Random initial S with size ~200
        target_size = 200 + rng.randint(0, 50)
        S = np.zeros(total_points, dtype=bool)
        indices = rng.choice(total_points, size=target_size, replace=False)
        S[indices] = True
        
        # Repair
        S = greedy_repair(S)
        
        size = np.sum(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()
        
        if best_size >= 250:  # Good enough, stop early
            break
    
    if best_S is None:
        # Fallback: return empty S (all points in N)
        best_S = np.zeros(total_points, dtype=bool)
    
    # Build the result: N = all points not in S
    points = []
    for idx in range(total_points):
        if not best_S[idx]:
            x_elem, y_elem = idx_to_xy[idx]
            xa, xb = x_elem
            ya, yb = y_elem
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
