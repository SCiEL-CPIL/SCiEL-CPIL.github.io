import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 590  # safety margin before 630s kill

    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Point (xa, xb, ya, yb) represents (xa + xb*alpha, ya + yb*alpha)
    # Directions: (1, ma + mb*alpha) for ma,mb in 0..6, plus vertical (0,1)
    # Total 50 directions

    n_points = 2401
    n_dirs = 50

    def point_to_idx(xa, xb, ya, yb):
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb

    def idx_to_point(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return (xa, xb, ya, yb)

    # Precompute line structures
    # For each direction d and line l (0..48), store the 49 point indices on that line
    # line_id[d][p] = which line of direction d contains point p

    line_id = np.zeros((n_dirs, n_points), dtype=np.int32)
    line_points = []  # line_points[d][l] = array of 49 int64

    for d in range(n_dirs):
        if d < 49:
            ma, mb = divmod(d, 7)
        else:
            ma, mb = -1, -1  # vertical

        lids = np.zeros(n_points, dtype=np.int32)
        for idx in range(n_points):
            xa, xb, ya, yb = idx_to_point(idx)
            if d < 49:
                # Line through (xa,xb,ya,yb) with direction (1, ma+mb*alpha)
                # y = x*(ma+mb*alpha) + c, where c = ya + yb*alpha - (xa+xb*alpha)*(ma+mb*alpha)
                # c_a = ya - xa*ma - 3*xb*mb
                # c_b = yb - xa*mb - xb*ma
                c1 = (ya - xa * ma - 3 * xb * mb) % 7
                c2 = (yb - xa * mb - xb * ma) % 7
            else:
                c1 = xa
                c2 = xb
            lids[idx] = c1 * 7 + c2
        line_id[d] = lids

        groups = [[] for _ in range(49)]
        for idx in range(n_points):
            groups[lids[idx]].append(idx)
        line_points.append([np.array(g, dtype=np.int64) for g in groups])

    # Maximum flow formulation:
    # Nodes:
    #   Source S
    #   Point nodes P_0..P_2400 (capacity 1 edge from S)
    #   Line nodes L_{d,l} for d in 0..49, l in 0..48 (2450 nodes)
    #   Constraint nodes C_0..C_2400 (one per point, must be "satisfied")
    #   Sink T
    #
    # Edges:
    #   S -> P_i (capacity 1): selecting point i to be "excluded" (removed from N)
    #   P_i -> L_{d,l} (capacity 1): point i lies on line (d,l)
    #   L_{d,l} -> C_j (capacity 1): point j lies on line (d,l) [j != i]
    #   C_j -> T (capacity 1): constraint j is satisfied
    #
    # Wait, this doesn't quite work. Let me rethink.
    #
    # Actually, the constraint is: for each point x, at least one line through x must be clean.
    # A line L is clean for point x if all points on L except x are in N.
    # Equivalently: the number of points NOT in N on line L (including x) is at most 1, and if x is not in N, it must be the only one.
    #
    # Let me think differently. Let S be the complement (points not in N).
    # For each point x and direction d, let L(x,d) be the line through x in direction d.
    # L(x,d) is "clean for x" if |S ∩ L(x,d)| <= 1 and if x ∉ N then x is the only one in S on that line.
    # Actually: L(x,d) is clean for x means all points on L(x,d) except x are in N, i.e., S ∩ (L(x,d) \ {x}) = ∅.
    # This is equivalent to: if x ∈ S, then S ∩ L(x,d) = {x}; if x ∉ S, then S ∩ L(x,d) = ∅.
    # In both cases: |S ∩ L(x,d)| <= 1.
    #
    # So the constraint is: for each x, there exists d such that |S ∩ L(x,d)| <= 1.
    #
    # This is a hitting/covering problem. Let me use a different flow formulation.
    #
    # Alternative: Think of it as a constraint satisfaction / ILP and solve via min-cut.
    # Actually, let me try a direct greedy approach with better heuristics first,
    # then use flow for local improvement.

    # Greedy approach: Start with all points in N, greedily remove points.
    # A point p can be removed if:
    #   1. p has a witness direction d where L(p,d) \ {p} ⊆ N (all other points on line are in N)
    #   2. No other point q loses its last witness
    #
    # For efficiency, track for each point the number of "clean" directions.
    # clean_count[q] = number of directions d where L(q,d) \ {q} ⊆ N
    # A point q can be removed iff clean_count[q] >= 1 after removal.
    #
    # When we remove point p:
    #   For each direction d, let L = L(p,d). For each q on L (q ≠ p):
    #     L(q,d) = L, so removing p from N means p is now in S.
    #     L(q,d) \ {q} still ⊆ N iff all other points on L (except q) are in N.
    #     Before removal: L \ {q} ⊆ N iff S ∩ (L \ {q}) = ∅
    #     After removal: S ∩ (L \ {q}) = {p} (since p was just removed), so this is no longer clean for q.
    #     So clean_count[q] decreases by 1 for each q on L(p,d), q ≠ p.
    #   For p itself: p is now in S. L(p,d) \ {p} ⊆ N iff S ∩ (L(p,d) \ {p}) = ∅.
    #     Before: S ∩ (L(p,d) \ {p}) = ∅ (since p was in N, and we're checking if line is clean)
    #     Actually we need to check: after removing p, is there a direction d where S ∩ (L(p,d) \ {p}) = ∅?
    #     S ∩ (L(p,d) \ {p}) = (old S) ∩ (L(p,d) \ {p}) since p is new to S.
    #     So p's clean directions after removal = directions d where old S ∩ (L(p,d) \ {p}) = ∅.

    in_N = np.ones(n_points, dtype=bool)

    # clean_count[q] = number of directions d where all points on L(q,d) except q are in N
    # i.e., S ∩ (L(q,d) \ {q}) = ∅
    clean_count = np.zeros(n_points, dtype=np.int32)

    # Initialize: all in N, so S is empty, all lines are clean for all points
    clean_count[:] = 50

    # Greedy removal
    removed = 0
    max_passes = 500
    for pass_num in range(max_passes):
        if time.time() - start_time > deadline:
            break

        progress = False
        # Try removing points in order of lowest clean_count (most constrained first)
        # Actually, try points that are "safe" to remove
        candidates = np.where(in_N)[0]
        if len(candidates) == 0:
            break

        # Sort by clean_count ascending (remove most constrained first)
        candidates = candidates[np.argsort(clean_count[candidates])]

        for p in candidates:
            if time.time() - start_time > deadline:
                break
            if not in_N[p]:
                continue

            # Check if p can be removed
            # After removal, p is in S. p needs a direction d where S ∩ (L(p,d) \ {p}) = ∅
            # S ∩ (L(p,d) \ {p}) = (current S) ∩ (L(p,d) \ {p})
            # Since p is not yet in S, current S = {points already removed}
            # So we need: exists d such that no already-removed point is on L(p,d) \ {p}
            # i.e., exists d such that clean_count_p_after >= 1
            # clean_count for p after removal = number of d where (S ∪ {p}) ∩ (L(p,d) \ {p}) = ∅
            # = number of d where S ∩ (L(p,d) \ {p}) = ∅ (since p ∉ L(p,d)\{p})
            # = current clean_count[p] (which counts directions where S ∩ (L(p,d)\{p}) = ∅)
            # Wait, clean_count[p] currently = number of d where all points on L(p,d) except p are in N
            # = number of d where S ∩ (L(p,d) \ {p}) = ∅
            # So p can be removed iff clean_count[p] >= 1

            if clean_count[p] < 1:
                continue

            # Simulate removal
            # For each direction d, find line L(p,d), and for each q on L(p,d) with q != p:
            #   clean_count[q] -= 1 (because p is now in S, so L(q,d) = L(p,d) now has p in S)
            #   But only if p was the "cause" - actually, clean_count[q] counts directions where S ∩ (L(q,d)\{q}) = ∅
            #   After adding p to S: S_new ∩ (L(q,d)\{q}) = S_old ∩ (L(q,d)\{q}) ∪ ({p} if p ∈ L(q,d)\{q})
            #   Since p ∈ L(p,d) and L(q,d) = L(p,d) when q ∈ L(p,d), p ∈ L(q,d)
            #   And p ≠ q, so p ∈ L(q,d)\{q}
            #   So if S_old ∩ (L(q,d)\{q}) = ∅, then S_new ∩ (L(q,d)\{q}) = {p} ≠ ∅
            #   So clean_count[q] decreases by 1 for all q on L(p,d), q ≠ p
            #   But only for directions d where p is on the same line as q.

            # For each direction d:
            #   lid = line_id[d, p]
            #   For each q in line_points[d][lid], q != p:
            #     clean_count[q] -= 1
            #   Check if any q now has clean_count[q] == 0 (invalid)

            affected = []
            for d in range(n_dirs):
                lid = line_id[d, p]
                for q in line_points[d][lid]:
                    if q != p:
                        affected.append((q, d))

            # Apply changes
            for q, d in affected:
                clean_count[q] -= 1

            # Check validity: no point should have clean_count < 1
            # (points in N need clean_count >= 1; points not in N also need clean_count >= 1)
            # Actually ALL points need clean_count >= 1
            min_cc = clean_count.min()
            if min_cc < 1:
                # Rollback
                for q, d in affected:
                    clean_count[q] += 1
                continue

            # Commit removal
            in_N[p] = False
            removed += 1
            progress = True

        if not progress:
            break

    # Build result
    points = []
    for idx in range(n_points):
        if in_N[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        points = [[[0, 0], [0, 0]]]

    return np.array(points, dtype=np.int64)
