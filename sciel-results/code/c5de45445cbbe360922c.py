import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 600
    p = 7
    q = 49
    k = 3  # alpha^2 = 3

    # Precompute field tables
    add_tbl = np.zeros((q, q), dtype=np.int64)
    sub_tbl = np.zeros((q, q), dtype=np.int64)
    mul_tbl = np.zeros((q, q), dtype=np.int64)

    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            add_tbl[i, j] = ((a0 + b0) % p) * p + ((a1 + b1) % p)
            sub_tbl[i, j] = ((a0 - b0) % p) * p + ((a1 - b1) % p)
            r0 = (a0 * b0 + a1 * b1 * k) % p
            r1 = (a0 * b1 + a1 * b0) % p
            mul_tbl[i, j] = r0 * p + r1

    total_points = q * q  # 2401

    # Construct Unital U: 2*xa + ya^2 - k*yb^2 = 0
    U = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        x_idx = xa * p + xb
                        y_idx = ya * p + yb
                        U.add(x_idx * q + y_idx)

    all_points_set = set(range(total_points))
    outside_U = all_points_set - U

    # Build all lines
    lines = []

    # Vertical lines: x = c
    for c in range(q):
        pts = [c * q + y for y in range(q)]
        lines.append(pts)

    # Slanted lines: y = m*x + b
    for m in range(q):
        for b in range(q):
            pts = []
            for x in range(q):
                y = add_tbl[mul_tbl[m, x], b]
                pts.append(x * q + y)
            lines.append(pts)

    num_lines = len(lines)

    # Precompute line properties
    line_U_intersections = []
    line_outside_coverage = []
    line_point_sets = []

    for L in lines:
        I = set(pt for pt in L if pt in U)
        cov = set(pt for pt in L if pt in outside_U)
        line_U_intersections.append(I)
        line_outside_coverage.append(cov)
        line_point_sets.append(set(L))

    # Greedy Set Cover with improved scoring
    uncovered = set(outside_U)
    H = set()
    chosen_lines = []

    max_iterations = 5000
    iteration = 0

    while uncovered and iteration < max_iterations:
        if time.time() - start_time > DEADLINE - 60:
            break

        best_line_idx = -1
        best_score = -1

        for i in range(num_lines):
            if not line_outside_coverage[i]:
                continue

            actual_new_cov = 0
            for pt in line_outside_coverage[i]:
                if pt in uncovered:
                    actual_new_cov += 1

            if actual_new_cov == 0:
                continue

            I_L = line_U_intersections[i]
            new_H_cost = 0
            for pt in I_L:
                if pt not in H:
                    new_H_cost += 1

            # Improved score: balance coverage vs cost more carefully
            # Use a smoother function to avoid over-penalizing
            score = actual_new_cov * 5.0 / (new_H_cost + 2) + actual_new_cov * 0.5

            if score > best_score:
                best_score = score
                best_line_idx = i

        if best_line_idx == -1:
            break

        I_L = line_U_intersections[best_line_idx]
        H.update(I_L)
        chosen_lines.append(best_line_idx)

        for pt in lines[best_line_idx]:
            if pt in uncovered:
                uncovered.remove(pt)

        iteration += 1

    # Local search refinement: try to improve by removing redundant lines
    if time.time() - start_time < DEADLINE - 30:
        for _ in range(200):
            if time.time() - start_time > DEADLINE - 15:
                break

            improved = False
            for i in range(len(chosen_lines)):
                if time.time() - start_time > DEADLINE - 15:
                    break

                # Try removing line i
                removed_line = chosen_lines[i]
                removed_U = set(line_U_intersections[removed_line])
                removed_pts = set(lines[removed_line])

                # Check if all points in removed_pts are still covered by other chosen lines
                still_covered = True
                for pt in removed_pts:
                    if pt in outside_U:
                        covered = False
                        for j in range(len(chosen_lines)):
                            if j != i and pt in line_point_sets[chosen_lines[j]]:
                                covered = True
                                break
                        if not covered:
                            still_covered = False
                            break

                if still_covered:
                    # Remove this line
                    chosen_lines.pop(i)
                    # Update H
                    for pt in removed_U:
                        # Check if pt is in any other chosen line
                        in_other = False
                        for j in range(len(chosen_lines)):
                            if pt in line_U_intersections[chosen_lines[j]]:
                                in_other = True
                                break
                        if not in_other:
                            H.discard(pt)
                    improved = True
                    break

            if not improved:
                break

    # Try to add lines that reduce H size (local search for H minimization)
    if time.time() - start_time < DEADLINE - 15:
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 15:
                break

            best_swap = None
            best_improvement = 0

            # Try adding each line not already chosen
            chosen_set = set(chosen_lines)
            for new_line in range(num_lines):
                if new_line in chosen_set:
                    continue
                if time.time() - start_time > DEADLINE - 15:
                    break

                new_line_U = line_U_intersections[new_line]
                new_line_pts = lines[new_line]

                # Count how many new points this line covers
                new_cov = 0
                for pt in new_line_pts:
                    if pt in outside_U and pt not in uncovered:
                        # Check if this point is covered by existing lines
                        already_covered = False
                        for j in range(len(chosen_lines)):
                            if pt in line_point_sets[chosen_lines[j]]:
                                already_covered = True
                                break
                        if not already_covered:
                            new_cov += 1

                # Count how many U points would be added to H
                new_H_pts = 0
                for pt in new_line_U:
                    if pt not in H:
                        new_H_pts += 1

                if new_cov > 0 and new_H_pts < 3:
                    # Check if this reduces total H size when combined with removal
                    # Simplified: just check if new_cov > new_H_pts
                    if new_cov > new_H_pts:
                        improvement = new_cov - new_H_pts
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = new_line

            if best_swap is not None:
                chosen_lines.append(best_swap)
                H.update(line_U_intersections[best_swap])
                for pt in lines[best_swap]:
                    if pt in uncovered:
                        uncovered.discard(pt)
            else:
                break

    # Compute S = U - H
    S = U - H
    N = all_points_set - S

    # Convert to array
    points = []
    for pt in N:
        x = pt // q
        y = pt % q
        xa, xb = x // p, x % p
        ya, yb = y // p, y % p
        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
