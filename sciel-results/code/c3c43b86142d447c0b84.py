import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3 # alpha^2 = 3

    # Field operations
    def add(x, y):
        a1, b1 = x // p, x % p
        a2, b2 = y // p, y % p
        return ((a1 + a2) % p) * p + ((b1 + b2) % p)

    def sub(x, y):
        a1, b1 = x // p, x % p
        a2, b2 = y // p, y % p
        return ((a1 - a2) % p) * p + ((b1 - b2) % p)

    def mul(x, y):
        a1, b1 = x // p, x % p
        a2, b2 = y // p, y % p
        real = (a1 * a2 + k * b1 * b2) % p
        imag = (a1 * b2 + a2 * b1) % p
        return real * p + imag

    # Precompute lines
    # Directions: (1, m) for m in 0..48, and (0, 1)
    directions = [(1, m) for m in range(q)] + [(0, 1)]
    
    # Represent a line by (point_on_line, direction)
    # Or just generate all lines
    # Total lines: q * (q + 1) = 49 * 50 = 2450
    
    lines = []
    line_points = []
    for x0 in range(q):
        for y0 in range(q):
            for dx, dy in directions:
                # Skip duplicate directions? (1, m) and (0,1) are distinct.
                # But (1, m) and (1, m') are distinct.
                # A line is defined by point and direction.
                # To avoid duplicates, we can canonicalize.
                # Or just store all.
                pts = set()
                for t in range(q):
                    tx = add(x0, mul(t, dx))
                    ty = add(y0, mul(t, dy))
                    pts.add(tx * q + ty) # Encode point as int
                lines.append((x0, y0, dx, dy))
                line_points.append(pts)

    # Initial S from current program
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                # Field element x = xa + xb*alpha
                # Field element y = ya + yb*alpha
                # Unital: 2*xa + ya*ya - yb*yb*k == 0 ?
                # Wait, the current program uses:
                # in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                # This is Trace(x) + Norm(y) = 0?
                # Trace(x) = x + x^7 = 2 * xa.
                # Norm(y) = y * y^7 = ya^2 - k yb^2.
                # So 2xa + ya^2 - k yb^2 = 0.
                # Yes.
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        # This point is in S
                        x_enc = xa * p + xb
                        y_enc = ya * p + yb
                        S.add(x_enc * q + y_enc)

    # Optimization: Try to add points
    # Precompute isolating lines for each point in S
    # For q in S, IL(q) = { line_idx | line contains q and line contains no other point in S }
    
    def get_isolating_lines(S_set):
        # Returns dict: point_enc -> list of line_idx
        il = {pt: [] for pt in S_set}
        for idx, pts in enumerate(line_points):
            # Count how many points of S_set are in this line
            count = 0
            for pt in S_set:
                if pt in pts:
                    count += 1
            if count == 1:
                # Find the point
                for pt in S_set:
                    if pt in pts:
                        il[pt].append(idx)
                        break
        return il

    # This is O(|S| * |Lines| * |S|) which is too slow.
    # Better: For each line, find intersection with S.
    
    def compute_il(S_set):
        il = {pt: [] for pt in S_set}
        S_list = list(S_set)
        for idx, pts in enumerate(line_points):
            # Find points in S_set that are in this line
            in_line = [pt for pt in S_list if pt in pts]
            if len(in_line) == 1:
                pt = in_line[0]
                il[pt].append(idx)
        return il

    start_time = time.time()
    deadline = start_time + 550
    
    # Initial IL
    il = compute_il(S)
    
    # Candidates: points not in S
    all_points = set(range(q * q))
    candidates = list(all_points - S)
    random.seed(42)
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in candidates:
            if time.time() > deadline:
                break
            # Try adding p
            # Check if p has an isolating line in S ∪ {p}
            # i.e., a line through p with 0 points from S.
            has_iso_p = False
            for idx, pts in enumerate(line_points):
                if p in pts:
                    # Check if any other point in S is in pts
                    # We can precompute for each line the set of points in S?
                    # Or just check.
                    # Since S is static during the check, we can use a flag.
                    # But we need to know if pts intersects S.
                    # Let's just iterate.
                    is_empty = True
                    for s_pt in S:
                        if s_pt in pts:
                            is_empty = False
                            break
                    if is_empty:
                        has_iso_p = True
                        break
            
            if not has_iso_p:
                continue
            
            # Check if any q in S loses its last isolating line
            # q loses a line if the line passes through p.
            # So we need to check for each q in S:
            # Does q have an isolating line that does NOT pass through p?
            
            valid = True
            for q in S:
                # Check if all isolating lines of q pass through p
                # i.e., for all idx in il[q], p is in line_points[idx]
                # If there is at least one idx in il[q] such that p NOT in line_points[idx], q is safe.
                safe = False
                for idx in il[q]:
                    if p not in line_points[idx]:
                        safe = True
                        break
                if not safe:
                    valid = False
                    break
            
            if valid:
                # Add p to S
                S.add(p)
                # Update IL
                # We can recompute IL for efficiency if S is small, but it's ~150.
                # Recomputing is O(|Lines| * |S|). 2450 * 150 = 367500.
                # This is fast.
                il = compute_il(S)
                improved = True
                # Remove p from candidates
                candidates.remove(p)
                
    # Construct N
    N = []
    for x in range(q):
        for y in range(q):
            pt_enc = x * q + y
            if pt_enc not in S:
                N.append([[x // p, x % p], [y // p, y % p]])
                
    return np.array(N, dtype=np.int64)
