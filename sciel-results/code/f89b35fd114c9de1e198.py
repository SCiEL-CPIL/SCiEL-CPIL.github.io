import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590
    
    p = 7
    k = 3  # alpha^2 = 3 in F_49 = F_7[alpha]
    
    # Field operations in F_49 = F_7[alpha]/(alpha^2 - 3)
    def mul(a, b):
        a1, a2 = a
        b1, b2 = b
        return ((a1*b1 + k*a2*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All 2401 points indexed 0..2400
    # Point i = (xa, xb, ya, yb) where i = ((xa*7 + xb)*7 + ya)*7 + yb
    n = p**4  # 2401
    all_points = np.zeros((n, 4), dtype=np.int64)
    for i in range(n):
        yb = i % p
        ya = (i // p) % p
        xb = (i // (p*p)) % p
        xa = (i // (p*p*p)) % p
        all_points[i] = [xa, xb, ya, yb]
    
    # Precompute all lines: 2450 lines total (49 slopes * 49 parallel classes + 49 vertical)
    # For each direction, we have 49 parallel lines, each containing 49 points
    # Total: 50 directions * 49 lines = 2450 lines
    
    # Build line membership: for each point, list of (direction, line_idx) pairs
    # direction 0 = vertical, directions 1..49 = slopes (ma, mb) where not both zero
    
    # For efficiency, precompute line assignments
    # Line key for direction d: 
    #   vertical (d=0): key = (xa, xb)
    #   slope (ma,mb) (d>0): key = (ya - mx[0], yb - mx[1]) where mx = mul((xa,xb),(ma,mb))
    
    # We'll store for each point a list of line IDs
    # Line ID: direction * 49 + line_idx_within_direction
    
    # First, count how many lines per direction (should be 49 each)
    num_dirs = 50  # 1 vertical + 49 slopes
    
    # Precompute all lines as arrays of point indices
    # lines[dir][line_idx] = array of 49 point indices
    lines_by_dir = [None] * num_dirs
    
    # Vertical lines (dir = 0)
    vert_dict = {}
    for i in range(n):
        xa, xb = all_points[i, 0], all_points[i, 1]
        key = (xa, xb)
        if key not in vert_dict:
            vert_dict[key] = []
        vert_dict[key].append(i)
    lines_by_dir[0] = [np.array(v, dtype=np.int64) for v in vert_dict.values()]
    
    # Slope lines (dir = 1..49)
    dir_idx = 1
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines_dict = {}
            for i in range(n):
                xa, xb = all_points[i, 0], all_points[i, 1]
                ya, yb = all_points[i, 2], all_points[i, 3]
                mx = mul((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                if c not in lines_dict:
                    lines_dict[c] = []
                lines_dict[c].append(i)
            lines_by_dir[dir_idx] = [np.array(v, dtype=np.int64) for v in lines_dict.values()]
            dir_idx += 1
    
    total_lines = sum(len(ld) for ld in lines_by_dir)  # should be 2450
    
    # For each point, precompute which lines contain it
    # point_lines[i] = list of (dir, line_idx)
    point_lines = [[] for _ in range(n)]
    for d in range(num_dirs):
        for li, line in enumerate(lines_by_dir[d]):
            for pt in line:
                point_lines[pt].append((d, li))
    
    # Witness count: for each point, how many lines have all other points in N
    # Initially all points in N, so witness_count[i] = num_dirs for all i
    
    def is_valid_exclusion(removed_set):
        """Check if removing removed_set from the full plane still gives a valid Nikodym set."""
        # A point x is "safe" if there exists a line through x where all OTHER points on that line are in N
        # i.e., all other points on that line are NOT in removed_set
        for x in range(n):
            if x in removed_set:
                continue  # x is not in N, doesn't need a witness
            has_witness = False
            for (d, li) in point_lines[x]:
                line = lines_by_dir[d][li]
                clean = True
                for q in line:
                    if q != x and q in removed_set:
                        clean = False
                        break
                if clean:
                    has_witness = True
                    break
            if not has_witness:
                return False
        return True
    
    # Greedy approach with witness tracking
    # State: in_N[i] = True if point i is in N
    # witness_count[i] = number of lines through i where all other points are in N
    
    def greedy_remove(order, max_time=None):
        """Try to remove points in the given order, tracking witness counts."""
        in_N = np.ones(n, dtype=bool)
        witness_count = np.full(n, num_dirs, dtype=np.int32)
        removed = set()
        
        for pt in order:
            if max_time is not None and time.time() > max_time:
                break
            if not in_N[pt]:
                continue
            
            # Check if removing pt keeps all other points safe
            # For each line through pt, check if pt is the only witness for any other point on that line
            can_remove = True
            for (d, li) in point_lines[pt]:
                line = lines_by_dir[d][li]
                for x in line:
                    if x == pt:
                        continue
                    if not in_N[x]:
                        continue
                    # Check if this line is a witness for x
                    # (all other points on line are in N)
                    if witness_count[x] > 0:
                        # Need to check if THIS specific line is a witness
                        # A line is a witness for x if all other points on it are in N
                        # Since we're about to remove pt, after removal, this line will NOT be a witness for x
                        # unless there are no other points being removed
                        # Actually, we need: after removing pt, does x still have at least one witness line?
                        # The line (d, li) will no longer be a witness for x (since pt is removed from N)
                        # So x loses one potential witness from this line
                        # But x might have other witness lines
                        # We need: witness_count[x] >= 2 OR (witness_count[x] == 1 AND this line is NOT x's only witness)
                        # Simpler: simulate - check if x has another witness line besides (d,li)
                        # For efficiency, just check: if witness_count[x] == 1, then we need to verify
                        # that (d,li) is not the only witness
                        # Actually, the cleanest check: after removing pt, for each x on this line (x != pt),
                        # x still needs at least one witness line. The line (d,li) is no longer a witness for x.
                        # So we need x to have another witness line.
                        # This is expensive to check naively. Let's use the witness_count approach:
                        # Before removal: witness_count[x] counts lines where all OTHER points are in N
                        # The line (d,li) IS a witness for x currently (since all other points are in N, including pt)
                        # After removing pt: line (d,li) is NO LONGER a witness for x
                        # So x's witness count decreases by 1 for this line
                        # If witness_count[x] == 1, then x would have 0 witnesses -> invalid
                        # If witness_count[x] >= 2, x still has at least one witness -> OK
                        if witness_count[x] == 1:
                            # This line is x's only witness, removing pt breaks it
                            can_remove = False
                            break
                if not can_remove:
                    break
            
            if can_remove:
                in_N[pt] = False
                removed.add(pt)
                # Update witness counts for all other points on lines through pt
                for (d, li) in point_lines[pt]:
                    line = lines_by_dir[d][li]
                    for x in line:
                        if x == pt:
                            continue
                        if in_N[x]:
                            # Line (d,li) was a witness for x, now it's not
                            witness_count[x] -= 1
        
        return removed
    
    # Try multiple random orders
    best_removed = set()
    best_count = 0
    
    num_trials = 12
    for trial in range(num_trials):
        if time.time() > deadline - 30:
            break
        
        order = list(range(n))
        random.seed(1000 + trial * 997)
        random.shuffle(order)
        
        removed = greedy_remove(order, max_time=deadline - 15)
        
        if len(removed) > best_count:
            best_count = len(removed)
            best_removed = removed.copy()
    
    # Local search: try to remove additional points one by one
    if time.time() < deadline - 20 and best_count > 0:
        in_N = np.ones(n, dtype=bool)
        for pt in best_removed:
            in_N[pt] = False
        witness_count = np.zeros(n, dtype=np.int32)
        
        # Recompute witness counts
        for x in range(n):
            if not in_N[x]:
                continue
            cnt = 0
            for (d, li) in point_lines[x]:
                line = lines_by_dir[d][li]
                clean = True
                for q in line:
                    if q != x and not in_N[q]:
                        clean = False
                        break
                if clean:
                    cnt += 1
            witness_count[x] = cnt
        
        # Try to remove more points
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for pt in range(n):
                if time.time() > deadline - 10:
                    break
                if not in_N[pt]:
                    continue
                
                can_remove = True
                for (d, li) in point_lines[pt]:
                    line = lines_by_dir[d][li]
                    for x in line:
                        if x == pt:
                            continue
                        if not in_N[x]:
                            continue
                        if witness_count[x] == 1:
                            can_remove = False
                            break
                    if not can_remove:
                        break
                
                if can_remove:
                    in_N[pt] = False
                    best_removed.add(pt)
                    for (d, li) in point_lines[pt]:
                        line = lines_by_dir[d][li]
                        for x in line:
                            if x == pt:
                                continue
                            if in_N[x]:
                                witness_count[x] -= 1
                    improved = True
    
    # Final validation (safety check)
    if time.time() < deadline - 5:
        if not is_valid_exclusion(best_removed):
            # Fallback: empty removal (full plane is always valid)
            best_removed = set()
    
    # Build result
    pts = [all_points[i] for i in range(n) if i not in best_removed]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result
