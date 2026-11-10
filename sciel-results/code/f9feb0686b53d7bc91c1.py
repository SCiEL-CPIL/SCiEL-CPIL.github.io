import numpy as np
import time
import heapq

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
            sub_tbl[i, j] = ((a0 - b0) % p) * p + ((a1 + b1) % p) # Note: sub is (a-b), so (a0-b0), (a1-b1)
            # Correction: sub_tbl[i,j] = i - j
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
    # Vertical lines: x = c
    # Slanted lines: y = m*x + b
    lines = []
    line_id = 0
    point_to_lines = {pt: [] for pt in range(total_points)}
    
    # Map point to line indices
    def add_line(pts):
        nonlocal line_id
        lines.append(pts)
        for pt in pts:
            point_to_lines[pt].append(line_id)
        line_id += 1

    # Vertical lines
    for c in range(q):
        pts = [c * q + y for y in range(q)]
        add_line(pts)
    
    # Slanted lines
    for m in range(q):
        for b in range(q):
            pts = []
            for x in range(q):
                y = add_tbl[mul_tbl[m, x], b]
                pts.append(x * q + y)
            add_line(pts)

    # For each point x not in U, find the unique line intersecting U at exactly one point
    # For each point x in U, find all lines through x
    
    # Precompute witness lines for x not in U
    witness_line = {}
    for x in outside_U:
        for lid, L in enumerate(lines):
            if x in L:
                # Check intersection with U
                inter = [pt for pt in L if pt in U]
                if len(inter) == 1:
                    witness_line[x] = lid
                    break
    
    # For x in U, we need the costs for each line through x
    # Initialize avail
    avail = np.ones(total_points, dtype=np.int8)
    
    # Heap for available points
    # (cost, point_id, version)
    heap = []
    version = np.zeros(total_points, dtype=np.int32)
    
    # Precompute line costs for U points
    # cost_u[x][lid] = sum of avail[y] for y in L \ {x}
    # This is expensive to initialize fully if we do it for all x in U and all lines.
    # But we can compute it on the fly or initialize efficiently.
    
    # Let's initialize costs.
    # For x not in U:
    costs_non_u = np.zeros(total_points, dtype=np.int32)
    for x in outside_U:
        lid = witness_line[x]
        L = lines[lid]
        # Sum avail[y] for y in L, y != x
        # Since all are avail initially, it's 48
        costs_non_u[x] = 48
        
    # For x in U:
    # We need to track the cost for each line through x.
    # Let's store a list of (cost, lid) for each x in U?
    # Or just the min cost and the best line.
    # But when avail changes, we need to update.
    
    # To simplify, let's use a simpler greedy without dynamic updates first, 
    # but with a good ordering.
    # Order points by: 
    # 1. x not in U: by degree (48)
    # 2. x in U: by min degree (48)
    # Since all degrees are 48 initially, order is arbitrary.
    # Let's just iterate and pick the one that forbids the fewest *currently available* points.
    
    S = set()
    forbidden = np.zeros(total_points, dtype=np.int8)
    
    # We will iterate total_points times.
    # In each iteration, we find the best x.
    # To speed up, we can precompute the list of points to check.
    
    # Actually, the dynamic update approach is better for quality.
    # Let's implement the heap-based greedy with updates.
    
    # Data structures for U points:
    # u_lines[x] = list of line ids through x
    u_lines = {x: [] for x in U}
    for lid, L in enumerate(lines):
        for x in L:
            if x in U:
                u_lines[x].append(lid)
                
    # For each x in U, maintain costs for each line
    # u_costs[x] = dict {lid: cost}
    # u_min_cost[x] = min cost
    # u_best_line[x] = lid with min cost
    
    u_costs = {}
    u_min_cost = {}
    u_best_line = {}
    
    for x in U:
        u_costs[x] = {}
        min_c = 100
        best_l = -1
        for lid in u_lines[x]:
            L = lines[lid]
            # Cost is sum of avail[y] for y in L, y != x
            # Initially 48
            c = 48
            u_costs[x][lid] = c
            if c < min_c:
                min_c = c
                best_l = lid
        u_min_cost[x] = min_c
        u_best_line[x] = best_l
        heapq.heappush(heap, (min_c, x, 0))
        
    for x in outside_U:
        heapq.heappush(heap, (costs_non_u[x], x, 0))
        
    while heap:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        cost, x, ver = heapq.heappop(heap)
        
        if forbidden[x]:
            continue
        if version[x] != ver:
            continue
            
        # Add x to S
        S.add(x)
        forbidden[x] = 1
        avail[x] = 0
        
        # Determine forbidden set F
        if x in outside_U:
            lid = witness_line[x]
            L = lines[lid]
            F = [y for y in L if y != x]
        else:
            lid = u_best_line[x]
            L = lines[lid]
            F = [y for y in L if y != x]
            
        # Update costs for all y in F
        # y becomes unavailable (if it wasn't already)
        # We need to update costs of all z that depend on y
        # z depends on y if y is in F_z (for z not in U) or y is in a line through z (for z in U)
        
        # To avoid double processing, we can just mark y as unavailable and update neighbors.
        # But we need to update the heap entries for neighbors.
        
        for y in F:
            if forbidden[y]:
                continue
            # y is becoming forbidden (unavailable)
            # Wait, y is in F, so if we pick x, y cannot be in S.
            # So y is effectively "removed" from consideration.
            # But y might have been in the heap.
            # We don't need to push y again.
            # But we need to update the costs of other points that depend on y.
            
            # Who depends on y?
            # 1. z not in U such that y in F_z.
            #    This means z and y are on the same witness line.
            #    Let L_w be the witness line of y (if y not in U) or lines through y (if y in U).
            #    Actually, if y not in U, F_z contains y iff z and y are on the same witness line.
            #    If y in U, F_z contains y iff z is on a line through y?
            #    No. F_z is defined by the line chosen for z.
            #    If z not in U, F_z is fixed. y in F_z iff z, y on same witness line.
            #    If z in U, F_z is the best line. y in F_z iff y is on the best line of z.
            
            # This dependency is complex.
            # Simpler: When avail[y] changes from 1 to 0:
            # For each line L containing y:
            #   For each z in L \ {y}:
            #     If z not in U:
            #        If L is the witness line of z, then cost[z] decreases by 1.
            #     If z in U:
            #        cost_u[z][lid_L] decreases by 1.
            #        Update min_cost[z].
            
            # Let's iterate over lines containing y.
            for lid_y in point_to_lines[y]:
                L_y = lines[lid_y]
                for z in L_y:
                    if z == y:
                        continue
                    if forbidden[z]:
                        continue
                    
                    if z in outside_U:
                        # Check if L_y is the witness line of z
                        if witness_line[z] == lid_y:
                            costs_non_u[z] -= 1
                            version[z] += 1
                            heapq.heappush(heap, (costs_non_u[z], z, version[z]))
                    else:
                        # z in U
                        # Update cost for line lid_y
                        u_costs[z][lid_y] -= 1
                        # Check if this line is the best line for z
                        if u_best_line[z] == lid_y:
                            # Recalculate min
                            new_min = 100
                            new_best = -1
                            for l_id, c in u_costs[z].items():
                                if c < new_min:
                                    new_min = c
                                    new_best = l_id
                            u_min_cost[z] = new_min
                            u_best_line[z] = new_best
                        else:
                            # If the decreased cost is now lower than current min
                            if u_costs[z][lid_y] < u_min_cost[z]:
                                u_min_cost[z] = u_costs[z][lid_y]
                                u_best_line[z] = lid_y
                                
                        version[z] += 1
                        heapq.heappush(heap, (u_min_cost[z], z, version[z]))

    # N = All - S
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
