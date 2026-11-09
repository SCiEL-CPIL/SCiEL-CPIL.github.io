"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START

def _sq_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _greedy_construct_fast(timeout):
    """
    Fast greedy construction using O(n) validity check with smart ordering.
    """
    start_time = time.time()
    best_points = []
    
    # Pre-generate all grid points
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    
    # Run multiple random greedy attempts
    while time.time() - start_time < timeout:
        current_points = []
        current_dist_sets = []
        
        # Use different ordering strategies for diversity
        strategy = random.random()
        if strategy < 0.3:
            shuffled = all_positions[:]
            random.shuffle(shuffled)
        elif strategy < 0.5:
            # Spiral from center
            cx, cy = N // 2, N // 2
            shuffled = sorted(all_positions, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        elif strategy < 0.7:
            # Row-major with random row offset
            shuffled = []
            row_offset = random.randint(0, N-1)
            for i in range(N):
                y = (row_offset + i) % N
                row = [(x, y) for x in range(N)]
                if random.random() < 0.5:
                    row.reverse()
                shuffled.extend(row)
        else:
            # Random with slight bias toward corners
            shuffled = all_positions[:]
            random.shuffle(shuffled)
            # Move some corner points to front
            corners = [(0,0), (N-1,0), (0,N-1), (N-1,N-1)]
            shuffled = [c for c in shuffled if c in corners] + [p for p in shuffled if p not in corners]
        
        for pos in shuffled:
            if time.time() - start_time > timeout:
                break
            
            valid = True
            dists_from_new = set()
            dists_list = []
            
            # Check if pos is apex
            for p in current_points:
                d2 = _sq_dist(pos, p)
                dists_list.append(d2)
                if d2 in dists_from_new:
                    valid = False
                    break
                dists_from_new.add(d2)
            
            # Check if any existing point is apex
            if valid:
                for i, d2 in enumerate(dists_list):
                    if d2 in current_dist_sets[i]:
                        valid = False
                        break
            
            if valid:
                current_points.append(pos)
                new_dists = set(dists_list)
                for i, d2 in enumerate(dists_list):
                    current_dist_sets[i].add(d2)
                current_dist_sets.append(new_dists)
                
                if len(current_points) > len(best_points):
                    best_points = current_points[:]

    return best_points

def _local_search_improve(points, timeout):
    """
    Try to add more points to an existing set via systematic search.
    """
    start_time = time.time()
    current_points = list(points)
    current_dist_sets = []
    
    # Build distance sets from existing points
    for i, p in enumerate(current_points):
        dists = set()
        for j, q in enumerate(current_points):
            if i != j:
                dists.add(_sq_dist(p, q))
        current_dist_sets.append(dists)
    
    all_positions = [(x, y) for x in range(N) for y in range(N)]
    existing_set = set(current_points)
    candidates = [p for p in all_positions if p not in existing_set]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        for pos in candidates[:]:
            if time.time() - start_time > timeout:
                break
            
            valid = True
            dists_from_new = set()
            dists_list = []
            
            # Check if pos is apex
            for p in current_points:
                d2 = _sq_dist(pos, p)
                dists_list.append(d2)
                if d2 in dists_from_new:
                    valid = False
                    break
                dists_from_new.add(d2)
            
            # Check if any existing point is apex
            if valid:
                for i, d2 in enumerate(dists_list):
                    if d2 in current_dist_sets[i]:
                        valid = False
                        break
            
            if valid:
                current_points.append(pos)
                new_dists = set(dists_list)
                for i, d2 in enumerate(dists_list):
                    current_dist_sets[i].add(d2)
                current_dist_sets.append(new_dists)
                candidates.remove(pos)
                improved = True
    
    return current_points

def _sidon_line():
    """Construct points on a line using Sidon-like positions."""
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    return [(x, 0) for x in xs]

def _two_row_construction():
    """Try to construct points on two rows with careful spacing."""
    points = []
    
    # Place some points on row 0
    xs_row0 = []
    used_diffs_row0 = set()
    for x in range(N):
        diffs = {abs(x - y) for y in xs_row0}
        if not (diffs & used_diffs_row0):
            used_diffs_row0 |= diffs
            xs_row0.append(x)
    
    for x in xs_row0:
        points.append((x, 0))
    
    # Try to add points on row 1
    # A point (x, 1) with (a, 0) and (b, 0) forms isosceles if:
    # dist((x,1),(a,0)) == dist((x,1),(b,0)) => (x-a)^2 + 1 == (x-b)^2 + 1 => (x-a)^2 == (x-b)^2
    # => x-a = ±(x-b) => either a=b (impossible) or x-a = -(x-b) => 2x = a+b => x = (a+b)/2
    # So we must avoid x = (a+b)/2 for any pair a,b in xs_row0
    
    forbidden_x = set()
    for i in range(len(xs_row0)):
        for j in range(i+1, len(xs_row0)):
            s = xs_row0[i] + xs_row0[j]
            if s % 2 == 0:
                forbidden_x.add(s // 2)
    
    # Also check within row 1: points (a,1) and (b,1) with (x,0)
    # dist((x,0),(a,1)) == dist((x,0),(b,1)) => (x-a)^2 + 1 == (x-b)^2 + 1 => same as above
    # So we need to also avoid creating isosceles with two row-1 points and one row-0 point
    
    xs_row1 = []
    for x in range(N):
        if x in forbidden_x:
            continue
        # Check with existing row 1 points
        valid = True
        for y in xs_row1:
            # Check isosceles between (x,1), (y,1) and any row 0 point
            for a in xs_row0:
                if _is_isosceles((x,1), (y,1), (a,0)):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            xs_row1.append(x)
            # Update forbidden for future row 1 points
            for a in xs_row0:
                s = x + a
                if s % 2 == 0:
                    forbidden_x.add(s // 2)
    
    for x in xs_row1:
        points.append((x, 1))
    
    return points

def _is_isosceles(p1, p2, p3):
    """Check if three points form an isosceles triangle (including degenerate)."""
    d12_sq = _sq_dist(p1, p2)
    d13_sq = _sq_dist(p1, p3)
    d23_sq = _sq_dist(p2, p3)
    
    if d12_sq == d13_sq and d12_sq != 0:
        return True
    if d12_sq == d23_sq and d12_sq != 0:
        return True
    if d13_sq == d23_sq and d13_sq != 0:
        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave safety margin
    
    # Start with the best known deterministic constructions
    candidates = []
    
    # Sidon line
    sidon_pts = _sidon_line()
    candidates.append(sidon_pts)
    
    # Two-row construction
    two_row_pts = _two_row_construction()
    candidates.append(two_row_pts)
    
    best_points = max(candidates, key=len)
    
    # Try greedy randomized search with time budget
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        # Split time: 70% for greedy, 30% for local search polish
        greedy_time = (time_remaining - 2) * 0.7
        polish_time = (time_remaining - 2) * 0.3
        
        greedy_pts = _greedy_construct_fast(greedy_time)
        if len(greedy_pts) > len(best_points):
            best_points = greedy_pts
        
        # Local search polish on the best solution
        if len(best_points) > 0:
            polished = _local_search_improve(best_points, min(polish_time, deadline - time.time()))
            if len(polished) > len(best_points):
                best_points = polished
    
    return best_points

# EVOLVE-BLOCK-END
