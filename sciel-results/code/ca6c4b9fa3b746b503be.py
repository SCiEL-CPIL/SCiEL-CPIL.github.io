<<<<<<< SEARCH
import random
import time
from itertools import combinations

N = 64
DEADLINE = 320  # safety margin before 330s

def _squared_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _is_valid_point(candidate, points):
    """Check if candidate can be added to points without forming isosceles triangle."""
    n = len(points)
    cand_dists = [_squared_dist(candidate, p) for p in points]
    
    # Check: for each pair (i, j), candidate doesn't create isosceles
    # dist(cand, i) == dist(cand, j) -> isosceles with apex at cand
    # dist(cand, i) == dist(i, j) -> isosceles with apex at i or j
    # dist(cand, j) == dist(i, j) -> isosceles with apex at i or j
    
    for i in range(n):
        for j in range(i + 1, n):
            dij2 = _squared_dist(points[i], points[j])
            if cand_dists[i] == cand_dists[j]:
                return False
            if cand_dists[i] == dij2:
                return False
            if cand_dists[j] == dij2:
                return False
    return True

def _greedy_with_random_order(points, random_order, deadline):
    """Try to add points in the given random order, return best found."""
    current = list(points)
    used_positions = set(points)
    
    for pos in random_order:
        if time.time() > deadline:
            break
        if pos in used_positions:
            continue
        if _is_valid_point(pos, current):
            current.append(pos)
            used_positions.add(pos)
    
    return current

def _generate_random_order():
    """Generate a random ordering of all grid positions."""
    positions = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(positions)
    return positions

def _greedy_with_priority(deadline):
    """Greedy approach with priority based on distance to existing points."""
    current = []
    used_positions = set()
    
    # Start with a random point
    start = (random.randint(0, N-1), random.randint(0, N-1))
    current.append(start)
    used_positions.add(start)
    
    while time.time() < deadline:
        # Find the best candidate: one that maximizes minimum distance to existing points
        best_pos = None
        best_min_dist = -1
        
        # Sample a subset of candidates for speed
        candidates = []
        for _ in range(500):
            x = random.randint(0, N-1)
            y = random.randint(0, N-1)
            pos = (x, y)
            if pos not in used_positions:
                candidates.append(pos)
        
        for pos in candidates:
            if time.time() > deadline:
                break
            min_dist = float('inf')
            for p in current:
                d = _squared_dist(pos, p)
                if d < min_dist:
                    min_dist = d
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = pos
        
        if best_pos is None or not _is_valid_point(best_pos, current):
            # Try a few more random candidates
            found = False
            for _ in range(200):
                x = random.randint(0, N-1)
                y = random.randint(0, N-1)
                pos = (x, y)
                if pos not in used_positions and _is_valid_point(pos, current):
                    current.append(pos)
                    used_positions.add(pos)
                    found = True
                    break
            if not found:
                break
        else:
            current.append(best_pos)
            used_positions.add(best_pos)
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_points = []
    
    # Multiple restarts with different strategies
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Strategy 1: Random order greedy
        random_order = _generate_random_order()
        result = _greedy_with_random_order([], random_order, deadline)
        if len(result) > len(best_points):
            best_points = result
        
        # Strategy 2: Priority-based greedy
        if time.time() > deadline - 5:
            break
        result = _greedy_with_priority(deadline)
        if len(result) > len(best_points):
            best_points = result
    
    return best_points
=======
import random
import time

N = 64
DEADLINE = 320  # safety margin before 330s

def _squared_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _build_conflict_structures():
    """
    Precompute for each grid point:
    - For every other grid point, the squared distance
    - Group distances by value to enable fast isosceles checks
    
    Returns:
        dist_matrix: 2D list where dist_matrix[i][j] = squared distance from point i to point j
        points: list of (x,y) tuples
    """
    points = [(x, y) for x in range(N) for y in range(N)]
    n = len(points)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = _squared_dist(points[i], points[j])
    return points, dist_matrix

def _is_valid_with_structures(candidate_idx, selected_indices, dist_matrix):
    """
    Check if candidate_idx can be added to selected_indices without forming isosceles triangle.
    
    Uses the distance matrix for O(n) check instead of O(n^2).
    For a new point c and existing points p_i, p_j:
    - isosceles at c: dist(c,i) == dist(c,j)
    - isosceles at i: dist(c,i) == dist(i,j)  
    - isosceles at j: dist(c,j) == dist(i,j)
    
    We track:
    - seen_cand_dists: set of distances from candidate to selected points
    - For each selected point i, we need to check if dist(c,i) equals any dist(i,j) for j != i
    """
    n_sel = len(selected_indices)
    if n_sel == 0:
        return True
    
    cand_dists = {}
    for i in selected_indices:
        d = dist_matrix[candidate_idx][i]
        if d in cand_dists:
            return False  # isosceles with apex at candidate
        cand_dists[d] = i
    
    # For each pair of selected points, check against candidate distances
    # More efficient: for each selected point i, check if dist(c,i) matches dist(i,j) for any j
    # Build a set of all pairwise distances among selected points, indexed by the "apex" point
    # Actually, let's use a different approach:
    # For the new point c, for each existing point i:
    #   - dist(c,i) must not equal dist(c,j) for any j != i (checked above via cand_dists)
    #   - dist(c,i) must not equal dist(i,j) for any j != i, j in selected
    #   - dist(c,j) must not equal dist(i,j) for any i != j, i in selected
    
    # The last two conditions can be checked as:
    # For each pair (i,j) in selected: dist(c,i) != dist(i,j) and dist(c,j) != dist(i,j)
    
    # Let's precompute: for each selected point i, what distances does it have to other selected points?
    # Then for candidate c, dist(c,i) must not be in that set.
    
    # Build: for each i in selected, set of distances from i to other selected points
    dist_to_others = {}
    for i in selected_indices:
        dists = set()
        for j in selected_indices:
            if i != j:
                dists.add(dist_matrix[i][j])
        dist_to_others[i] = dists
    
    for i in selected_indices:
        d_ci = dist_matrix[candidate_idx][i]
        if d_ci in dist_to_others[i]:
            return False
    
    return True

def _greedy_build(points, dist_matrix, deadline, strategy='random'):
    """
    Greedy construction using precomputed structures.
    strategy: 'random', 'farthest', 'balanced'
    """
    n = len(points)
    selected = []
    selected_set = set()
    
    # Initial point
    if strategy == 'random':
        start = random.randint(0, n - 1)
    else:
        start = random.randint(0, n - 1)
    selected.append(start)
    selected_set.add(start)
    
    # Maintain a list of candidate indices
    candidates = list(range(n))
    candidates.remove(start)
    random.shuffle(candidates)
    
    while time.time() < deadline:
        if not candidates:
            break
        
        # Try to find a valid point from candidates
        added = False
        # Try a batch of candidates
        batch_size = min(200, len(candidates))
        for _ in range(batch_size):
            idx = candidates.pop(0)
            if _is_valid_with_structures(idx, selected, dist_matrix):
                selected.append(idx)
                selected_set.add(idx)
                added = True
                break
        
        if not added:
            # Shuffle remaining and try again, or stop if we've tried enough
            if len(candidates) < 50:
                break
            random.shuffle(candidates)
    
    return selected

def _local_search_refine(points, dist_matrix, selected, deadline):
    """
    Local search: try swapping one selected point with one unselected point.
    A swap is valid if the new set has no isosceles triangles.
    We also try just adding a point if possible.
    """
    n = len(points)
    selected_set = set(selected)
    unselected = [i for i in range(n) if i not in selected_set]
    random.shuffle(unselected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        unselected_idx = 0
        random.shuffle(unselected)
        
        for rem_idx in range(len(unselected)):
            if time.time() > deadline:
                break
            new_point = unselected[rem_idx]
            
            # Try to add new_point (if not already in selected)
            if new_point not in selected_set:
                if _is_valid_with_structures(new_point, selected, dist_matrix):
                    selected.append(new_point)
                    selected_set.add(new_point)
                    improved = True
                    break
            
            # Try swap: remove one selected, add new_point
            if time.time() > deadline - 2:
                break
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                old_point = selected[i]
                if old_point == new_point:
                    continue
                
                # Check if removing old_point and adding new_point is valid
                # New set: (selected - {old_point}) + {new_point}
                new_selected = [selected[j] for j in range(len(selected)) if j != i]
                
                # Check if new_point is valid against new_selected
                if not _is_valid_with_structures(new_point, new_selected, dist_matrix):
                    continue
                
                # Also need to verify that new_selected itself is valid (it should be, since it's a subset)
                # But we need to check that the full new set is valid
                # _is_valid_with_structures checks new_point against new_selected
                # But we also need to ensure no isosceles among new_selected (inherited from original)
                # Since new_selected is a subset of the original valid set, it's valid.
                
                # Perform the swap
                selected[i] = new_point
                selected_set.discard(old_point)
                selected_set.add(new_point)
                improved = True
                break
            
            if improved:
                break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute structures
    points, dist_matrix = _build_conflict_structures()
    
    best_selected = []
    
    # Multiple greedy restarts with different strategies
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Use a portion of time for greedy, then local search
        greedy_deadline = start_time + min(DEADLINE - 30, (restart + 1) * (DEADLINE - 40) / num_restarts)
        
        strategy = ['random', 'random', 'random', 'balanced'][restart % 4]
        selected = _greedy_build(points, dist_matrix, greedy_deadline, strategy)
        
        # Local search refinement
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            selected = _local_search_refine(points, dist_matrix, selected, deadline - 2)
        
        if len(selected) > len(best_selected):
            best_selected = list(selected)
    
    # Convert indices to (x, y) tuples
    result = [points[i] for i in best_selected]
    
    return result
>>>>>>> REPLACE
<<<<<<< SEARCH
    return best_points
=======
    return result
>>>>>>> REPLACE

Wait, I need to check the last line. The original returns `best_points` but my new code returns `result`. Let me fix that properly.

