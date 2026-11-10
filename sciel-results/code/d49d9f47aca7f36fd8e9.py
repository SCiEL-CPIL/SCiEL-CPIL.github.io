# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using local search on the complement S.
    We want to maximize |S| such that the complement N is a Nikodym set.
    Condition: For every x, there exists a line L through x such that (L \ {x}) \cap S = empty.
    
    Strategy: Start with a valid baseline (greedy add), then iteratively try to remove
    points from N (i.e., add to S) to maximize |S|. Use local search with single and
    pair removals to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 570  # 60s safety margin
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines for all 50 directions
    direction_lines = []
    
    # Vertical
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[pt][dir] = line_idx
    point_line = [[0] * 50 for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
    
    # For each point and direction, get the set of other points on that line
    # line_neighbors[pt][dir] = list of other points on the line through pt in direction dir
    line_neighbors = [[[] for _ in range(50)] for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                line_neighbors[pt_idx][dir_idx] = [y for y in line if y != pt_idx]
    
    # in_S[i] = True if i in S (not in N)
    # witness_count[i] = number of directions d such that (Line(i, d) \ {i}) \cap S is empty
    
    def compute_witness_count(in_S):
        """Compute witness counts for a given in_S array."""
        witness_count = [0] * n
        for i in range(n):
            for d in range(50):
                # Check if line through i in direction d has no S points other than i
                has_S = False
                for y in line_neighbors[i][d]:
                    if in_S[y]:
                        has_S = True
                        break
                if not has_S:
                    witness_count[i] += 1
        return witness_count
    
    def is_valid(in_S):
        """Check if in_S defines a valid Nikodym set (complement N)."""
        witness_count = compute_witness_count(in_S)
        return all(w >= 1 for w in witness_count)
    
    def try_add_to_S(in_S, witness_count, idx):
        """Try to add idx to S. Returns True if successful, updates in_S and witness_count."""
        # Check if adding idx to S would make any point have witness_count 0
        for d in range(50):
            for y in line_neighbors[idx][d]:
                if witness_count[y] == 1:
                    return False
        # Safe to add
        in_S[idx] = True
        for d in range(50):
            for y in line_neighbors[idx][d]:
                witness_count[y] -= 1
        return True
    
    def try_remove_from_S(in_S, witness_count, idx):
        """Try to remove idx from S. Returns True if successful, updates in_S and witness_count."""
        # Check if removing idx from S would be valid
        # When we remove idx from S, points on lines through idx gain back a witness
        # But we need to ensure that no point currently has witness_count 0
        # Actually, removing from S only increases witness counts, so it's always valid
        # if the current state is valid. But we should verify.
        in_S[idx] = False
        for d in range(50):
            for y in line_neighbors[idx][d]:
                witness_count[y] += 1
        return True
    
    # Phase 1: Greedy Add to get a baseline
    in_S = [False] * n
    witness_count = [50] * n
    current_size = 0
    
    order = list(range(n))
    random.shuffle(order)
    
    for idx in order:
        if time.time() > deadline - 30:
            break
        if try_add_to_S(in_S, witness_count, idx):
            current_size += 1
    
    best_size = current_size
    best_S = in_S[:]
    best_witness = witness_count[:]
    
    # Phase 2: Local search - try to add more points to S
    # First, try single additions
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        # Shuffle order for diversity
        candidates = [i for i in range(n) if not in_S[i]]
        random.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline - 20:
                break
            if try_add_to_S(in_S, witness_count, idx):
                current_size += 1
                improved = True
    
    if current_size > best_size:
        best_size = current_size
        best_S = in_S[:]
        best_witness = witness_count[:]
    
    # Phase 3: Try pair additions to escape local minima
    # Reset to best state
    in_S = best_S[:]
    witness_count = best_witness[:]
    current_size = best_size
    
    # Try pairs of points that are both not in S
    not_in_S = [i for i in range(n) if not in_S[i]]
    random.shuffle(not_in_S)
    
    # Limit pair attempts to avoid timeout
    max_pair_attempts = min(len(not_in_S) * 20, 5000)
    pair_attempts = 0
    
    while pair_attempts < max_pair_attempts and time.time() < deadline - 10:
        # Pick two random points not in S
        if len(not_in_S) < 2:
            break
        i1, i2 = random.sample(not_in_S, 2)
        
        # Try adding both
        # First check if we can add i1
        can_add_i1 = True
        for d in range(50):
            for y in line_neighbors[i1][d]:
                if witness_count[y] == 1:
                    can_add_i1 = False
                    break
            if not can_add_i1:
                break
        
        if can_add_i1:
            # Temporarily add i1
            old_witness = witness_count[:]
            in_S[i1] = True
            for d in range(50):
                for y in line_neighbors[i1][d]:
                    witness_count[y] -= 1
            
            # Now check if we can add i2
            can_add_i2 = True
            for d in range(50):
                for y in line_neighbors[i2][d]:
                    if witness_count[y] == 1:
                        can_add_i2 = False
                        break
                if not can_add_i2:
                    break
            
            if can_add_i2:
                # Add i2
                in_S[i2] = True
                for d in range(50):
                    for y in line_neighbors[i2][d]:
                        witness_count[y] -= 1
                current_size += 2
                # Remove from not_in_S list
                not_in_S.remove(i1)
                not_in_S.remove(i2)
            else:
                # Revert i1
                in_S[i1] = False
                for d in range(50):
                    for y in line_neighbors[i1][d]:
                        witness_count[y] += 1
        pair_attempts += 1
    
    if current_size > best_size:
        best_size = current_size
        best_S = in_S[:]
        best_witness = witness_count[:]
    
    # Phase 4: More single additions after pair moves
    in_S = best_S[:]
    witness_count = best_witness[:]
    current_size = best_size
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        candidates = [i for i in range(n) if not in_S[i]]
        random.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline - 5:
                break
            if try_add_to_S(in_S, witness_count, idx):
                current_size += 1
                improved = True
    
    if current_size > best_size:
        best_size = current_size
        best_S = in_S[:]
        best_witness = witness_count[:]
    
    # Build result N = complement of S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    if not pts:
        # Fallback: full set
        pts = all_points
        
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
