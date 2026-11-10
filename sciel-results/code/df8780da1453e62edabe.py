import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    N = 11  # 11x11 grid
    # Points are (x, y) where x, y in [0, N-1]
    # We map (x, y) to index i = x * N + y
    
    points = []
    for x in range(N):
        for y in range(N):
            points.append((x, y))
            
    n_points = N * N
    
    # Precompute all lines (as bitmasks)
    # A line is defined by two points. We generate unique lines.
    # To avoid duplicates, we can normalize the line equation or just use a set of frozensets of indices, then convert to masks.
    # Given N=11, n_points=121. Pairs = 121*120/2 = 7260.
    # For each pair, find all points on the line.
    
    line_masks = set()
    
    # Optimization: iterate over all pairs, compute line, add to set of masks
    # This might be slow if done naively in Python for 121 points.
    # 7260 pairs * 121 checks = ~877k operations. This is fine.
    
    for i in range(n_points):
        x1, y1 = points[i]
        for j in range(i + 1, n_points):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction? Not strictly needed for finding points on line, 
            # but helps in identifying the same line from different pairs.
            # We just need the set of points.
            
            mask = (1 << i) | (1 << j)
            
            for k in range(n_points):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                # Check collinearity: (x2-x1)(y3-y1) == (y2-y1)(x3-x1)
                if dx * (y3 - y1) == dy * (x3 - x1):
                    mask |= (1 << k)
            
            # Only consider lines with at least 2 points (which is always true here)
            # We add the mask to the set to remove duplicates (same line found by different pairs)
            line_masks.add(mask)
            
    line_masks = list(line_masks)
    
    # We can filter out lines that cannot possibly have exactly 4 points selected?
    # No, because we select 16 points. A line with 11 points in grid could have 4 selected.
    # A line with 2 points in grid can have at most 2 selected. So it never contributes to score.
    # We only need to consider lines where len(line) >= 4 for scoring?
    # Wait. If a line has 3 points in the grid, and we select all 3, it doesn't count (need 4).
    # If a line has 4 points in the grid, and we select all 4, it counts.
    # If a line has 5 points in the grid, and we select 4, it counts.
    # If a line has 5 points in the grid, and we select 5, it is INVALID.
    
    # So, for scoring:
    # For each line mask L:
    # common = L & S
    # c = common.bit_count()
    # if c >= 5: invalid
    # if c == 4: score += 1
    
    # We can optimize by only checking lines that have at least 4 bits in the grid?
    # If a line has 3 bits in grid, c <= 3, so c != 4 and c < 5.
    # So we can ignore lines with < 4 points in the grid for both validity and scoring.
    
    filtered_line_masks = []
    for m in line_masks:
        if m.bit_count() >= 4:
            filtered_line_masks.append(m)
            
    # Further optimization:
    # If a line has >= 5 points, it's a "danger" line for validity.
    # If a line has >= 4 points, it's a "score" line.
    
    def evaluate(mask):
        score = 0
        for lm in filtered_line_masks:
            c = (lm & mask).bit_count()
            if c >= 5:
                return -1, 0 # Invalid
            if c == 4:
                score += 1
        return 1, score # Valid, score

    # Simulated Annealing
    best_score = -1
    best_mask = 0
    current_mask = 0
    current_score = -1
    
    # Initialize
    indices = list(range(n_points))
    random.shuffle(indices)
    initial_indices = indices[:16]
    for idx in initial_indices:
        current_mask |= (1 << idx)
    
    valid, current_score = evaluate(current_mask)
    if valid:
        best_score = current_score
        best_mask = current_mask
    else:
        # Try to find a valid initial state
        attempts = 0
        while attempts < 1000:
            random.shuffle(indices)
            initial_indices = indices[:16]
            current_mask = 0
            for idx in initial_indices:
                current_mask |= (1 << idx)
            valid, current_score = evaluate(current_mask)
            if valid:
                best_score = current_score
                best_mask = current_mask
                break
            attempts += 1
            
    if best_mask == 0:
        # Fallback: just return first 16, even if invalid? No, must be valid.
        # The problem requires no 5 collinear.
        # If we can't find one quickly, we might fail.
        # But 11x11 grid is large, random 16 points are unlikely to have 5 collinear.
        pass

    T = 10.0
    cooling = 0.999
    
    # Precompute list of all indices for random selection
    all_indices = list(range(n_points))
    
    while time.time() < deadline:
        # Propose move
        # Remove one, add one
        # Get list of set bits and unset bits
        # Doing this efficiently is key.
        
        # Instead of building lists every time, we can just pick random i and j
        # and check if i is in mask and j is not.
        
        i = random.choice(all_indices)
        j = random.choice(all_indices)
        
        if (current_mask & (1 << i)) and not (current_mask & (1 << j)):
            new_mask = current_mask ^ (1 << i) ^ (1 << j)
            valid, new_score = evaluate(new_mask)
            
            if valid:
                if new_score > current_score:
                    current_mask = new_mask
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_mask = new_mask
                else:
                    # Accept worse solution with probability
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
            # If invalid, reject
            
        T *= cooling
        if T < 0.01:
            # Restart with higher T or just continue hill climbing
            T = 1.0
            # Or just break and do hill climbing?
            # Let's just keep going with low T (greedy)
            T = 0.01 

    # Convert best_mask to list of points
    result = []
    for i in range(n_points):
        if best_mask & (1 << i):
            result.append([points[i][0], points[i][1]])
            
    return result

