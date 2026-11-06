import time
import random
import itertools

def _score(points):
    """
    Calculate the maximum multiplicity of a positive doubled area.
    """
    counts = {}
    n = len(points)
    # Unpack for speed
    coords = points
    for i in range(n):
        x1, y1 = coords[i]
        for j in range(i + 1, n):
            x2, y2 = coords[j]
            for k in range(j + 1, n):
                x3, y3 = coords[k]
                det = abs(x1 * (y2 - y3) - x2 * (y1 - y3) + x3 * (y1 - y2))
                if det > 0:
                    counts[det] = counts.get(det, 0) + 1
    if not counts:
        return 0
    return max(counts.values())

def construct():
    start_time = time.time()
    # Deadline: 85 seconds to be safe (limit is 90s, evaluator kills at 90s)
    # But task says "Keep an EXPLICIT deadline guard... safety margin".
    # Let's use 80s.
    deadline = 80.0
    
    # Initial configuration: 6x5 grid
    # Centered at origin?
    # x: -2, -1, 0, 1, 2, 3
    # y: -2, -1, 0, 1, 2
    # This fits in [-30, 30].
    # Actually, the seed was [(x,y) for x in range(6) for y in range(5)]
    # Which is x: 0..5, y: 0..4.
    # Shifting doesn't change the relative areas.
    # We'll stick to the seed's coordinates to ensure we start at the same baseline.
    
    initial_points = [(x, y) for x in range(6) for y in range(5)]
    
    best_points = initial_points[:]
    best_score = _score(best_points)
    
    current_points = best_points[:]
    
    # Local Search
    # We will try to move one point at a time.
    # To speed up, we can precompute the set of occupied points for O(1) lookup.
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx]
        
        # Generate a list of candidate moves
        # Strategy: Try a few local moves and a few global moves
        candidates = []
        
        # Local moves: within distance 3
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                np = (old_p[0] + dx, old_p[1] + dy)
                if -30 <= np[0] <= 30 and -30 <= np[1] <= 30:
                    candidates.append(np)
        
        # Global moves: random in box
        for _ in range(5):
            np = (random.randint(-30, 30), random.randint(-30, 30))
            candidates.append(np)
        
        # Shuffle candidates
        random.shuffle(candidates)
        
        # Try each candidate
        # We need to check if the candidate is already occupied by another point
        # We can do this by checking if np is in current_points (excluding idx)
        
        # Optimization: Keep a set of current points
        # But updating the set on every move is overhead.
        # Since n=30, checking `if np in current_points` is O(30).
        # 30 * 100 candidates = 3000 ops. Negligible.
        
        best_candidate = None
        best_candidate_score = -1
        best_candidate_idx = -1
        
        # We only want to accept if it improves the score
        # Or if it's equal? No, we want to maximize.
        
        for np in candidates:
            if np == old_p:
                continue
            
            # Check collision
            # Remove old_p from consideration for collision?
            # If np is in current_points, it must be at a different index.
            # If np is at idx, it's the same point (skipped above).
            # So if np in current_points, it's a collision.
            if np in current_points:
                continue
            
            # Apply move
            current_points[idx] = np
            
            # Calculate score
            # This is the bottleneck.
            s = _score(current_points)
            
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = np
                best_candidate_idx = idx
            
            # Revert for next candidate
            current_points[idx] = old_p
        
        # Apply the best candidate if it improved the global best
        if best_candidate_score > best_score:
            best_score = best_candidate_score
            best_points = current_points[:] # Wait, current_points was reverted to old_p in the loop
            # We need to re-apply the best candidate
            best_points[best_candidate_idx] = best_candidate
            current_points = best_points[:]
            
    return best_points
