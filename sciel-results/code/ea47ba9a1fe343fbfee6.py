import time
import random
import numpy as np

N = 64
NUM_POINTS = N * N

# Precompute coordinates
xs = np.arange(N, dtype=np.int32)
ys = np.arange(N, dtype=np.int32)
all_x = np.repeat(xs, N)
all_y = np.tile(ys, N)

# Conflict graph construction
# conflict[i] is a bitmask of points j that conflict with i
def build_conflict_graph():
    # We use a list of integers as bitmasks
    conflict = [0] * NUM_POINTS
    
    # For each potential apex A
    for A in range(NUM_POINTS):
        ax = all_x[A]
        ay = all_y[A]
        
        # Calculate squared distances from A to all other points
        # dx = all_x - ax, dy = all_y - ay
        dx = all_x - ax
        dy = all_y - ay
        dists = dx*dx + dy*dy
        
        # Group points by distance
        # Since distances are small integers, we can use a dictionary or a list of lists
        # Max distance is 2*N*N = 8192. We can use a dict.
        groups = {}
        for j in range(NUM_POINTS):
            if j == A:
                continue
            d = int(dists[j])
            if d not in groups:
                groups[d] = []
            groups[d].append(j)
            
        # For each group of points at the same distance from A,
        # any pair (P, Q) in the group forms an isosceles triangle with A as apex.
        # So P and Q conflict.
        for d, pts in groups.items():
            k = len(pts)
            if k < 2:
                continue
            # Add conflicts for all pairs in pts
            for i_idx in range(k):
                p = pts[i_idx]
                # Conflict with all other points in the group
                # We can do this by OR-ing bits for all other points in the group
                # But doing this pair by pair is O(k^2).
                # Better: for each p in pts, conflict[p] |= bitmask of (pts \ {p})
                # Let's just iterate pairs for simplicity, k is usually small.
                for j_idx in range(i_idx + 1, k):
                    q = pts[j_idx]
                    conflict[p] |= (1 << q)
                    conflict[q] |= (1 << p)
                    
    return conflict

# Precompute the conflict graph at module level to save time in construct()
# This takes a few seconds, which is acceptable.
CONFLICT_GRAPH = build_conflict_graph()

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Local state
    best_set_bits = 0
    best_size = 0
    
    # Helper: check if point P can be added to set represented by bits
    def can_add(P, current_bits):
        return (CONFLICT_GRAPH[P] & current_bits) == 0

    def greedy_fill(current_bits, current_size, remaining_time):
        # Try to add points in random order
        candidates = list(range(NUM_POINTS))
        random.shuffle(candidates)
        
        added_any = False
        for P in candidates:
            if time.time() > remaining_time:
                break
            if (current_bits >> P) & 1:
                continue
            if can_add(P, current_bits):
                current_bits |= (1 << P)
                current_size += 1
                added_any = True
        
        return current_bits, current_size, added_any

    def local_search(initial_bits, initial_size, time_limit):
        current_bits = initial_bits
        current_size = initial_size
        best_bits = initial_bits
        best_size = initial_size
        
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < time_limit and no_improve < max_no_improve:
            # 1. Try to add any valid point
            added = False
            # Sample random candidates
            candidates = random.sample(range(NUM_POINTS), min(200, NUM_POINTS - current_size))
            for P in candidates:
                if (current_bits >> P) & 1:
                    continue
                if can_add(P, current_bits):
                    current_bits |= (1 << P)
                    current_size += 1
                    added = True
                    if current_size > best_size:
                        best_size = current_size
                        best_bits = current_bits
                        no_improve = 0
                    break
            
            if not added:
                no_improve += 1
                
                # 2. Kick: Remove 1-3 random points, then try to add many
                if current_size > 5:
                    kick_size = random.randint(1, 3)
                    kick_size = min(kick_size, current_size - 1) # Keep at least 1
                    
                    # Select points to remove
                    points_in_set = [i for i in range(NUM_POINTS) if (current_bits >> i) & 1]
                    points_to_remove = random.sample(points_in_set, kick_size)
                    
                    for P in points_to_remove:
                        current_bits &= ~(1 << P)
                        current_size -= 1
                    
                    # Try to fill up with greedy
                    # We don't check time limit inside greedy_fill strictly here, 
                    # but the outer loop will check.
                    candidates = list(range(NUM_POINTS))
                    random.shuffle(candidates)
                    
                    for P in candidates:
                        if (current_bits >> P) & 1:
                            continue
                        if can_add(P, current_bits):
                            current_bits |= (1 << P)
                            current_size += 1
                            
                            if current_size > best_size:
                                best_size = current_size
                                best_bits = current_bits
                                no_improve = 0
                    
                    # Reset no_improve after kick to allow more tries
                    no_improve = 0
        
        return best_bits, best_size

    # Main search loop
    # Run multiple rounds of greedy + local search
    for round_num in range(50):
        if time.time() > deadline:
            break
            
        # Start with a random greedy construction
        bits = 0
        size = 0
        
        # Greedy fill
        candidates = list(range(NUM_POINTS))
        random.shuffle(candidates)
        
        for P in candidates:
            if can_add(P, bits):
                bits |= (1 << P)
                size += 1
        
        if size > best_size:
            best_size = size
            best_set_bits = bits
        
        # Local search
        # Allocate remaining time divided among rounds
        # Simple heuristic: give each round a fixed chunk, or just run until deadline
        # Let's give local search a fraction of remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        search_time = time.time() + min(10, remaining / 2)
        
        bits, size = local_search(bits, size, search_time)
        
        if size > best_size:
            best_size = size
            best_set_bits = bits

    # Convert bitset back to list of (x, y) tuples
    result = []
    for i in range(NUM_POINTS):
        if (best_set_bits >> i) & 1:
            x = i % N
            y = i // N
            result.append((x, y))
            
    return result

if __name__ == "__main__":
    points = construct()
    print(f"Found set of size {len(points)}")
    print(points[:10])
