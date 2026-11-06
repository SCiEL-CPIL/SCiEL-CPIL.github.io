import time
import random
import itertools

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute isosceles pairs for each point
# iso_pairs[i] is a list of tuples (j, k) such that triangle (POINTS[i], POINTS[j], POINTS[k]) is isosceles
# Note: We only need to check if {j, k} are in S.
iso_pairs = [[] for _ in range(NUM_POINTS)]

def _is_isosceles(a, b, c):
    d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    return d1 == d2 or d1 == d3 or d2 == d3

# Precomputation
for i in range(NUM_POINTS):
    a = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        b = POINTS[j]
        for k in range(j + 1, NUM_POINTS):
            c = POINTS[k]
            if _is_isosceles(a, b, c):
                # Triangle (a, b, c) is isosceles.
                # This means if a is in S, then b and c cannot both be in S?
                # No, the condition is: if we add p, and there exist a, b in S such that (p, a, b) is isosceles.
                # So for point i, the "blocking" pairs are (j, k) such that (i, j, k) is isosceles.
                iso_pairs[i].append((j, k))
                iso_pairs[j].append((i, k))
                iso_pairs[k].append((i, j))

def is_safe(p_idx, S_set):
    """
    Check if adding p_idx to S_set maintains the no-isosceles property.
    S_set is a set of indices.
    """
    for j, k in iso_pairs[p_idx]:
        if j in S_set and k in S_set:
            return False
    return True

def greedy_construct(rng):
    S = []
    S_set = set()
    # Shuffle points
    points = list(range(NUM_POINTS))
    rng.shuffle(points)
    
    for p in points:
        if is_safe(p, S_set):
            S.append(p)
            S_set.add(p)
            
    # Convert indices back to coordinates
    return [POINTS[i] for i in S]

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = None
    best_len = float('inf')
    
    # Use multiple random seeds/restarts
    # We can estimate how many restarts we can do.
    # One restart is roughly O(N^2 * avg_pairs). 
    # Let's just loop until deadline.
    
    rng = random.Random(42) # Fixed seed for reproducibility of the "random" strategy, or use system time?
    # The task says "random" is allowed. Using a fixed seed makes it deterministic, which is good for debugging, 
    # but for optimization, varying seeds helps. 
    # However, the evaluator might run it once. We should use a fixed seed to ensure consistent "best" behavior 
    # if the random walk is good, or use time-based seed to explore. 
    # Given the stagnation, a fixed seed with a good strategy is safer. 
    # But we can also use a counter to vary the seed per restart.
    
    restart_count = 0
    while time.time() < deadline:
        # Create a new RNG with a different seed each restart
        current_rng = random.Random(restart_count)
        
        S = greedy_construct(current_rng)
        
        if len(S) < best_len:
            best_len = len(S)
            best_set = S
            
        restart_count += 1
        
        # Optional: If we find a very good solution, we might want to do local search?
        # For now, simple restarts.
        
    return best_set
