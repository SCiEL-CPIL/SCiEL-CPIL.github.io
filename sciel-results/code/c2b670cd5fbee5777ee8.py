import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 280.0  # Safety margin
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def dist2(a, b):
        return (a[0] - b[0])**2 + (a[1] - b[1])**2

    def is_valid_addition(p, S, dist_sets):
        """
        S: list of points currently in the set
        dist_sets: dict mapping point index in S to set of squared distances to other points in S
        p: candidate point
        """
        k = len(S)
        if k == 0:
            return True, []
        
        # Calculate distances from p to all points in S
        distances = []
        for i, a in enumerate(S):
            d = dist2(p, a)
            distances.append(d)
            
        # Check 1: Apex at existing point A.
        # dist(p, A) must not be in dist_sets[A]
        # If dist(p, A) == dist(A, B) for some B in S, then dist(p, A) is in dist_sets[A]
        for i, a in enumerate(S):
            if distances[i] in dist_sets[i]:
                return False, []
                
        # Check 2: Apex at p.
        # No two distances from p to S should be equal.
        # i.e., len(set(distances)) should be k
        if len(set(distances)) < k:
            return False, []
            
        return True, distances

    best_set = []
    best_len = 0
    seed = 0
    
    while time.time() - start_time < time_limit:
        # Shuffle points
        random.seed(seed)
        shuffled_points = all_points.copy()
        random.shuffle(shuffled_points)
        
        S = []
        # dist_sets[i] will hold the set of squared distances from S[i] to other points in S
        dist_sets = []
        
        for p in shuffled_points:
            valid, distances = is_valid_addition(p, S, dist_sets)
            if valid:
                # Add p to S
                idx = len(S)
                S.append(p)
                # Update existing dist_sets
                for i in range(idx):
                    dist_sets[i].add(distances[i])
                # Add new dist_set for p
                dist_sets.append(set(distances))
                
        if len(S) > best_len:
            best_len = len(S)
            best_set = S.copy()
            
        # Early exit if we found a very large set? 
        # Max possible is N*N = 4096, but likely much lower.
        # If we get > 100, it's a huge improvement.
        if best_len > 200:
            break
            
        seed += 1
        
    return best_set
# EVOLVE-BLOCK-END
