import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 290.0  # Safety margin
    
    best_set = []
    
    # Precompute all points
    points = [(x, y) for x in range(N) for y in range(N)]
    
    # Helper to check validity of adding p to S
    # S is a list of points
    # dist_sets is a list of sets of squared distances corresponding to S
    def try_add(S, dist_sets, p):
        # Calculate squared distances from p to all q in S
        new_dists = []
        for q in S:
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            new_dists.append(dx*dx + dy*dy)
        
        # Check 1: Distances from p to others must be distinct
        if len(set(new_dists)) != len(new_dists):
            return False, None
            
        # Check 2: For each q in S, the distance to p must not be already present in dist_sets[q]
        for i in range(len(S)):
            if new_dists[i] in dist_sets[i]:
                return False, None
                
        return True, new_dists

    # Greedy construction with random order
    def greedy_random():
        S = []
        dist_sets = []
        
        # Randomly shuffle points
        candidates = points[:]
        random.shuffle(candidates)
        
        for p in candidates:
            valid, new_dists = try_add(S, dist_sets, p)
            if valid:
                S.append(p)
                # Update dist_sets for existing points
                for i in range(len(S) - 1):
                    dist_sets[i].add(new_dists[i])
                # Add dist set for new point p
                dist_sets.append(set(new_dists))
                
        return S

    # Run multiple restarts
    while time.time() - start_time < time_limit:
        current_set = greedy_random()
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
# EVOLVE-BLOCK-END
