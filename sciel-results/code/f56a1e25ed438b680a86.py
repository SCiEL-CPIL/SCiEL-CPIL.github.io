import time
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit

    # Pre-generate all grid points
    grid_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    
    # Helper to compute squared distance
    def dist2(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def can_add_optimized(p, S, forbidden_dists):
        """
        Check if p can be added to S.
        S: list of points
        forbidden_dists: list of sets. forbidden_dists[i] contains dist2(S[i], S[j]) for all j != i.
        Returns True if p can be added.
        """
        K = len(S)
        if K < 2:
            return True
        
        px, py = p
        
        # 1. Compute distances from p to all points in S
        d_p = [0] * K
        for i, a in enumerate(S):
            d_p[i] = (px - a[0])**2 + (py - a[1])**2
            
        # 2. Check if p is the apex: d(p, S[i]) == d(p, S[j])
        # This is equivalent to checking if d_p has any duplicates
        seen_dists = set()
        for d in d_p:
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
        # 3. Check if any S[i] is the apex: d(p, S[i]) == d(S[i], S[j])
        # We precomputed forbidden_dists[i] which contains all d(S[i], S[j]) for j != i
        for i in range(K):
            if d_p[i] in forbidden_dists[i]:
                return False
                
        return True

    def add_point_optimized(S, forbidden_dists, P):
        """
        Add P to S and update forbidden_dists.
        """
        K = len(S)
        
        # Compute distances from P to all points in S
        new_dists = [0] * K
        for i, a in enumerate(S):
            new_dists[i] = (P[0] - a[0])**2 + (P[1] - a[1])**2
        
        # Update forbidden_dists for existing points
        for i in range(K):
            forbidden_dists[i].add(new_dists[i])
            
        # Add new point to S
        S.append(P)
        
        # Add forbidden distances for the new point P (index K)
        # forbidden_dists[K] will contain distances from P to all other points in S
        new_forbidden = set(new_dists)
        forbidden_dists.append(new_forbidden)

    # Random Seed for reproducibility
    random.seed(42)

    # We will run multiple random restarts
    while time.time() < deadline:
        # Create a shuffled list of points for this restart
        points = grid_points[:]
        random.shuffle(points)
        
        S = []
        forbidden_dists = []
        
        for P in points:
            if can_add_optimized(P, S, forbidden_dists):
                add_point_optimized(S, forbidden_dists, P)
        
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # If we've already found a very large set, we might break early?
        # Or continue to try to find even larger.
        # Given the optimization, we can run many restarts.
        
    return best_set
# EVOLVE-BLOCK-END
