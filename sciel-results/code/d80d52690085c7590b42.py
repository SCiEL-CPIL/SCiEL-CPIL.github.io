import random
import time

N = 64

def construct():
    # Precompute all grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def can_add(existing, p):
        # Check if adding p to existing creates an isosceles triangle
        # For any pair A, B in existing, check if (p, A, B) is isosceles
        # Isosceles if d(p,A) == d(p,B) or d(p,A) == d(A,B) or d(p,B) == d(A,B)
        # We use squared distances to avoid sqrt
        n = len(existing)
        if n == 0:
            return True
        if n == 1:
            return True # 2 points cannot form a triangle
            
        # Optimization: precompute distances from p to all existing points
        # d_p[i] = sq_dist(p, existing[i])
        d_p = [sq_dist(p, ex) for ex in existing]
        
        # Check apex at p: d_p[i] == d_p[j] for i != j
        # This means p is equidistant to two existing points.
        seen_dp = set()
        for i in range(n):
            if d_p[i] in seen_dp:
                return False
            seen_dp.add(d_p[i])
            
        # Check apex at existing points A_i
        # For each i, we need to check if d_p[i] == d(A_i, A_j) for any j != i
        # d(A_i, A_j) is sq_dist(existing[i], existing[j])
        # We can precompute the set of squared distances between all pairs in existing?
        # That would be O(n^2) memory/time per check, which is too slow if done inside the loop.
        # But n is small.
        
        # Alternative: for each i, iterate j
        for i in range(n):
            di = d_p[i]
            # If di is 0, p == existing[i], but p is a candidate from grid, 
            # and we ensure p is not in existing? 
            # The greedy loop should ensure p is not already in existing.
            # Actually, we iterate candidates not in existing.
            
            # Check if di == d(existing[i], existing[j]) for any j != i
            for j in range(n):
                if i == j:
                    continue
                d_ij = sq_dist(existing[i], existing[j])
                if di == d_ij:
                    return False
        return True

    best_size = 0
    best_set = []
    
    # Run multiple random restarts
    # Time limit is 300s. We can run many iterations.
    # Each iteration: shuffle points, greedy add.
    # Greedy add is O(M * N_grid * S^2) where S is set size.
    # With S ~ 20, S^2 = 400. N_grid = 4096. M=4096.
    # 4096 * 4096 * 400 ~ 6 * 10^9 operations? Too slow for Python.
    
    # We need to optimize the check.
    
    def can_add_fast(existing, p, existing_dists):
        # existing_dists is a list of sets of squared distances from existing[i] to all other existing points?
        # Or just check on the fly but efficiently.
        
        n = len(existing)
        if n < 2:
            return True
            
        d_p = [sq_dist(p, ex) for ex in existing]
        
        # Check apex at p
        if len(set(d_p)) < n:
            return False
            
        # Check apex at existing points
        # For each i, d_p[i] must not be in the set of distances from existing[i] to any other existing point
        # Let's precompute dist_sets[i] = {sq_dist(existing[i], existing[j]) for j != i}
        # This can be updated incrementally? No, we are rebuilding from scratch each restart.
        # But inside the greedy loop, we add points one by one.
        # We can maintain a list of dist_sets.
        
        pass

    # Let's restructure the greedy loop to maintain state.
    
    def greedy_random():
        current_set = []
        # dist_sets[i] contains squared distances from current_set[i] to all other points in current_set
        dist_sets = []
        
        # Random order of all points
        candidates = points[:]
        random.shuffle(candidates)
        
        for p in candidates:
            # Check if p can be added
            ok = True
            n = len(current_set)
            
            if n >= 2:
                # Calculate distances from p to all points in current_set
                d_p = [sq_dist(p, ex) for ex in current_set]
                
                # 1. Check if p is equidistant to any two points in current_set
                # i.e., len(set(d_p)) < n
                if len(set(d_p)) < n:
                    ok = False
                else:
                    # 2. Check if for any i, d_p[i] equals distance between current_set[i] and current_set[j]
                    for i in range(n):
                        di = d_p[i]
                        # Check if di is in dist_sets[i]
                        if di in dist_sets[i]:
                            ok = False
                            break
            
            if ok:
                # Add p to current_set
                # Update dist_sets
                # New point is at index n
                # dist_sets[n] will be the set of distances from p to all previous points
                new_dist_set = set(d_p) if n > 0 else set()
                # Wait, if n==0 or 1, we didn't compute d_p in the if block properly?
                # Let's compute d_p outside if needed.
                
                # Recompute d_p if not already done
                if n == 0:
                    d_p = []
                elif n == 1:
                    d_p = [sq_dist(p, current_set[0])]
                # If n >= 2, d_p is already computed.
                
                current_set.append(p)
                
                # Update existing dist_sets
                for i in range(n):
                    dist_sets[i].add(d_p[i])
                
                dist_sets.append(set(d_p))
                
        return len(current_set), current_set

    # Optimization: The above is still O(N_grid * S^2) per restart.
    # 4096 * 400 = 1.6M ops per restart.
    # We can do ~1000 restarts in 300s?
    # 1.6M * 1000 = 1.6B ops. Might be too slow in Python.
    # We need to speed it up.
    
    # We can precompute squared distances between all pairs of grid points?
    # 4096 * 4096 = 16M pairs. Memory: 16M integers. ~128MB. Acceptable.
    # Time to compute: 16M * (dx*dx + dy*dy). Fast in numpy? Or just list of lists.
    # Let's use numpy for distance matrix if possible, or just precompute a dict.
    
    # Actually, we can just run fewer restarts or optimize the inner loop.
    # Let's try 100 restarts first to see how fast it is.
    
    # Better: Use a set of points.
    # The check `if di in dist_sets[i]` is O(1).
    # The calculation of `d_p` is O(S).
    # So per candidate, it's O(S).
    # Total per restart: O(N_grid * S).
    # 4096 * 20 = 80k ops.
    # 1000 restarts = 80M ops. This is very fast in Python (< 1s).
    
    # So my previous complexity analysis was wrong. 
    # I thought checking apex at existing points was O(S^2) per candidate.
    # But with `dist_sets`, it's O(S) per candidate (iterate i, check set membership).
    # So total is O(N_grid * S).
    
    best_size, best_set = 0, []
    
    # Run 2000 restarts
    for _ in range(2000):
        s, s_set = greedy_random()
        if s > best_size:
            best_size = s
            best_set = s_set
            # Optional: print progress if needed, but stdout might be captured.
            
    return best_set

# Run construct to get the result
# The function construct is the entry point.
