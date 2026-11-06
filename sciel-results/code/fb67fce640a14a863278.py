import random
import math
import heapq
from collections import defaultdict

N = 64

def construct():
    best_points = []
    
    # Try multiple random initializations
    for trial in range(20):
        # Randomly select a subset of points to start with
        # Start with a moderate size, e.g., 150-300 points
        # Larger initial sets might take longer to prune but could yield better results
        num_initial = random.randint(150, 300)
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
        S = all_points[:num_initial]
        
        # Map point to index in S for fast lookup? 
        # Or just use the tuple as key in dictionaries.
        
        # dist_counts[p] is a dict: distance -> count
        dist_counts = {}
        # conflict_count[p] is the number of pairs (q, r) such that d(p,q)=d(p,r)
        conflict_count = {}
        
        # Initialize distances and counts
        for i, p in enumerate(S):
            dist_counts[p] = defaultdict(int)
            for j in range(len(S)):
                if i == j:
                    continue
                q = S[j]
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                d_sq = dx * dx + dy * dy
                dist_counts[p][d_sq] += 1
            
            # Calculate conflict count
            cc = 0
            for d_sq, cnt in dist_counts[p].items():
                if cnt >= 2:
                    cc += cnt * (cnt - 1) // 2
            conflict_count[p] = cc
            
        # Max heap of (-conflict_count, point)
        # We need a way to invalidate old entries in the heap
        # We'll just check if the conflict count matches when popping
        heap = []
        for p in S:
            heapq.heappush(heap, (-conflict_count[p], p))
        
        active_points = set(S)
        
        while heap:
            neg_cc, p = heapq.heappop(heap)
            cc = -neg_cc
            
            # Check if this entry is stale
            if p not in active_points:
                continue
            if conflict_count[p] != cc:
                continue
            
            if cc == 0:
                break # No more conflicts
            
            # Remove p
            active_points.remove(p)
            
            # Update neighbors
            for q in list(active_points):
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                d_sq = dx * dx + dy * dy
                
                # Get current count for this distance for q
                cnt = dist_counts[q].get(d_sq, 0)
                if cnt == 0:
                    continue
                
                # The number of conflicts contributed by this distance to q was cnt*(cnt-1)//2
                # After removing p, the count becomes cnt-1
                # New conflicts from this distance: (cnt-1)*(cnt-2)//2
                # Decrease in conflicts: cnt-1
                old_conflicts_from_d = cnt * (cnt - 1) // 2
                new_conf = cnt - 1
                new_conflicts_from_d = new_conf * (new_conf - 1) // 2
                
                delta_cc = old_conflicts_from_d - new_conflicts_from_d
                conflict_count[q] -= delta_cc
                
                # Update the count
                dist_counts[q][d_sq] -= 1
                if dist_counts[q][d_sq] == 0:
                    del dist_counts[q][d_sq]
                
                # Push updated q to heap
                heapq.heappush(heap, (-conflict_count[q], q))
        
        # After pruning, active_points contains a valid set
        if len(active_points) > len(best_points):
            best_points = list(active_points)
            
        # Optional: Try to add more points greedily to the resulting set?
        # The current set is "maximal" in the sense that no point can be removed to fix a conflict?
        # No, it's a set with NO conflicts. We can try to add more points.
        
        # Greedy addition
        remaining_points = [p for p in all_points if p not in active_points]
        random.shuffle(remaining_points)
        
        for p in remaining_points:
            # Check if adding p creates a conflict
            # p creates a conflict if:
            # 1. There are two points q, r in active_points such that d(p,q) = d(p,r)
            # 2. There are two points q, r in active_points such that d(q,r) = d(q,p) (apex q)
            # 3. There are two points q, r in active_points such that d(q,r) = d(r,p) (apex r)
            
            # Efficient check:
            # For condition 1: Check if p has duplicate distances to points in active_points.
            # For conditions 2 & 3: For each q in active_points, check if d(q,p) equals any other distance from q to active_points.
            
            is_valid = True
            
            # Check 1: Duplicates from p
            dists_from_p = {}
            for q in active_points:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                d_sq = dx * dx + dy * dy
                if d_sq in dists_from_p:
                    is_valid = False
                    break
                dists_from_p[d_sq] = True
            
            if not is_valid:
                continue
                
            # Check 2 & 3: Conflicts at existing points
            # For each q in active_points, the distance d(q,p) must be unique among distances from q to other points in active_points.
            # We already know that distances from q to other points in active_points are unique (since active_points is valid).
            # So we just need to check if d(q,p) is already present in the set of distances from q to active_points.
            
            for q in active_points:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                d_sq = dx * dx + dy * dy
                
                # Check if d_sq is in dist_counts[q]
                # Note: dist_counts[q] contains counts of distances from q to active_points.
                # If d_sq is in dist_counts[q], it means there is already a point r in active_points such that d(q,r) = d(q,p).
                # This would form an isosceles triangle (q, r, p) with apex q.
                if d_sq in dist_counts[q]:
                    is_valid = False
                    break
            
            if is_valid:
                # Add p to active_points
                active_points.add(p)
                
                # Update dist_counts for all existing points to include p
                for q in list(active_points):
                    if q == p:
                        continue
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    d_sq = dx * dx + dy * dy
                    dist_counts[q][d_sq] = dist_counts[q].get(d_sq, 0) + 1
                
                # Initialize dist_counts for p
                dist_counts[p] = defaultdict(int)
                for q in active_points:
                    if q == p:
                        continue
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    d_sq = dx * dx + dy * dy
                    dist_counts[p][d_sq] = dist_counts[p].get(d_sq, 0) + 1
        
        if len(active_points) > len(best_points):
            best_points = list(active_points)

    return best_points
