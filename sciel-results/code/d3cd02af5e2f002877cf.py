import random

N = 64

def construct():
    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    best_set = []
    
    # Try multiple random orders
    for attempt in range(50):
        random.seed(attempt)
        points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(points)
        
        current_set = []
        # Set of squared distances between all pairs in current_set
        pair_dists = set()
        
        for p in points:
            valid = True
            
            # 1. Check if P forms isosceles with any pair (A,B) in S
            # This is equivalent to:
            # a) d(P,A) == d(P,B) for some A,B in S
            # b) d(P,A) == d(A,B) for some A,B in S
            
            # Compute distances from P to all points in S
            d_to_s = []
            for s in current_set:
                d = dist_sq(p, s)
                d_to_s.append(d)
                
            # Check condition (a): any two distances from P to S are equal
            # If len(d_to_s) > 1, check for duplicates
            if len(d_to_s) > 1:
                if len(set(d_to_s)) < len(d_to_s):
                    valid = False
            
            # Check condition (b): any distance from P to S is equal to a pair distance in S
            if valid:
                for d in d_to_s:
                    if d in pair_dists:
                        valid = False
                        break
            
            if valid:
                # Add P to current_set
                current_set.append(p)
                # Update pair_dists with distances from P to all existing points
                for s in current_set[:-1]:
                    pair_dists.add(dist_sq(p, s))
                    
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
