import time
import random
import math

N = 64

def construct():
        start_time = time.time()
        deadline = start_time + 290
        
        all_points = [(x, y) for x in range(N) for y in range(N)]
        best_set = []
        best_size = 0
        
        def greedy_select(points_order):
            selected = []
            # dist_from[i] is a set of squared distances from selected[i] to all other selected points
            dist_from = []
            
            for p in points_order:
                px, py = p
                valid = True
                d_p = []
                
                # Calculate distances from P to all selected points
                for i in range(len(selected)):
                    ax, ay = selected[i]
                    d_sq = (px - ax)**2 + (py - ay)**2
                    d_p.append(d_sq)
                
                # Check 1: P is apex. All d_p must be unique.
                if len(set(d_p)) != len(d_p):
                    valid = False
                else:
                    # Check 2: For each A, d(P,A) must not be equal to d(A,B) for any B
                    for i in range(len(selected)):
                        d_pa = d_p[i]
                        # dist_from[i] contains distances from selected[i] to all OTHER selected points
                        if d_pa in dist_from[i]:
                            valid = False
                            break
                            
                if valid:
                    selected.append(p)
                    # Update dist_from
                    new_dist_set = set()
                    for i in range(len(selected) - 1):
                        new_dist_set.add(d_p[i])
                    dist_from.append(new_dist_set)
                    
                    # Update existing dist_from sets with new distance to P
                    for i in range(len(selected) - 1):
                        dist_from[i].add(d_p[i])
                        
            return selected

        # Run multiple restarts
        # We can also try different random seeds or shuffling strategies
        while time.time() < deadline:
            random.shuffle(all_points)
            current_set = greedy_select(all_points)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set
                
            # If we have a lot of time, we could try local search (swapping points)
            # But for now, let's maximize restarts.
            
        return best_set
