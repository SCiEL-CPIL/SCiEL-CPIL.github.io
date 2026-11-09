import time
import random

N = 64

def construct():
        start_time = time.time()
        deadline = start_time + 300  # 300s limit
        
        # Precompute all points
        all_points = [(x, y) for y in range(N) for x in range(N)]
        n_points = len(all_points)
        
        # Helper to compute squared distance
        def dist2(i, j):
            x1, y1 = all_points[i]
            x2, y2 = all_points[j]
            dx = x1 - x2
            dy = y1 - y2
            return dx*dx + dy*dy

        best_set = []
        
        for restart in range(1000): # Upper bound, will be cut by time
            if time.time() > deadline - 5:
                break
            
            random.seed(restart)
            order = list(range(n_points))
            random.shuffle(order)
            
            S = []
            D = [] # D[k] is set of dists from S[k] to other points in S
            
            for p in order:
                if time.time() > deadline - 5:
                    break
                
                # Check if p can be added
                valid = True
                dists_from_p = []
                
                for k, q in enumerate(S):
                    d = dist2(p, q)
                    # Apex at P: d must be unique among dists_from_p
                    if d in dists_from_p:
                        valid = False
                        break
                    # Apex at Q: d must not be in D[k]
                    if d in D[k]:
                        valid = False
                        break
                    dists_from_p.append(d)
                
                if valid:
                    S.append(p)
                    # Update D for existing points
                    for k, q in enumerate(S[:-1]): # S[:-1] is old S
                        D[k].add(dist2(p, q))
                    # Add D for new point p
                    D.append(set(dists_from_p))
            
            if len(S) > len(best_set):
                best_set = S.copy()
        
        # Convert indices to coordinates
        return [all_points[i] for i in best_set]
