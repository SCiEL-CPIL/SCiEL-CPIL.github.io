import numpy as np
import random
import time
from collections import defaultdict

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Generate all grid points
    points = np.array([(x, y) for x in range(N) for y in range(N)])
    num_points = len(points)
    
    # Precompute squared distances matrix
    # dist[i, j] = squared distance between points[i] and points[j]
    dx = points[:, 0:1] - points[1:0, 0]  # This is wrong, let's do it properly
    xs = points[:, 0]
    ys = points[:, 1]
    dx_mat = xs[:, None] - xs[None, :]
    dy_mat = ys[:, None] - ys[None, :]
    dist_sq = dx_mat**2 + dy_mat**2
    
    # dist_sq[i, j] is squared distance from point i to point j
    # dist_sq[i, i] = 0
    
    best_set = []
    best_size = 0
    
    # Precompute for each point, the list of (distance, other_point) sorted by distance
    # Actually, we want to try to add points in a way that minimizes conflicts.
    
    # Let's use a different strategy:
    # 1. Start with a random point
    # 2. Maintain for each candidate point, the set of distances to already selected points
    # 3. When adding a point, we check if it creates an isosceles triangle
    
    # The isosceles condition: three points A, B, C form an isosceles triangle if:
    # - dist(A,B) == dist(A,C) [A is apex]
    # - dist(A,B) == dist(B,C) [B is apex]
    # - dist(A,C) == dist(B,C) [C is apex]
    
    # So when adding point P to set S, we need to check:
    # 1. For each pair (A, B) in S, check if dist(P,A) == dist(P,B) [P is apex]
    #    This means all distances from P to S must be distinct.
    # 2. For each A in S, check if there exists B in S such that dist(P,A) == dist(B,A)
    #    This means for each A, dist(P,A) must not equal dist(B,A) for any B in S\{A}
    # 3. For each A in S, check if there exists B in S such that dist(P,B) == dist(A,B)
    #    This is symmetric to 2.
    
    # Condition 1: all d(P, S) are distinct
    # Condition 2: for each A in S, d(P,A) not in {d(B,A) for B in S, B != A}
    
    # Let's maintain:
    # - current_set: list of selected points
    # - dist_to_selected: for each point in current_set, a set of distances to other selected points
    #   dist_to_selected[i] = set of dist_sq[i, j] for j in current_set, j != i
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        random.seed(num_restarts)
        
        # Random ordering of points
        indices = np.arange(num_points)
        random.shuffle(indices)
        
        current_set = []
        # For each point in current_set, maintain set of distances to other points in current_set
        # dist_sets[i] where i is index in current_set
        dist_sets = []
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
            
            if not current_set:
                current_set.append(idx)
                dist_sets.append(set())
                continue
            
            # Check condition 1: all distances from idx to current_set are distinct
            d_to_set = dist_sq[idx, current_set]
            
            # Check for duplicates in d_to_set
            if len(set(d_to_set)) < len(d_to_set):
                continue
            
            # Check condition 2: for each A in current_set, d(idx, A) not in dist_sets[i]
            # where i is the index of A in current_set
            valid = True
            for i, a_idx in enumerate(current_set):
                d = dist_sq[idx, a_idx]
                if d in dist_sets[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add idx to current_set
            current_set.append(idx)
            
            # Create new dist set for idx
            new_dist_set = set(d_to_set)
            dist_sets.append(new_dist_set)
            
            # Update existing dist sets
            for i in range(len(current_set) - 1):
                dist_sets[i].add(dist_sq[current_set[i], idx])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Convert indices to coordinates
    result = [(int(points[i][0]), int(points[i][1])) for i in best_set]
    return result
