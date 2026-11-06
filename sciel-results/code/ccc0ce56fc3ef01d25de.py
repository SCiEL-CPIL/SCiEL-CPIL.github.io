"""
Strategy: Simulated Annealing with Vectorized Incremental Scoring.
1. Vectorize the area computation for a set of points.
2. Use an incremental update strategy for moves: when a point is moved, 
   only the triangles involving that point change. This reduces evaluation 
   cost from O(N^3) to O(N^2) per step.
3. Use Simulated Annealing to explore the configuration space, 
   accepting worse moves to escape local maxima.
"""

import time
import numpy as np
from collections import Counter
import random

def _compute_areas_vectorized(points):
    """
    Compute absolute doubled areas for all triples in a set of points.
    Returns a numpy array of areas (excluding zeros).
    """
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    
    # We want |det(B-A, C-A)| for i<j<k
    # det(B-A, C-A) = (Bx-Ax)(Cy-Ay) - (By-Ay)(Cx-Ax)
    
    # To vectorize efficiently for all triples is tricky without huge memory.
    # N=30, triples = 4060.
    # We can generate indices.
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    # Vectors
    # B - A
    bx_ax = pts[j_idx, 0] - pts[i_idx, 0]
    by_ay = pts[j_idx, 1] - pts[i_idx, 1]
    # C - A
    cx_ax = pts[k_idx, 0] - pts[i_idx, 0]
    cy_ay = pts[k_idx, 1] - pts[i_idx, 1]
    
    dets = bx_ax * cy_ay - by_ay * cx_ax
    areas = np.abs(dets)
    
    # Exclude zero areas
    non_zero_areas = areas[areas > 0]
    return non_zero_areas

def _get_max_multiplicity(areas):
    if len(areas) == 0:
        return 0
    # Using Counter on numpy array is slow. 
    # Better: np.unique with return_counts
    unique, counts = np.unique(areas, return_counts=True)
    return int(np.max(counts))

def _score(points):
    areas = _compute_areas_vectorized(points)
    return _get_max_multiplicity(areas)

def _incremental_score_update(points, moved_idx, old_pos, new_pos):
    """
    Calculate the change in score when point moved_idx changes from old_pos to new_pos.
    This is an approximation or exact?
    Actually, calculating the exact new max multiplicity requires updating the histogram.
    
    Let's maintain a global histogram of areas.
    When a point moves, the areas of all triangles involving that point change.
    Number of such triangles: O(N^2) ~ 30*29/2 = 435.
    
    We can:
    1. Identify all triples involving moved_idx.
    2. Remove their old areas from the histogram.
    3. Calculate their new areas.
    4. Add new areas to the histogram.
    5. Find the new max count.
    
    This is O(N^2) per move, which is much faster than O(N^3).
    """
    n = len(points)
    
    # We need a global state for the histogram to make this fast.
    # But for simplicity and robustness in a single function, 
    # let's just re-compute if N is small? 
    # No, the directive asks for structural change. 
    # Let's implement a class or closure to hold the histogram.
    
    pass # Placeholder, will implement in construct

class Annealer:
    def __init__(self, points):
        self.points = np.array(points, dtype=np.int64)
        self.n = len(self.points)
        self.histogram = Counter()
        self.areas_cache = {} # (i,j,k) -> area
        
        # Precompute indices for all triples
        self.triples = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                for k in range(j+1, self.n):
                    self.triples.append((i, j, k))
                    
        # Initialize histogram
        for i, j, k in self.triples:
            a = self._calc_area(i, j, k)
            if a > 0:
                self.histogram[a] += 1
                self.areas_cache[(i,j,k)] = a
            else:
                self.areas_cache[(i,j,k)] = 0
                
        self.current_score = max(self.histogram.values()) if self.histogram else 0

    def _calc_area(self, i, j, k):
        p1 = self.points[i]
        p2 = self.points[j]
        p3 = self.points[k]
        # det(p2-p1, p3-p1)
        d = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
        return d

    def _get_triples_involving(self, idx):
        """Yield triples and their position (0,1,2) where idx is located."""
        for t in self.triples:
            if t[0] == idx:
                yield t, 0
            elif t[1] == idx:
                yield t, 1
            elif t[2] == idx:
                yield t, 2

    def move_point(self, idx, new_pos):
        """
        Move point idx to new_pos. Update histogram and score.
        Returns the score difference.
        """
        old_pos = self.points[idx].copy()
        
        # 1. Remove old areas for triples involving idx
        affected_triples = list(self._get_triples_involving(idx))
        
        for t, _pos in affected_triples:
            old_area = self.areas_cache[t]
            if old_area > 0:
                self.histogram[old_area] -= 1
                if self.histogram[old_area] == 0:
                    del self.histogram[old_area]
        
        # 2. Update point position
        self.points[idx] = new_pos
        
        # 3. Calculate and add new areas
        for t, _pos in affected_triples:
            new_area = self._calc_area(t[0], t[1], t[2])
            self.areas_cache[t] = new_area
            if new_area > 0:
                self.histogram[new_area] = self.histogram.get(new_area, 0) + 1
        
        # 4. Update score
        if self.histogram:
            new_score = max(self.histogram.values())
        else:
            new_score = 0
            
        diff = new_score - self.current_score
        self.current_score = new_score
        
        return diff

    def revert_move(self, idx, old_pos):
        """Revert the last move for point idx."""
        # This is complex to track exactly without storing state.
        # Easier: just apply the move back to old_pos.
        self.move_point(idx, old_pos)
        
def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Initial configuration: 3x10 grid
    initial_points = []
    for y in range(3):
        for x in range(10):
            initial_points.append((x, y))
            
    annealer = Annealer(initial_points)
    best_score = annealer.current_score
    best_points = [tuple(p) for p in annealer.points]
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    max_steps = 100000
    
    # Bounds
    min_coord = -30
    max_coord = 30
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        if T < min_T:
            break
            
        # Choose a random point to move
        idx = random.randint(0, annealer.n - 1)
        old_pos = annealer.points[idx].copy()
        
        # Generate a random new position
        # To keep points distinct, we might need to check.
        # For now, allow overlap in intermediate steps? 
        # The task requires "exactly 30 distinct integer coordinate pairs".
        # If we allow overlaps, the final result must be distinct.
        # Better to ensure distinctness during the move.
        
        # Random perturbation: move by -2 to 2 in x and y
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        new_pos = (old_pos[0] + dx, old_pos[1] + dy)
        
        # Check bounds
        if new_pos[0] < min_coord or new_pos[0] > max_coord:
            continue
        if new_pos[1] < min_coord or new_pos[1] > max_coord:
            continue
            
        # Check distinctness
        # This is O(N) check.
        is_distinct = True
        for i in range(annealer.n):
            if i == idx:
                continue
            if tuple(annealer.points[i]) == new_pos:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        # Perform move
        delta_score = annealer.move_point(idx, new_pos)
        
        # Acceptance criterion
        if delta_score > 0:
            # Always accept improvement
            if annealer.current_score > best_score:
                best_score = annealer.current_score
                best_points = [tuple(p) for p in annealer.points]
        else:
            # Accept with probability exp(delta / T)
            prob = np.exp(delta_score / T)
            if random.random() < prob:
                # Accept
                if annealer.current_score > best_score:
                    best_score = annealer.current_score
                    best_points = [tuple(p) for p in annealer.points]
            else:
                # Revert
                annealer.revert_move(idx, old_pos)
                
        T *= cooling_rate
        
    return best_points

# Fallback if Annealer fails or takes too long to initialize
if __name__ == "__main__":
    # This block is just for local testing
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Points: {pts}")
