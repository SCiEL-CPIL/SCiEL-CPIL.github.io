import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 58.0  # Safety margin before 60s limit
    
    def elapsed():
        return time.time() - t0

    def compute_max_area_count_fast(pts):
        """
        Compute the maximum multiplicity of any non-zero doubled area.
        """
        n = len(pts)
        if n < 3:
            return 0

        P = np.array(pts, dtype=np.int64)

        # Build all pairwise difference vectors (i, j) with i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]
        dx = d[:, 0].astype(np.int64)
        dy = d[:, 1].astype(np.int64)
        M = len(dx)

        # Compute all pairwise cross products at once
        cross = np.outer(dx, dy) - np.outer(dy, dx)
        
        # Take upper triangle (u < v) to avoid double counting
        tri = np.triu(cross, k=1)
        nz = tri[tri != 0]
        if len(nz) == 0:
            return 0
        nz_abs = np.abs(nz)
        vals, cnts = np.unique(nz_abs, return_counts=True)
        return int(cnts.max())

    def score_of(pts):
        return compute_max_area_count_fast(pts)

    # --- Strategy: Target-Area Constraint Satisfaction ---
    # We hypothesize that forcing many triangles to have a specific area T
    # will yield a high multiplicity.
    # We do this by starting with a base configuration and greedily adding points
    # that maximize the count of triangles with area T involving the new point.
    
    best_points = None
    best_score = 0
    
    # Candidate base configurations to start the greedy process
    # We use small structured sets that are likely to have many parallel lines
    bases = []
    
    # 1. Two parallel lines
    bases.append([(x, 0) for x in range(6)] + [(x, 1) for x in range(6)])
    
    # 2. Three parallel lines
    bases.append([(x, 0) for x in range(5)] + [(x, 1) for x in range(5)] + [(x, 2) for x in range(5)])
    
    # 3. Grid 4x4
    bases.append([(x, y) for x in range(4) for y in range(4)])
    
    # 4. Grid 5x4
    bases.append([(x, y) for x in range(5) for y in range(4)])
    
    # 5. Lattice with spacing 2
    bases.append([(2*x, 0) for x in range(5)] + [(2*x, 2) for x in range(5)])
    
    # 6. Diagonal lines
    bases.append([(x, x) for x in range(6)] + [(x, x+1) for x in range(6)])

    # Candidate target areas
    # Small integers are more likely to appear frequently in bounded grids
    target_areas = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    
    # Pre-compute all valid points in the box
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    for base in bases:
        if elapsed() > DEADLINE:
            break
            
        for T in target_areas:
            if elapsed() > DEADLINE:
                break
                
            # Start with the base configuration
            current_pts = list(base)
            used_set = set(current_pts)
            
            # Greedy expansion
            # We want to add points p such that for many existing points q,
            # cross(p, q) = T or -T (relative to some origin? No, relative to each other)
            # Actually, for a new point P_new, and existing points P_i, P_j,
            # the new triangles are (P_new, P_i, P_j).
            # This is O(N^2) per candidate point, which is 30^2 = 900 ops.
            # There are ~3600 candidate points. 900 * 3600 = 3.2M ops per step.
            # 24 steps -> 76M ops. Might be slow in Python.
            
            # Optimization:
            # Instead of checking all pairs (i, j), we can check for each existing point i,
            # what vectors v from i to P_new satisfy cross(v, w) = T for some w?
            # No, the area is determined by the triple.
            # Let's stick to a simpler heuristic:
            # For a new point P, count how many existing pairs (A, B) have area T with P.
            # Area(P, A, B) = 0.5 * | (A-P) x (B-P) |
            # (A-P) x (B-P) = A x B - A x P - P x B + P x P (which is 0)
            # = A x B - P x (B - A)
            # We want | A x B - P x (B-A) | = 2T
            
            # Let's iterate through all points in the box as candidates for the next addition.
            # To speed this up, we can sample candidates or use a smarter scoring.
            
            # Alternative: Just add points that are "close" to forming the target area with existing points.
            # Let's try a simpler greedy: Add the point that maximizes the number of existing points
            # that form a triangle of area T with it and ANY other existing point.
            
            # Actually, let's just do a fast local search on the base, but biased towards
            # configurations that already have high multiplicity.
            
            # Let's try a different tactic:
            # Start with the base.
            # While we have room, pick a random point from the box.
            # Calculate the score if added.
            # Keep the best.
            
            # To make it faster, we'll only consider a subset of candidates or use a heuristic.
            # Heuristic: Points that are collinear with existing points are bad (area 0).
            # Points that are "orthogonal" to many difference vectors might be good?
            
            # Let's just run a fast local search starting from the base, but with a twist:
            # We will try to "push" points to specific lattice sites that are known to be good.
            
            # Actually, the most effective strategy from the "inspiration" was just good initialization + local search.
            # The current program already does that.
            # The new strategy is to explicitly target a specific area T.
            
            # Let's implement a specific greedy:
            # 1. Start with base.
            # 2. For each empty slot, evaluate all possible points in a local neighborhood of the centroid
            #    or a set of "good" lattice points.
            # 3. Pick the one that maximizes the count of triangles with area T.
            
            # To avoid O(N^2 * Candidates) cost, we will limit candidates to a random sample of 100 points per step.
            
            pts = list(base)
            used = set(pts)
            
            for _ in range(30 - len(pts)):
                if elapsed() > DEADLINE:
                    break
                
                best_candidate = None
                best_candidate_score = -1
                
                # Sample candidates
                candidates = random.sample(all_points, min(100, len(all_points) - len(used)))
                
                for cand in candidates:
                    if cand in used:
                        continue
                    
                    # Calculate score if cand is added
                    # We only care about the increase in count for area T
                    # New triangles: (cand, p_i, p_j) for all i < j in pts
                    # Area = 0.5 * | (p_i - cand) x (p_j - cand) |
                    
                    count_T = 0
                    # Vectorized check for speed
                    P_arr = np.array(pts, dtype=np.int64)
                    C_arr = np.array(cand, dtype=np.int64)
                    
                    # Difference vectors from C
                    diffs = P_arr - C_arr
                    dx = diffs[:, 0]
                    dy = diffs[:, 1]
                    
                    # Cross product of all pairs
                    # This is O(N^2)
                    cross = np.outer(dx, dy) - np.outer(dy, dx)
                    tri = np.triu(cross, k=1)
                    nz = tri[tri != 0]
                    if len(nz) > 0:
                        count_T = np.sum(np.abs(nz) == 2*T)
                    
                    if count_T > best_candidate_score:
                        best_candidate_score = count_T
                        best_candidate = cand
                
                if best_candidate is not None:
                    pts.append(best_candidate)
                    used.add(best_candidate)
                else:
                    # Fallback: random point
                    while True:
                        p = random.choice(all_points)
                        if p not in used:
                            pts.append(p)
                            used.add(p)
                            break
            
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = list(pts)
                
            # Local search refinement on the result
            current_pts = list(pts)
            current_score = s
            random.seed(42) # Reset seed for consistency in refinement
            
            no_improve = 0
            while elapsed() < DEADLINE and no_improve < 500:
                idx = random.randint(0, 29)
                old_x, old_y = current_pts[idx]
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
                
                if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                    continue
                new_pt = (new_x, new_y)
                if new_pt == (old_x, old_y):
                    continue
                is_valid = True
                for j in range(30):
                    if j == idx and current_pts[j] == new_pt:
                        continue
                    if current_pts[j] == new_pt:
                        is_valid = False
                        break
                if not is_valid:
                    continue
                
                current_pts[idx] = new_pt
                new_score = score_of(current_pts)
                
                if new_score > current_score:
                    current_score = new_score
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_points = list(current_pts)
                else:
                    current_pts[idx] = (old_x, old_y)
                    no_improve += 1

    # Final fallback if nothing was found (shouldn't happen)
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Found {len(pts)} points")
    print(f"Score: {compute_max_area_count_fast(pts)}")
