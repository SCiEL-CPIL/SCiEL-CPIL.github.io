import numpy as np
import time
from scipy.spatial.distance import cdist

def _hamming_weight(v):
    return int(np.count_nonzero(v))

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580

    def _make_candidate_sets():
        # Base set: all 2^11 binary vectors mapped to {+1, -1}
        # Binary code: all vectors in {0,1}^11
        # Map: 0 -> -1, 1 -> +1
        # Then min distance in Hamming >= d means min squared distance >= 2*d (since each differing coordinate contributes 4 to squared distance)
        # Norm squared is always 11 (since all entries are +/-1)
        # We need min_dist_sq >= max_norm_sq = 11
        # So we need 2*d >= 11, i.e., d >= 5.5, so d >= 6
        # Hamming code of length 11 with minimum distance 6?
        
        # Actually, let's think differently. The known kissing number in dimension 11 is 165.
        # A construction uses the even-weight binary code or related constructions.
        
        # Let me use a different approach: start with a known good construction and try to add points.
        
        # Simple construction: all vectors with entries in {0, +1, -1} with norm squared <= some bound
        # and pairwise distances large enough.
        
        # Let's use the D11 root system approach:
        # All vectors in {+1, -1}^11 (2048 vectors) - but we need min_dist >= norm
        # For vectors in {+1,-1}^11, norm_sq = 11, dist_sq = 2*(11 - dot) = 22 - 2*dot
        # We need 22 - 2*dot >= 11, i.e., 2*dot <= 11, i.e., dot <= 5.5, i.e., dot <= 5
        # So we need a code in {+1,-1}^11 with pairwise inner product <= 5
        # This is equivalent to a binary code in {0,1}^11 with pairwise Hamming distance >= 3 (since dot = 11 - 2*dist, dot <= 5 means 11-2*dist <= 5, 2*dist >= 6, dist >= 3)
        
        # Maximum size of binary code of length 11 with minimum distance 3?
        # This is A(11, 3). The bound gives A(11,3) <= 2^11/11 = 186. 
        # Actually A(11,3) = 512 (extended Hamming? No). Let me think...
        # Hamming code [7,4,3] has 16 codewords. For length 11, we can use shortening or other codes.
        # Actually, the maximum size A(11,3) is known. Let's just enumerate greedily.
        
        # For now, let's use a simpler construction: take all vectors in {+1,-1}^11 that have even weight (sum of bits in {0,1} representation is even).
        # That gives 1024 vectors. But they don't all satisfy the distance constraint.
        
        # Let me just do a greedy construction with the full set of {+1,-1}^11 vectors.
        
        return None

    def _greedy_select(points_pool, max_norm_sq_target):
        """Greedy: select points from pool such that min_dist_sq >= max_norm_sq of selected."""
        n = len(points_pool)
        if n == 0:
            return np.array([]).reshape(0, 11)
        
        # Compute all pairwise distances and norms
        norms_sq = np.sum(points_pool**2, axis=1)
        
        selected = []
        selected_indices = []
        
        # Sort by norm squared, pick low-norm points first to keep max_norm_sq low
        order = np.argsort(norms_sq)
        
        for i in order:
            if len(selected) > 0:
                # Check distance to all selected
                diffs = points_pool[selected] - points_pool[i]
                dists_sq = np.sum(diffs**2, axis=1)
                min_dist = np.min(dists_sq)
                if min_dist < max(np.max(norms_sq[selected]), norms_sq[i]):
                    continue
            selected.append(i)
            selected_indices.append(i)
        
        return points_pool[selected_indices]

    # Strategy 1: Use {+1, -1}^11 vectors with greedy selection
    # Generate all 2^11 = 2048 vectors
    all_binary = np.array([[1 if (i >> b) & 1 else -1 for b in range(11)] for i in range(2048)], dtype=np.int64)
    
    # Sort by norm (all have norm^2 = 11)
    # Greedy: we want pairwise inner product <= 5 (equivalent to Hamming dist >= 3)
    
    # Use a better greedy: sort by some heuristic
    # Compute dot products with a reference and pick antipodal pairs first
    
    # Simple approach: use greedy with random restarts
    best_k = 2
    best_points = np.array([np.ones(11), -np.ones(11)], dtype=np.int64)
    
    def _greedy_code(points, T, time_budget):
        """Find large subset with pairwise inner product <= T."""
        n = len(points)
        if n == 0:
            return np.array([]).reshape(0, 11)
        
        M = points @ points.T
        best_set = list(range(min(2, n)))
        best_size = len(best_set)
        
        while time.time() < time_budget:
            perm = np.random.permutation(n)
            current = []
            for i in perm:
                if not current:
                    current.append(i)
                else:
                    if np.all(M[i, current] <= T):
                        current.append(i)
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = current
                if best_size >= 100:
                    break
        
        return points[best_set]

    # For {+1,-1}^11: T = 5 (inner product <= 5)
    # But we also need min_dist_sq >= max_norm_sq. Since all norms are 11, we need min_dist_sq >= 11.
    # dist_sq = 22 - 2*dot >= 11 => dot <= 5.5 => dot <= 5. So T=5 works.
    
    t_limit1 = min(deadline, time.time() + 300)
    if time.time() < t_limit1:
        pts = _greedy_code(all_binary, 5, t_limit1)
        if len(pts) > best_k:
            best_k = len(pts)
            best_points = pts.copy()
    
    # Strategy 2: Try with T=4 (inner product <= 4, stricter)
    # dist_sq = 22 - 2*dot >= 22-8 = 14 >= 11, so T=4 also works and gives better separation
    t_limit2 = min(deadline, time.time() + 200)
    if time.time() < t_limit2:
        pts2 = _greedy_code(all_binary, 4, t_limit2)
        if len(pts2) > best_k:
            best_k = len(pts2)
            best_points = pts2.copy()
    
    # Verify
    if best_k >= 2:
        k = len(best_points)
        norms = np.sum(best_points**2, axis=1)
        max_norm_sq = int(np.max(norms))
        
        min_dist_sq = float('inf')
        for i in range(k):
            for j in range(i+1, k):
                diff = best_points[i] - best_points[j]
                d_sq = int(np.sum(diff**2))
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
        
        if min_dist_sq >= max_norm_sq:
            return np.around(best_points).astype(np.int64)
    
    # Fallback
    return np.array([np.ones(11), -np.ones(11)], dtype=np.int64)
