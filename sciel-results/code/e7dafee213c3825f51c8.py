import numpy as np
import itertools
import time
import heapq
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    def optimize_sphere_points(n_points, max_iters=2000, lr_init=0.01, lr_decay=0.999, seed=42):
        """Place n_points on sphere of radius 2 in R^11, maximizing min angular separation."""
        rng = np.random.RandomState(seed)
        R = 2.0
        
        # Random initialization on sphere
        points = rng.randn(n_points, D)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms * R
        
        lr = lr_init
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Compute pairwise distances and forces
            # Force on point i is sum of (p_i - p_j) / ||p_i - p_j||^3 for all j != i
            # (inverse square repulsion, projected to tangent plane)
            
            # For efficiency, batch process
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, D)
            dist_sq = np.sum(diff**2, axis=2)  # (n, n)
            np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
            
            # Repulsive force: (p_i - p_j) / dist^3
            inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))  # 1/dist^3
            np.fill_diagonal(inv_dist_cubed, 0.0)
            
            forces = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)  # (n, D)
            
            # Project to tangent plane (remove radial component)
            point_norms = np.linalg.norm(points, axis=1, keepdims=True)
            radial = (points / point_norms) * np.sum(points * forces, axis=1, keepdims=True) / point_norms
            forces = forces - radial
            
            # Update
            points = points + lr * forces
            
            # Re-normalize to sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms * R
            
            lr *= lr_decay
            
        return points
    
    def round_and_prune(points):
        """Round to integers, remove zeros and conflicts."""
        # Round to nearest integer
        int_points = np.round(points).astype(np.int64)
        
        # Remove zero vectors
        mask = np.any(int_points != 0, axis=1)
        int_points = int_points[mask]
        
        # Remove duplicates
        if len(int_points) == 0:
            return int_points
        
        # Use a set-based approach to remove duplicates
        seen = set()
        unique = []
        for v in int_points:
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                unique.append(v)
        
        if len(unique) == 0:
            return np.array([]).reshape(0, D)
        
        int_points = np.array(unique, dtype=np.int64)
        
        # Prune conflicts: keep vectors where dot with all previously kept <= 2
        kept = []
        for v in int_points:
            if len(kept) == 0:
                kept.append(v)
                continue
            kept_arr = np.array(kept)
            dots = kept_arr @ v
            if np.all(dots <= 2):
                kept.append(v)
        
        return np.array(kept, dtype=np.int64) if kept else np.array([]).reshape(0, D)
    
    best_result = np.array([]).reshape(0, D)
    
    # Try different numbers of points on the sphere
    for n_points in [200, 300, 400, 500, 600]:
        if time.time() > deadline - 10:
            break
        
        # Try a few seeds
        for seed in [42, 123, 456]:
            if time.time() > deadline - 5:
                break
            
            points = optimize_sphere_points(n_points, max_iters=1500, seed=seed)
            result = round_and_prune(points)
            
            if len(result) > len(best_result):
                best_result = result
    
    # Also try the original approach as a fallback/complement
    if time.time() < deadline - 30:
        # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
        pos_sets = list(itertools.combinations(range(D), 4))
        signs_list = list(itertools.product([1, -1], repeat=4))
        
        vectors = []
        for p in pos_sets:
            for s in signs_list:
                v = [0] * D
                for pos, sig in zip(p, s):
                    v[pos] = sig
                vectors.append(v)
        
        N4 = len(vectors)
        vectors_v4 = np.array(vectors, dtype=np.int64)
        
        # Build conflict graph
        groups = {p: [] for p in pos_sets}
        idx = 0
        for p in pos_sets:
            for s in signs_list:
                groups[p].append(idx)
                idx += 1
        
        conflict_pairs = []
        for i, p1 in enumerate(pos_sets):
            for j, p2 in enumerate(pos_sets):
                if i >= j:
                    continue
                inter = tuple(set(p1) & set(p2))
                if len(inter) == 3:
                    conflict_pairs.append((p1, p2, inter))
        
        adj = [[] for _ in range(N4)]
        
        for p1, p2, inter in conflict_pairs:
            idxs1 = groups[p1]
            idxs2 = groups[p2]
            
            for i1 in idxs1:
                v1 = vectors[i1]
                s1 = tuple(v1[pos] for pos in inter)
                
                for i2 in idxs2:
                    v2 = vectors[i2]
                    s2 = tuple(v2[pos] for pos in inter)
                    
                    if s1 == s2:
                        adj[i1].append(i2)
                        adj[i2].append(i1)
        
        # Greedy MIS
        alive = [True] * N4
        degrees = [len(a) for a in adj]
        
        heap = [(degrees[i], i) for i in range(N4)]
        heapq.heapify(heap)
        
        selected = []
        
        while heap and time.time() < deadline - 5:
            deg, i = heapq.heappop(heap)
            
            if not alive[i] or degrees[i] != deg:
                continue
                
            selected.append(i)
            alive[i] = False
            
            for j in adj[i]:
                if alive[j]:
                    alive[j] = False
                    degrees[j] -= 1
                    heapq.heappush(heap, (degrees[j], j))
        
        result_v4 = vectors_v4[selected]
        
        # V1 vectors
        v1_candidates = []
        for pos in range(D):
            for sign in [1, -1]:
                v = [0] * D
                v[pos] = 2 * sign
                v1_candidates.append(v)
        
        vectors_v1 = np.array(v1_candidates, dtype=np.int64)
        
        final_v4 = np.vstack([result_v4, vectors_v1])
        
        if len(final_v4) > len(best_result):
            best_result = final_v4
    
    return best_result

