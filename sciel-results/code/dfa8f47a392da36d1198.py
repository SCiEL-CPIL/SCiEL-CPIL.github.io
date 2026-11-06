"""No-isosceles set construction on 64x64 grid."""
import numpy as np
import time
import random

N = 64
NUM_POINTS = N * N
DEADLINE = time.time() + 290

xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)
_all_points = np.column_stack([xs, ys]).astype(np.int32)

def construct():
    global DEADLINE
    DEADLINE = time.time() + 290
    
    random.seed(42)
    np.random.seed(42)
    
    best_S = []
    best_len = 0
    
    # Greedy with restarts
    for restart in range(500):
        if time.time() > DEADLINE - 30:
            break
        
        order = np.random.permutation(NUM_POINTS)
        S = []
        S_pts = np.empty((0, 2), dtype=np.int32)
        pairwise_dists = set()  # All pairwise squared distances in S
        
        for p_idx in order:
            if time.time() > DEADLINE - 15:
                break
            p_idx = int(p_idx)
            K = len(S)
            
            if K <= 1:
                S.append(p_idx)
                if K == 1:
                    S_pts = np.vstack([S_pts, _all_points[p_idx]])
                else:
                    S_pts = _all_points[p_idx].reshape(1, 2)
                continue
            
            p = _all_points[p_idx]
            
            # Fast check: distances from p to S
            diff = S_pts - p
            dists = np.sum(diff * diff, axis=1).astype(np.int64)
            
            # Check distinctness
            if len(np.unique(dists)) != K:
                continue
            
            # Check against pairwise distances
            valid = True
            for d in dists:
                if d in pairwise_dists:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add to set
            S.append(p_idx)
            S_pts = np.vstack([S_pts, p])
            for j in range(K):
                diff_j = S_pts[j] - p
                d_j = int(np.sum(diff_j * diff_j))
                pairwise_dists.add(d_j)
        
        if len(S) > best_len:
            best_len = len(S)
            best_S = S[:]
    
    # Local search
    S = best_S[:]
    
    for iteration in range(2000):
        if time.time() > DEADLINE - 5:
            break
        
        improved = False
        S_set = set(S)
        
        # Pre-compute pairwise distances for S
        K = len(S)
        S_pts = _all_points[S]
        pairwise_dists = set()
        for i in range(K):
            for j in range(i + 1, K):
                diff = S_pts[j] - S_pts[i]
                pairwise_dists.add(int(np.sum(diff * diff)))
        
        for i in range(K):
            if time.time() > DEADLINE - 5:
                break
            
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            K_temp = len(S_temp)
            S_temp_pts = _all_points[S_temp] if K_temp > 0 else np.empty((0, 2), dtype=np.int32)
            
            # Recompute pairwise distances for S_temp
            pairwise_dists_temp = set()
            for a in range(K_temp):
                for b in range(a + 1, K_temp):
                    diff = S_temp_pts[b] - S_temp_pts[a]
                    pairwise_dists_temp.add(int(np.sum(diff * diff)))
            
            candidates = [p for p in range(NUM_POINTS) if p not in S_temp_set]
            random.shuffle(candidates)
            
            new_points = []
            for p_idx in candidates:
                if time.time() > DEADLINE - 5:
                    break
                
                p = _all_points[p_idx]
                current_set = S_temp + new_points
                current_K = len(current_set)
                
                if current_K <= 1:
                    new_points.append(p_idx)
                    if len(new_points) == 2:
                        break
                    continue
                
                current_pts = _all_points[current_set]
                diff = current_pts - p
                dists = np.sum(diff * diff, axis=1).astype(np.int64)
                
                if len(np.unique(dists)) != current_K:
                    continue
                
                valid = True
                for d in dists:
                    if d in pairwise_dists_temp:
                        valid = False
                        break
                    # Also check against distances involving new_points
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Need to also check that new distances don't conflict with each other
                # This is already covered by the distinctness check
                new_points.append(p_idx)
                
                # Update pairwise_dists_temp with new distances
                for np_idx in new_points[:-1]:
                    np_p = _all_points[np_idx]
                    diff_np = np_p - p
                    pairwise_dists_temp.add(int(np.sum(diff_np * diff_np)))
                
                if len(new_points) == 2:
                    break
            
            if len(new_points) == 2:
                S = S_temp + new_points
                improved = True
                break
        
        if not improved:
            break
    
    result = [tuple(_all_points[i]) for i in S]
    return result
