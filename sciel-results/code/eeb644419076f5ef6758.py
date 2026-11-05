"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
_ISO_MASKS = None
_POINTS = None

def _init_data():
    global _ISO_MASKS, _POINTS
    if _ISO_MASKS is not None:
        return _ISO_MASKS, _POINTS
    
    # Generate points
    pts = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    _POINTS = pts
    
    # Compute squared distance matrix D (N*N x N*N)
    # D[i, j] = squared distance between point i and point j
    # Using broadcasting: (pts[:, None, :] - pts[None, :, :])**2
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.sum(diff * diff, axis=2)
    
    num_pts = N * N
    _ISO_MASKS = [[0] * num_pts for _ in range(num_pts)]
    
    # Precompute masks
    # For each pair (i, j), find all k such that (i, j, k) is isosceles.
    # Condition: D[i,j] == D[i,k] or D[i,j] == D[j,k] or D[i,k] == D[j,k]
    
    for i in range(num_pts):
        for j in range(num_pts):
            if i == j:
                continue
            
            dij = D[i, j]
            
            # M1: k such that D[i,k] == dij
            m1 = (D[i, :] == dij).astype(np.int64)
            
            # M2: k such that D[j,k] == dij
            m2 = (D[j, :] == dij).astype(np.int64)
            
            # M3: k such that D[i,k] == D[j,k]
            m3 = (D[i, :] == D[j, :]).astype(np.int64)
            
            # Union
            mask_bool = m1 | m2 | m3
            
            # Exclude i and j themselves? 
            # The definition of isosceles triangle usually requires distinct vertices.
            # If k==i, D[i,k]=0. If dij!=0, M1/M2/M3 might include i/j if distances match 0?
            # D[i,i]=0. If dij==0 (i==j, handled), or if there are other points at same dist.
            # We should exclude i and j from the mask to ensure k is distinct.
            mask_bool[i] = False
            mask_bool[j] = False
            
            # Convert boolean mask to integer bitmask
            # mask_bool is 1D array of length 256
            # We need to map True indices to bits
            # Using sum of (mask << idx) is slow in Python loop.
            # Use numpy to create powers of 2?
            # Or just loop over True indices.
            # Since N=16, 256 points. Number of True bits is small (perp bisector etc).
            # Let's just loop.
            mask_int = 0
            indices = np.where(mask_bool)[0]
            for k in indices:
                mask_int |= (1 << k)
            
            _ISO_MASKS[i][j] = mask_int
            _ISO_MASKS[j][i] = mask_int # Symmetric
            
    return _ISO_MASKS, _POINTS

def _greedy_domination(masks, points, seed):
    random.seed(seed)
    num_pts = len(points)
    available = (1 << num_pts) - 1
    S = []
    current_union = 0
    
    while available:
        valid_candidates = []
        scores = []
        
        avail = available
        while avail:
            l = avail & -avail
            p = l.bit_length() - 1
            avail ^= l
            
            # Check if p is valid (not in current_union)
            if (1 << p) & current_union:
                continue
                
            # Calculate new_union
            new_union = current_union
            for a in S:
                new_union |= masks[p][a]
                
            # Score: how many available points are removed
            removed = new_union & available
            score = removed.bit_count()
            
            valid_candidates.append(p)
            scores.append(score)
            
        if not valid_candidates:
            break
            
        # Pick candidate with max score, random tie-breaking
        max_score = max(scores)
        best_candidates = [p for p, s in zip(valid_candidates, scores) if s == max_score]
        p = random.choice(best_candidates)
        
        # Update S and available
        S.append(p)
        
        # Recalculate new_union for the chosen p to update current_union
        # We already calculated it in the loop, but we didn't store it.
        # Let's just recalculate for the chosen p.
        new_union = current_union
        for a in S:
            new_union |= masks[p][a]
            
        current_union = new_union
        
        # Remove p from available
        available &= ~(1 << p)
        # Remove all points in current_union from available
        available &= ~current_union
        
    return S

def construct():
    masks, points = _init_data()
    num_pts = len(points)
    
    best_set = None
    best_size = num_pts + 1
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    seed = 0
    while time.time() < deadline:
        S = _greedy_domination(masks, points, seed)
        size = len(S)
        
        if size < best_size:
            best_size = size
            best_set = S
            
        seed += 1
        
    return best_set
# EVOLVE-BLOCK-END
