import numpy as np
import random
import time
import sys

N = 16
NUM_POINTS = N * N
POINTS = [(x, y) for y in range(N) for x in range(N)]
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

# Precomputation
# 1. Distances for NumPy checks
coords = np.array(POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
DIST_SQ = np.sum(diffs * diffs, axis=-1)

# 2. Block masks for fast reduction checks
# block_mask[i][j] is a bitmask of points x such that {i, j, x} is isosceles
BLOCK_MASK = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]

def precompute_masks():
    for i in range(NUM_POINTS):
        for j in range(i + 1, NUM_POINTS):
            mask = 0
            d_ij = DIST_SQ[i, j]
            # Vectorized check for x
            # d_ix == d_jx OR d_ix == d_ij OR d_jx == d_ij
            d_i = DIST_SQ[i]
            d_j = DIST_SQ[j]
            
            # Create boolean masks for conditions
            # We exclude i and j themselves
            valid_x = np.ones(NUM_POINTS, dtype=bool)
            valid_x[i] = False
            valid_x[j] = False
            
            cond1 = (d_i == d_j) & valid_x
            cond2 = (d_i == d_ij) & valid_x
            cond3 = (d_j == d_ij) & valid_x
            
            is_isosceles = cond1 | cond2 | cond3
            
            # Convert boolean mask to integer bitmask
            # We iterate over the set bits to build the int mask
            # Since NUM_POINTS=256, we can use Python int
            m = 0
            # np.where returns indices
            idxs = np.where(is_isosceles)[0]
            for x in idxs:
                m |= (1 << x)
            BLOCK_MASK[i][j] = m
            BLOCK_MASK[j][i] = m

precompute_masks()

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    def fast_greedy_construct(rng):
        """
        Construct a maximal no-isosceles set using NumPy for speed.
        Returns a list of indices.
        """
        S = []
        # used_dist[i, d] is True if point i has a neighbor in S at squared distance d
        # Max dist is (N-1)^2 + (N-1)^2 = 2*(15)^2 = 450
        MAX_D = 2 * (N - 1) ** 2
        used_dist = np.zeros((NUM_POINTS, MAX_D + 1), dtype=bool)
        
        # Random order
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        
        for p in order:
            if not S:
                S.append(p)
                # No distances to mark yet for the first point in terms of conflicts with others
                continue
            
            # Get distances from p to all points in S
            # S is a list of indices
            s_arr = np.array(S)
            dists_p_to_s = DIST_SQ[p, s_arr]
            
            # Check 1: Uniqueness of distances from p to S
            # If any two points in S are at the same distance from p, conflict.
            if len(dists_p_to_s) != len(np.unique(dists_p_to_s)):
                continue
            
            # Check 2: For each b in S, d(p, b) must not be in used_dist[b]
            # used_dist[b] tracks distances from b to other points in S.
            # If d(p, b) is already used by b (i.e., there exists c in S such that d(b, c) == d(p, b)),
            # then triangle (p, b, c) is isosceles (d(b,c)==d(b,p)).
            # We need to check if dists_p_to_s[k] is in used_dist[S[k]]
            
            # Vectorized check:
            # We want to check for each k: used_dist[S[k], dists_p_to_s[k]]
            # s_arr is S indices. dists_p_to_s is distances.
            conflicts = used_dist[s_arr, dists_p_to_s]
            
            if np.any(conflicts):
                continue
            
            # Valid! Add p to S
            S.append(p)
            
            # Update used_dist
            # 1. Mark distances from p to old S as used for p
            used_dist[p, dists_p_to_s] = True
            
            # 2. Mark distance from old S to p as used for old S
            # For each b in old S, d(b, p) is dists_p_to_s[k]
            # used_dist[b, dists_p_to_s[k]] = True
            used_dist[s_arr, dists_p_to_s] = True
            
        return S

    def reduce_to_minimal(S_indices):
        """
        Reduce a maximal set to a minimal maximal set using bitmask counts.
        """
        if not S_indices:
            return []
            
        S = set(S_indices)
        n_s = len(S)
        
        if n_s < 2:
            return list(S)
        
        # counts[x] = number of pairs (a, b) in S such that x is in block_mask[a][b]
        counts = [0] * NUM_POINTS
        
        # Initialize counts
        s_list = list(S)
        for i in range(n_s):
            a = s_list[i]
            for j in range(i + 1, n_s):
                b = s_list[j]
                mask = BLOCK_MASK[a][b]
                m = mask
                while m:
                    lsb = m & -m
                    x_idx = lsb.bit_length() - 1
                    counts[x_idx] += 1
                    m ^= lsb
        
        improved = True
        while improved:
            improved = False
            # Try to remove points in random order
            s_list = list(S)
            random.shuffle(s_list)
            
            for p_idx in s_list:
                if p_idx not in S:
                    continue
                
                # Simulate removal of p_idx
                changes = []
                
                for q_idx in S:
                    if q_idx == p_idx:
                        continue
                    mask = BLOCK_MASK[p_idx][q_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        x_idx = lsb.bit_length() - 1
                        counts[x_idx] -= 1
                        changes.append(x_idx)
                        m ^= lsb
                
                # Check if S \ {p_idx} is still maximal
                # 1. p_idx must be blocked by S \ {p_idx}
                if counts[p_idx] == 0:
                    # Revert
                    for x_idx in changes:
                        counts[x_idx] += 1
                    continue
                
                # 2. All x not in S \ {p_idx} must be blocked
                # x not in S \ {p_idx} means x not in S OR x == p_idx
                valid = True
                for x_idx in range(NUM_POINTS):
                    if x_idx in S and x_idx != p_idx:
                        continue
                    if counts[x_idx] == 0:
                        valid = False
                        break
                
                if not valid:
                    for x_idx in changes:
                        counts[x_idx] += 1
                    continue
                
                # Removal is valid
                S.remove(p_idx)
                improved = True
                break
        
        return list(S)

    def try_swap_improvement(S_indices, rng):
        """
        Attempt to improve the set by swapping points.
        This is a heuristic local search.
        """
        S = set(S_indices)
        n_s = len(S)
        
        if n_s < 2:
            return S_indices
            
        # We want to find a point u in S and v not in S such that
        # S' = (S \ {u}) U {v} is valid and maximal (or leads to a smaller minimal set).
        
        # For speed, we only try a few swaps.
        # Pick a random u in S
        s_list = list(S)
        
        for _ in range(10): # Limit swap attempts
            u = rng.choice(s_list)
            # Pick a random v not in S
            candidates = [i for i in range(NUM_POINTS) if i not in S]
            if not candidates:
                break
            v = rng.choice(candidates)
            
            S_test = (S - {u}) | {v}
            
            # Quick validity check using NumPy? 
            # Or just use the slow is_valid? 
            # Since S is small, is_valid is O(|S|^3) which is manageable for |S|~10-20.
            
            if is_valid_set(list(S_test)):
                # Check if it's maximal? 
                # If it's valid but not maximal, we can reduce it.
                # If it's valid and maximal, we can reduce it.
                reduced = reduce_to_minimal(list(S_test))
                if len(reduced) < len(S_indices):
                    return reduced
            
        return S_indices

    def is_valid_set(S_indices):
        """
        Check if a set is no-isosceles.
        """
        n = len(S_indices)
        if n < 3:
            return True
        s_set = set(S_indices)
        # Use bitmask for fast intersection
        # For each pair a, b in S, check if block_mask[a][b] intersects S \ {a, b}
        # Since S is small, we can just iterate
        for i in range(n):
            a = S_indices[i]
            for j in range(i + 1, n):
                b = S_indices[j]
                mask = BLOCK_MASK[a][b]
                # Check if any k in S is in mask
                # mask & S_mask?
                # Let's just iterate S
                for k in S_indices:
                    if k == a or k == b:
                        continue
                    if (mask >> k) & 1:
                        return False
        return True

    # Main Loop
    rng = random.Random(42)
    
    while time.time() < deadline:
        # 1. Generate a maximal set
        S_gen = fast_greedy_construct(rng)
        
        # 2. Reduce to minimal
        S_min = reduce_to_minimal(S_gen)
        
        current_size = len(S_min)
        
        if current_size < best_size:
            best_size = current_size
            best_set = S_min.copy()
            
            # 3. Try to improve via swaps
            S_improved = try_swap_improvement(S_min, rng)
            if len(S_improved) < best_size:
                best_size = len(S_improved)
                best_set = S_improved.copy()
        
        # Early exit if we find a very small set (theoretical lower bound is likely around 8-12)
        if best_size <= 10:
            break

    if best_set is None:
        # Fallback
        return []
        
    return [POINTS[i] for i in best_set]

if __name__ == "__main__":
    s = construct()
    print(len(s))
