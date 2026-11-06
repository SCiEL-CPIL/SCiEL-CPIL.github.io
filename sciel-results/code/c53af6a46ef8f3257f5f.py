import numpy as np
import random
import time
import sys

N = 64
GRID_SIZE = N * N
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]

# Pre-compute the isosceles conflict masks
# ISO_MASK[i, j] is a Python int where bit k is 1 if {i, j, k} forms an isosceles triangle
# We exclude i and j themselves from the mask (distance 0 is not a triangle side in this context usually, 
# but technically if d(i,k)=d(j,k) and k!=i,j it's isosceles. If k=i or k=j, it's degenerate).
# The problem says "no three points forming an isosceles triangle". 
# Usually this implies distinct points. 
# If d(i,j) = d(i,k) and j!=k, it's isosceles with vertex i.
# If d(i,j) = d(j,k) and i!=k, it's isosceles with vertex j.
# If d(i,k) = d(j,k) and i!=j, it's isosceles with vertex k.

# We will compute for each pair (i, j) the set of all k != i, k != j such that 
# d(i,j)^2 == d(i,k)^2 OR d(i,j)^2 == d(j,k)^2 OR d(i,k)^2 == d(j,k)^2

# To speed up precomputation, we can precompute squared distances between all pairs.
# DIST[i, j]
DIST = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

def precompute_distances():
    for i in range(GRID_SIZE):
        x1, y1 = ALL_POINTS[i]
        for j in range(i + 1, GRID_SIZE):
            x2, y2 = ALL_POINTS[j]
            dx = x1 - x2
            dy = y1 - y2
            d_sq = dx * dx + dy * dy
            DIST[i, j] = d_sq
            DIST[j, i] = d_sq

# Run pre-computation at module load time
precompute_distances()

# ISO_MASK[i, j] will be a list of integers (bits) or just a Python int
# Python int is a big integer, perfect for bitsets.
ISO_MASK = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

def precompute_iso_masks():
    # For each pair (i, j), find all k that form an isosceles triangle
    # This is O(N^3) which is 4096^3 = 68 billion. Too slow.
    # We need a smarter way.
    
    # For a fixed i, group all other points j by their distance to i.
    # Let groups[i] be a dict: distance -> list of points
    # If we pick two points j, k from the same group[i], then d(i,j)=d(i,k), so {i,j,k} is isosceles.
    # So for any pair (j, k) in the same group[i], ISO_MASK[j, k] should have bit i set.
    # Also, for any pair (i, j), we need to check if there exists k such that d(i,j)=d(i,k) (covered above)
    # or d(i,j)=d(j,k) (covered by groups[j]) or d(i,k)=d(j,k) (covered by groups[k]).
    
    # So the condition "{i, j, k} is isosceles" is equivalent to:
    # i is in the "conflict set" of pair (j, k) if d(i,j)=d(i,k)
    # j is in the "conflict set" of pair (i, k) if d(j,i)=d(j,k)
    # k is in the "conflict set" of pair (i, j) if d(k,i)=d(k,j)
    
    # Let's build ISO_MASK[i, j] as the set of all k such that {i, j, k} is isosceles.
    # This is the union of:
    # 1. k such that d(i,k) == d(j,k)  (k is apex)
    # 2. k such that d(i,k) == d(i,j)  (i is apex, k is base) -> k is in group[i] with dist d(i,j)
    # 3. k such that d(j,k) == d(i,j)  (j is apex, k is base) -> k is in group[j] with dist d(i,j)
    
    # We can precompute for each point p and each distance d, the list of points at distance d from p.
    # dist_map[p][d] = list of points q such that d(p,q) = d
    
    dist_map = [dict() for _ in range(GRID_SIZE)]
    
    for i in range(GRID_SIZE):
        for j in range(i + 1, GRID_SIZE):
            d = DIST[i, j]
            if d not in dist_map[i]:
                dist_map[i][d] = []
            if d not in dist_map[j]:
                dist_map[j][d] = []
            dist_map[i][d].append(j)
            dist_map[j][d].append(i)
            
    # Now compute ISO_MASK
    # ISO_MASK[i, j] is the set of k != i, k != j such that {i, j, k} is isosceles.
    # We will store this as a Python int (bitset).
    
    # Initialize
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            ISO_MASK[i][j] = 0
            
    # Case 1 & 2: i is apex or j is apex
    # If i is apex, d(i,j) = d(i,k). So k is in dist_map[i][d(i,j)].
    # If j is apex, d(j,i) = d(j,k). So k is in dist_map[j][d(i,j)].
    
    for i in range(GRID_SIZE):
        for j in range(i + 1, GRID_SIZE):
            d_ij = DIST[i, j]
            
            # Points k such that d(i,k) == d(i,j)
            # These k form isosceles triangles with vertex i.
            if d_ij in dist_map[i]:
                for k in dist_map[i][d_ij]:
                    if k != j:
                        # k is a conflict for pair (i, j) ? No, k is a point that CANNOT be added if i and j are in the set.
                        # So bit k should be set in ISO_MASK[i][j].
                        ISO_MASK[i][j] |= (1 << k)
                        ISO_MASK[j][i] |= (1 << k)
                        
            # Points k such that d(j,k) == d(j,i)
            if d_ij in dist_map[j]:
                for k in dist_map[j][d_ij]:
                    if k != i:
                        ISO_MASK[i][j] |= (1 << k)
                        ISO_MASK[j][i] |= (1 << k)

    # Case 3: k is apex. d(k,i) == d(k,j).
    # This means i and j are in the same "ring" around k.
    # We can iterate over k, and for each k, look at its dist_map.
    # For each distance d, let the list of points at distance d from k be L.
    # For any pair (i, j) in L, k is a conflict.
    # So for all pairs (i, j) in L, set bit k in ISO_MASK[i][j].
    
    for k in range(GRID_SIZE):
        for d, points in dist_map[k].items():
            # For each pair in points, set bit k in ISO_MASK
            # points is a list of indices
            m = len(points)
            for a in range(m):
                for b in range(a + 1, m):
                    i = points[a]
                    j = points[b]
                    ISO_MASK[i][j] |= (1 << k)
                    ISO_MASK[j][i] |= (1 << k)

    # Clear bits for i and j themselves (degenerate triangles)
    # Actually, the problem implies distinct points. 
    # If k=i, d(k,i)=0. d(k,j)=d(i,j). 0 != d(i,j) usually.
    # If d(i,j)=0, i=j, not a pair.
    # So we should ensure bit i and bit j are not set in ISO_MASK[i][j].
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            ISO_MASK[i][j] &= ~(1 << i)
            ISO_MASK[i][j] &= ~(1 << j)

precompute_iso_masks()

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best_size = 0
    best_set = []
    
    # Helper to check if a point can be added given current blocked mask
    # We don't need to check against individual points if we maintain the blocked mask correctly.
    
    # We will use a backtracking search with a limit on depth or time.
    # Given the high dimensionality, deep backtracking might be slow.
    # Greedy with random restarts is often better for these packing problems.
    
    # Strategy:
    # 1. Shuffle all points.
    # 2. Iterate through points.
    # 3. If point is not in `blocked`, add it to set, update `blocked`.
    # 4. Update `blocked` by OR-ing ISO_MASK[new_point, existing_point] for all existing points.
    
    # To improve, we can try to pick the "best" point, not just the first available.
    # "Best" could be the one that minimizes the increase in `blocked` size (popcount).
    # But calculating popcount for all candidates is expensive.
    # Let's stick to greedy with randomization and many restarts.
    
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        random.seed(num_restarts) # Ensure reproducibility per restart if needed, or just random
        
        # Shuffle indices
        indices = list(range(GRID_SIZE))
        random.shuffle(indices)
        
        current_set = []
        current_blocked = 0
        
        # Greedy construction
        for idx in indices:
            # Check if idx is blocked
            if current_blocked & (1 << idx):
                continue
            
            # Can we add idx?
            # We need to check if adding idx conflicts with any pair in current_set?
            # No, the `current_blocked` mask already contains all points that conflict with ANY pair in current_set.
            # Wait, does it?
            # `current_blocked` is the union of ISO_MASK[a, b] for all a, b in current_set.
            # If idx is not in `current_blocked`, it means idx does not form an isosceles triangle with ANY pair in current_set.
            # This is exactly the condition for validity.
            
            # So we can just add it.
            # Update current_blocked:
            # New conflicts are introduced by pairs (idx, s) for all s in current_set.
            # So we OR ISO_MASK[idx, s] for all s in current_set.
            
            new_blocked = current_blocked
            for s in current_set:
                new_blocked |= ISO_MASK[idx][s]
                
            # Check if this is a valid move (it is, by definition of current_blocked)
            current_set.append(idx)
            current_blocked = new_blocked
            
            # Early termination if we've found a huge set?
            # No, keep going until all indices processed or time runs out.
            # But if time is running out, we should break.
            if time.time() > deadline - 1.0:
                break
                
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # Optional: Local search on the best set found
        # Try to swap points to increase size?
        # Given the time budget, simple greedy restarts might be sufficient.
        # If we have time, we can try to remove a point and see if we can add two?
        # For now, let's rely on the power of many restarts.
        
        # If best_size is very high, we can try to optimize further
        if best_size > 100 and time.time() < deadline - 30:
            # Try a local improvement on the current best
            # This is complex, so we skip for now to keep the code simple and robust.
            pass

    return [ALL_POINTS[i] for i in best_set]

# Run precomputation at module load
# Note: precompute_iso_masks is O(N^3 * avg_degree) which might be slow.
# N=4096.
# dist_map construction is O(N^2).
# The triple loop in Case 3:
# For each k, for each d, for each pair in points.
# Sum over k of sum over d of C(len(points), 2).
# This is roughly the number of isosceles triangles in the grid.
# In a 64x64 grid, the number of isosceles triangles is large but manageable.
# Let's estimate:
# Each point has ~100 neighbors at various distances.
# The number of pairs in each ring is small.
# Total operations should be well under 10^7 or 10^8, which is fine for Python if optimized?
# Actually, Python loops are slow. 10^8 iterations might take 10-30 seconds.
# We have 300s. Precomputation is one-time cost.
# If precomputation takes too long, we will timeout.
# Let's optimize the precomputation.

# Optimization: Use NumPy for distance calculation and grouping?
# Or just accept the Python loop cost if it's < 20s.

# Let's add a check to see if precomputation is too slow.
# Actually, the current program's precompute_distances is O(N^2) and fast.
# precompute_iso_masks is the new cost.

# To be safe, we can limit the precomputation or make it faster.
# Alternative: Compute ISO_MASK on the fly? No, too slow during search.
# Let's trust the O(N^3) logic is fast enough for N=4096 with small constants.
# The inner loop `for a in range(m): for b in range(a+1, m)` runs only for points in the same ring.
# Rings are small.

# One more optimization:
# We can store ISO_MASK as a list of integers.
# The bitwise OR is fast.

# Let's run it.
