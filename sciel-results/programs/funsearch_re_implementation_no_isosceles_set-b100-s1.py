import time
import random
import numpy as np

N = 64
GRID_SIZE = N * N

# Pre-compute points
points_arr = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
ALL_POINTS = [tuple(p) for p in points_arr]

xs = points_arr[:, 0].astype(np.int64)
ys = points_arr[:, 1].astype(np.int64)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST = (dx * dx + dy * dy)  # shape (GRID_SIZE, GRID_SIZE)

# For fast validity checking, we can precompute for each point k, 
# the list of points that would conflict with k if k were added to a set S.
# But since S changes, we need a dynamic check.
# However, the "forbidden distance" bitmask idea from the inspiration is powerful.
# Let's try a different structural change: Simulated Annealing with a 
# precomputed "conflict graph" or similar? No, the conflict depends on the set S.
#
# The key insight from the cell findings is: "Vectorized O(1) validity checks 
# using a forbidden-distance bitmask allow for 10-50x more greedy passes".
# How to implement this?
# For a given set S, a point k is valid if:
# 1. No two points in S are at the same distance from k.
# 2. No point a in S has two "neighbors" in {k} U S at the same distance.
#
# Let's stick to a robust Simulated Annealing strategy but optimize the 
# validity check as much as possible using numpy.
#
# Strategy:
# 1. Start with a random valid set (using a fast greedy).
# 2. Perform SA: randomly pick a point in S to remove, and a point not in S to add.
# 3. Check if the new set is valid. If valid, accept if it improves size (always) 
#    or with probability exp(-delta/T) if it worsens (but size is fixed in swap, 
#    so we need a different objective or variable size?).
#
# Actually, the objective is to maximize size. So we should allow variable size.
# Moves:
# A. Add a random valid point (if exists).
# B. Remove a random point and add a random point (swap) if the new set is valid.
#
# To make this fast, we need a very fast `is_valid` function.
# Let's use the bitmask idea if possible, or just highly optimized numpy.
#
# Let's try a "repair" heuristic combined with SA.
# Or better: Use the "forbidden distance" bitmask.
# For a point k, and a set S, k is invalid if there exist a,b in S such that d(k,a)==d(k,b) 
# OR d(k,a)==d(a,b) OR d(k,b)==d(a,b).
#
# Let's precompute for each k, the distances to all other points.
# DIST[k, :] is a vector of length GRID_SIZE.
#
# In the SA loop, we maintain a boolean array `in_set` of size GRID_SIZE.
# When checking if k can be added to S:
# 1. Get distances from k to all points in S: dks = DIST[k, :][in_set]
# 2. Check for duplicates in dks. (If len(dks) > len(set(dks)), invalid)
# 3. For each a in S, check if d(k,a) is equal to d(a,b) for any b in S.
#    This is the hard part.
#
# Let's try a different angle: **Greedy with Random Restarts and a better ordering**.
# The current program uses random shuffle. What if we use a geometric ordering?
# Or what if we use a **Genetic Algorithm**?
#
# Let's go with **Simulated Annealing** but with a **variable size** set.
# We start with a greedy set.
# Move 1: Try to add a random point. If valid, accept.
# Move 2: Try to swap a random point in S with a random point not in S. 
#         If the new set is valid, accept if it doesn't reduce size (it doesn't, it's a swap).
#         But swaps don't increase size. We need to find "holes" that can be filled.
#
# Actually, the most effective structural change for these packing problems is often 
# **Local Search with Random Restarts** but with a **much faster validity check**.
# The cell finding mentions "forbidden-distance bitmask". 
#
# Let's implement a validity check that is as fast as possible.
# We will precompute `DIST` as int32 to save memory and speed up ops.
#
# Let's try this strategy:
# 1. Precompute DIST.
# 2. Use a greedy algorithm to get an initial set.
# 3. Use a local search that tries to **remove** a point and then **add** multiple points.
#    This is a "1-out, k-in" move.
#
# But the most promising lead is the bitmask. 
# For each point k, we can precompute a set of "forbidden configurations".
#
# Let's try a **Genetic Algorithm** approach.
# Represent a solution as a bitmask of GRID_SIZE bits.
# Fitness = number of 1s, if valid. Else 0.
# Crossover: Bitwise OR/AND? No, that creates invalid sets.
# Mutation: Flip a bit.
#
# This is too slow.
#
# Let's go back to **Simulated Annealing** with a **highly optimized validity check**.
# We will use a set of indices for S.
#
# To speed up the check "is k valid given S":
# 1. `dks = DIST[k, S_array]` (numpy array)
# 2. `if len(dks) != len(np.unique(dks)): return False`
# 3. For the second condition: For each a in S, we need to check if `d(k,a)` is in `DIST[a, S_array]`.
#    We can do this by:
#    `for a in S_array: if d(k,a) in set(DIST[a, S_array]): return False`
#    This is still O(N^2) in the size of S.
#
# Can we vectorize step 3?
# We have a matrix `D_S` where `D_S[i, j] = DIST[S[i], S[j]]`.
# We have a vector `dks` where `dks[i] = DIST[k, S[i]]`.
# We want to know if for any i, `dks[i]` is in `D_S[i, :]` or `D_S[:, i]`.
# Note `D_S[i, i] = 0`.
#
# This is a matrix membership problem.
#
# Let's try a different structural change: **Constructing the set row by row or column by column**.
# Or **Divide and Conquer**.
#
# Actually, let's look at the "base" mode question. 
# "Does mode 'base' change the chance that a child improves its island?"
# This suggests that the base program's behavior is a reference point.
#
# Let's try a **Randomized Local Search** with a **different neighborhood**.
# Instead of just adding points, we will use a **Hill Climbing** algorithm that 
# starts from an empty set and adds points in an order determined by a **fitness function** 
# that estimates how "isolated" a point is.
#
# Let's try **Simulated Annealing** with **variable size** and a **fast check**.
# We will accept the slowness of the check but try to make the search smarter.
#
# Wait, the cell finding says: "Vectorized O(1) validity checks using a forbidden-distance bitmask".
# How is that O(1)? 
# It must be that for a fixed set S, we precompute a bitmask of "forbidden" points.
# But S changes.
#
# Maybe the bitmask is for the distances?
# For a point k, the distances to points in S are a set of values.
# If we precompute for each k, the histogram of distances to all other points?
#
# Let's try a **Greedy Algorithm with a Priority Queue**.
# We maintain a list of "candidate" points.
# For each candidate, we compute a score based on how many constraints it violates.
# We pick the one with the fewest violations.
#
# This is **Min-Conflicts** heuristic.
#
# Let's implement **Min-Conflicts**:
# 1. Start with a random set of points (maybe all points, or a random subset).
# 2. Compute the number of isosceles triangles formed by each point.
# 3. If any point is involved in a triangle, remove it (or the one with the most conflicts).
# 4. Repeat until no conflicts.
# 5. Then try to add points.
#
# This is a different design family: **Constraint Satisfaction Problem (CSP) solver**.
#
# Let's implement this.

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    # We will use a Min-Conflicts style approach.
    # Start with a random subset, then remove conflicting points.
    # Then try to add more.
    
    best_set = []
    best_size = 0
    
    # Helper to check if a set is valid and return the set of conflicting points
    def get_conflicts(S_indices):
        m = len(S_indices)
        if m < 3:
            return set()
        
        S_set = set(S_indices)
        conflicts = set()
        
        # We need to find all points involved in an isosceles triangle.
        # Iterate over all triples? O(N^3) is too slow.
        # Iterate over pairs (a, b). For each pair, find all c such that 
        # d(a,c) == d(a,b) or d(b,c) == d(a,b) or d(a,c) == d(b,c).
        
        # This is still hard to vectorize easily for "conflicts".
        
        # Let's stick to the validity check for a single point addition, 
        # but use it in a smarter search.
        
        # Actually, let's just return a boolean for validity, and use 
        # a different search strategy.
        
        # Let's try **Random Restarts with Greedy + Swap Local Search**.
        # The key is to make the swap local search much more effective.
        
        pass
    
    # Let's implement a robust **Simulated Annealing** with a **fast validity check**.
    # We will use a list `S` for the current set.
    # We will use a numpy array `in_set` for O(1) membership.
    
    in_set = np.zeros(GRID_SIZE, dtype=bool)
    S = []
    
    def is_valid_addition(k):
        m = len(S)
        if m == 0:
            return True
        if m == 1:
            return True
        
        # Get indices of S as a numpy array for fast access
        S_arr = np.array(S, dtype=np.int32)
        
        # 1. Check duplicate distances from k
        dks = DIST[k, S_arr]
        if len(dks) != len(np.unique(dks)):
            return False
        
        # 2. Check if d(k,a) == d(a,b) for any a,b in S
        # We need to check for each a in S, if d(k,a) is in {d(a,b) for b in S}
        # This is equivalent to checking if d(k,a) is in DIST[a, S_arr]
        
        # Vectorized approach:
        # Create a matrix M where M[i, j] = DIST[S[i], S[j]]
        # We want to know if for any i, dks[i] is in M[i, :]
        
        # This is a loop over i, but we can do it in numpy if we are careful.
        # However, the set membership check is the bottleneck.
        
        # Let's use a set for the distances from each a.
        # This is O(m^2) in the worst case, but with small constants.
        
        for i in range(m):
            a = S[i]
            d_ka = dks[i]
            # Check if d_ka is in DIST[a, S_arr]
            # Note: DIST[a, a] = 0, so we should exclude a itself? 
            # No, d(a,a)=0, and d_ka > 0, so it's fine.
            if d_ka in set(DIST[a, S_arr]):
                return False
        
        return True
    
    # Greedy initialization
    def greedy_init():
        nonlocal S, in_set
        S = []
        in_set[:] = False
        
        indices = list(range(GRID_SIZE))
        random.shuffle(indices)
        
        for k in indices:
            if is_valid_addition(k):
                S.append(k)
                in_set[k] = True
    
    # Local Search: Try to swap points
    def local_search():
        nonlocal S, in_set
        improved = True
        while improved:
            improved = False
            # Try to add a random point
            for _ in range(1000):
                k = random.randint(0, GRID_SIZE - 1)
                if not in_set[k] and is_valid_addition(k):
                    S.append(k)
                    in_set[k] = True
                    improved = True
            
            # Try to swap: remove a random point, try to add a random point
            if len(S) > 0:
                for _ in range(1000):
                    idx = random.randint(0, len(S) - 1)
                    a = S[idx]
                    # Temporarily remove a
                    S.pop(idx)
                    in_set[a] = False
                    
                    # Try to add a random point
                    for _ in range(10):
                        k = random.randint(0, GRID_SIZE - 1)
                        if not in_set[k] and is_valid_addition(k):
                            S.append(k)
                            in_set[k] = True
                            improved = True
                            break
                    else:
                        # If we couldn't add a point, put a back
                        S.append(a)
                        in_set[a] = True
    
    # Main loop: Random restarts with local search
    while time.time() < deadline:
        greedy_init()
        local_search()
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # If we are getting close to the deadline, break
        if time.time() > deadline - 5:
            break
    
    return [ALL_POINTS[i] for i in best_set]
