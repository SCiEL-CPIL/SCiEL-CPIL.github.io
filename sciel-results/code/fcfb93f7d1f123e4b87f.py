"""No-isosceles set: Vectorized Simulated Annealing over isosceles triangle counts."""
import numpy as np
import random
import time

N = 64
TIME_LIMIT = 280.0
TOTAL = N * N

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 5.0
    
    # Grid points
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    xs = all_points[:, 0].astype(np.int32)
    ys = all_points[:, 1].astype(np.int32)

    # Pairwise squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int64)
    np.fill_diagonal(dist2, 0)

    # Precompute isosceles constraints:
    # A triangle (i, j, k) is isosceles if d(i,j)==d(i,k), d(i,j)==d(j,k), or d(i,k)==d(j,k).
    # We want to minimize the number of such triangles in the selected set.
    
    # For efficiency, we can precompute for each point i, the list of pairs (j, k) 
    # such that d(i,j) == d(i,k). If i is selected, and both j and k are selected, 
    # we have an isosceles triangle (i,j,k) with apex i.
    # Note: This counts each isosceles triangle potentially multiple times if it has 
    # multiple equal sides (equilateral) or if we don't handle symmetry carefully.
    # Actually, an isosceles triangle has at least two equal sides.
    # If d(i,j)==d(i,k), it's isosceles with apex i.
    # If d(i,j)==d(j,k), it's isosceles with apex j.
    # If d(i,k)==d(j,k), it's isosceles with apex k.
    
    # To avoid double counting in the objective function, we can just sum over all i 
    # the number of pairs (j,k) with j,k > i (or some ordering) such that d(i,j)==d(i,k) 
    # and both are selected? No, that's complex.
    
    # Simpler Objective:
    # Let S be the set of selected points.
    # Cost = sum_{i in S} ( number of pairs (j,k) in S\{i} such that d(i,j)==d(i,k) )
    # This counts an isosceles triangle with apex i once.
    # An equilateral triangle is counted 3 times (once for each apex).
    # A non-equilateral isosceles triangle is counted 1 time (only the apex has equal legs).
    # So minimizing this cost minimizes the number of isosceles triangles.
    
    # For each i, let neighbors_by_dist[i] be a dictionary mapping distance d to list of points j with d(i,j)=d.
    # If i is selected, the contribution to cost from apex i is sum_{d} C(count_d, 2) where count_d is the number of selected neighbors at distance d.
    
    # This is still expensive to update if we do it naively.
    # Alternative:
    # For each pair (j,k) with j<k, let d_jk = d(j,k).
    # The set of points i such that d(i,j)==d(i,k) or d(i,j)==d(j,k) or d(i,k)==d(j,k) forms the "conflict" for pair (j,k).
    # Actually, let's stick to the apex-based cost.
    
    # Precomputation:
    # For each i, we want to quickly calculate the cost contribution if i is added/removed.
    # Cost_i = sum_{d} C(n_{i,d}, 2) where n_{i,d} is number of selected points at distance d from i.
    
    # This requires knowing n_{i,d} for all i,d.
    # Updating n_{i,d} when a point p is added:
    # For all i, n_{i, d(i,p)} increases by 1.
    # This is O(N^2) per update. With 280s, we can do many updates.
    # N^2 = 4096. 280s / (4096 ops) is plenty if ops are fast.
    # But Python loops are slow. We need vectorization.
    
    # Let's use a different representation.
    # State: boolean array S of length TOTAL.
    # We maintain an array `counts` of shape (TOTAL, MAX_DIST) ? No, MAX_DIST is large (2*N^2).
    # Instead, for each point i, we store a histogram of distances to selected points?
    
    # Let's try a different angle:
    # Cost = sum_{i<j<k} [IsIsosceles(i,j,k)]
    # This is hard to vectorize directly.
    
    # Let's go back to: Cost = sum_{i} sum_{j<k, j,k != i} [d(i,j) == d(i,k)]
    # This is sum_{i} sum_{d} C(n_{i,d}, 2).
    
    # We can maintain `n_{i,d}` implicitly.
    # For each i, we have a list of distances to all other points.
    # Let's precompute for each i, the indices of points at each distance.
    # dist_to_i = dist2[i, :]
    # unique_dists, counts = np.unique(dist_to_i, return_counts=True)
    # This is static.
    
    # When we add point p:
    # For each i, the distance d = dist2[i, p].
    # The term C(n_{i,d}, 2) changes to C(n_{i,d}+1, 2) = C(n_{i,d}, 2) + n_{i,d}.
    # So Delta Cost = sum_{i != p} n_{i, dist2[i,p]}.
    
    # We need to maintain n_{i,d} for all i and relevant d.
    # Since d is determined by i and p, we only need to update n_{i, dist2[i,p]} for all i.
    # This is an array update of length N^2.
    # We can store `n_selected_at_dist` as a 2D array? No, d varies.
    
    # Let's store `neighbor_counts` of shape (TOTAL, TOTAL) ? No, that's distances.
    # We can store for each i, a dictionary of distance -> count.
    # But updating a dictionary for all i is slow in Python.
    
    # Vectorized approach:
    # We have `dist2` matrix (N^2 x N^2).
    # We have `S` boolean vector (N^2).
    # We want to compute `n_{i,d}` for all i?
    # Actually, we just need the current total cost.
    
    # Let's try a simpler, albeit less precise, objective that is easier to vectorize:
    # For each pair (j,k), let D_jk = dist2[j,k].
    # The number of points i such that d(i,j)==D_jk is fixed (let's call it cnt_jk).
    # If j and k are selected, the number of isosceles triangles with base jk is the number of selected points i (i!=j,k) such that d(i,j)==D_jk.
    # Let's precompute `base_counts` matrix B where B[j,k] = number of points i (i!=j,k) with d(i,j)==d(j,k).
    # Wait, d(i,j)==d(j,k) means i is at distance D_jk from j.
    # Also d(i,k)==d(j,k) means i is at distance D_jk from k.
    # An isosceles triangle with base jk has apex i such that d(i,j)==d(i,k).
    # This is different.
    
    # Let's stick to the apex definition:
    # Triangle (i,j,k) is isosceles if d(i,j)==d(i,k) OR d(i,j)==d(j,k) OR d(i,k)==d(j,k).
    
    # Let's use the property:
    # Number of isosceles triangles = sum_{i} (Number of pairs (j,k) in S\{i} with d(i,j)==d(i,k))
    # This counts each isosceles triangle exactly once if it's not equilateral, and 3 times if equilateral.
    # Since we want to minimize isosceles triangles, minimizing this sum is a good proxy.
    
    # We will maintain `hist[i]` which is a histogram of distances from i to selected points.
    # Since N=64, max dist is 2*63^2 ~ 8000.
    # We can have an array `hist` of shape (TOTAL, MAX_DIST+1) ? 4096 * 8000 = 32M integers. 32M * 8 bytes = 256MB. Too much?
    # Maybe. But we don't need full histogram for all i.
    
    # Alternative:
    # Just compute the cost from scratch every step?
    # Cost = sum_{i in S} sum_{d} C(n_{i,d}, 2).
    # To compute this, for each i in S, we need the distances to all other points in S.
    # If |S| ~ 50, this is 50 * 50 = 2500 distance lookups. Very fast.
    # So we can just recompute the cost from scratch after each move!
    
    def compute_cost(S):
        """S is a boolean array of length TOTAL. Returns total isosceles count (proxy)."""
        if not np.any(S):
            return 0
        indices = np.where(S)[0]
        k = len(indices)
        if k < 3:
            return 0
        
        # Extract submatrix of distances
        # dist2 is (TOTAL, TOTAL)
        sub_dist = dist2[np.ix_(indices, indices)]
        
        # For each i (row), count pairs (j,k) with j<k and sub_dist[i,j] == sub_dist[i,k]
        # We can do this by sorting or using unique.
        # Since k is small (~50-100), we can loop over i in Python.
        cost = 0
        for i in range(k):
            d_i = sub_dist[i]
            # d_i[i] is 0. We should ignore it?
            # The condition is for j,k != i.
            # d_i has 0 at index i.
            # We want pairs (j,k) from {0..k-1} \ {i} such that d_i[j] == d_i[k].
            
            # Remove self-distance
            d_i_no_self = np.delete(d_i, i)
            
            # Count frequencies of each distance
            # Since k is small, np.unique is fine.
            vals, counts = np.unique(d_i_no_self, return_counts=True)
            # Sum C(count, 2)
            cost += np.sum(counts * (counts - 1) // 2)
            
        return cost

    def get_points(S):
        return [(int(all_points[i, 0]), int(all_points[i, 1])) for i in np.where(S)[0]]

    # Simulated Annealing
    S = np.zeros(TOTAL, dtype=bool)
    
    # Initialize with random points
    init_size = 30
    init_indices = np.random.choice(TOTAL, init_size, replace=False)
    S[init_indices] = True
    
    current_cost = compute_cost(S)
    best_cost = current_cost
    best_S = S.copy()
    
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline:
        # Generate a random move: flip a random bit
        idx = np.random.randint(TOTAL)
        
        # Calculate delta cost
        # This is expensive if we recompute from scratch.
        # But k is small, so compute_cost is fast (~ms).
        # Let's just recompute.
        
        S[idx] = not S[idx]
        new_cost = compute_cost(S)
        
        if new_cost < current_cost:
            current_cost = new_cost
            S = S.copy() # Commit
            if current_cost < best_cost:
                best_cost = current_cost
                best_S = S.copy()
                if best_cost == 0:
                    break # Found perfect solution
        else:
            # Accept with probability exp(-delta/T)
            delta = new_cost - current_cost
            if delta < 0:
                pass
            else:
                if np.random.rand() < np.exp(-delta / T):
                    current_cost = new_cost
                    # S is already flipped
                else:
                    # Revert
                    S[idx] = not S[idx]
        
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: reheat if stuck?
            # For now, just keep going.
        
        # Check time
        if time.time() > deadline - 10:
            break

    # Final local search: try to add points if cost remains 0
    if best_cost == 0:
        # Greedy add
        candidates = np.where(~best_S)[0]
        random.shuffle(candidates.tolist())
        for idx in candidates:
            if time.time() > deadline:
                break
            best_S[idx] = True
            new_cost = compute_cost(best_S)
            if new_cost == 0:
                pass
            else:
                best_S[idx] = False

    # Return best set
    return get_points(best_S)

if __name__ == "__main__":
    construct()
