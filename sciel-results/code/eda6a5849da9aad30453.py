import time
import numpy as np
import random

N = 64
TOTAL = N * N

# Precompute squared distances
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

def is_valid_set(points):
    """Check if a set of points contains no isosceles triangles."""
    if len(points) < 3:
        return True
    
    pts = np.array(points)
    # Compute all pairwise squared distances
    # Shape: (M, M)
    d2 = dist2[np.ix_(pts, pts)]
    
    # For each point i, the distances to all other points j (j != i) must be distinct.
    # An isosceles triangle exists if there are i, j, k such that d(i,j) == d(i,k) or 
    # d(i,j) == d(j,k) or d(j,k) == d(i,k).
    # Actually, the condition "no three points forming an isosceles triangle" is equivalent to:
    # For every point P, the distances from P to all other points in the set are distinct.
    # AND for every pair of points P, Q, the distance PQ is unique to that pair in the sense that
    # no other point R exists such that PR == QR (which would make PQR isosceles with apex R? No, apex R means RP=RQ).
    # Wait, let's re-verify the definition.
    # Isosceles triangle: at least two sides equal.
    # Sides are d(A,B), d(B,C), d(C,A).
    # Condition: d(A,B) != d(A,C) AND d(A,B) != d(B,C) AND d(A,C) != d(B,C).
    # This is equivalent to:
    # 1. For any fixed A, all d(A, X) for X in S\{A} are distinct. (Prevents d(A,B)=d(A,C))
    # 2. For any fixed pair A, B, the value d(A,B) does not appear as d(A,C) or d(B,C) for any C != A,B.
    #    Actually, condition 1 implies that for any A, the map X -> d(A,X) is injective.
    #    If d(A,B) = d(A,C), then B=C. So d(A,B) cannot equal d(A,C) for distinct B,C.
    #    What about d(A,B) = d(B,C)?
    #    This means B is the apex. The distances from B to A and C are equal.
    #    This is covered by condition 1 applied to point B: d(B,A) must be distinct from d(B,C).
    #    So, the condition "For every point P in S, the distances from P to all other points in S are distinct"
    #    is NECESSARY and SUFFICIENT for no isosceles triangles.
    #    Proof:
    #    If there is an isosceles triangle A,B,C, then two sides are equal.
    #    Case 1: AB = AC. Then d(A,B) = d(A,C). Since B!=C, the distances from A are not distinct. Contradiction.
    #    Case 2: BA = BC. Then d(B,A) = d(B,C). Since A!=C, the distances from B are not distinct. Contradiction.
    #    Case 3: CA = CB. Then d(C,A) = d(C,B). Since A!=B, the distances from C are not distinct. Contradiction.
    #    Conversely, if for some P, d(P,A) = d(P,B) with A!=B, then P,A,B form an isosceles triangle.
    
    for i in range(len(pts)):
        # Get distances from pts[i] to all other points
        # Exclude self (distance 0)
        d_row = d2[i, :]
        d_row = d_row[d_row > 0]
        if len(d_row) != len(np.unique(d_row)):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 295
    
    # Strategy:
    # 1. Start with a structured set. A common construction for no-isosceles (or "distinct distance") sets
    #    in 2D grids involves using points (i, i^2 mod p) or similar.
    #    However, N=64 is not prime. We can use N=61 (prime) subset or just use a greedy deterministic start.
    #    Let's try a greedy deterministic start based on "least distance conflict" or just a simple pattern.
    #    Actually, the random greedy in the current program is decent but slow.
    #    Let's try a deterministic greedy: always pick the next point that maximizes the minimum distance to existing points?
    #    Or simpler: iterate through points in a fixed order (e.g., Hilbert curve or just row-major) and add if valid.
    #    Row-major greedy:
    
    initial_set = []
    # Try row-major order
    for idx in range(TOTAL):
        candidate = (idx // N, idx % N)
        new_set = initial_set + [candidate]
        # Optimization: Instead of full check, we can check incrementally.
        # But for simplicity and correctness in the first pass, let's use the vectorized check on the whole set.
        # This is O(K^2) per addition, so O(K^3) total. For K~10, it's fine.
        if is_valid_set(new_set):
            initial_set.append(candidate)
            
    best_set = initial_set
    best_size = len(best_set)
    
    # 2. Local Search / Hill Climbing
    # Try to swap a point in the set with a point outside the set to improve... wait, we just want to maximize size.
    # So we should try to ADD points.
    # If we can't add any point, we are stuck.
    # Local search strategy:
    # 1. Try to add any remaining point.
    # 2. If no point can be added, try to remove one point and see if we can add two points (net gain +1).
    #    Or remove one point and add one different point (no gain, but might unblock).
    
    # Let's implement a "repair" local search.
    # State: set of points.
    # Move: Remove point P from set. Try to add point Q (not in set).
    # If valid, accept if size increases (impossible with 1 remove 1 add) or if it allows further additions.
    
    # Simpler: Randomized Local Search with Deterministic Initialization.
    # Since the deterministic start might be small, let's combine.
    
    # Actually, the random restarts in the original code were the main driver.
    # Let's try a different structural approach:
    # Use a known construction for "Golomb rulers" or "Sidon sets" in 2D.
    # A Sidon set in Z_n x Z_n has the property that all pairwise sums are distinct? No, that's for no parallelograms.
    # For isosceles, we need distinct distances from each point.
    
    # Let's try a "greedy with lookahead" or "simulated annealing" style on the set membership.
    # But time is tight.
    
    # Let's refine the deterministic greedy.
    # The row-major greedy might be suboptimal.
    # Let's try multiple deterministic orderings.
    
    orderings = [
        list(range(TOTAL)),
        list(range(TOTAL-1, -1, -1)),
        # Diagonal order
        [i*N + j for i in range(N) for j in range(N) if (i+j) % 2 == 0] + 
        [i*N + j for i in range(N) for j in range(N) if (i+j) % 2 == 1]
    ]
    
    for order in orderings:
        s = []
        for idx in order:
            # Quick check: is idx already in s? (It shouldn't be if order is a permutation)
            # Check validity incrementally
            valid = True
            for q in s:
                d = dist2[idx, q]
                # Check if d is already a distance from idx to some other point in s? 
                # No, we need to check if adding idx creates an isosceles triangle with any pair in s.
                # This is equivalent to:
                # 1. Distances from idx to all points in s must be distinct.
                # 2. For each point q in s, the distance d(idx, q) must not be equal to any distance d(q, r) for r in s, r != idx, q.
                #    Wait, condition 2 is: d(idx, q) != d(q, r) for all r in s, r != idx, q.
                #    This means d(idx, q) is not in the set of distances from q to other points in s.
                
                # We need to maintain for each point q in s, the set of distances to other points in s.
                # This is the `ud` structure from the original code.
                # Let's use that structure for speed.
                pass
            
            # The incremental check with `ud` is fast. Let's implement it properly here.
            # This is essentially the same as the original greedy, but with a fixed order instead of random.
            # To make it different, we can use a "best fit" heuristic:
            # Instead of fixed order, pick the next point that minimizes the number of conflicts?
            # Or maximizes the minimum distance to existing points?
            
            # Let's stick to the `ud` structure but iterate in a specific smart order.
            # Order: Points far from existing points first?
            
        break # Just do one ordering for now to keep code simple? No, let's do the random restarts but faster.
        
    # The original code's bottleneck was the Python loop overhead for 5000 restarts.
    # Let's try to reduce restarts but make each one smarter.
    # Or, let's try a completely different construction:
    # Use points (x, y) where x is from a Sidon set in [0, N) and y is determined by x?
    # If we pick x values such that differences are unique (Sidon), does that help?
    # If x_i are Sidon, then |x_i - x_j| are distinct for pairs? No, Sidon means sums are distinct.
    # Differences being distinct is a Golomb ruler.
    # If we have a Golomb ruler on x-axis, then for any pair (x_i, y_i), (x_j, y_j), the dx is unique.
    # This doesn't prevent isosceles triangles where dy compensates.
    
    # Let's go back to the `ud` greedy but optimize the inner loop.
    # The original code did 5000 restarts.
    # Let's do 100 restarts but with a "local improvement" phase at the end of each greedy pass.
    
    best_set = []
    best_size = 0
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        perm = list(range(TOTAL))
        random.shuffle(perm)
        
        si = []
        ud = [] # List of sets of distances used by each point
        
        for p_idx in perm:
            if not si:
                si.append(p_idx)
                ud.append(set())
                continue
            
            valid = True
            dists = []
            # Compute distances
            for q_idx in si:
                d = dist2[p_idx, q_idx]
                dists.append(d)
            
            # Check distinctness among new distances
            if len(set(dists)) != len(dists):
                continue
            
            # Check against existing used distances
            for i in range(len(si)):
                if dists[i] in ud[i]:
                    valid = False
                    break
            
            if valid:
                si.append(p_idx)
                new_ud = set(dists)
                ud.append(new_ud)
                
                # Update existing points' used distance sets
                for i in range(len(si) - 1):
                    ud[i].add(dists[i])
        
        if len(si) > best_size:
            best_size = len(si)
            best_set = [(idx // N, idx % N) for idx in si]
            
            # Local Search: Try to add more points
            # Keep track of which points were tried and failed?
            # Or just try all remaining points.
            remaining = [p for p in range(TOTAL) if p not in si]
            # Shuffle remaining to randomize
            random.shuffle(remaining)
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for p_idx in remaining:
                    if p_idx in si:
                        continue
                    valid = True
                    dists = []
                    for q_idx in si:
                        d = dist2[p_idx, q_idx]
                        dists.append(d)
                    
                    if len(set(dists)) != len(dists):
                        continue
                        
                    for i in range(len(si)):
                        if dists[i] in ud[i]:
                            valid = False
                            break
                    
                    if valid:
                        si.append(p_idx)
                        new_ud = set(dists)
                        ud.append(new_ud)
                        for i in range(len(si) - 1):
                            ud[i].add(dists[i])
                        remaining.remove(p_idx)
                        improved = True
                        if len(si) > best_size:
                            best_size = len(si)
                            best_set = [(idx // N, idx % N) for idx in si]
                        break # Restart the loop over remaining with updated state
            
    return best_set

if __name__ == "__main__":
    # Quick test
    s = construct()
    print(len(s))
