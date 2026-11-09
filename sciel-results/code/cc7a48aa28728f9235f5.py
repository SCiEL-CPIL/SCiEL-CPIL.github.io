# EVOLVE-BLOCK-START
import time
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set N in F_49^2 by maximizing the complement S.
    S is a set of points such that for every x in S and every direction d,
    the line through x in direction d contains at least one other point of S.
    """
    p = 7
    q = p * p
    total = q * q
    deadline = time.time() + 600.0
    deadline_soft = deadline - 30.0

    # Precompute field operations
    alpha2 = 3
    add = np.zeros((q, q), dtype=np.int64)
    mul = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        ai = i // p
        bi = i % p
        for j in range(q):
            aj = j // p
            bj = j % p
            add[i, j] = ((ai + aj) % p) * p + ((bi + bj) % p)
            ra = (ai * aj + bi * bj * alpha2) % p
            rb = (ai * bj + bi * aj) % p
            mul[i, j] = ra * p + rb

    # Precompute lines
    # lines[x][d] is a list of q point indices on the line through x in direction d.
    # We need the "other" points for updates.
    # To save memory and time, we precompute the list of OTHER points for each (x, d).
    # other_lines[x][d] is a list of q-1 point indices.
    
    # This precomputation takes 2401 * 50 * 49 ~ 5.9M operations. Fast.
    other_lines = np.zeros((total, q + 1, q - 1), dtype=np.int64)
    
    for x in range(q):
        for y in range(q):
            idx = x * q + y
            for d in range(q):
                m = d
                # Line: (x+t, y+m*t)
                pts = []
                for t in range(q):
                    if t == 0: continue # Skip self? No, we need to identify self.
                    xt = add[x, t]
                    yt = add[y, mul[m, t]]
                    pts.append(xt * q + yt)
                other_lines[idx, d, :] = pts
            
            # Vertical
            pts = []
            for t in range(q):
                if t == 0: continue
                yt = add[y, t]
                pts.append(x * q + yt)
            other_lines[idx, q, :] = pts

    best_complement_size = 0
    best_S = None
    rng = np.random.default_rng(42)

    # Try multiple random initializations
    for trial in range(20):
        if time.time() > deadline_soft:
            break
            
        S = np.zeros(total, dtype=bool)
        # Start with a random subset to seed the process
        init_size = 200
        init_pts = rng.choice(total, size=init_size, replace=False)
        for pt in init_pts:
            S[pt] = True
            
        # Initialize counts
        # count_in_S[x, d] = number of points in S on the line through x in dir d (excluding x)
        count_in_S = np.zeros((total, q + 1), dtype=np.int32)
        
        # Update counts for the initial S
        for pt in init_pts:
            for d in range(q + 1):
                others = other_lines[pt, d]
                count_in_S[others, d] += 1
                
        # Greedy expansion
        added_any = True
        while added_any and time.time() < deadline_soft:
            added_any = False
            # Find all points that can be added
            # Condition: min(count_in_S[x, :]) > 0
            mins = np.min(count_in_S, axis=1)
            candidates = np.where((mins > 0) & (~S))[0]
            
            if len(candidates) == 0:
                break
                
            # Add all candidates at once? 
            # If we add multiple points, they might block each other?
            # No, adding to S only increases counts for others. It never decreases.
            # So if a point is addable now, it remains addable after adding other points.
            # Thus we can add ALL candidates in one batch.
            
            new_pts = candidates
            S[new_pts] = True
            added_any = True
            
            # Update counts for all neighbors of new_pts
            # This is the expensive part.
            # For each new_pt, for each d, for each other in other_lines[new_pt, d]:
            #    count_in_S[other, d] += 1
            
            # Vectorized update might be hard due to variable lengths, but lengths are fixed.
            # We can loop over new_pts.
            for pt in new_pts:
                for d in range(q + 1):
                    others = other_lines[pt, d]
                    count_in_S[others, d] += 1
                    
        current_size = np.sum(S)
        if current_size > best_complement_size:
            best_complement_size = current_size
            best_S = S.copy()
            
        # If we reached a very large size, we might stop early
        if best_complement_size > 1500:
             # Try a few more to see if we can go higher, but don't waste too much time
             pass

    # Build N from best_S
    if best_S is None:
        # Fallback
        points = []
        for x in range(q):
            for y in range(q):
                xa = x // p
                xb = x % p
                ya = y // p
                yb = y % p
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    points = []
    for x in range(q):
        for y in range(q):
            idx = x * q + y
            if not best_S[idx]:
                xa = x // p
                xb = x % p
                ya = y // p
                yb = y % p
                points.append([[xa, xb], [ya, yb]])

    if len(points) == 0:
        points = [[[0, 0], [0, 0]]]

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
