# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Point representation: (xa, xb, ya, yb)
    # Index: ((xa * p + xb) * p + ya) * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb

    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Generate all points
    total = p * p * p * p
    
    # Initial S construction (Unital based)
    k = 3 # QNR
    qr = set((u * u) % p for u in range(p))
    
    def in_unital(xa, xb, ya, yb):
        return (2 * xa + ya * ya - k * yb * yb) % p == 0

    S = set()
    for idx in range(total):
        xa, xb, ya, yb = idx_to_point(idx)
        if in_unital(xa, xb, ya, yb) and ya in qr:
            S.add(idx)
            
    # Lines representation
    # A line is defined by a point (x0, y0) and a slope m (or vertical)
    # We need to map lines to sets of point indices.
    # There are q*(q+1) lines? No, q^2 + q lines in affine plane?
    # Number of lines in F_q^2 is q(q+1).
    # For each point, q+1 lines pass through it.
    
    # Precompute lines passing through each point
    # Line ID: 
    # Vertical lines: x = c. ID: c (0..q-1)
    # Non-vertical lines: y = m*x + c. ID: q + m*q + c?
    # Let's just generate lists of points for each line.
    
    # We need to iterate through all lines.
    # Total lines = q*(q+1) = 49*50 = 2450.
    
    lines = [] # List of lists of point indices
    point_to_lines = [[] for _ in range(total)]
    
    # Vertical lines
    for c in range(q):
        pts = []
        for y in range(q):
            # x = c, y = y
            # x is (xa, xb) = divmod(c, p)
            xa, xb = divmod(c, p)
            ya, yb = divmod(y, p)
            idx = point_to_idx(xa, xb, ya, yb)
            pts.append(idx)
        line_id = len(lines)
        lines.append(pts)
        for idx in pts:
            point_to_lines[idx].append(line_id)
            
    # Non-vertical lines
    for m in range(q):
        for c in range(q):
            pts = []
            for x in range(q):
                # y = m*x + c
                # x, y in F_q
                # We need to do arithmetic in F_q
                # Represent x, y, m, c as ints 0..48
                # Need mul and add in F_q
                
                # Helper for F_q arithmetic
                def fq_add(a, b):
                    aa, ab = divmod(a, p)
                    ba, bb = divmod(b, p)
                    ra = (aa + ba) % p
                    rb = (ab + bb) % p
                    return ra * p + rb
                    
                def fq_mul(a, b):
                    aa, ab = divmod(a, p)
                    ba, bb = divmod(b, p)
                    # (aa + ab*alpha) * (ba + bb*alpha)
                    # = aa*ba + (aa*bb + ab*ba)*alpha + ab*bb*alpha^2
                    # alpha^2 = 3
                    ra = (aa * ba + 3 * ab * bb) % p
                    rb = (aa * bb + ab * ba) % p
                    return ra * p + rb

                y = fq_add(fq_mul(m, x), c)
                xa, xb = divmod(x, p)
                ya, yb = divmod(y, p)
                idx = point_to_idx(xa, xb, ya, yb)
                pts.append(idx)
            line_id = len(lines)
            lines.append(pts)
            for idx in pts:
                point_to_lines[idx].append(line_id)

    # Initial clean counts
    # clean_count[x] = number of lines L through x such that L is "clean" for x given S
    clean_count = np.zeros(total, dtype=int)
    
    # Precompute S membership boolean array for fast access
    in_S = np.zeros(total, dtype=bool)
    for idx in S:
        in_S[idx] = True
        
    # For each line, count S points
    line_S_count = np.zeros(len(lines), dtype=int)
    for L in range(len(lines)):
        cnt = 0
        for idx in lines[L]:
            if in_S[idx]:
                cnt += 1
        line_S_count[L] = cnt
        
    # Calculate initial clean_count
    for x in range(total):
        for L in point_to_lines[x]:
            cnt = line_S_count[L]
            if in_S[x]:
                # Clean if cnt == 1 (and it's x)
                # Since x is in S, cnt >= 1.
                if cnt == 1:
                    clean_count[x] += 1
            else:
                # Clean if cnt == 0
                if cnt == 0:
                    clean_count[x] += 1
                    
    # Greedy growth
    # Candidates: points not in S
    candidates = [i for i in range(total) if not in_S[i]]
    
    # Shuffle candidates to avoid bias?
    # np.random.shuffle(candidates) # Maybe deterministic is better for reproducibility, but random might find better local optima.
    # Let's try random order.
    np.random.seed(42)
    np.random.shuffle(candidates)
    
    for z in candidates:
        if in_S[z]:
            continue
            
        # Tentatively add z
        # Check if adding z breaks any clean line for any point
        
        # Affected lines: lines passing through z
        affected_lines = point_to_lines[z]
        
        # We need to check if any point x has clean_count[x] == 1 
        # and the only clean line is one of the affected lines, 
        # and that line becomes dirty.
        
        # Optimization: 
        # 1. Identify lines L in affected_lines that are currently "clean" for z?
        #    If z is added, z is in S.
        #    For z to be valid, there must be a line L through z such that 
        #    (L \ {z}) \cap S = empty.
        #    This means L \cap S = {z} in the NEW set.
        #    In OLD set, L \cap S must be empty (since z was not in S).
        #    So we need a line L through z such that line_S_count[L] == 0.
        
        valid_for_z = False
        for L in affected_lines:
            if line_S_count[L] == 0:
                valid_for_z = True
                break
                
        if not valid_for_z:
            continue # z cannot be added, no clean line for z itself
            
        # Now check other points
        # For any point x != z, if its clean count drops to 0, it's invalid.
        # A line L becomes "dirty" for x if:
        # - x in S: old cnt 1 -> new cnt 2. (Was clean, now dirty)
        # - x not in S: old cnt 0 -> new cnt 1. (Was clean, now dirty)
        
        # We only need to check points x that had exactly 1 clean line, 
        # and that line is in affected_lines, and that line becomes dirty.
        
        # Let's just simulate the update for affected lines and check counts.
        # To be safe and simple, we can just re-evaluate the condition for all points 
        # on the affected lines.
        
        # But we need to REVERT if it fails.
        # So we store the old state of line_S_count for affected lines.
        old_counts = {}
        for L in affected_lines:
            old_counts[L] = line_S_count[L]
            
        # Apply change
        for L in affected_lines:
            line_S_count[L] += 1
        in_S[z] = True
            
        # Check validity
        valid = True
        for L in affected_lines:
            # Check all points on this line
            for x in lines[L]:
                # Determine if L is clean for x in NEW state
                new_clean = False
                if in_S[x]:
                    if line_S_count[L] == 1:
                        new_clean = True
                else:
                    if line_S_count[L] == 0:
                        new_clean = True
                        
                # We don't have old_clean_count stored per line per point easily.
                # But we know that only lines in affected_lines changed.
                # So for x, the number of clean lines is:
                # (Old clean lines not in affected_lines) + (New clean lines in affected_lines)
                
                # This is getting complicated to update incrementally without storing per-line cleanliness.
                # Given the small size, let's just recompute clean_count for all points?
                # No, that's 2401 * 50 = 120k ops.
                # Candidates ~ 2200.
                # 2200 * 120k = 264M ops. Might be too slow in Python.
                
                # Alternative: Only check points x where clean_count[x] == 1 
                # AND x is on an affected line.
                # If clean_count[x] > 1, losing one clean line is fine.
                # If clean_count[x] == 1, we must ensure the one clean line is NOT made dirty.
                
                # So:
                # 1. Find all x on affected lines with clean_count[x] == 1.
                # 2. For each such x, check if its unique clean line is in affected_lines 
                #    and if that line is now dirty.
                
                pass # Logic below
                
        # Let's implement the check properly
        
        # Reset in_S[z] to False temporarily for logic? No, we set it True.
        # We need to know which line was the "unique clean line" for each x.
        # We didn't store that.
        
        # Let's try a simpler check:
        # For each x in union of affected lines:
        #   Recalculate clean_count[x] from scratch?
        #   No, just check if clean_count[x] == 1.
        #   If so, check if the line that WAS clean is now dirty.
        #   How to know which line was clean?
        #   We can just check: Is there ANY line through x that is clean in NEW state?
        #   If yes, x is valid.
        #   This is O(50) per x.
        #   Number of x to check: size of union of affected lines.
        #   Affected lines: 50 lines. Each has 49 points.
        #   Union size <= 50*49 = 2450.
        #   So 2450 * 50 = 122,500 ops per candidate.
        #   2200 * 122,500 = 270M ops.
        #   This is borderline for 60s in Python.
        
        # Let's optimize the check.
        # We only care if valid becomes False.
        
        for x in set(lines[L] for L in affected_lines for _ in [0]): # Flatten
             # This is inefficient.
             pass

        # Let's just do the full recheck for affected points.
        affected_points = set()
        for L in affected_lines:
            affected_points.update(lines[L])
            
        for x in affected_points:
            # Check if x has at least one clean line in NEW state
            has_clean = False
            for L2 in point_to_lines[x]:
                if in_S[x]:
                    if line_S_count[L2] == 1:
                        has_clean = True
                        break
                else:
                    if line_S_count[L2] == 0:
                        has_clean = True
                        break
            if not has_clean:
                valid = False
                break
                
        if valid:
            # Accept z
            # Update clean_count? 
            # We don't strictly need to maintain clean_count if we do full recheck for next candidates?
            # No, we need clean_count for the "valid_for_z" check? 
            # Actually, "valid_for_z" just checks if there is a line with count 0.
            # We can just check that directly.
            
            # But we need clean_count for the next iteration's "valid_for_z"?
            # No, "valid_for_z" only depends on line_S_count.
            # So we don't need to maintain clean_count array at all!
            # We just need to check "has_clean" for affected points.
            
            # So the algorithm is:
            # 1. Check if z has a line with count 0.
            # 2. If yes, tentatively add z.
            # 3. Check all affected points if they still have a clean line.
            # 4. If yes, commit.
            
            pass # Already committed in simulation?
            # We set in_S[z]=True and line_S_count updated.
            # If valid, we keep it.
            # If not valid, we revert.
            
        else:
            # Revert
            in_S[z] = False
            for L in affected_lines:
                line_S_count[L] = old_counts[L]

    # Construct result
    N_indices = [i for i in range(total) if not in_S[i]]
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        xa, xb, ya, yb = idx_to_point(idx)
        result[i] = [[xa, xb], [ya, yb]]
        
    return result

# EVOLVE-BLOCK-END
