import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    # We keep track of original indices to return in the correct order
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # State: list of bins. Each bin is a list of free spaces.
    # A free space is a tuple (x, y, z, w, h, d) representing a cuboid.
    bins = []
    bin_volumes = []
    
    def add_space(bin_idx, x, y, z, w, h, d):
        """
        Add a new free space to the bin, merging with existing spaces if possible.
        If the new space is contained within an existing space, we don't add it (it's redundant).
        If an existing space is contained within the new space, we remove the existing one.
        If they overlap but neither contains the other, we split them.
        """
        new_space = (x, y, z, w, h, d)
        i = 0
        while i < len(bins[bin_idx]):
            ex, ey, ez, ew, eh, ed = bins[bin_idx][i]
            
            # Check if new_space is inside existing space
            if x >= ex and y >= ey and z >= ez and x + w <= ex + ew and y + h <= ey + eh and z + d <= ez + ed:
                return # New space is redundant
            
            # Check if existing space is inside new_space
            if ex >= x and ey >= y and ez >= z and ex + ew <= x + w and ey + eh <= y + h and ez + ed <= z + d:
                bins[bin_idx].pop(i)
                continue
            
            # Check for overlap
            # Overlap if: not (x+w <= ex or ex+ew <= x or y+h <= ey or ey+eh <= y or z+d <= ez or ez+ed <= z)
            if not (x + w <= ex or ex + ew <= x or
                    y + h <= ey or ey + eh <= y or
                    z + d <= ez or ez + ed <= z):
                # They overlap but don't contain each other. Split existing space.
                # We need to split the existing space into up to 6 non-overlapping parts
                # that do not intersect the new space.
                
                # Calculate intersection boundaries
                ix1 = max(x, ex)
                iy1 = max(y, ey)
                iz1 = max(z, ez)
                ix2 = min(x + w, ex + ew)
                iy2 = min(y + h, ey + eh)
                iz2 = min(z + d, ez + ed)
                
                new_spaces = []
                
                # Split along X
                if ex < x:
                    new_spaces.append((ex, ey, ez, x - ex, eh, ed))
                if ex + ew > x + w:
                    new_spaces.append((x + w, ey, ez, ex + ew - (x + w), eh, ed))
                
                # Split along Y (only for parts not already split by X? 
                # Actually, standard maximal space partitioning splits recursively.
                # To avoid duplicates and ensure disjointness, we can generate candidates 
                # and then filter, or use a specific splitting logic.
                # A simpler robust method: 
                # 1. Split X into Left and Right (relative to new box)
                # 2. Split Y of the Middle part (where X overlaps)
                # 3. Split Z of the Middle-Middle part
                
                # Let's use the "slice" method.
                # Region 1: X < new_x
                if ex < x:
                    new_spaces.append((ex, ey, ez, x - ex, eh, ed))
                # Region 2: X > new_x + new_w
                if ex + ew > x + w:
                    new_spaces.append((x + w, ey, ez, ex + ew - (x + w), eh, ed))
                
                # Middle X region: [max(ex, x), min(ex+ew, x+w)]
                mx1 = max(ex, x)
                mx2 = min(ex + ew, x + w)
                mw = mx2 - mx1
                
                if mw > 0:
                    # Within this X strip, split by Y
                    # Region 2a: Y < new_y
                    if ey < y:
                        new_spaces.append((mx1, ey, ez, mw, y - ey, ed))
                    # Region 2b: Y > new_y + new_h
                    if ey + eh > y + h:
                        new_spaces.append((mx1, y + h, ez, mw, ey + eh - (y + h), ed))
                    
                    # Middle Y strip: [max(ey, y), min(ey+eh, y+h)]
                    my1 = max(ey, y)
                    my2 = min(ey + eh, y + h)
                    mh = my2 - my1
                    
                    if mh > 0:
                        # Within this XY strip, split by Z
                        # Region 2b1: Z < new_z
                        if ez < z:
                            new_spaces.append((mx1, my1, ez, mw, mh, z - ez))
                        # Region 2b2: Z > new_z + new_d
                        if ez + ed > z + d:
                            new_spaces.append((mx1, my1, z + d, mw, mh, ez + ed - (z + d)))
                        
                        # The center part [mx1..mx2, my1..my2, max(ez,z)..min(ez+ed, z+d)] is occupied by new box.
                        # So no space there.
                
                # Remove the existing space
                bins[bin_idx].pop(i)
                # Add the new split spaces
                # Note: These new spaces might overlap with each other? 
                # No, the slicing logic ensures they are disjoint.
                # But they might overlap with *other* existing spaces in the bin.
                # So we must re-insert them using add_space logic.
                # To avoid infinite recursion or complexity, we append them to a list 
                # and then process them. 
                # However, add_space modifies the list it's iterating on.
                # Let's collect them and insert them one by one at the current index.
                
                # We popped i. So now we insert the new spaces at i.
                # We need to be careful about the order.
                for ns in reversed(new_spaces):
                    bins[bin_idx].insert(i, ns)
                    i += 1
                # After inserting, i points to the next space to check against the *original* new_space?
                # Actually, the new_space is being added. We are effectively merging new_space into the bin's space list.
                # If we split an existing space, we replace it with the fragments.
                # The "new_space" itself is NOT added as a free space, because it's occupied by the box.
                # Wait, the function `add_space` is meant to add a free space.
                # In the context of placing a box:
                # 1. Find a free space that fits the box.
                # 2. Remove that free space.
                # 3. Calculate the remaining free spaces around the box within that free space.
                # 4. Add those remaining spaces back to the bin's free space list.
                
                # So `add_space` here is actually `add_free_space`.
                # If we are splitting an existing free space because it overlaps with the *new box*,
                # we are effectively removing the part of the free space occupied by the box.
                # The logic above generates the *remaining* parts of the existing space.
                # We pop the existing space and insert the remaining parts.
                # We do NOT insert the new box's volume as a free space.
                
                # Continue checking the next space (which is now at index i)
                # The loop continues.
                
            else:
                i += 1
        
        # If we reached here, new_space did not overlap with any existing space in a way that required splitting,
        # or it was contained. If it was contained, we returned.
        # If it didn't overlap at all, we just add it.
        # Wait, the logic above pops and inserts if overlap. 
        # If no overlap, we just increment i.
        # At the end, if we didn't return (contained), we need to add the space if it wasn't merged/removed.
        # Actually, if we split, we replaced the existing space with fragments. The "new_space" argument 
        # in this context is the box being placed? No.
        
        # Let's refactor the placement logic to be clearer.
        
    def place_in_bin(bin_idx, w, h, d):
        """
        Try to place a box (w,h,d) in bin_idx.
        Returns (success, pos, new_free_spaces) or None.
        """
        # Find the best free space
        best_idx = -1
        best_score = float('inf')
        best_pos = None
        
        for i, (sx, sy, sz, sw, sh, sd) in enumerate(bins[bin_idx]):
            if w <= sw and h <= sh and d <= sd:
                # It fits. Score: prefer tightest fit (smallest remaining volume? or smallest space volume?)
                # Best Fit usually minimizes the remaining space in the chosen slot.
                # Volume of space: sw*sh*sd
                # Volume of box: w*h*d
                # Remaining: sw*sh*sd - w*h*d
                # Smaller remaining is better (tighter fit).
                score = sw * sh * sd
                if score < best_score:
                    best_score = score
                    best_idx = i
                    best_pos = (sx, sy, sz)
        
        if best_idx == -1:
            return None
        
        # Place the box at best_pos
        # We need to split the free space at best_idx
        sx, sy, sz, sw, sh, sd = bins[bin_idx][best_idx]
        
        # The box is at (sx, sy, sz) with size (w, h, d)
        # We remove the free space at best_idx
        bins[bin_idx].pop(best_idx)
        
        # Generate new free spaces
        new_spaces = []
        
        # X split
        if sx < sx + w: # Always true if w>0
             # Left part
             if sx + w > sx: # w > 0
                 pass
             # Actually, box is at sx. So left is 0 width.
             # Right part: from sx+w to sx+sw
             if sx + w < sx + sw:
                 new_spaces.append((sx + w, sy, sz, sx + sw - (sx + w), sh, sd))
        
        # Wait, the box is placed at the corner of the free space (sx, sy, sz).
        # So:
        # 1. Right of box: (sx+w, sy, sz) size (sw-w, sh, sd)
        # 2. Above box (in Y): (sx, sy+h, sz) size (sw, sh-h, sd) -- but this overlaps with 1?
        #    No, 1 is at x >= sx+w. 2 is at y >= sy+h. They are disjoint in X?
        #    1 is x in [sx+w, sx+sw]. 2 is x in [sx, sx+sw].
        #    They overlap in x [sx+w, sx+sw] and y [sy+h, sy+sh].
        #    So we must be careful.
        
        # Standard slicing for a box placed at the corner (sx, sy, sz) of a free space (sx, sy, sz, sw, sh, sd):
        
        # Space 1: To the right (X+)
        if sw > w:
            new_spaces.append((sx + w, sy, sz, sw - w, sh, sd))
        
        # Space 2: Above (Y+)
        # This space is in the region where X < sx+w? No, the whole Y+ region.
        # But part of it overlaps with Space 1.
        # To keep them disjoint, we can define:
        # Space 2: (sx, sy+h, sz) size (sw, sh-h, sd)
        # But this overlaps with Space 1 in (sx+w..sx+sw, sy+h..sy+sh, sz..sz+sd).
        # So we subtract that overlap.
        # Alternatively, use the recursive split logic from `add_space` but adapted for removal.
        
        # Let's use the same slicing logic as before, but "inverted".
        # We have a free space F and a box B inside F at corner.
        # We want to find F \ B.
        
        # F: [sx, sx+sw], [sy, sy+sh], [sz, sz+sd]
        # B: [sx, sx+w], [sy, sy+h], [sz, sz+d]
        
        # 1. X > sx+w:
        #    [sx+w, sx+sw], [sy, sy+sh], [sz, sz+sd]
        if sw > w:
            new_spaces.append((sx + w, sy, sz, sw - w, sh, sd))
            
        # 2. X <= sx+w AND Y > sy+h:
        #    [sx, sx+w], [sy+h, sy+sh], [sz, sz+sd]
        if sh > h:
            new_spaces.append((sx, sy + h, sz, w, sh - h, sd))
            
        # 3. X <= sx+w AND Y <= sy+h AND Z > sz+d:
        #    [sx, sx+w], [sy, sy+h], [sz+d, sz+sd]
        if sd > d:
            new_spaces.append((sx, sy, sz + d, w, h, sd - d))
            
        # Note: This partitioning is disjoint and covers F \ B.
        # Are these spaces maximal? Not necessarily. They might be splittable further 
        # if they touch other free spaces. But for correctness, disjointness is enough.
        # For efficiency, we might want to merge them later, but for 40 boxes, 
        # the number of free spaces is small (O(N^2) worst case, but usually much less).
        
        # Add the new spaces to the bin
        for ns in new_spaces:
            # We don't need to check for overlap with *other* spaces in the bin 
            # because F was disjoint from all other spaces, and F \ B is a subset of F.
            # So these new spaces are disjoint from all other spaces in the bin.
            bins[bin_idx].append(ns)
            
        return (sx, sy, sz)

    def place_box(w, h, d):
        # 1. Try to place in existing bins (Best Fit)
        best_bin_idx = -1
        best_score = float('inf')
        best_pos = None
        
        for i in range(len(bins)):
            res = place_in_bin(i, w, h, d)
            if res is not None:
                pos = res
                # We need a score to compare bins.
                # The score in place_in_bin was the volume of the free space used.
                # We should return that score to compare across bins.
                # Let's modify place_in_bin to return (pos, score)
                pass
        
        # Let's refine place_in_bin to return (pos, score)
        
    # Refactor place_in_bin
    
    def place_in_bin_v2(bin_idx, w, h, d):
        best_idx = -1
        best_score = float('inf')
        best_pos = None
        
        for i, (sx, sy, sz, sw, sh, sd) in enumerate(bins[bin_idx]):
            if w <= sw and h <= sh and d <= sd:
                score = sw * sh * sd
                if score < best_score:
                    best_score = score
                    best_idx = i
                    best_pos = (sx, sy, sz)
        
        if best_idx == -1:
            return None
        
        sx, sy, sz, sw, sh, sd = bins[bin_idx][best_idx]
        bins[bin_idx].pop(best_idx)
        
        new_spaces = []
        if sw > w:
            new_spaces.append((sx + w, sy, sz, sw - w, sh, sd))
        if sh > h:
            new_spaces.append((sx, sy + h, sz, w, sh - h, sd))
        if sd > d:
            new_spaces.append((sx, sy, sz + d, w, h, sd - d))
            
        for ns in new_spaces:
            bins[bin_idx].append(ns)
            
        return best_pos

    # Main Loop
    placements = [None] * n
    
    for idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
            
        placed = False
        for i in range(len(bins)):
            pos = place_in_bin_v2(i, w, h, d)
            if pos is not None:
                placements[idx] = (i, pos[0], pos[1], pos[2], w, h, d)
                placed = True
                break
        
        if not placed:
            # Open new bin
            new_bin_idx = len(bins)
            # Initial free space is the whole container
            bins.append([(0, 0, 0, W, H, D)])
            bin_volumes.append(0) # Not strictly needed if we track volumes separately, but good for debugging
            
            # Place box at (0,0,0) in the new bin
            # We can just call place_in_bin_v2 on the new bin
            pos = place_in_bin_v2(new_bin_idx, w, h, d)
            if pos is None:
                # Box doesn't fit in container? Problem states it should.
                # If not, we have an error.
                raise ValueError("Box does not fit in container")
            placements[idx] = (new_bin_idx, pos[0], pos[1], pos[2], w, h, d)
            
    # Handle timeout fallback
    if time.time() > deadline:
        unplaced_indices = [i for i in range(n) if placements[i] is None]
        for i in unplaced_indices:
            w, h, d = boxes[i]
            b = len(bins)
            bins.append([(0, 0, 0, W, H, D)])
            pos = place_in_bin_v2(b, w, h, d)
            if pos is None:
                # Should not happen
                pos = (0,0,0)
            placements[i] = (b, pos[0], pos[1], pos[2], w, h, d)
            
    return placements
