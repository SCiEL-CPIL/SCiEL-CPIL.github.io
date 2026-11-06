import time
import random
import itertools
import math

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    
    # Safety margin: The evaluator kills at 150s, but the task says 120s limit.
    # We use 115s to be safe.
    deadline = time.time() + 115.0
    
    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]
    
    # --- Strategy: Max-Contact Local Search ---
    # The current program uses a greedy "lowest point" heuristic for placement.
    # This new program uses a "Max Contact" heuristic: for each box, it tries all
    # valid positions (derived from existing box corners) and picks the one that
    # maximizes the surface area in contact with other boxes (or container walls).
    # This tends to create denser, more compact packs by filling gaps.
    # We then use Simulated Annealing on the box ordering to find a sequence
    # that allows this heuristic to produce a high-volume packing.
    
    def pack_max_contact(order):
        placed = [] # list of (idx, x, y, z, w, h, d)
        candidates = set([(0, 0, 0)])
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_contact = -1
            best_score = float('inf') # tie-breaker: lowest z, y, x
            
            for (cx, cy, cz) in candidates:
                # Check if anchor is inside any placed box
                inside = False
                for (p_idx, px, py, pz, pw, ph, pd) in placed:
                    if px <= cx < px + pw and py <= cy < py + ph and pz <= cz < pz + pd:
                        inside = True
                        break
                if inside:
                    continue
                
                for (bw, bh, bd) in orient_cache[idx]:
                    # Check bounds
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    # Check overlap
                    valid = True
                    contact_area = 0
                    for (p_idx, px, py, pz, pw, ph, pd) in placed:
                        # Overlap check
                        if (cx < px + pw and cx + bw > px and
                            cy < py + ph and cy + bh > py and
                            cz < pz + pd and cz + bd > pz):
                            valid = False
                            break
                        
                        # Contact area calculation (if not overlapping, check face contact)
                        # Face contact requires touching on one axis and overlapping on the other two.
                        
                        # X-face contact
                        if (cx == px + pw or cx + bw == px):
                            # Overlap in Y and Z
                            y_overlap = min(cy + bh, py + ph) - max(cy, py)
                            z_overlap = min(cz + bd, pz + pd) - max(cz, pz)
                            if y_overlap > 0 and z_overlap > 0:
                                contact_area += y_overlap * z_overlap
                                
                        # Y-face contact
                        if (cy == py + ph or cy + bh == py):
                            # Overlap in X and Z
                            x_overlap = min(cx + bw, px + pw) - max(cx, px)
                            z_overlap = min(cz + bd, pz + pd) - max(cz, pz)
                            if x_overlap > 0 and z_overlap > 0:
                                contact_area += x_overlap * z_overlap
                                
                        # Z-face contact
                        if (cz == pz + pd or cz + bd == pz):
                            # Overlap in X and Y
                            x_overlap = min(cx + bw, px + pw) - max(cx, px)
                            y_overlap = min(cy + bh, py + ph) - max(cy, py)
                            if x_overlap > 0 and y_overlap > 0:
                                contact_area += x_overlap * y_overlap
                            
                    if not valid:
                        continue
                    
                    # Add container wall contact (optional, but helps pack against walls)
                    if cx == 0: contact_area += bh * bd
                    if cx + bw == W: contact_area += bh * bd
                    if cy == 0: contact_area += bw * bd
                    if cy + bh == H: contact_area += bw * bd
                    if cz == 0: contact_area += bw * bh
                    if cz + bd == D: contact_area += bw * bh
                    
                    # Score: Maximize contact, then minimize z, y, x
                    score = -contact_area * 10000 + 100 * cz + cy + cx
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz, bw, bh, bd)
                        best_contact = contact_area
            
            if best_pos:
                x, y, z, bw, bh, bd = best_pos
                placed.append((idx, x, y, z, bw, bh, bd))
                
                # Update candidates
                new_corners = [
                    (x, y, z), (x+bw, y, z), (x, y+bh, z), (x+bw, y+bh, z),
                    (x, y, z+bd), (x+bw, y, z+bd), (x, y+bh, z+bd), (x+bw, y+bh, z+bd)
                ]
                for nc in new_corners:
                    if 0 <= nc[0] < W and 0 <= nc[1] < H and 0 <= nc[2] < D:
                        candidates.add(nc)
                
                # Prune candidates that are now inside the newly placed box
                filtered = set()
                for c in candidates:
                    if not (x <= c[0] < x + bw and y <= c[1] < y + bh and z <= c[2] < z + bd):
                        filtered.add(c)
                candidates = filtered
                
        return placed

    # --- Search Strategy: Simulated Annealing on Order ---
    
    indexed = list(range(n))
    
    # Initial diverse orders
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    order_sum = sorted(indexed, key=lambda i: sum(boxes[i]), reverse=True)
    
    def aspect_ratio(i):
        w, h, d = boxes[i]
        return max(w, h, d) / min(w, h, d)
    order_aspect = sorted(indexed, key=aspect_ratio)
    
    all_orders = [order_vol, order_max, order_sum, order_aspect]
    
    # Add a few random orders
    for _ in range(5):
        rand_order = indexed[:]
        random.shuffle(rand_order)
        all_orders.append(rand_order)
        
    best_placed = []
    best_vol = 0
    best_order = None
    
    # Evaluate initial orders
    for order in all_orders:
        if time.time() > deadline:
            break
        p = pack_max_contact(order)
        v = sum(p[4]*p[5]*p[6] for p in p)
        if v > best_vol:
            best_vol = v
            best_placed = p[:]
            best_order = order[:]
            if len(p) == n:
                break # Perfect pack
    
    # If we have time, run Simulated Annealing
    if best_order and len(best_placed) < n and time.time() < deadline:
        current_order = best_order[:]
        current_vol = best_vol
        
        # SA Parameters
        T = 100.0
        T_min = 1.0
        cooling_rate = 0.995
        
        no_improve = 0
        max_no_improve = 100 # Plateau detection
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Generate neighbor: Swap two boxes in the order
            i, j = random.sample(range(n), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            p = pack_max_contact(new_order)
            v = sum(p[4]*p[5]*p[6] for p in p)
            
            delta = v - current_vol
            
            if delta > 0:
                # Accept
                current_vol = v
                current_order = new_order
                no_improve = 0
                if v > best_vol:
                    best_vol = v
                    best_placed = p[:]
                    if len(p) == n:
                        break
            else:
                # Metropolis Criterion
                if random.random() < math.exp(delta / T):
                    current_vol = v
                    current_order = new_order
                else:
                    no_improve += 1
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # If we hit T_min and no improvement for a while, restart with higher T?
                # Or just stop if no_improve is high.
                # Let's just keep going until deadline or plateau.
                
    # Finalize result
    result = [None] * n
    for p in best_placed:
        idx, x, y, z, w, h, d = p
        result[idx] = (0, x, y, z, w, h, d)
    
    for i in range(n):
        if result[i] is None:
            result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return result

# EVOLVE-BLOCK-END
