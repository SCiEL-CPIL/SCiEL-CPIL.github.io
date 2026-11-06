import time
import random
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    t0 = time.time()
    DEADLINE = 115
    
    n = len(boxes)
    
    box_orientations = []
    box_volumes = []
    for i in range(n):
        w, h, d = boxes[i]
        box_volumes.append(w * h * d)
        orients = list(set(permutations([w, h, d])))
        box_orientations.append(orients)
        
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: -box_volumes[i])
    
    best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    best_volume = 0
    
    def run_greedy(seed_val, temperature=0.1):
        random.seed(seed_val)
        # Randomize order of boxes with equal volume
        sorted_indices = sorted(box_indices, key=lambda i: (-box_volumes[i], random.random()))
        
        placed_boxes = []
        candidate_set = {(0, 0, 0)}
        candidates = [(0, 0, 0)]
        placements = [None] * n
        total_vol = 0
        
        def check_overlap(x, y, z, ow, oh, od):
            for px, py, pz, pw, ph, pd, _ in placed_boxes:
                if (x < px + pw and px < x + ow and
                    y < py + ph and py < y + oh and
                    z < pz + pd and pz < z + od):
                    return True
            return False
        
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            valid_positions = []
            
            for cx, cy, cz in candidates:
                for ow, oh, od in orients:
                    if ow > W or oh > H or od > D:
                        continue
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    if check_overlap(cx, cy, cz, ow, oh, od):
                        continue
                    
                    valid_positions.append((cz, cy, cx, ow, oh, od))
            
            if valid_positions:
                # Sort by (z, y, x)
                valid_positions.sort(key=lambda p: (p[0], p[1], p[2]))
                
                # Stochastic selection
                # Probability of picking index i is proportional to exp(-i * temperature)
                # Or simply: pick index 0 with prob 0.8, index 1 with 0.1, etc.
                # Let's use a simple geometric distribution
                # r = random.random()
                # if r < 0.7: idx_pick = 0
                # elif r < 0.9: idx_pick = 1
                # else: idx_pick = random.randint(2, len(valid_positions)-1)
                
                # Better: Weighted choice
                # Weights decay exponentially
                weights = [1.0 / (i + 1) for i in range(len(valid_positions))]
                # Normalize
                sum_w = sum(weights)
                weights = [w / sum_w for w in weights]
                
                # Choose
                r = random.random()
                cumulative = 0
                chosen_idx = 0
                for i, wgt in enumerate(weights):
                    cumulative += wgt
                    if r <= cumulative:
                        chosen_idx = i
                        break
                
                bz, by, bx, bw, bh, bd = valid_positions[chosen_idx]
                # Note: valid_positions stores (z, y, x, w, h, d)
                # So bx, by, bz are x, y, z
                
                placements[idx] = (0, bx, by, bz, bw, bh, bd)
                placed_boxes.append((bx, by, bz, bw, bh, bd, idx))
                total_vol += box_volumes[idx]
                
                new_candidates = [
                    (bx, by, bz),
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx + bw, by + bh, bz),
                    (bx, by, bz + bd),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                ]
                
                for nc in new_candidates:
                    if nc not in candidate_set:
                        candidate_set.add(nc)
                        candidates.append(nc)
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
        return placements, total_vol

    iterations = 0
    while time.time() - t0 < DEADLINE:
        seed = iterations
        placements, vol = run_greedy(seed)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
        iterations += 1
        
    return best_placements
