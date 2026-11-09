"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    W, H, D = container
    n = len(boxes)
    
    # Pre-compute all 6 orientations for each box
    def get_orientations(w, h, d):
        return list(set(permutations([w, h, d])))
    
    # Sort boxes by volume descending (largest first)
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    
    # State: list of placed boxes as (x, y, z, w, h, d)
    def can_place(state, x, y, z, bw, bh, bd):
        # Check bounds
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        # Check overlap with all placed boxes
        for (px, py, pz, pw, ph, pd) in state:
            if not (x + bw <= px or px + pw <= x or
                    y + bh <= py or py + ph <= y or
                    z + bd <= pz or pz + pd <= z):
                return False
        return True
    
    def place_box(state, box_idx, bw, bh, bd):
        """Try to place a box at the best position (bottom-left-front priority)"""
        # Generate candidate positions: try to place at lowest possible z, then y, then x
        # Use a simple approach: scan from bottom-left-front
        best_pos = None
        best_score = None
        
        # Try placing at origin first, then shift
        # For efficiency, try a limited set of candidate positions
        # The key insight: for each axis, the box can be placed at position 0 or at 
        # (existing_box_edge) positions
        
        # Collect candidate x positions
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for (px, py, pz, pw, ph, pd) in state:
            x_candidates.append(px + pw)
            y_candidates.append(py + ph)
            z_candidates.append(pz + pd)
        
        # Deduplicate and sort
        x_candidates = sorted(set(x_candidates))
        y_candidates = sorted(set(y_candidates))
        z_candidates = sorted(set(z_candidates))
        
        # Try all combinations but limit to reasonable count
        # Sort by (z, y, x) to prefer bottom-left-front
        candidates = []
        for zc in z_candidates:
            if zc + bd > D:
                continue
            for yc in y_candidates:
                if yc + bh > H:
                    continue
                for xc in x_candidates:
                    if xc + bw > W:
                        continue
                    if can_place(state, xc, yc, zc, bw, bh, bd):
                        candidates.append((zc, yc, xc))
        
        if not candidates:
            return None
        
        # Pick the one with lowest (z, y, x)
        candidates.sort(key=lambda p: (p[0], p[1], p[2]))
        return candidates[0][2], candidates[0][1], candidates[0][0]  # x, y, z
    
    def compute_filled_volume(state):
        return sum(bw * bh * bd for (_, _, _, bw, bh, bd) in state)
    
    # Greedy construction with multiple strategies
    def greedy_placement(order, orientation_strategy='first'):
        """Place boxes in given order with given orientation strategy"""
        state = []
        results = [-1] * n  # -1 means not placed
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = get_orientations(w, h, d)
            
            placed = False
            for (bw, bh, bd) in orients:
                pos = place_box(state, idx, bw, bh, bd)
                if pos is not None:
                    x, y, z = pos
                    state.append((x, y, z, bw, bh, bd))
                    results[idx] = 0
                    placed = True
                    break
            
            if not placed:
                results[idx] = -1
        
        return state, results
    
    # Try multiple orderings and pick the best
    best_state = None
    best_results = None
    best_volume = 0
    
    # Strategy 1: Volume descending
    state1, res1 = greedy_placement(box_indices, 'first')
    vol1 = compute_filled_volume(state1)
    if vol1 > best_volume:
        best_state, best_results, best_volume = state1, res1, vol1
    
    # Strategy 2: Volume ascending (sometimes helps pack tighter)
    order2 = sorted(box_indices, key=lambda i: box_volumes[i])
    state2, res2 = greedy_placement(order2, 'first')
    vol2 = compute_filled_volume(state2)
    if vol2 > best_volume:
        best_state, best_results, best_volume = state2, res2, vol2
    
    # Strategy 3: Sort by max dimension descending
    order3 = sorted(box_indices, key=lambda i: max(boxes[i]), reverse=True)
    state3, res3 = greedy_placement(order3, 'first')
    vol3 = compute_filled_volume(state3)
    if vol3 > best_volume:
        best_state, best_results, best_volume = state3, res3, vol3
    
    # Strategy 4: Sort by min dimension descending
    order4 = sorted(box_indices, key=lambda i: min(boxes[i]), reverse=True)
    state4, res4 = greedy_placement(order4, 'first')
    vol4 = compute_filled_volume(state4)
    if vol4 > best_volume:
        best_state, best_results, best_volume = state4, res4, vol4
    
    # Try random perturbations if we have time
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        rng = random.Random(42)
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Create a perturbed order
            order = box_indices[:]
            # Randomly swap some elements
            for _ in range(5):
                i, j = rng.randint(0, n-1), rng.randint(0, n-1)
                order[i], order[j] = order[j], order[i]
            
            state_r, res_r = greedy_placement(order, 'first')
            vol_r = compute_filled_volume(state_r)
            if vol_r > best_volume:
                best_state, best_results, best_volume = state_r, res_r, vol_r
    
    # Build final placement list
    placements = []
    for i in range(n):
        if best_results[i] == 0:
            # Find the box in state
            # We need to track which state entry corresponds to which box
            # Re-do: track properly
            pass
    
    # Actually, let's redo this more carefully with proper tracking
    def greedy_placement_tracked(order):
        """Place boxes in given order, tracking which box goes where"""
        state = []  # list of (box_idx, x, y, z, bw, bh, bd)
        results = [-1] * n
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = get_orientations(w, h, d)
            
            placed = False
            for (bw, bh, bd) in orients:
                pos = place_box([(s[1], s[2], s[3], s[4], s[5], s[6]) for s in state], idx, bw, bh, bd)
                if pos is not None:
                    x, y, z = pos
                    state.append((idx, x, y, z, bw, bh, bd))
                    results[idx] = 0
                    placed = True
                    break
            
            if not placed:
                results[idx] = -1
        
        return state, results
    
    # Redo all strategies with tracking
    best_state_t = None
    best_results_t = None
    best_volume_t = 0
    
    for order in [box_indices, order2, order3, order4]:
        state_t, res_t = greedy_placement_tracked(order)
        vol_t = compute_filled_volume([(s[4], s[5], s[6]) for s in state_t])
        if vol_t > best_volume_t:
            best_state_t, best_results_t, best_volume_t = state_t, res_t, vol_t
    
    # Try random perturbations
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        rng = random.Random(42)
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            order = box_indices[:]
            for _ in range(5):
                i, j = rng.randint(0, n-1), rng.randint(0, n-1)
                order[i], order[j] = order[j], order[i]
            
            state_t, res_t = greedy_placement_tracked(order)
            vol_t = compute_filled_volume([(s[4], s[5], s[6]) for s in state_t])
            if vol_t > best_volume_t:
                best_state_t, best_results_t, best_volume_t = state_t, res_t, vol_t
    
    # Build final output
    placements = []
    for i in range(n):
        if best_results_t[i] == 0:
            # Find in state
            for s in best_state_t:
                if s[0] == i:
                    placements.append((0, s[1], s[2], s[3], s[4], s[5], s[6]))
                    break
        else:
            w, h, d = boxes[i]
            placements.append((-1, 0, 0, 0, w, h, d))
    
    return placements

# EVOLVE-BLOCK-END
