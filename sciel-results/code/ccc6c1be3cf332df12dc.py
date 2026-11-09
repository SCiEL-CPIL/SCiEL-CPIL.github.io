"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-compute volumes and dimensions
    volumes = [w * h * d for w, h, d in boxes]
    container_vol = W * H * D
    
    # Sort indices by volume descending for initial heuristic and B&B ordering
    sorted_indices = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    # --- Data Structures ---
    # We represent a solution as a list of bins.
    # Each bin is a list of (box_idx, x, y, z).
    
    def get_initial_solution():
        bins = []
        
        def find_place_in_bin(bin_items, box_idx):
            w, h, d = boxes[box_idx]
            # Candidate points: 0 and right/back/top faces of existing boxes
            xs = {0}
            ys = {0}
            zs = {0}
            for b_idx, bx, by, bz in bin_items:
                bw, bh, bd = boxes[b_idx]
                if bx + bw <= W: xs.add(bx + bw)
                if by + bh <= H: ys.add(by + bh)
                if bz + bd <= D: zs.add(bz + bd)
            
            valid_xs = sorted([x for x in xs if x + w <= W])
            if not valid_xs: return None
            valid_ys = sorted([y for y in ys if y + h <= H])
            if not valid_ys: return None
            valid_zs = sorted([z for z in zs if z + d <= D])
            if not valid_zs: return None
            
            best_pos = None
            best_score = float('inf')
            
            for x in valid_xs:
                for y in valid_ys:
                    for z in valid_zs:
                        # Check overlap
                        overlap = False
                        for b_idx, bx, by, bz in bin_items:
                            bw, bh, bd = boxes[b_idx]
                            if not (x + w <= bx or bx + bw <= x or
                                    y + h <= by or by + bh <= y or
                                    z + d <= bz or bz + bd <= z):
                                overlap = True
                                break
                        if overlap:
                            continue
                        
                        # Prefer positions closer to origin (compactness)
                        score = x * (H * D) + y * D + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
            
            return best_pos

        for idx in sorted_indices:
            placed = False
            for b_idx in range(len(bins)):
                pos = find_place_in_bin(bins[b_idx], idx)
                if pos is not None:
                    bins[b_idx].append((idx, pos[0], pos[1], pos[2]))
                    placed = True
                    break
            if not placed:
                # New bin at (0,0,0)
                # Check if box fits in empty bin
                w, h, d = boxes[idx]
                if w <= W and h <= H and d <= D:
                    bins.append([(idx, 0, 0, 0)])
                else:
                    # Box doesn't fit in any bin? Should not happen if container is valid.
                    # But if it does, we must place it somewhere to avoid invalidity, 
                    # though it will likely result in a penalty or error.
                    # For robustness, we assume it fits.
                    bins.append([(idx, 0, 0, 0)])
        
        return bins

    def calculate_score(bins):
        if not bins:
            return 0.0
        
        occupied_bins = [b for b in bins if b]
        B = len(occupied_bins)
        
        if B == 0:
            return 0.0
        
        # Last bin is the one with the highest index in the compacted list
        last_bin = occupied_bins[-1]
        last_bin_vol = sum(volumes[box_idx] for box_idx, x, y, z in last_bin)
        
        if container_vol == 0:
            return B
        fraction = last_bin_vol / container_vol
        return B + 0.1 * (1 - fraction)

    # --- Branch and Bound ---
    
    # We will try to solve the problem exactly or near-exactly using B&B.
    # State: (unplaced_indices, current_bins)
    # To make B&B feasible, we need a strong lower bound on the number of additional bins needed.
    
    def lower_bound(unplaced, current_bin_count):
        """
        Estimate the minimum number of bins needed for the unplaced boxes.
        Returns (min_bins_needed, is_lower_bound_tight)
        We use volume and max-dimension heuristics.
        """
        if not unplaced:
            return 0, True
        
        total_vol = sum(volumes[i] for i in unplaced)
        
        # Volume bound
        # Remaining capacity in current bins?
        # For simplicity and speed, we assume we start with 'current_bin_count' bins full? 
        # No, we are adding to existing bins or opening new ones.
        # A simple bound: ceil(total_vol / container_vol)
        vol_bound = math.ceil(total_vol / container_vol) if container_vol > 0 else len(unplaced)
        
        # Max dimension bound
        # If a box has dimension > W, it needs its own bin (or impossible).
        # If multiple boxes have dimension > W/2, they can't share a bin along that axis.
        # This is complex. Let's stick to a simpler dual bound.
        
        # A better bound: 
        # 1. Volume
        # 2. Count of boxes that don't fit in remaining space of current bins?
        # Since we don't track remaining space precisely in this simplified B&B state,
        # we rely on the volume bound primarily.
        
        # We can also use a "packing density" heuristic.
        # But for B&B to work, we need to prune.
        
        return vol_bound, False

    def branch_and_bound(unplaced, current_bins, best_score, best_bins, time_limit):
        """
        Recursive B&B.
        unplaced: list of indices
        current_bins: list of bins (list of (idx, x, y, z))
        """
        if time.time() > time_limit:
            return
        
        if not unplaced:
            score = calculate_score(current_bins)
            if score < best_score[0]:
                best_score[0] = score
                best_bins[0] = [list(b) for b in current_bins]
            return
        
        # Lower bound check
        lb_bins, _ = lower_bound(unplaced, len(current_bins))
        # The total bins will be at least len(current_bins) + lb_bins? 
        # No, unplaced boxes can go into existing bins.
        # So the total bins is at least len(current_bins) (if all fit) 
        # but we are minimizing B. 
        # Actually, B is the number of occupied bins.
        # If we put everything in existing bins, B stays same.
        # If we need new bins, B increases.
        
        # A better bound: 
        # The final B is at least the number of bins currently occupied (len(current_bins)) 
        # PLUS the number of NEW bins needed.
        # How many new bins? At least ceil(total_unplaced_vol / container_vol) is NOT correct 
        # because unplaced boxes can fill existing bins.
        
        # Let's use a simpler pruning:
        # If current B (occupied bins) + 0 > best_score (ignoring the 0.1 term), prune?
        # No, best_score includes the 0.1 term.
        
        # Let's try a different approach: 
        # Since n=40 is too large for pure B&B without strong heuristics, 
        # we will use B&B to optimize the *last few* boxes or to verify if a solution with B-1 bins is possible.
        
        # Alternative Strategy: 
        # 1. Get initial solution with B0 bins.
        # 2. Try to find a solution with B0-1 bins using B&B/CP-SAT style logic.
        # 3. If found, try B0-2, etc.
        # 4. If not found, optimize the last bin of the B0 solution.
        
        pass # Placeholder for complex B&B logic which is hard to implement robustly in short time
    
    # Given the complexity and time limit, a pure B&B on 40 boxes is risky.
    # However, the directive asks for a structural change to B&B/Exact.
    # Let's implement a "Best-First Search" with strong pruning, which is a form of B&B.
    
    # We will use a priority queue (min-heap) based on a heuristic estimate of the final score.
    # State: (unplaced_indices_tuple, bins_tuple, current_score_estimate)
    
    # To make this efficient, we need fast placement checks.
    
    def find_place_fast(bin_items, box_idx):
        w, h, d = boxes[box_idx]
        xs = {0}
        ys = {0}
        zs = {0}
        for b_idx, bx, by, bz in bin_items:
            bw, bh, bd = boxes[b_idx]
            if bx + bw <= W: xs.add(bx + bw)
            if by + bh <= H: ys.add(by + bh)
            if bz + bd <= D: zs.add(bz + bd)
        
        valid_xs = sorted([x for x in xs if x + w <= W])
        if not valid_xs: return None
        valid_ys = sorted([y for y in ys if y + h <= H])
        if not valid_ys: return None
        valid_zs = sorted([z for z in zs if z + d <= D])
        if not valid_zs: return None
        
        best_pos = None
        best_score = float('inf')
        
        for x in valid_xs:
            for y in valid_ys:
                for z in valid_zs:
                    overlap = False
                    for b_idx, bx, by, bz in bin_items:
                        bw, bh, bd = boxes[b_idx]
                        if not (x + w <= bx or bx + bw <= x or
                                y + h <= by or by + bh <= y or
                                z + d <= bz or bz + bd <= z):
                            overlap = True
                            break
                    if overlap:
                        continue
                    
                    score = x * (H * D) + y * D + z
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
        
        return best_pos

    # Initialize with the greedy solution
    initial_bins = get_initial_solution()
    initial_score = calculate_score(initial_bins)
    
    best_score = [initial_score]
    best_bins = [initial_bins]
    
    # We will use a simple beam search / best-first search to explore alternative placements.
    # This is effectively a bounded B&B.
    
    import heapq
    
    # Heap items: (heuristic_score, unplaced_indices, bins)
    # We start with the first box unplaced? No, we start with all boxes unplaced?
    # That's too big. 
    # Let's restart the construction using Best-First Search.
    
    # State: (idx_of_next_box_to_place, bins)
    # We process boxes in sorted order (largest first).
    
    def bfs_pack():
        # State: (unplaced_index, bins_tuple)
        # We use a priority queue to explore the most promising partial solutions.
        # Priority: estimated final score.
        
        # To keep memory and time low, we limit the beam width.
        beam_width = 50
        
        # Initial state
        # unplaced: all boxes in sorted order
        # bins: []
        
        # We will use a list of states instead of a heap if beam search is simpler.
        # State: (bins, next_idx)
        
        # Let's try a different B&B approach: 
        # Fix the first K boxes (largest) using the greedy method.
        # Then use B&B for the remaining N-K boxes.
        # This reduces the branching factor.
        
        K = 20 # Fix first 20 largest boxes greedily
        if n <= K:
            K = n
            
        # Greedy placement for first K boxes
        bins = []
        for i in range(K):
            idx = sorted_indices[i]
            placed = False
            for b_idx in range(len(bins)):
                pos = find_place_fast(bins[b_idx], idx)
                if pos is not None:
                    bins[b_idx].append((idx, pos[0], pos[1], pos[2]))
                    placed = True
                    break
            if not placed:
                w, h, d = boxes[idx]
                if w <= W and h <= H and d <= D:
                    bins.append([(idx, 0, 0, 0)])
        
        # Now B&B for remaining n-K boxes
        remaining_indices = sorted_indices[K:]
        
        def solve_remaining(unplaced, current_bins):
            if time.time() > deadline:
                return
            
            if not unplaced:
                score = calculate_score(current_bins)
                if score < best_score[0]:
                    best_score[0] = score
                    best_bins[0] = [list(b) for b in current_bins]
                return
            
            # Lower Bound Pruning
            # Current number of bins
            curr_B = len([b for b in current_bins if b])
            
            # If curr_B > best_score[0] (ignoring 0.1 term), prune?
            # best_score is B + 0.1*(1-f). Min score is B.
            # So if curr_B >= best_score[0], we can prune?
            # Yes, because score >= B. If B >= best_score, we can't improve.
            # Note: best_score[0] might be e.g. 5.1. If curr_B is 5, we might still improve to 5.05.
            # So prune if curr_B > best_score[0] is wrong.
            # Prune if curr_B >= ceil(best_score[0])? No.
            # Prune if curr_B > best_score[0] is safe? 
            # If best_score is 5.1, and curr_B is 6, score >= 6 > 5.1. Prune.
            # If curr_B is 5, score >= 5. Could be 5.05 < 5.1. Keep.
            
            if curr_B > best_score[0]:
                return
            
            # Try to place the next box (unplaced[0])
            next_idx = unplaced[0]
            rest_unplaced = unplaced[1:]
            
            # Option 1: Place in existing bins
            for b_idx in range(len(current_bins)):
                pos = find_place_fast(current_bins[b_idx], next_idx)
                if pos is not None:
                    # Create new state
                    new_bins = [list(b) for b in current_bins]
                    new_bins[b_idx].append((next_idx, pos[0], pos[1], pos[2]))
                    solve_remaining(rest_unplaced, new_bins)
                    if time.time() > deadline:
                        return
                    # Optimization: If we found a solution with B < curr_B, we might prune other branches?
                    # Not easily in this recursive structure without passing best_B down.
            
            # Option 2: Place in a new bin
            w, h, d = boxes[next_idx]
            if w <= W and h <= H and d <= D:
                new_bins = [list(b) for b in current_bins]
                new_bins.append([(next_idx, 0, 0, 0)])
                
                # Pruning: If new_B > best_score[0], skip
                new_B = len([b for b in new_bins if b])
                if new_B <= best_score[0]:
                    solve_remaining(rest_unplaced, new_bins)
                    if time.time() > deadline:
                        return
        
        solve_remaining(remaining_indices, bins)

    # Run the B&B
    bfs_pack()
    
    # If B&B didn't improve or timed out, we still have the initial greedy solution in best_bins
    # But wait, best_bins is initialized with initial_bins.
    # The B&B might have updated best_bins.
    
    # Finalize: Compact the bins
    new_bins = []
    old_to_new = {}
    for i, b in enumerate(best_bins[0]):
        if b:
            old_to_new[i] = len(new_bins)
            new_bins.append(b)
    
    # Generate placements
    placements = [None] * n
    for old_idx, b in enumerate(best_bins[0]):
        if not b:
            continue
        new_idx = old_to_new[old_idx]
        for box_idx, x, y, z in b:
            w, h, d = boxes[box_idx]
            placements[box_idx] = (new_idx, x, y, z, w, h, d)
            
    # Check for any missing placements
    for i in range(n):
        if placements[i] is None:
            # Fallback
            w, h, d = boxes[i]
            placements[i] = (len(new_bins), 0, 0, 0, w, h, d)
            new_bins.append([(i, 0, 0, 0)])
            
    return placements

# EVOLVE-BLOCK-END
