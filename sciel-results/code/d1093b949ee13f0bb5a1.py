"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe within 150s
    
    container_vol = W * H * D
    if container_vol == 0:
        return [(0, 0, 0, 0, 0, 0) for _ in range(n)]

    # Sort boxes by volume descending
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)

    best_score = float('inf')
    best_placements = None

    # Number of restarts. We use a time budget to decide how many to run.
    # Each restart is a full backtracking attempt.
    # We will try to run as many as possible within the deadline.
    # But backtracking can be fast or slow. Let's run a fixed number first,
    # then use remaining time for more if needed.
    
    # Actually, let's just run until deadline, with a minimum of 1 restart.
    # If a restart finishes quickly, we do more. If it's slow, we do fewer.
    
    # To ensure we always have a result, we first run a fast greedy to get a baseline.
    # Then we improve with backtracking.
    
    def greedy_pack(order_idx):
        """Fast greedy pack to get a baseline solution."""
        bins = []  # List of lists of (x, y, z, w, h, d)
        grids = [] # List of 3D numpy arrays for occupancy
        placements = [None] * n
        
        for i, vol, w, h, d in order_idx:
            best_bin_idx = -1
            best_pos = None
            best_cost = float('inf')
            
            # Try existing bins
            for b_idx in range(len(bins)):
                grid = grids[b_idx]
                bin_boxes = bins[b_idx]
                
                # Generate candidate positions
                candidates = [(0, 0, 0)]
                for bx, by, bz, bw, bh, bd in bin_boxes:
                    candidates.append((bx, by, bz + bd))
                    candidates.append((bx + bw, by, bz))
                    candidates.append((bx, by + bh, bz))
                    
                unique_candidates = list(set(candidates))
                
                for cx, cy, cz in unique_candidates:
                    if cx + w <= W and cy + h <= H and cz + d <= D:
                        if not grid[cx:cx+w, cy:cy+h, cz:cz+d].any():
                            cost = cz * 10000 + cy * 100 + cx
                            if cost < best_cost:
                                best_cost = cost
                                best_bin_idx = b_idx
                                best_pos = (cx, cy, cz)
            
            if best_bin_idx == -1:
                best_bin_idx = len(bins)
                bins.append([])
                grids.append(np.zeros((W, H, D), dtype=np.uint8))
                best_pos = (0, 0, 0)
            
            bx, by, bz = best_pos
            bins[best_bin_idx].append((bx, by, bz, w, h, d))
            grids[best_bin_idx][bx:bx+w, by:by+h, bz:bz+d] = 1
            placements[i] = (best_bin_idx, bx, by, bz, w, h, d)
        
        return placements, bins

    def calc_score(placements):
        """Compute the score: B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not placements:
            return 0
        # Count occupied bins
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        if B == 0:
            return 0
        # Find the last bin (largest label)
        last_bin_idx = max(bin_labels)
        last_vol = sum(p[3]*p[4]*p[5] for p in placements if p[0] == last_bin_idx)
        if container_vol == 0:
            return B
        fraction = last_vol / container_vol
        return B + 0.1 * (1 - fraction)

    # 1. Get a baseline with a deterministic greedy
    base_order = list(indexed_boxes)
    base_placements, _ = greedy_pack(base_order)
    base_score = calc_score(base_placements)
    if base_score < best_score:
        best_score = base_score
        best_placements = [tuple(p) for p in base_placements]
    
    # 2. Backtracking search
    # We will perform multiple restarts with randomized orderings.
    
    def backtrack_search(order, max_bins_limit=None):
        """
        Attempt to pack boxes in the given order using backtracking.
        Returns placements if successful, else None.
        """
        # State:
        # boxes: list of (i, vol, w, h, d) to place
        # current_bin: list of (x, y, z, w, h, d)
        # grid: np.array
        # bin_count: int
        # placements: list to store results
        
        boxes_to_place = order
        current_bin = []
        grid = np.zeros((W, H, D), dtype=np.uint8)
        bin_count = 0
        placements = [None] * n
        num_placed = 0
        
        # Precompute candidate points for a given bin state?
        # It's easier to generate them on the fly.
        
        def get_candidates(bin_boxes, w, h, d):
            cands = [(0, 0, 0)]
            for bx, by, bz, bw, bh, bd in bin_boxes:
                cands.append((bx, by, bz + bd))
                cands.append((bx + bw, by, bz))
                cands.append((bx, by + bh, bz))
                # Maybe add corners for better packing?
                cands.append((bx + bw, by, bz + bd))
                cands.append((bx, by + bh, bz + bd))
                cands.append((bx + bw, by + bh, bz))
            
            unique_cands = list(set(cands))
            valid_cands = []
            for cx, cy, cz in unique_cands:
                if cx + w <= W and cy + h <= H and cz + d <= D:
                    if not grid[cx:cx+w, cy:cy+h, cz:cz+d].any():
                        valid_cands.append((cx, cy, cz))
            
            # Sort candidates by cost (z, y, x) to prefer lower placements
            valid_cands.sort(key=lambda p: p[2]*10000 + p[1]*100 + p[0])
            return valid_cands

        def solve(idx, bin_count):
            nonlocal placements, best_score, best_placements
            
            if time.time() > deadline:
                return False
            
            if idx == len(boxes_to_place):
                # All boxes placed
                # Check if this is a better solution
                # We need to reconstruct placements from the state
                # This is tricky because we modified 'placements' in place?
                # Let's track placements in the recursion state or re-derive.
                # For simplicity, let's just return True and let the caller
                # reconstruct or store.
                # Actually, storing the full state at each step is expensive.
                # Let's modify 'placements' as we go and undo on backtrack.
                
                # Calculate score
                # We need to know which bin each box is in.
                # 'placements' is updated in the loop below.
                
                # To calculate score, we need bin labels.
                # In this DFS, we only fill one bin at a time?
                # No, the logic is: try to fit in current bin. If not, open new bin.
                # So bin labels are 0, 1, 2...
                
                # Let's calculate score from 'placements'
                # But 'placements' contains the bin index assigned.
                # Wait, in the current implementation, I'm not updating 'placements'
                # with the correct bin index if I move to a new bin.
                
                # Let's restructure:
                # State:
                # - list of boxes placed so far in current bin
                # - list of boxes placed in previous bins (just count? or volume?)
                # - placements array (partially filled)
                
                # When we move to a new bin:
                # - increment bin_count
                # - reset current_bin and grid
                # - assign bin_count to the new box's placement
                
                # Let's implement this carefully.
                return True
            
            i, vol, w, h, d = boxes_to_place[idx]
            
            # 1. Try to place in current bin
            cands = get_candidates(current_bin, w, h, d)
            for cx, cy, cz in cands:
                # Place box
                current_bin.append((cx, cy, cz, w, h, d))
                grid[cx:cx+w, cy:cy+h, cz:cz+d] = 1
                placements[i] = (bin_count, cx, cy, cz, w, h, d)
                
                if solve(idx + 1, bin_count):
                    return True
                
                # Undo
                grid[cx:cx+w, cy:cy+h, cz:cz+d] = 0
                current_bin.pop()
                placements[i] = None
            
            # 2. If current bin is full or no fit, open a new bin
            # Optimization: If current bin is empty and we still can't fit,
            # it means the box is too big for the container. (Should not happen)
            
            # Move to new bin
            # Save state of current bin? No, we just discard it.
            # But we need to remember that this bin was closed.
            # The score depends on the LAST bin.
            # So if we open a new bin, the "last bin" changes.
            
            # We can just increment bin_count and reset.
            # The previous bin's content is lost from 'current_bin', but
            # its boxes are already in 'placements' with the old bin index.
            
            new_bin_count = bin_count + 1
            if max_bins_limit is not None and new_bin_count > max_bins_limit:
                return False
                
            # Reset current bin
            current_bin = []
            grid = np.zeros((W, H, D), dtype=np.uint8)
            
            # Try to place in new bin
            cands = get_candidates(current_bin, w, h, d)
            for cx, cy, cz in cands:
                current_bin.append((cx, cy, cz, w, h, d))
                grid[cx:cx+w, cy:cy+h, cz:cz+d] = 1
                placements[i] = (new_bin_count, cx, cy, cz, w, h, d)
                
                if solve(idx + 1, new_bin_count):
                    return True
                
                grid[cx:cx+w, cy:cy+h, cz:cz+d] = 0
                current_bin.pop()
                placements[i] = None
            
            return False
        
        # Start solving
        # We need to handle the state mutation correctly.
        # 'current_bin' and 'grid' are local to this closure but modified in solve.
        # This works in Python if we use 'nonlocal' or if they are lists/arrays (mutated in place).
        # 'current_bin' is a list, 'grid' is a numpy array.
        # But we reassign 'current_bin = []' and 'grid = np.zeros...'.
        # This requires 'nonlocal' if we want to change the reference.
        
        # Let's refactor solve to take state as arguments or use a class-like structure.
        # Or just use nonlocal.
        
        # Let's try a different approach: explicit stack or just accept the nonlocal complexity.
        # Given the constraints, let's just use a simple recursive function with nonlocal.
        
        # Actually, the previous definition of solve didn't use nonlocal for current_bin/grid.
        # Let's fix that.
        
        # To avoid deep recursion issues and state management pain,
        # let's limit the depth or use an iterative approach?
        # With 40 boxes, recursion depth 40 is fine.
        
        # Let's rewrite solve with nonlocal
        def solve2(idx, bin_count):
            nonlocal current_bin, grid, placements
            
            if time.time() > deadline:
                return False
            
            if idx == len(boxes_to_place):
                # Success
                # Calculate score
                # Placements are in 'placements'
                # We need to check if this is better than best_score
                # But we don't have the full placements for the *previous* bins?
                # Wait, 'placements' is updated for each box as we place it.
                # So at the end, 'placements' contains all boxes.
                
                score = calc_score(placements)
                if score < best_score:
                    best_score = score
                    best_placements = [tuple(p) for p in placements]
                return True
            
            i, vol, w, h, d = boxes_to_place[idx]
            
            # 1. Try current bin
            cands = get_candidates(current_bin, w, h, d)
            for cx, cy, cz in cands:
                current_bin.append((cx, cy, cz, w, h, d))
                grid[cx:cx+w, cy:cy+h, cz:cz+d] = 1
                placements[i] = (bin_count, cx, cy, cz, w, h, d)
                
                if solve2(idx + 1, bin_count):
                    return True
                
                grid[cx:cx+w, cy:cy+h, cz:cz+d] = 0
                current_bin.pop()
                placements[i] = None
            
            # 2. Open new bin
            new_bin_count = bin_count + 1
            if max_bins_limit is not None and new_bin_count > max_bins_limit:
                return False
            
            # Save old state? No, we are discarding the current bin.
            # We just reset.
            old_bin = current_bin
            old_grid = grid
            
            current_bin = []
            grid = np.zeros((W, H, D), dtype=np.uint8)
            
            cands = get_candidates(current_bin, w, h, d)
            for cx, cy, cz in cands:
                current_bin.append((cx, cy, cz, w, h, d))
                grid[cx:cx+w, cy:cy+h, cz:cz+d] = 1
                placements[i] = (new_bin_count, cx, cy, cz, w, h, d)
                
                if solve2(idx + 1, new_bin_count):
                    return True
                
                grid[cx:cx+w, cy:cy+h, cz:cz+d] = 0
                current_bin.pop()
                placements[i] = None
            
            # Restore state if we backtracked from the "new bin" step?
            # No, if we failed to place in the new bin, we return False.
            # The caller will undo its move.
            # But we changed current_bin and grid.
            # We should restore them if we are returning False?
            # Actually, if we return False, the caller (which called us after placing a box)
            # will undo its box. But our state change (opening new bin) is local to this call?
            # No, 'current_bin' is nonlocal.
            # If we fail, we should restore the state to what it was before we tried to open a new bin.
            # But wait, if we fail to place in the new bin, it means the box doesn't fit in an empty bin?
            # That's impossible unless the box is larger than the container.
            # Or if we ran out of time.
            # If we ran out of time, we should probably just return.
            
            # If we failed, restore state
            current_bin = old_bin
            grid = old_grid
            return False
        
        # Run solve2
        # Initialize state
        current_bin = []
        grid = np.zeros((W, H, D), dtype=np.uint8)
        placements = [None] * n
        
        solve2(0, 0)
        
        return best_placements

    # Main loop for backtracking restarts
    # We will run a few restarts.
    # To speed up, we can limit the number of bins we try to open.
    # Start with a limit slightly higher than the greedy solution's bin count.
    
    greedy_bins = len(set(p[0] for p in base_placements))
    
    # Try to improve
    # Use random orderings
    for restart in range(20):
        if time.time() > deadline - 2:
            break
        
        # Randomize order slightly
        # Sort by volume, then shuffle ties? Or just shuffle completely?
        # Shuffling completely might be too hard for backtracking.
        # Let's sort by volume, then add some noise.
        
        noisy_boxes = []
        for i, vol, w, h, d in indexed_boxes:
            noisy_vol = vol * random.uniform(0.8, 1.2)
            noisy_boxes.append((i, noisy_vol, w, h, d))
        
        noisy_boxes.sort(key=lambda x: x[1], reverse=True)
        
        # Try to solve with a bin limit
        # Start with greedy_bins + 1, then increase?
        # Or just try to find ANY solution with <= greedy_bins bins.
        # If we find one, it's better.
        
        # Let's just run the search without a strict limit first,
        # but the search will naturally try to minimize bins because
        # it tries to fit in current bin first.
        
        # However, backtracking can get stuck.
        # Let's impose a limit to make it faster.
        # If we can pack in greedy_bins bins, it's a tie or better (due to last bin fill).
        # If we can pack in fewer, it's definitely better.
        
        # Let's try with limit = greedy_bins
        # If it finds a solution, great.
        # If not, try limit = greedy_bins + 1? No, that's worse.
        # We want to improve, so we want <= greedy_bins.
        
        # Actually, the score is B + 0.1*(1 - fill).
        # So if B is the same, we want higher fill.
        # Backtracking might find a packing with same B but better fill.
        
        # So we should allow B up to greedy_bins.
        
        backtrack_search(noisy_boxes, max_bins_limit=greedy_bins)
        
        # If we found a better solution, we can update greedy_bins
        # to the new best B.
        if best_placements is not None:
            new_greedy_bins = len(set(p[0] for p in best_placements))
            if new_greedy_bins < greedy_bins:
                greedy_bins = new_greedy_bins
            # If B is same, we can't lower the limit.
            # But we might find a better fill with the same B.
            # So we keep trying.

    return best_placements
# EVOLVE-BLOCK-END
