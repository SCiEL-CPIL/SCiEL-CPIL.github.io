import time
import sys
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) 0-indexed
    ap_indices = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_indices.append((a, a + d, a + 2 * d, a + 3 * d))
    
    ap_array = np.array(ap_indices, dtype=np.int64)  # shape (num_aps, 4)
    num_aps = len(ap_array)
    
    # For each position, which APs does it appear in?
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_indices):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations_vectorized(colors_np):
        """Count violations using numpy vectorization."""
        # Extract colors at AP positions: shape (num_aps, 4)
        vals = colors_np[ap_array]  # (num_aps, 4)
        # A 4-AP is valid if it has at least 3 distinct colors
        # Invalid if all 4 same, or exactly 2 distinct colors
        # Check: are all 4 the same?
        all_same = np.all(vals == vals[:, 0:1], axis=1)
        # Check: exactly 2 distinct? This is harder. 
        # Alternative: count distinct values per row
        # For each row, sort and count unique
        sorted_vals = np.sort(vals, axis=1)
        # Count unique: number of positions where val != previous
        diffs = np.diff(sorted_vals, axis=1)
        num_unique = 1 + np.sum(diffs != 0, axis=1)
        violations = np.sum(num_unique < 3)
        return int(violations)
    
    def get_violation_positions_vectorized(colors_np):
        """Return boolean mask of which APs are violated."""
        vals = colors_np[ap_array]
        sorted_vals = np.sort(vals, axis=1)
        diffs = np.diff(sorted_vals, axis=1)
        num_unique = 1 + np.sum(diffs != 0, axis=1)
        return num_unique < 3
    
    # Greedy solution first
    greedy_colors = []
    for x in range(1, n + 1):
        idx = x - 1
        best_color = -1
        for color in range(128):
            valid = True
            for d in range(1, (idx // 3) + 1):
                a = idx - 3 * d
                c1 = greedy_colors[a]
                c2 = greedy_colors[a + d]
                c3 = greedy_colors[a + 2 * d]
                if len({color, c1, c2, c3}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            for d in range(1, (idx // 2) + 1):
                a = idx - 2 * d
                c1 = greedy_colors[a]
                c2 = greedy_colors[a + d]
                if color == c1 and c1 == c2:
                    valid = False
                    break
            if valid:
                best_color = color
                break
        if best_color == -1:
            best_color = 0
        greedy_colors.append(best_color)
        
    num_greedy_colors = len(set(greedy_colors))
    
    if num_greedy_colors <= 3:
        return greedy_colors
    
    # Vectorized local search
    def local_search_vectorized():
        best_colors = None
        best_violations = float('inf')
        
        num_attempts = 30
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Random initial coloring with 3 colors
            colors = np.random.randint(0, 3, size=n).astype(np.int64)
            colors_list = colors.tolist()
            violations = count_violations_vectorized(colors)
            
            if violations == 0:
                return colors_list
            
            if violations < best_violations:
                best_violations = violations
                best_colors = colors_list[:]
            
            # Hill climbing with numpy-based delta calculation
            current_violations = violations
            max_iters = 100000
            plateau_count = 0
            max_plateau = 5000
            
            for i in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Pick a random position
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, 2)
                
                if new_color == old_color:
                    continue
                
                # Calculate delta using only affected APs
                affected_aps = pos_to_aps[pos]
                if not affected_aps:
                    continue
                
                # Get the AP indices as numpy array
                ap_idx_arr = np.array(affected_aps, dtype=np.int64)
                
                # Before: check these APs
                vals_before = colors[ap_array[ap_idx_arr]]
                sorted_before = np.sort(vals_before, axis=1)
                diffs_before = np.diff(sorted_before, axis=1)
                unique_before = 1 + np.sum(diffs_before != 0, axis=1)
                viol_before = np.sum(unique_before < 3)
                
                # After: temporarily change
                colors[pos] = new_color
                vals_after = colors[ap_array[ap_idx_arr]]
                sorted_after = np.sort(vals_after, axis=1)
                diffs_after = np.diff(sorted_after, axis=1)
                unique_after = 1 + np.sum(diffs_after != 0, axis=1)
                viol_after = np.sum(unique_after < 3)
                
                delta = int(viol_after - viol_before)
                new_violations = current_violations + delta
                
                if new_violations < current_violations:
                    current_violations = new_violations
                    plateau_count = 0
                    if current_violations == 0:
                        return colors.tolist()
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors = colors.tolist()[:]
                elif new_violations == current_violations:
                    if random.random() < 0.3:
                        plateau_count += 1
                    else:
                        colors[pos] = old_color
                else:
                    colors[pos] = old_color
                    plateau_count += 1
                
                if plateau_count > max_plateau:
                    break
            
            if current_violations == 0:
                return colors.tolist()
        
        if best_colors is not None:
            return best_colors
        return None
    
    result = local_search_vectorized()
    if result is not None:
        # Verify the result
        colors_np = np.array(result, dtype=np.int64)
        v = count_violations_vectorized(colors_np)
        if v == 0:
            return result
    
    # Fallback: try a few more random restarts with different strategies
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        # Use a structured initial coloring based on modular arithmetic
        mod = random.choice([3, 4, 5, 6, 7, 8])
        colors = [(i % mod) % 3 for i in range(n)]
        colors_np = np.array(colors, dtype=np.int64)
        violations = count_violations_vectorized(colors_np)
        
        if violations == 0:
            return colors
        
        # Quick hill climb
        current_violations = violations
        for i in range(50000):
            if time.time() > deadline:
                break
            
            pos = random.randint(0, n - 1)
            old_color = colors_np[pos]
            new_color = random.randint(0, 2)
            
            if new_color == old_color:
                continue
            
            affected_aps = pos_to_aps[pos]
            if not affected_aps:
                continue
            
            ap_idx_arr = np.array(affected_aps, dtype=np.int64)
            vals_before = colors_np[ap_array[ap_idx_arr]]
            sorted_before = np.sort(vals_before, axis=1)
            diffs_before = np.diff(sorted_before, axis=1)
            unique_before = 1 + np.sum(diffs_before != 0, axis=1)
            viol_before = np.sum(unique_before < 3)
            
            colors_np[pos] = new_color
            vals_after = colors_np[ap_array[ap_idx_arr]]
            sorted_after = np.sort(vals_after, axis=1)
            diffs_after = np.diff(sorted_after, axis=1)
            unique_after = 1 + np.sum(diffs_after != 0, axis=1)
            viol_after = np.sum(unique_after < 3)
            
            delta = int(viol_after - viol_before)
            new_violations = current_violations + delta
            
            if new_violations <= current_violations:
                current_violations = new_violations
                if current_violations == 0:
                    return colors_np.tolist()
            else:
                colors_np[pos] = old_color
        
        if current_violations == 0:
            return colors_np.tolist()
    
    # Return greedy as fallback
    return greedy_colors
