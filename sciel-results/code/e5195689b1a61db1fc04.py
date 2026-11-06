import time
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    
    # Pre-compute all 4-term APs
    aps = []
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
            
    num_aps = len(aps)
    aps_np = np.array(aps, dtype=np.int32)
    
    def count_violations(colors):
        """
        Vectorized count of 4-term APs with < 3 distinct colors.
        colors: numpy array of shape (N,)
        """
        # Get colors at all positions of all APs
        ap_colors = colors[aps_np] # shape (num_aps, 4)
        
        c0, c1, c2, c3 = ap_colors[:, 0], ap_colors[:, 1], ap_colors[:, 2], ap_colors[:, 3]
        
        # A 4-term AP uses < 3 distinct colors if:
        # 1. All 4 are same
        # 2. Exactly 2 distinct colors (patterns: AABB, ABAB, ABBA, AABC, etc.)
        # It is easier to count distinct colors directly or check for 1 or 2 distinct.
        
        # Check for 1 distinct color
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # Check for exactly 2 distinct colors
        # This happens if not all_same and (sorted unique count == 2)
        # We can check pairs:
        # If c0==c1 and c2==c3 and c0!=c2 -> 2 distinct
        # If c0==c2 and c1==c3 and c0!=c1 -> 2 distinct
        # If c0==c3 and c1==c2 and c0!=c1 -> 2 distinct
        # If c0==c1 and c0!=c2 and c0!=c3 and c2==c3 -> covered by first? No.
        # Let's just count distincts.
        
        # For each AP, count distinct colors
        # Since it's small (4 elements), we can do:
        # distinct_count = 1 + (c0!=c1) + (c0!=c2 & c1!=c2) + (c0!=c3 & c1!=c3 & c2!=c3)
        
        d1 = (c0 != c1)
        d2 = (c0 != c2) & (c1 != c2)
        d3 = (c0 != c3) & (c1 != c3) & (c2 != c3)
        
        distinct_count = 1 + d1 + d2 + d3
        
        # Violation if distinct_count < 3
        return np.sum(distinct_count < 3)

    def get_violated_indices(colors):
        """
        Returns indices of APs that are violated.
        """
        ap_colors = colors[aps_np]
        c0, c1, c2, c3 = ap_colors[:, 0], ap_colors[:, 1], ap_colors[:, 2], ap_colors[:, 3]
        
        d1 = (c0 != c1)
        d2 = (c0 != c2) & (c1 != c2)
        d3 = (c0 != c3) & (c1 != c3) & (c2 != c3)
        
        distinct_count = 1 + d1 + d2 + d3
        return np.where(distinct_count < 3)[0]

    def run_seed():
        colors=[]
        for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
        return colors

    # Get initial seed solution
    seed_colors = np.array(run_seed(), dtype=np.int32)
    best_colors = seed_colors[:]
    best_k = len(np.unique(seed_colors))
    
    # Local Search to repair violations
    def local_search_repair(colors, target_k, time_limit):
        """
        Attempts to reduce violations to 0.
        Returns (colors, violations)
        """
        colors = colors.copy()
        current_violations = count_violations(colors)
        best_violations = current_violations
        best_colors = colors.copy()
        
        if current_violations == 0:
            return colors, 0
            
        steps_since_improve = 0
        max_steps_no_improve = 1000
        
        while time.time() < time_limit:
            if current_violations == 0:
                return colors, 0
                
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                steps_since_improve = 0
            else:
                steps_since_improve += 1
                if steps_since_improve > max_steps_no_improve:
                    # Random restart within this repair phase? Or break?
                    # Let's just break and return best
                    break
            
            # Pick a random violated AP
            violated_indices = get_violated_indices(colors)
            if len(violated_indices) == 0:
                return colors, 0
                
            ai = int(random.choice(violated_indices))
            ap = aps_np[ai]
            
            # Pick a random position in the AP to change
            pos = random.randint(0, 3)
            idx = int(ap[pos])
            old_c = colors[idx]
            
            # Try all possible new colors in 0..target_k-1
            best_new_c = old_c
            best_new_viol = current_violations
            
            for c in range(target_k):
                if c == old_c:
                    continue
                colors[idx] = c
                new_viol = count_violations(colors)
                if new_viol < best_new_viol:
                    best_new_viol = new_viol
                    best_new_c = c
            
            # Apply best move
            if best_new_c != old_c:
                colors[idx] = best_new_c
                current_violations = best_new_viol
            else:
                # No improvement, maybe random move to escape local minima
                if random.random() < 0.1:
                    new_c = random.randint(0, target_k - 1)
                    if new_c != old_c:
                        colors[idx] = new_c
                        current_violations = count_violations(colors)
        
        if best_violations < current_violations:
            return best_colors, best_violations
        return colors, current_violations

    # Color Merging Strategy
    def reduce_colors(colors, time_limit):
        """
        Tries to reduce the number of distinct colors in a valid coloring.
        """
        colors = colors.copy()
        k = len(np.unique(colors))
        
        # Map colors to 0..k-1 for consistency
        unique_colors = np.unique(colors)
        mapping = {c: i for i, c in enumerate(unique_colors)}
        colors = np.array([mapping[c] for c in colors], dtype=np.int32)
        k = len(unique_colors)
        
        while k > 3 and time.time() < time_limit:
            # Count frequency of each color
            freq = Counter(colors.tolist())
            # Find least frequent color
            c_min = min(freq, key=freq.get)
            
            merged = False
            # Try merging c_min into every other color
            for c_other in range(k):
                if c_other == c_min:
                    continue
                
                # Create candidate
                candidate = colors.copy()
                candidate[candidate == c_min] = c_other
                
                # Check validity
                viol = count_violations(candidate)
                
                if viol == 0:
                    # Success
                    colors = candidate
                    k -= 1
                    merged = True
                    # Remap to 0..k-1
                    unique_colors = np.unique(colors)
                    mapping = {c: i for i, c in enumerate(unique_colors)}
                    colors = np.array([mapping[c] for c in colors], dtype=np.int32)
                    break
                else:
                    # Try to repair
                    # Only try repair if violations are low enough to be promising
                    if viol < 10:
                        repaired_colors, repaired_viol = local_search_repair(candidate, k, time.time() + 0.5)
                        if repaired_viol == 0:
                            colors = repaired_colors
                            k -= 1
                            merged = True
                            unique_colors = np.unique(colors)
                            mapping = {c: i for i, c in enumerate(unique_colors)}
                            colors = np.array([mapping[c] for c in colors], dtype=np.int32)
                            break
            
            if not merged:
                # Stuck, break
                break
                
        return colors

    # Main Logic
    current_colors = seed_colors
    current_k = len(np.unique(current_colors))
    
    # Try to reduce colors from the seed
    improved_colors = reduce_colors(current_colors, time.time() + 10.0)
    improved_k = len(np.unique(improved_colors))
    
    if improved_k < current_k:
        best_colors = improved_colors
        best_k = improved_k
    else:
        best_colors = current_colors
        best_k = current_k

    # If time remains, try random restarts with reduction
    # This is useful if the seed got stuck in a local minimum for reduction
    restarts = 0
    while time.time() < deadline - 2.0:
        # Random valid initialization? 
        # Generating a random valid coloring is hard.
        # Instead, perturb the best known valid coloring and repair, then reduce.
        
        # Perturb best_colors
        perturbed = best_colors.copy()
        # Change some random positions
        num_changes = 5
        indices = random.sample(range(N), num_changes)
        for idx in indices:
            perturbed[idx] = random.randint(0, best_k - 1)
            
        # Repair
        repaired, viol = local_search_repair(perturbed, best_k, time.time() + 2.0)
        
        if viol == 0:
            # Try to reduce
            reduced = reduce_colors(repaired, time.time() + 2.0)
            reduced_k = len(np.unique(reduced))
            
            if reduced_k < best_k:
                best_colors = reduced
                best_k = reduced_k
            
        restarts += 1
        if restarts > 50: # Safety cap on iterations
            break

    return best_colors.tolist()
# EVOLVE-BLOCK-END
