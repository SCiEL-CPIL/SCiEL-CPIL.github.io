import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    N = 128
    
    # Precompute all 4-term progressions (a, a+d, a+2d, a+3d) with 1-based indexing for logic
    # We will use 0-based indexing for the array `colors`
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N // 3 + 1):
            end = a + 3 * d
            if end <= N:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map position (1-based) to list of AP indices containing it
    # We will use 0-based indices in the `colors` array, so pos-1
    ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices[pos - 1].append(idx)

    def check_ap(ap_idx, colors):
        """
        Check if the AP at ap_idx can still be satisfied given current colors.
        Returns True if satisfiable, False otherwise.
        """
        a, b, c, d = aps[ap_idx]
        ca = colors[a-1]
        cb = colors[b-1]
        cc = colors[c-1]
        cd = colors[d-1]
        
        assigned_colors = []
        unassigned_count = 0
        for val in (ca, cb, cc, cd):
            if val != -1:
                assigned_colors.append(val)
            else:
                unassigned_count += 1
        
        distinct_assigned = set(assigned_colors)
        num_distinct = len(distinct_assigned)
        
        # We need at least 3 distinct colors in total.
        # If we already have >= 3, it's fine.
        if num_distinct >= 3:
            return True
        
        # If we have < 3, we need the unassigned positions to provide the missing distinct colors.
        missing = 3 - num_distinct
        
        # We need at least 'missing' unassigned positions to be assigned distinct colors not in distinct_assigned.
        if unassigned_count < missing:
            return False
        
        return True

    def solve(k, colors, assigned_count, domains):
        """
        Backtracking solver with MRV heuristic and Forward Checking.
        k: max colors allowed (0..k-1)
        colors: list of length N, -1 for unassigned
        assigned_count: number of positions already assigned
        domains: list of sets of available colors for each position
        """
        if time.time() > deadline:
            return None
        
        if assigned_count == N:
            return colors[:]
        
        # Find the unassigned position with the smallest domain (MRV)
        best_pos = -1
        min_domain_size = float('inf')
        
        for pos in range(N):
            if colors[pos] == -1:
                ds = len(domains[pos])
                if ds < min_domain_size:
                    min_domain_size = ds
                    best_pos = pos
                    if ds == 1: # Unit propagation opportunity
                        break
        
        if best_pos == -1:
            return colors[:]
        
        # Get available values for best_pos
        vals = list(domains[best_pos])
        
        # Symmetry breaking: limit the color values we can assign
        # Count how many distinct colors have been used so far
        used_colors = set()
        for c in colors:
            if c != -1:
                used_colors.add(c)
        max_color_used = max(used_colors) if used_colors else -1
        
        # The new color can be at most max_color_used + 1 (to avoid symmetry)
        # But we must also respect the domain
        # Note: domains might contain colors > max_color_used + 1, but we can prune them
        # Actually, we should just iterate through vals, but prune if v > max_color_used + 1
        # However, if we are forced to use a new color, it must be max_color_used + 1
        
        for v in vals:
            # Prune values that break symmetry
            if v > max_color_used + 1:
                continue
                
            colors[best_pos] = v
            
            # Forward Checking: Check if any other unassigned position has an empty domain
            valid = True
            # Save domains to restore if invalid
            saved_domains = [d.copy() for d in domains]
            
            # Update domains for neighbors (positions in same APs)
            # Actually, we just need to check if the current assignment makes any neighbor unsatisfiable
            # A neighbor is unsatisfiable if its domain becomes empty OR if it's part of an AP that is now unsatisfiable
            # Simpler: Just check check_ap for all APs involving best_pos
            # And for each such AP, check if the remaining unassigned positions in that AP can still satisfy the constraint
            
            for ap_idx in ap_indices[best_pos]:
                if not check_ap(ap_idx, colors):
                    valid = False
                    break
            
            if valid:
                # Recurse
                result = solve(k, colors, assigned_count + 1, domains)
                if result is not None:
                    return result
            
            # Restore
            colors[best_pos] = -1
            # Note: In this simplified version, we didn't explicitly shrink domains in the loop, 
            # so we don't strictly need to restore them if we don't modify them in place 
            # in a way that persists. But for a proper FC implementation, we would.
            # Here, check_ap is the primary constraint propagation.
            
        return None

    # Try k from 3 upwards
    # We expect k=4 to work (since W(3,4) <= 128)
    for k in range(3, 129):
        colors = [-1] * N
        # Initialize domains
        domains = [set(range(k)) for _ in range(N)]
        
        # Fix first position to 0 for symmetry breaking
        colors[0] = 0
        domains[0] = {0}
        
        # Check time before starting search
        if time.time() > deadline:
            break
            
        result = solve(k, colors, 1, domains) # Start with 1 position already assigned
        
        if result is not None:
            # Ensure we return a list of integers
            # The colors are 0..k-1
            # We should remap the colors to be consecutive 0..m-1 where m is the number of distinct colors used
            unique_colors = sorted(set(result))
            color_map = {c: i for i, c in enumerate(unique_colors)}
            remapped = [color_map[c] for c in result]
            return remapped
        
        # If k=3 failed, try k=4, etc.
        
    # Fallback: use the seed's greedy approach if we haven't found a solution
    # This is extremely unlikely to be reached if k=3 or 4 works, but good for safety.
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
# EVOLVE-BLOCK-END
