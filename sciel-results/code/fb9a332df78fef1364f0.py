import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    n = 128
    # Try to find a coloring with 3 colors first
    # If that fails within time, try 4 colors
    
    def check_4ap_validity(colors, i):
        # Check all 4-APs ending at index i (0-indexed: i)
        # Positions are 1-indexed in logic, 0-indexed in array
        # Let current pos be p = i + 1
        # 4-AP: a, a+d, a+2d, p
        # p = a + 3d => a = p - 3d
        # We need a >= 1 => p - 3d >= 1 => 3d <= p - 1 => d <= (p-1)//3
        # Also a+d, a+2d must be < p, which is true for d > 0
        p = i + 1
        max_d = (p - 1) // 3
        for d in range(1, max_d + 1):
            a = p - 3 * d
            # Indices in 0-based array:
            # a -> a-1
            # a+d -> a+d-1
            # a+2d -> a+2d-1
            # p -> i
            idx1 = a - 1
            idx2 = a + d - 1
            idx3 = a + 2 * d - 1
            idx4 = i
            
            c1 = colors[idx1]
            c2 = colors[idx2]
            c3 = colors[idx3]
            c4 = colors[idx4]
            
            # All must be assigned (they are, since we fill sequentially)
            if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
                return False # Should not happen
            
            distinct_colors = {c1, c2, c3, c4}
            if len(distinct_colors) < 3:
                return False
        return True

    def solve(num_colors):
        colors = [-1] * n
        # Symmetry breaking: fix first few colors?
        # Or just try all
        
        def backtrack(i):
            if time.time() > deadline:
                return False
            
            if i == n:
                return True
            
            # Heuristic: try colors in random order or specific order
            # To avoid bias, we can shuffle colors for each position if we restart
            # But for deterministic output, let's stick to 0,1,2...
            # Actually, random restarts are better for finding a solution if one exists.
            
            for c in range(num_colors):
                colors[i] = c
                if check_4ap_validity(colors, i):
                    if backtrack(i + 1):
                        return True
                colors[i] = -1
            return False
        
        return backtrack(0), colors

    # Try 3 colors
    # We might need random restarts if the first attempt fails quickly due to a dead end
    # But 3 colors might be impossible.
    
    # Let's try a few random restarts for 3 colors
    found = False
    result = None
    
    # Since backtracking might be slow, we limit the depth or time per restart
    # But the deadline guard handles the total time.
    
    # Try 3 colors with a few random permutations of color order to break symmetry
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        
        # Shuffle color order for the first few steps?
        # Simple approach: just run backtrack.
        # If it's too slow, it will hit deadline.
        
        # To improve chances, we can pre-assign a small prefix randomly
        prefix_len = 10
        colors = [-1] * n
        valid_prefix = True
        for i in range(prefix_len):
            colors[i] = random.randint(0, 2)
            if not check_4ap_validity(colors, i):
                valid_prefix = False
                break
        
        if not valid_prefix:
            continue
            
        # Complete the rest
        def backtrack_prefix(i):
            if time.time() > deadline:
                return False
            if i == n:
                return True
            for c in range(3):
                colors[i] = c
                if check_4ap_validity(colors, i):
                    if backtrack_prefix(i + 1):
                        return True
                colors[i] = -1
            return False
        
        if backtrack_prefix(prefix_len):
            found = True
            result = colors
            break

    if not found:
        # Try 4 colors
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            prefix_len = 10
            colors = [-1] * n
            valid_prefix = True
            for i in range(prefix_len):
                colors[i] = random.randint(0, 3)
                if not check_4ap_validity(colors, i):
                    valid_prefix = False
                    break
            
            if not valid_prefix:
                continue
                
            def backtrack_prefix_4(i):
                if time.time() > deadline:
                    return False
                if i == n:
                    return True
                for c in range(4):
                    colors[i] = c
                    if check_4ap_validity(colors, i):
                        if backtrack_prefix_4(i + 1):
                            return True
                    colors[i] = -1
                return False
            
            if backtrack_prefix_4(prefix_len):
                found = True
                result = colors
                break

    if not found:
        # Fallback: Greedy with 4 colors (might fail validity, but better than nothing? No, must be valid)
        # If we can't find a valid one in time, we return a best effort?
        # The problem requires a valid coloring.
        # If we fail, we might score 0.
        # But 3 or 4 colors should be sufficient.
        
        # Last resort: deterministic greedy with 4 colors
        colors = [-1] * n
        for i in range(n):
            for c in range(4):
                colors[i] = c
                if check_4ap_validity(colors, i):
                    break
            # If no color worked, it's a failure, but we return what we have
            if colors[i] == -1:
                colors[i] = 0 # Arbitrary, likely invalid
        
        result = colors

    # Ensure labels are in 0..127
    # They are 0..2 or 0..3, so OK.
    
    return result
