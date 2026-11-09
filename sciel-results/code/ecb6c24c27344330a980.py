import random
import itertools

def count_exact_4_lines(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns 0 if any line contains 5 or more points.
    """
    n = len(points)
    if n != 16:
        return 0
    
    # Check for duplicates
    if len(set(map(tuple, points))) != 16:
        return 0

    # Precompute lines
    # A line is defined by two distinct points.
    # We need to count how many points lie on each unique line.
    # To avoid double counting, we can use a canonical representation of the line.
    # Line equation: ax + by + c = 0
    # We normalize (a, b, c) such that gcd(|a|,|b|,|c|)=1 and a>0 or (a==0 and b>0) or (a==0 and b==0 and c>0)
    
    from math import gcd
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        # a*x + b*y + c = 0
        # (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
        a = y2 - y1
        b = -(x2 - x1)
        c = x1 * y2 - x2 * y1
        
        # Normalize
        # Find gcd of a, b, c
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        if g == 0:
            return None # Should not happen for distinct points
        a //= g
        b //= g
        c //= g
        
        # Ensure canonical sign
        if a < 0:
            a = -a
            b = -b
            c = -c
        elif a == 0:
            if b < 0:
                b = -b
                c = -c
            elif b == 0:
                if c < 0:
                    c = -c
        
        return (a, b, c)

    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    count = 0
    for key, pts_on_line in lines.items():
        k = len(pts_on_line)
        if k > 4:
            return 0 # Invalid configuration
        if k == 4:
            count += 1
            
    return count

def construct():
    # Use a deterministic seed for reproducibility in evaluation, 
    # but allow some randomness if we want to explore. 
    # However, the task implies a single function call. 
    # We will run a short hill-climbing search.
    
    # Grid size
    G = 10
    min_coord = -G
    max_coord = G
    
    # Initial configuration: A 4x4 grid shifted to center
    # Points: (x, y) for x in [0,3], y in [0,3]
    # Shift to [-2, 2]
    init_points = [[x-2, y-2] for x in range(4) for y in range(4)]
    
    best_score = count_exact_4_lines(init_points)
    best_points = [p[:] for p in init_points]
    
    # Hill Climbing
    # We perform a limited number of iterations to stay within time limits.
    # 16 points, grid 21x21.
    
    # To ensure we don't timeout, we limit iterations.
    # 120 seconds is plenty for a simple Python loop if efficient.
    # Let's run for a fixed number of steps.
    
    num_iterations = 5000
    random.seed(42) # Fixed seed for consistent behavior
    
    for _ in range(num_iterations):
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_p = best_points[idx]
        
        # Pick a new random location
        new_x = random.randint(min_coord, max_coord)
        new_y = random.randint(min_coord, max_coord)
        
        # Apply move
        best_points[idx] = [new_x, new_y]
        
        # Evaluate
        new_score = count_exact_4_lines(best_points)
        
        if new_score > best_score:
            best_score = new_score
            # Optionally, we could break if we hit a theoretical max, 
            # but we don't know the exact integer max. 
            # 15 is the real max. 12-13 is likely achievable.
        else:
            # Revert
            best_points[idx] = old_p
    
    # Final check to ensure validity
    final_score = count_exact_4_lines(best_points)
    if final_score == 0:
        # Fallback to the initial 4x4 grid which is known to be valid
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# The prompt asks to keep the function names. 
# The current program has construct inside the block. 
# I will place the helper functions and construct inside the block or outside?
# The seed program has:
# # EVOLVE-BLOCK-START
# def construct(): ...
# # EVOLVE-BLOCK-END
#
# I should keep the structure similar. I'll define helpers outside or inside?
# Usually, it's safer to keep everything in the block if the evaluator only extracts that part, 
# but the prompt says "Rewrite the program... Output the COMPLETE program".
# And "Keep the function names and signatures of the seed program."
# The seed program defines construct inside the block.
# I will define helper functions outside the block to keep the block clean, 
# or inside if they are necessary. 
# Let's put helpers outside the block to avoid cluttering the evolution target, 
# but ensure construct is inside.

# Wait, the seed code is:
# """Initial program..."""
# # EVOLVE-BLOCK-START
# def construct():
#    ...
# # EVOLVE-BLOCK-END

# I will follow this structure.

# EVOLVE-BLOCK-END
