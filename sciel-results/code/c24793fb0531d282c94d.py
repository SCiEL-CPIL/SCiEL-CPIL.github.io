import time

def construct():
    # Start time
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Base: Paley graph of order 13
    # Squares mod 13
    squares_13 = {x * x % 13 for x in range(1, 13)}
    n_base = 13
    
    # Blow up by 2
    n = 2 * n_base
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Determine original indices
            i_orig = i // 2
            j_orig = j // 2
            
            if i_orig == j_orig:
                # Edge within the same twin pair
                # Can be 0 or 1. Let's use 0.
                # Note: If we use 0, we must ensure this doesn't create K222.
                # The internal edges of the twin pairs form a perfect matching of color 0.
                # This is safe as long as the "super-edges" don't form K222 with them.
                colors.append(0)
            else:
                # Edge between different twin pairs
                # Color based on base graph
                diff = (j_orig - i_orig) % n_base
                if diff in squares_13:
                    colors.append(0)
                else:
                    colors.append(1)
    
    return {'n': n, 'colors': colors}
