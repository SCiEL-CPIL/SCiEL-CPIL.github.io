import time

def count_mono_ap(colors):
    """Count monochromatic 4-term arithmetic progressions."""
    n = len(colors)
    count = 0
    # d max is (n-1)//3
    for d in range(1, (n - 1) // 3 + 1):
        # a max is n - 3*d
        for a in range(n - 3 * d):
            if colors[a] == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    return [bin(i).count('1') % 2 for i in range(n)]

def construct():
    n = 256
    # Start with Thue-Morse
    colors = thue_morse(n)
    current_score = count_mono_ap(colors)
    
    # Hill Climbing
    # We try to improve by flipping bits
    # To avoid getting stuck in a very bad local minimum, we can do a few restarts
    # but Thue-Morse is a very good start.
    
    max_time = 100  # seconds, leaving buffer for overhead
    start_time = time.time()
    
    # Local search: flip bit if it reduces score
    # We iterate through bits multiple times
    sweeps = 0
    while time.time() - start_time < max_time:
        improved = False
        for i in range(n):
            # Flip
            colors[i] ^= 1
            new_score = count_mono_ap(colors)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Unflip
                colors[i] ^= 1
        
        if not improved:
            # Try a random restart with some noise if we have time
            if time.time() - start_time < max_time - 10:
                # Add random noise
                import random
                for _ in range(10):
                    idx = random.randint(0, n - 1)
                    colors[idx] ^= 1
                current_score = count_mono_ap(colors)
                # Continue loop to try to improve again
            else:
                break
        sweeps += 1

    return colors
