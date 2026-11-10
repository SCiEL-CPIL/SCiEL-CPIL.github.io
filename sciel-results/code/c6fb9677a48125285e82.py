import itertools

def construct(k, g):
    # Target: 4-regular, girth 10, n=320
    if k != 4 or g != 10:
        # Fallback for other parameters if needed, but task specifies (4,10)
        # Use a generic construction or raise error
        # For now, return a trivial graph or raise
        raise ValueError("Only (4,10) is implemented in this optimized block")

    n = 320
    # Search for generators a, b
    # S = {a, -a, b, -b}
    # Conditions for no cycle < 10
    # We check m=4, 6, 8
    
    def has_short_cycle(a, b):
        # Check m=4
        # 4a=0, 4b=0, 2a+2b=0, 2a-2b=0, a+3b=0, 3a+b=0, a-3b=0, 3a-b=0
        if (4 * a) % n == 0: return True
        if (4 * b) % n == 0: return True
        if (2 * a + 2 * b) % n == 0: return True
        if (2 * a - 2 * b) % n == 0: return True
        if (a + 3 * b) % n == 0: return True
        if (3 * a + b) % n == 0: return True
        if (a - 3 * b) % n == 0: return True
        if (3 * a - b) % n == 0: return True
        
        # Check m=6
        # 6a=0, 6b=0, 4a+2b=0, 4a-2b=0, 2a+4b=0, 2a-4b=0, a+5b=0, 5a+b=0, a-5b=0, 5a-b=0
        if (6 * a) % n == 0: return True
        if (6 * b) % n == 0: return True
        if (4 * a + 2 * b) % n == 0: return True
        if (4 * a - 2 * b) % n == 0: return True
        if (2 * a + 4 * b) % n == 0: return True
        if (2 * a - 4 * b) % n == 0: return True
        if (a + 5 * b) % n == 0: return True
        if (5 * a + b) % n == 0: return True
        if (a - 5 * b) % n == 0: return True
        if (5 * a - b) % n == 0: return True

        # Check m=8
        # 8a=0, 8b=0, 6a+2b=0, 6a-2b=0, 4a+4b=0, 4a-4b=0, 2a+6b=0, 2a-6b=0, 
        # 5a+3b=0, 5a-3b=0, 3a+5b=0, 3a-5b=0, a+7b=0, 7a+b=0, a-7b=0, 7a-b=0
        if (8 * a) % n == 0: return True
        if (8 * b) % n == 0: return True
        if (6 * a + 2 * b) % n == 0: return True
        if (6 * a - 2 * b) % n == 0: return True
        if (4 * a + 4 * b) % n == 0: return True
        if (4 * a - 4 * b) % n == 0: return True
        if (2 * a + 6 * b) % n == 0: return True
        if (2 * a - 6 * b) % n == 0: return True
        if (5 * a + 3 * b) % n == 0: return True
        if (5 * a - 3 * b) % n == 0: return True
        if (3 * a + 5 * b) % n == 0: return True
        if (3 * a - 5 * b) % n == 0: return True
        if (a + 7 * b) % n == 0: return True
        if (7 * a + b) % n == 0: return True
        if (a - 7 * b) % n == 0: return True
        if (7 * a - b) % n == 0: return True
        
        return False

    # Find valid a, b
    # a, b should be distinct and not negatives
    for a in range(1, n // 2):
        for b in range(a + 1, n):
            if b == n - a: continue # a = -b
            # Also ensure a != b (handled by range)
            # Ensure 2a != 0, 2b != 0 (handled by n even and a,b < n/2 or a,b not n/2)
            if a == n // 2 or b == n // 2: continue
            
            if not has_short_cycle(a, b):
                # Found a candidate
                # Construct edges
                edges = []
                for i in range(n):
                    edges.append((i, (i + a) % n))
                    edges.append((i, (i + b) % n))
                return n, edges
                
    # If no pair found, fallback to a known valid pair if we had one, or raise
    # But we expect to find one
    raise RuntimeError("No valid generators found")
