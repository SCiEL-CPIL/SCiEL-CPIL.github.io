import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_balls(q, n, radius, deadline):
    """Precompute for each word x the list of words within Hamming distance <= radius."""
    total = q ** n
    # Convert all words to base-q integers
    # We'll represent each word as an integer 0..total-1
    # For radius 1, ball(x) = {x} U {x with one digit changed}
    # We need a mapping from integer to digits and back, but for large total we avoid full digit arrays if possible.
    
    # For q=3, n=8, total=6561, radius=1:
    # Ball size = 1 + n*(q-1) = 1 + 8*2 = 17
    # We can store balls as a list of lists of integers.
    
    if radius == 1:
        balls = []
        # Powers of q for digit extraction
        # digits of x: x = d0*q^0 + d1*q^1 + ... + dn-1*q^(n-1)
        # Actually, let's use a consistent order.
        # Let's say word w = [w0, w1, ..., w_{n-1}]
        # Integer value = w0 + w1*q + w2*q^2 + ...
        
        # Precompute powers
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
            
        # Generate all balls
        # To speed up, we can generate digits for all x at once?
        # 6561 * 8 = 52488 digits. Small enough.
        
        # Let's just loop x.
        # Optimization: Precompute the "neighbors" for each position and digit?
        # For a given x, digit i is (x // powers[i]) % q.
        # Neighbor: x + (new_digit - old_digit) * powers[i]
        
        for x in range(total):
            ball = [x]
            for i in range(n):
                old_digit = (x // powers[i]) % q
                for new_digit in range(q):
                    if new_digit != old_digit:
                        y = x + (new_digit - old_digit) * powers[i]
                        ball.append(y)
            balls.append(ball)
            
            if time.time() > deadline - 10:
                break
        return balls
    else:
        # General case (not expected for this specific task instance likely)
        # Fall back to slower method or raise
        raise NotImplementedError("Only radius 1 is optimized")

def _greedy_cover(q, n, radius, deadline):
    total = q ** n
    balls = _precompute_balls(q, n, radius, deadline)
    
    # covered: boolean array of size total
    covered = np.zeros(total, dtype=bool)
    # uncovered_count: integer array, number of uncovered words in ball i?
    # Actually, for greedy set cover, we want to pick the set (ball) that covers the most *uncovered* elements.
    # Let count[i] = number of uncovered elements in ball[i].
    # Initially, count[i] = len(ball[i]).
    
    # However, as we pick a ball, we mark its elements as covered.
    # For each newly covered element y, we need to decrement count[i] for all balls i that contain y.
    # This requires an inverse index: for each y, list of balls i such that y in ball[i].
    
    # Let's build the inverse index.
    # inv: list of lists, inv[y] = list of ball indices i where y is in balls[i]
    inv = [[] for _ in range(total)]
    for i, ball in enumerate(balls):
        for y in ball:
            inv[y].append(i)
            
    # Initialize counts
    counts = np.array([len(b) for b in balls], dtype=np.int32)
    
    selected = []
    
    while True:
        if time.time() > deadline - 5:
            break
            
        # Find the ball with the maximum count of uncovered elements
        # If max count is 0, we are done (or stuck, but greedy should finish)
        max_idx = np.argmax(counts)
        if counts[max_idx] == 0:
            break
            
        selected.append(max_idx)
        
        # Mark elements in balls[max_idx] as covered
        ball = balls[max_idx]
        for y in ball:
            if not covered[y]:
                covered[y] = True
                # Decrement count for all balls containing y
                for i in inv[y]:
                    counts[i] -= 1
                    
        # Check if all covered
        if np.all(covered):
            break

    # Convert selected indices to actual codewords (lists of digits)
    # We need to convert integer index back to digits
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    codewords = []
    for idx in selected:
        digits = []
        x = idx
        for i in range(n):
            d = (x // powers[i]) % q
            digits.append(d)
        codewords.append(digits)
        
    return codewords

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: aim to finish by 140s to avoid timeout
    deadline = start_time + 140
    
    # Try greedy cover
    # For q=3, n=8, radius=1, this should be fast enough.
    # 6561 words, ball size 17.
    # Precomputation: 6561 * 17 ops ~ 100k ops. Fast.
    # Inverse index: 6561 * 17 ~ 100k entries. Fast.
    # Greedy loop:
    # Number of iterations = number of codewords selected.
    # Expected ~ 400-500.
    # Each iteration:
    #   np.argmax(counts): O(6561)
    #   Update: for y in ball (17 items):
    #       if not covered:
    #           for i in inv[y] (size ~ 17? No, inv[y] size is number of balls containing y.
    #           Since balls are radius 1 spheres, a word y is in ball[x] iff d(x,y)<=1.
    #           So |inv[y]| = 1 + n*(q-1) = 17.
    #   So update cost ~ 17 * 17 = 289 ops.
    # Total cost ~ 500 * (6561 + 289) ~ 3.4M ops.
    # This is very fast in Python/NumPy, likely < 1s.
    
    try:
        result = _greedy_cover(q, n, radius, deadline)
    except Exception:
        # Fallback to direct sum if greedy fails or times out unexpectedly
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Ensure distinctness (greedy ensures this by construction of selection, but let's be safe)
    # And validity is checked by evaluator.
    
    return result

# EVOLVE-BLOCK-END
