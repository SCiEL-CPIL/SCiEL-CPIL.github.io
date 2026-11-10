"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    # Then greedily remove redundant codewords.
    
    deadline = time.time() + 140  # Leave 10s safety margin
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Helper: generate ball of radius 1 around a codeword
    def get_ball(cw):
        ball = {tuple(cw)}
        for i in range(n):
            for v in range(q):
                if v != cw[i]:
                    new_cw = list(cw)
                    new_cw[i] = v
                    ball.add(tuple(new_cw))
        return ball
    
    # Precompute balls for all codewords
    balls = [get_ball(cw) for cw in codewords]
    
    # Coverage count: for each word, how many codewords cover it
    coverage = {}
    for i, ball in enumerate(balls):
        for w in ball:
            coverage[w] = coverage.get(w, 0) + 1
    
    # Greedy removal: try removing each codeword if all words in its ball are still covered
    removed = [False] * len(codewords)
    improved = True
    
    while improved:
        improved = False
        for i in range(len(codewords)):
            if removed[i]:
                continue
            if time.time() > deadline:
                break
            
            # Check if all words in ball[i] have coverage > 1 (i.e., covered by other codewords)
            can_remove = True
            for w in balls[i]:
                if coverage[w] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove codeword i
                removed[i] = True
                improved = True
                # Update coverage counts
                for w in balls[i]:
                    coverage[w] -= 1
        
        if time.time() > deadline:
            break
    
    # Build result
    result = [codewords[i] for i in range(len(codewords)) if not removed[i]]
    return result
# EVOLVE-BLOCK-END
