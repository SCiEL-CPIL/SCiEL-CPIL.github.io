import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Convert between int and word representations
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
    
    def to_word(val):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = val % q
            val //= q
        return word
    
    # Precompute balls
    def get_ball(word_int):
        word = to_word(word_int)
        ball = [word_int]
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val != orig:
                    word[pos] = new_val
                    ball.append(to_int(word))
            word[pos] = orig
        return ball
    
    all_ints = list(range(total))
    
    # Greedy set cover
    covered = [False] * total
    covered_count = 0
    codewords = []
    
    # Precompute all balls
    balls = [get_ball(w) for w in all_ints]
    
    while covered_count < total:
        if time.time() > deadline:
            break
        
        best_word = -1
        best_new = 0
        for w in all_ints:
            new_cov = 0
            for x in balls[w]:
                if not covered[x]:
                    new_cov += 1
            if new_cov > best_new:
                best_new = new_cov
                best_word = w
        
        if best_word < 0 or best_new == 0:
            break
        
        codewords.append(best_word)
        for x in balls[best_word]:
            if not covered[x]:
                covered[x] = True
                covered_count += 1
    
    # Try to remove redundant codewords
    if time.time() < deadline:
        # Build coverage count: for each word, how many codewords cover it
        cov_count = [0] * total
        for cw in codewords:
            for x in balls[cw]:
                cov_count[x] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_cov_count = cov_count[:]
            for i, cw in enumerate(codewords):
                if time.time() > deadline:
                    break
                # Check if removing cw leaves all its covered words still covered
                safe = True
                for x in balls[cw]:
                    if cov_count[x] == 1:
                        safe = False
                        break
                if safe:
                    # Remove cw
                    for x in balls[cw]:
                        cov_count[x] -= 1
                    codewords.pop(i)
                    changed = True
                    break
    
    result = [to_word(w) for w in codewords]
    
    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)
    
    return unique_result
