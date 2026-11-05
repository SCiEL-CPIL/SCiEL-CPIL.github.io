import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback for non-standard parameters or if greedy is too slow
    def seed_construct():
        if n < 4:
            # For small n, brute force or simple
            # If n=1, q words.
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if q != 3:
            # Generic fallback: all words (inefficient but valid)
            return [list(t) for t in itertools.product(range(q), repeat=n)]
            
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Check if we should use the specialized greedy for K_3(8,1)
    if q == 3 and n == 8 and radius == 1:
        try:
            return greedy_k3_8_1(deadline)
        except Exception:
            pass

    return seed_construct()

def greedy_k3_8_1(deadline):
    q = 3
    n = 8
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    def int_to_word(x):
        word = []
        for _ in range(n):
            word.append(x % q)
            x //= q
        return word

    def get_neighbors(x):
        nbs = []
        for i in range(n):
            current_digit = (x // powers[i]) % q
            for new_digit in range(q):
                if new_digit == current_digit:
                    continue
                delta = (new_digit - current_digit) * powers[i]
                nbs.append(x + delta)
        return nbs

    # Precompute neighbors
    neighbors = [get_neighbors(x) for x in range(total_words)]
    
    covered = [False] * total_words
    uncovered_count = [len(neighbors[x]) for x in range(total_words)]
    
    code = []
    
    while time.time() < deadline:
        # Find max
        max_count = -1
        best_word = -1
        # Optimization: skip if covered? No, covered words can still cover others.
        # But if a word is covered, its own "uncoveredness" is gone, but it might have uncovered neighbors.
        # The metric `uncovered_count[x]` is the number of uncovered words in B(x,1).
        # If x is covered, B(x,1) might still contain uncovered words.
        # So we must consider all x.
        
        for x in range(total_words):
            if uncovered_count[x] > max_count:
                max_count = uncovered_count[x]
                best_word = x
                
        if best_word == -1 or max_count == 0:
            break
            
        code.append(best_word)
        
        S = [best_word] + neighbors[best_word]
        for v in S:
            if not covered[v]:
                covered[v] = True
                for u in [v] + neighbors[v]:
                    uncovered_count[u] -= 1
                    
    return [int_to_word(x) for x in code]
