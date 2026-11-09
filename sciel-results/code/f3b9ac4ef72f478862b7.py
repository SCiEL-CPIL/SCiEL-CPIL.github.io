import time
import random

def _score_from_set(S):
    """Count distinct d > 0 such that a, a+d, a+2d in S for some a."""
    S_set = set(S)
    diffs = set()
    # For each pair (a, c) with a < c, check if midpoint is in S
    # d = (c - a) // 2, must be integer > 0
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i + 1, n):
            c = lst[j]
            diff = c - a
            if diff % 2 == 0:
                d = diff // 2
                if d > 0:
                    mid = a + d
                    if mid in S_set:
                        diffs.add(d)
    return len(diffs)

def _score_delta_add(S_set, x):
    """
    Given a set S_set and a new element x to add, compute the number of NEW distinct d's.
    Actually, we need the full score after adding x. Let's just compute it directly for simplicity
    but use the pair method.
    """
    new_set = S_set | {x}
    return _score_from_set(new_set)

def _score_delta_remove(S_set, x):
    """Compute score after removing x from S_set."""
    new_set = S_set - {x}
    return _score_from_set(new_set)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin: evaluator kills at 150s, we use 120s
    
    # Parameters
    N = 32
    MAX_VAL = 512
    
    # Initialize with a random set
    def random_set():
        return set(random.sample(range(1, MAX_VAL + 1), N))
    
    # Greedy initialization: build a set that maximizes 3-term progressions
    # Start with a structured set
    # A good initial guess: multiples of a small number, or a grid
    # Let's try a few initializations and pick the best
    
    initializations = []
    
    # 1. Random
    initializations.append(random_set())
    
    # 2. Arithmetic progression
    # 32 elements in 1..512. AP with difference d.
    # 1, 1+d, ..., 1+31d <= 512 => 31d <= 511 => d <= 16.
    # AP has 31 distinct differences? No, an AP {1, 1+d, ..., 1+(N-1)d} has differences d, 2d, ..., (N-1)d.
    # So it has N-1 = 31 distinct differences.
    # Wait, is that the maximum possible?
    # The score is the number of DISTINCT d.
    # An AP of length 32 has differences d, 2d, ..., 31d. That's 31 distinct values.
    # Can we do better?
    # The set of all differences of a 3-term AP in the set.
    # If we take the set {1, 2, ..., 32}, the differences are 1, 2, ..., 15 (since 1,1+d,1+2d <= 32 => 2d <= 31 => d <= 15.5).
    # So {1..32} has score 15.
    # An AP with d=1 is {1..32}, score 15.
    # An AP with d=2 is {1, 3, ..., 63}. Differences are 2, 4, ..., 62. Score 15.
    # An AP with d=16 is {1, 17, ..., 1+31*16=505}. Differences are 16, 32, ..., 31*16=496. Score 15.
    # Wait, the number of distinct differences in an AP of length N is N-1.
    # Because d, 2d, ..., (N-1)d are all distinct.
    # So ANY arithmetic progression of length 32 has score 31.
    # Is 31 the maximum possible?
    # The maximum possible number of distinct differences is bounded by the number of pairs / 2?
    # No, the score is the number of distinct d such that a, a+d, a+2d in S.
    # For a fixed d, there can be multiple a's.
    # The total number of 3-term APs is sum over d of (number of a such that a, a+d, a+2d in S).
    # But the score is just the COUNT of distinct d's.
    # So we want to maximize the number of distinct d's.
    # An AP of length 32 gives 31 distinct d's.
    # Can we get 32 or more?
    # The maximum d is limited by the range. If min(S)=1, max(S)=512, then 2d <= 511 => d <= 255.
    # So there are 255 possible values for d.
    # Can we have a set of 32 integers that contains a 3-term AP for all d=1..32?
    # Or d=1..40?
    # This is a known problem. The maximum number of distinct differences for a set of size N.
    # For N=32, an AP gives 31.
    # Let's check if we can get 32.
    # If we take an AP of length 32, we have d, 2d, ..., 31d.
    # If we modify one element, we might break some d's but gain others.
    # However, 31 is already quite high.
    # Let's verify the score of an AP.
    # S = {1, 17, 33, ..., 505} (d=16).
    # 3-term APs: (1, 17, 33) -> d=16. (1, 33, 55) -> d=32. ... (1, 489, 505) -> d=256? No, 1+2*256 = 513 > 505.
    # The largest d is when a=1, a+2d = 505 => 2d = 504 => d=252.
    # The differences are 16, 32, 48, ..., 252.
    # Number of terms: (252-16)/16 + 1 = 236/16 + 1 = 14.75 + 1.
    # Wait. 16 * 1 = 16. 16 * 15 = 240. 16 * 16 = 256.
    # 1 + 2*16*15 = 1 + 480 = 481 <= 505.
    # 1 + 2*16*16 = 1 + 512 = 513 > 505.
    # So d can be 16, 32, ..., 16*15 = 240.
    # That's 15 distinct differences.
    # My previous logic was wrong.
    # In an AP {a, a+d, ..., a+(N-1)d}, the 3-term APs are (a+i*d, a+(i+k)*d, a+(i+2k)*d).
    # The difference is k*d.
    # The possible values for k are 1, 2, ..., floor((N-1)/2).
    # For N=32, floor(31/2) = 15.
    # So an AP of length 32 has exactly 15 distinct differences.
    # The seed program {1..32} also has 15 distinct differences (1..15).
    # So the seed score is 15.
    # The observed scores in the prompt were 2.53, 2.13. This is score/seed.
    # If seed score is 15, then 2.53 * 15 = 37.95. So the best found so far has score ~38.
    # This means we CAN do much better than 15.
    # We need a set with ~38 distinct differences.
    
    # Let's use a local search to maximize the number of distinct differences.
    
    # Start with a random set or a heuristic set.
    # Heuristic: Start with a set that has many small differences.
    # A "dense" set like {1..32} has small differences.
    # A "sparse" set has large differences.
    # To maximize the COUNT of distinct differences, we want to cover as many d values as possible.
    # Small d's are easier to achieve if the set is dense.
    # Large d's require the set to span a large range.
    # So a good strategy might be to have a mix.
    
    # Let's try a local search starting from {1..32}.
    
    current_set = set(range(1, 33))
    current_score = _score_from_set(current_set)
    
    best_set = set(current_set)
    best_score = current_score
    
    # Local search: swap one element with another
    def local_search(S, max_iters=10000):
        S = set(S)
        score = _score_from_set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try swapping each element with a random new element
            for x in list(S):
                # Try replacing x with y
                # To be efficient, we can check a few y's
                for _ in range(5):
                    y = random.randint(1, MAX_VAL)
                    if y in S:
                        continue
                    new_S = S - {x} | {y}
                    new_score = _score_from_set(new_S)
                    if new_score > score:
                        S = new_S
                        score = new_score
                        improved = True
                        break
        return S, score
    
    # Run local search with a time limit
    while time.time() < deadline - 10.0:
        new_set, new_score = local_search(current_set, max_iters=5000)
        if new_score > best_score:
            best_score = new_score
            best_set = new_set
            current_set = new_set
        else:
            # Restart with a new random set
            current_set = random_set()
    
    return sorted(best_set)

# To ensure the function is fast, we might need to optimize _score_from_set.
# But with N=32, N^2 = 1024 operations. That's very fast.
# 10000 iterations * 1024 ops = 10^7 ops. That's < 1 second in Python.
# So we can run many iterations.
