import time
import random

# EVOLVE-BLOCK-START

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
NUM_PRIMES = len(PRIMES)

def factorize(n):
    vec = [0] * NUM_PRIMES
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

VECS = {i: factorize(i) for i in range(2, 129)}
ZERO_VEC = (0,) * NUM_PRIMES

def add_vecs(a, b):
    return tuple(x + y for x, y in zip(a, b))

def sub_vecs(a, b):
    return tuple(x - y for x, y in zip(a, b))

def is_valid_sequence(seq):
    n = len(seq)
    if n == 0:
        return True
    # Prefix sums
    prefix = [ZERO_VEC] * (n + 1)
    for i in range(n):
        prefix[i+1] = add_vecs(prefix[i], VECS[seq[i]])
    
    prods = set()
    for i in range(n):
        for j in range(i, n):
            prod = sub_vecs(prefix[j+1], prefix[i])
            if prod in prods:
                return False
            prods.add(prod)
    return True

def greedy_construct():
    seq = []
    for x in range(2, 129):
        new_seq = seq + [x]
        if is_valid_sequence(new_seq):
            seq = new_seq
    return seq

def local_search(seq, deadline):
    seq = list(seq)
    used = set(seq)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(2, 129):
            if x in used:
                continue
            for i in range(len(seq) + 1):
                if i > 0 and x <= seq[i-1]:
                    continue
                if i < len(seq) and x >= seq[i]:
                    continue
                new_seq = seq[:i] + [x] + seq[i:]
                if is_valid_sequence(new_seq):
                    seq = new_seq
                    used.add(x)
                    improved = True
                    break
            if improved:
                break
    return seq

def construct():
    deadline = time.time() + 55.0
    
    # 1. Greedy
    seq = greedy_construct()
    
    # 2. Local Search
    if time.time() < deadline:
        seq = local_search(seq, deadline)
        
    # 3. Randomized restarts
    if time.time() < deadline - 10:
        best_seq = seq
        random.seed(42)
        for _ in range(5):
            if time.time() > deadline:
                break
            candidates = list(range(2, 129))
            random.shuffle(candidates)
            seq_try = []
            for x in sorted(candidates):
                new_seq = seq_try + [x]
                if is_valid_sequence(new_seq):
                    seq_try = new_seq
            if len(seq_try) > len(best_seq):
                best_seq = seq_try
                if time.time() < deadline - 5:
                    best_seq = local_search(best_seq, deadline)
    
    return best_seq

# EVOLVE-BLOCK-END
