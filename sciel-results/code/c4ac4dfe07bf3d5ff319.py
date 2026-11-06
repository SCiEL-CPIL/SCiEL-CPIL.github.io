import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin

    best_seq = []
    best_len = 0

    def is_valid(x, P, D):
        """
        Check if adding x to the sequence is valid.
        P: list of prefix sums, P[0]=0, P[-1] is current sum.
        D: set of all differences P[j] - P[i] for i < j.
        """
        new_p = P[-1] + x
        for p in P:
            diff = new_p - p
            if diff in D:
                return False
        return True

    def get_new_diffs(x, P):
        """
        Return the set of new differences if x is added.
        """
        new_p = P[-1] + x
        return {new_p - p for p in P}

    def greedy_smallest():
        P = [0]
        D = set()
        X = []
        last = 0
        while True:
            found = False
            for x in range(last + 1, 257):
                if is_valid(x, P, D):
                    X.append(x)
                    D.update(get_new_diffs(x, P))
                    P.append(P[-1] + x)
                    last = x
                    found = True
                    break
            if not found:
                break
        return X

    def greedy_largest():
        P = [0]
        D = set()
        X = []
        last = 0
        while True:
            found = False
            for x in range(256, last, -1):
                if is_valid(x, P, D):
                    X.append(x)
                    D.update(get_new_diffs(x, P))
                    P.append(P[-1] + x)
                    last = x
                    found = True
                    break
            if not found:
                break
        return X

    def greedy_randomized(rng):
        P = [0]
        D = set()
        X = []
        last = 0
        while True:
            candidates = []
            for x in range(last + 1, 257):
                if is_valid(x, P, D):
                    candidates.append(x)
            
            if not candidates:
                break
            
            x = rng.choice(candidates)
            
            X.append(x)
            D.update(get_new_diffs(x, P))
            P.append(P[-1] + x)
            last = x
            
        return X

    # Run deterministic strategies
    seq = greedy_smallest()
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]

    seq = greedy_largest()
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]

    if time.time() > deadline - 5:
        return best_seq

    # Run randomized strategies
    rng = random.Random(42)
    
    for i in range(10000):
        if time.time() > deadline - 2:
            break
        
        seq = greedy_randomized(rng)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
    return best_seq
