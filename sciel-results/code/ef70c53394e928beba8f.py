import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    def is_valid_fast(seq):
        """Fast validity check using prefix sums"""
        if not seq:
            return True
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        
        sums = set()
        for i in range(L):
            pi = P[i]
            for j in range(i + 1, L + 1):
                s = P[j] - pi
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_build(order):
        """Greedy construction following a given order of candidates"""
        values = []
        sums = set()
        for x in order:
            suffix_sums = []
            current_sum = 0
            for v in reversed(values):
                current_sum += v
                suffix_sums.append(current_sum)
            
            potential_sums = [x] + [x + s for s in suffix_sums]
            
            valid = True
            for ps in potential_sums:
                if ps in sums:
                    valid = False
                    break
            
            if valid:
                values.append(x)
                for ps in potential_sums:
                    sums.add(ps)
        
        return values

    # Try multiple greedy orderings
    best_seq = []
    best_len = 0
    
    # Ordering 1: natural ascending
    seq1 = greedy_build(list(range(1, 257)))
    if len(seq1) > best_len:
        best_len = len(seq1)
        best_seq = seq1
    
    # Ordering 2: descending
    seq2 = greedy_build(list(range(256, 0, -1)))
    if len(seq2) > best_len:
        best_len = len(seq2)
        best_seq = seq2
    
    # Ordering 3: interleaved (small, large, small, large)
    interleaved = []
    for i in range(128):
        interleaved.append(i + 1)
        interleaved.append(256 - i)
    seq3 = greedy_build(interleaved)
    if len(seq3) > best_len:
        best_len = len(seq3)
        best_seq = seq3
    
    # Ordering 4: powers of 2 first, then others
    powers = [2**i for i in range(1, 9) if 2**i <= 256]
    others = [x for x in range(1, 257) if x not in powers]
    ordering4 = powers + others
    seq4 = greedy_build(ordering4)
    if len(seq4) > best_len:
        best_len = len(seq4)
        best_seq = seq4
    
    # Ordering 5: Fibonacci-like
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= 256:
        fibs.append(fibs[-1] + fibs[-2])
    others5 = [x for x in range(1, 257) if x not in fibs]
    ordering5 = fibs + others5
    seq5 = greedy_build(ordering5)
    if len(seq5) > best_len:
        best_len = len(seq5)
        best_seq = seq5
    
    # Random orderings
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        random_order = list(range(1, 257))
        random.shuffle(random_order)
        seq_r = greedy_build(random_order)
        if len(seq_r) > best_len:
            best_len = len(seq_r)
            best_seq = seq_r
    
    # Local search: try to add elements at different positions
    current_seq = list(best_seq)
    current_set = set(current_seq)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in range(1, 257):
            if x not in current_set:
                new_seq = sorted(current_seq + [x])
                if is_valid_fast(new_seq):
                    current_seq = new_seq
                    current_set.add(x)
                    improved = True
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = list(current_seq)
    
    # Try removing one element and adding a different one
    for i in range(len(current_seq)):
        if time.time() > deadline - 2:
            break
        for x in range(1, 257):
            if x not in current_set:
                new_seq = [v for j, v in enumerate(current_seq) if j != i]
                new_seq.append(x)
                new_seq.sort()
                if is_valid_fast(new_seq) and len(new_seq) == len(current_seq):
                    # Same length but different composition - might allow further additions
                    # Only keep if it enables adding another element
                    new_set = set(new_seq)
                    can_add_more = False
                    for y in range(1, 257):
                        if y not in new_set:
                            test_seq = sorted(new_seq + [y])
                            if is_valid_fast(test_seq):
                                can_add_more = True
                                break
                    if can_add_more:
                        current_seq = new_seq
                        current_set = set(current_seq)
                        improved = True
                        break
    
    # Final local search again
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for x in range(1, 257):
            if x not in current_set:
                new_seq = sorted(current_seq + [x])
                if is_valid_fast(new_seq):
                    current_seq = new_seq
                    current_set.add(x)
                    improved = True
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = list(current_seq)
    
    return sorted(best_seq)
