import random
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 1000
    max_sum = 2 * N
    
    def validate_set(S):
        """Return True if every sum has at most 2 representations."""
        counts = [0] * (max_sum + 1)
        for i in range(len(S)):
            a = S[i]
            counts[2 * a] += 1
            for j in range(i):
                s = a + S[j]
                counts[s] += 1
        return all(c <= 2 for c in counts)
    
    # 1. Deterministic greedy (ascending)
    def greedy_ascending():
        S = []
        counts = [0] * (max_sum + 1)
        for x in range(1, N + 1):
            ok = True
            for y in S:
                if counts[x + y] == 2:
                    ok = False
                    break
            if ok and counts[2 * x] == 2:
                ok = False
            if ok:
                for y in S:
                    counts[x + y] += 1
                counts[2 * x] += 1
                S.append(x)
        return S
    
    S_det = greedy_ascending()
    best_set = S_det[:]
    best_size = len(S_det)
    
    # 2. Randomized greedy with multiple trials
    def randomized_greedy(seed=None):
        rng = random.Random(seed)
        S = []
        counts = [0] * (max_sum + 1)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        for x in candidates:
            ok = True
            for y in S:
                if counts[x + y] == 2:
                    ok = False
                    break
            if ok and counts[2 * x] == 2:
                ok = False
            if ok:
                for y in S:
                    counts[x + y] += 1
                counts[2 * x] += 1
                S.append(x)
        return S
    
    # 3. Local search: try to swap elements to increase size or maintain size
    def try_swap_add(S, counts, in_set, x_out, x_in):
        """Try to remove x_out and add x_in. Return new S and counts if successful."""
        # Check if x_in can be added given S without x_out
        temp_counts = counts[:]  # We'll modify this
        # First, remove x_out's contributions
        for y in S:
            if y != x_out:
                temp_counts[x_out + y] -= 1
        temp_counts[2 * x_out] -= 1
        
        # Now check if x_in fits
        ok = True
        for y in S:
            if y != x_out:
                if temp_counts[x_in + y] == 2:
                    ok = False
                    break
        if ok and temp_counts[2 * x_in] == 2:
            ok = False
        
        if not ok:
            return None, None
        
        # Apply: add x_in
        for y in S:
            if y != x_out:
                temp_counts[x_in + y] += 1
        temp_counts[2 * x_in] += 1
        
        new_S = [y for y in S if y != x_out]
        new_S.append(x_in)
        return new_S, temp_counts
    
    # 4. Hill-climbing local search
    def local_search(S, counts, in_set, time_limit):
        S = S[:]
        counts = counts[:]
        size = len(S)
        
        def recompute_counts(new_S):
            c = [0] * (max_sum + 1)
            for i in range(len(new_S)):
                a = new_S[i]
                c[2 * a] += 1
                for j in range(i):
                    c[a + new_S[j]] += 1
            return c
        
        # Build in_set
        in_set = [False] * (N + 1)
        for x in S:
            in_set[x] = True
        
        # Try random swaps
        attempts = 0
        while time.time() < time_limit and attempts < 50000:
            attempts += 1
            # Pick random element to remove
            idx = random.randint(0, len(S) - 1)
            x_out = S[idx]
            
            # Pick random element to add (not in S)
            x_in = random.randint(1, N)
            retries = 0
            while in_set[x_in] and retries < 50:
                x_in = random.randint(1, N)
                retries += 1
            if in_set[x_in]:
                continue
            
            # Try the swap
            new_S, new_counts = try_swap_add(S, counts, in_set, x_out, x_in)
            if new_S is not None:
                S = new_S
                counts = new_counts
                in_set[x_out] = False
                in_set[x_in] = True
                # Size stays the same, but we might have room for more
        
        # After swaps, try to add more elements
        for x in range(1, N + 1):
            if in_set[x]:
                continue
            ok = True
            for y in S:
                if counts[x + y] == 2:
                    ok = False
                    break
            if ok and counts[2 * x] == 2:
                ok = False
            if ok:
                for y in S:
                    counts[x + y] += 1
                counts[2 * x] += 1
                S.append(x)
                in_set[x] = True
        
        return S, counts
    
    # Run randomized greedy several times and apply local search to best ones
    t0 = time.time()
    
    # First, get a few diverse starting points
    candidates = [S_det]
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        S_r = randomized_greedy(seed=trial * 12345)
        if len(S_r) >= best_size:
            candidates.append(S_r)
    
    # Apply local search to promising candidates
    for S_start in candidates:
        if time.time() > deadline - 10:
            break
        if len(S_start) < best_size - 2:
            continue
        
        # Compute counts for S_start
        counts = [0] * (max_sum + 1)
        for i in range(len(S_start)):
            a = S_start[i]
            counts[2 * a] += 1
            for j in range(i):
                counts[a + S_start[j]] += 1
        
        in_set = [False] * (N + 1)
        for x in S_start:
            in_set[x] = True
        
        S_ls, counts_ls = local_search(S_start, counts, in_set, deadline - 15)
        
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_set = S_ls[:]
    
    # Final validation
    if not validate_set(best_set):
        # Fallback to deterministic
        best_set = S_det[:]
        best_size = len(S_det)
    
    return best_set
