"""Two Sidon sets with disjoint difference sets – Bitset-based fast search.
Uses integer bitmasks for sum and difference collisions to enable high-throughput
local search within the time budget."""

import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 55.0  # Safety margin before 60s limit

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # State: lists of elements, and bitmasks for sums and diffs
    # Max sum is 256+256=512. Max diff is 255.
    # We use Python ints as bitsets.
    
    A = []
    B = []
    sums_A = 0
    sums_B = 0
    diffs_A = 0
    diffs_B = 0
    
    def can_add_A(x):
        # Check if x can be added to A
        # 1. Check x+x against sums_A
        if (sums_A >> (x + x)) & 1:
            return False
        # 2. Check x+a against sums_A and diffs against diffs_A and diffs_B
        for a in A:
            s = x + a
            d = abs(x - a)
            if (sums_A >> s) & 1:
                return False
            if (diffs_A >> d) & 1:
                return False
            if (diffs_B >> d) & 1:
                return False
        return True

    def add_A(x):
        nonlocal sums_A, diffs_A
        sums_A |= (1 << (x + x))
        for a in A:
            s = x + a
            d = abs(x - a)
            sums_A |= (1 << s)
            diffs_A |= (1 << d)
        A.append(x)

    def can_add_B(x):
        # Check if x can be added to B
        if (sums_B >> (x + x)) & 1:
            return False
        for b in B:
            s = x + b
            d = abs(x - b)
            if (sums_B >> s) & 1:
                return False
            if (diffs_B >> d) & 1:
                return False
            if (diffs_A >> d) & 1:
                return False
        return True

    def add_B(x):
        nonlocal sums_B, diffs_B
        sums_B |= (1 << (x + x))
        for b in B:
            s = x + b
            d = abs(x - b)
            sums_B |= (1 << s)
            diffs_B |= (1 << d)
        B.append(x)

    def remove_A(x):
        nonlocal sums_A, diffs_A
        sums_A &= ~(1 << (x + x))
        for a in A:
            if a == x: continue
            s = x + a
            d = abs(x - a)
            sums_A &= ~(1 << s)
            diffs_A &= ~(1 << d)
        A.remove(x)

    def remove_B(x):
        nonlocal sums_B, diffs_B
        sums_B &= ~(1 << (x + x))
        for b in B:
            if b == x: continue
            s = x + b
            d = abs(x - b)
            sums_B &= ~(1 << s)
            diffs_B &= ~(1 << d)
        B.remove(x)

    # Phase 1: Greedy
    for x in range(1, N + 1):
        if time.time() > deadline - 5.0:
            break
        if len(A) < MAX_SIZE and can_add_A(x):
            add_A(x)
        elif len(B) < MAX_SIZE and can_add_B(x):
            add_B(x)

    # Phase 2: Local Search with 1-for-2 swaps
    rng = random.Random(42)
    best_score = len(A) + len(B)
    best_A = list(A)
    best_B = list(B)
    
    def try_add_to_best(x, target='A'):
        """Try to add x to target set, update best if improved."""
        nonlocal best_score, best_A, best_B
        if target == 'A' and len(A) < MAX_SIZE and can_add_A(x):
            add_A(x)
            if len(A) + len(B) > best_score:
                best_score = len(A) + len(B)
                best_A = list(A)
                best_B = list(B)
            return True
        elif target == 'B' and len(B) < MAX_SIZE and can_add_B(x):
            add_B(x)
            if len(A) + len(B) > best_score:
                best_score = len(A) + len(B)
                best_A = list(A)
                best_B = list(B)
            return True
        return False
    
    # Precompute candidate pool: elements not in A or B
    def get_candidates():
        in_sets = set(A) | set(B)
        return [x for x in range(1, N + 1) if x not in in_sets]
    
    while time.time() < deadline - 2.0:
        candidates = get_candidates()
        
        # 1. Try to add a random element
        if candidates and (len(A) < MAX_SIZE or len(B) < MAX_SIZE):
            for _ in range(5):
                x = rng.choice(candidates)
                if len(A) < MAX_SIZE:
                    try_add_to_best(x, 'A')
                else:
                    try_add_to_best(x, 'B')
        
        # 2. 1-for-2 swap in A: remove 2 from A, try to add 3 candidates
        if len(A) >= 2:
            idx1 = rng.randint(0, len(A) - 1)
            idx2 = rng.randint(0, len(A) - 2)
            if idx2 >= idx1:
                idx2 += 1
            a1 = A[idx1]
            a2 = A[idx2]
            remove_A(a1)
            remove_A(a2)
            added = 0
            for x in candidates:
                if added >= 3:
                    break
                if len(A) < MAX_SIZE and can_add_A(x):
                    add_A(x)
                    added += 1
            if added > 2:
                if len(A) + len(B) > best_score:
                    best_score = len(A) + len(B)
                    best_A = list(A)
                    best_B = list(B)
            else:
                # Restore
                add_A(a1)
                add_A(a2)
        
        # 3. 1-for-2 swap in B: remove 2 from B, try to add 3 candidates
        if len(B) >= 2:
            idx1 = rng.randint(0, len(B) - 1)
            idx2 = rng.randint(0, len(B) - 2)
            if idx2 >= idx1:
                idx2 += 1
            b1 = B[idx1]
            b2 = B[idx2]
            remove_B(b1)
            remove_B(b2)
            added = 0
            for x in candidates:
                if added >= 3:
                    break
                if len(B) < MAX_SIZE and can_add_B(x):
                    add_B(x)
                    added += 1
            if added > 2:
                if len(A) + len(B) > best_score:
                    best_score = len(A) + len(B)
                    best_A = list(A)
                    best_B = list(B)
            else:
                # Restore
                add_B(b1)
                add_B(b2)
        
        # 4. Cross swap: remove 1 from A, 1 from B, try to add 2+1
        if A and B:
            a_old = A[rng.randint(0, len(A) - 1)]
            b_old = B[rng.randint(0, len(B) - 1)]
            remove_A(a_old)
            remove_B(b_old)
            added_a = 0
            added_b = 0
            for x in candidates:
                if added_a >= 2 and added_b >= 1:
                    break
                if added_a < 2 and len(A) < MAX_SIZE and can_add_A(x):
                    add_A(x)
                    added_a += 1
                if added_b < 1 and len(B) < MAX_SIZE and can_add_B(x):
                    add_B(x)
                    added_b += 1
            if added_a + added_b > 2:
                if len(A) + len(B) > best_score:
                    best_score = len(A) + len(B)
                    best_A = list(A)
                    best_B = list(B)
            else:
                # Restore
                add_A(a_old)
                add_B(b_old)

    # Restore best
    A = list(best_A)
    B = list(best_B)
    
    if not A: A = [1]
    if not B: B = [2]
    
    return [A, B]
