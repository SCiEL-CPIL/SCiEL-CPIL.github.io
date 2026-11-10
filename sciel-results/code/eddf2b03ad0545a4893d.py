import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s to be safe
    
    best_a = []
    best_b = []
    best_score = 0
    
    n = 256
    candidates = list(range(1, n + 1))
    
    # Helper to check and add
    def try_add(x, A, B, sums_A, sums_B, diffs_A, diffs_B, target):
        if target == 0: # A
            if len(A) >= 64: return False
            # Check sums
            for y in A:
                if x + y in sums_A:
                    return False
            # Check diffs against B
            for y in A:
                d = abs(x - y)
                if d > 0 and d in diffs_B:
                    return False
            # Add
            A.append(x)
            for y in A:
                sums_A.add(x + y)
            for y in A:
                d = abs(x - y)
                if d > 0:
                    diffs_A.add(d)
            return True
        else: # B
            if len(B) >= 64: return False
            for y in B:
                if x + y in sums_B:
                    return False
            for y in B:
                d = abs(x - y)
                if d > 0 and d in diffs_A:
                    return False
            B.append(x)
            for y in B:
                sums_B.add(x + y)
            for y in B:
                d = abs(x - y)
                if d > 0:
                    diffs_B.add(d)
            return True

    while time.time() < deadline:
        random.shuffle(candidates)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in candidates:
            # Randomize order of trying A and B to avoid bias
            if random.random() < 0.5:
                if not try_add(x, A, B, sums_A, sums_B, diffs_A, diffs_B, 0):
                    try_add(x, A, B, sums_A, sums_B, diffs_A, diffs_B, 1)
            else:
                if not try_add(x, A, B, sums_A, sums_B, diffs_A, diffs_B, 1):
                    try_add(x, A, B, sums_A, sums_B, diffs_A, diffs_B, 0)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_a = A[:]
            best_b = B[:]
            
        # Early stop if we hit theoretical max?
        # Max possible is limited by diff count.
        # 255 diffs total.
        # k(k-1)/2 + m(m-1)/2 <= 255.
        # If k=16, m=16 -> 120+120=240 <= 255.
        # If k=17, m=15 -> 136+105=241 <= 255.
        # If k=18, m=14 -> 153+91=244 <= 255.
        # If k=19, m=13 -> 171+78=249 <= 255.
        # If k=20, m=12 -> 190+66=256 > 255.
        # So max sum is likely around 32-34.
        # If best_score >= 34, we can break.
        if best_score >= 34:
            break

    return [best_a, best_b]
