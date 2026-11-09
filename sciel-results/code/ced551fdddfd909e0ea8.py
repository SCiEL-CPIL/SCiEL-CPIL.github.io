import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 115  # 35s margin for safety
    
    # State
    S = []
    in_S = [False] * (max_val + 1)
    counts = [0] * (2 * max_val + 1)
    
    def try_add(x):
        if counts[2 * x] >= 2:
            return False
        for y in S:
            if y != x and counts[x + y] >= 2:
                return False
        # Add
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    def try_remove(x):
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    def is_valid_swap(a, c):
        # Check if S - {a} + {c} is valid
        # Affected sums: 2*c, and c+y for all y in S\{a}
        
        # Check 2*c: new count = old count of 2*c, minus 1 if a+a was counted, plus 1 for c+c
        # But we're removing a, so sums involving a go away
        new_2c = counts[2 * c]
        # If a == c, then 2*a = 2*c, so removing a reduces 2*c by 1
        if a == c:
            new_2c -= 1
        # Adding c adds 1 to 2*c
        new_2c += 1
        if new_2c > 2:
            return False
        
        # For each y in S\{a}, check c+y
        for y in S:
            if y == a:
                continue
            s = c + y
            if s > 2 * max_val:
                continue
            # New count for sum s: old count + 1 (for c+y) - 1 if a+y == s (i.e., y == c)
            new_count = counts[s] + 1
            if y == c:
                # a+c was counted in old, but we're removing a, so subtract 1
                new_count -= 1
            if new_count > 2:
                return False
        return True

    def local_search():
        while time.time() < deadline:
            # 1. Try to add any unused element
            added = False
            unused = [x for x in range(1, max_val + 1) if not in_S[x]]
            
            if len(unused) > 50:
                sample = random.sample(unused, 50)
                random.shuffle(sample)
            else:
                sample = unused
                random.shuffle(sample)
            
            for c in sample:
                if try_add(c):
                    added = True
                    break
            
            if added:
                continue
            
            # 2. Try swaps
            swapped = False
            if not S or not unused:
                break
            
            swap_candidates = unused if len(unused) < 200 else random.sample(unused, 200)
            
            for _ in range(200):
                a = random.choice(S)
                c = random.choice(swap_candidates)
                if is_valid_swap(a, c):
                    try_remove(a)
                    try_add(c)
                    swapped = True
                    break
            
            if swapped:
                continue
            
            # 3. Perturbation: remove elements that are "constraining" others
            if len(S) > 10:
                constraint_scores = []
                for a in S:
                    score = 0
                    for y in S:
                        if y != a:
                            if counts[a + y] == 2:
                                score += 1
                    if counts[2*a] == 2:
                        score += 1
                    constraint_scores.append((score, a))
                
                constraint_scores.sort(reverse=True)
                k = min(random.randint(2, 4), len(S) - 5)
                for _, a in constraint_scores[:k]:
                    try_remove(a)
                
    # Main Loop: Multiple Restarts
    best_size = 0
    best_S = []
    
    gap_strategies = [5, 7, 11, 13, 17, 23, 29, 37]
    
    while time.time() < deadline:
        # Reset State
        S.clear()
        for i in range(max_val + 1):
            in_S[i] = False
        for i in range(2 * max_val + 1):
            counts[i] = 0
            
        # 1. Structured Greedy Initialization
        gap = random.choice(gap_strategies)
        start = random.randint(1, gap)
        
        for x in range(start, max_val + 1, gap):
            if try_add(x):
                pass
        
        # 2. Local Search
        local_search()
        
        # 3. Final polish: try to add any remaining elements
        for _ in range(5):
            unused = [x for x in range(1, max_val + 1) if not in_S[x]]
            random.shuffle(unused)
            added_any = False
            for c in unused:
                if try_add(c):
                    added_any = True
            if not added_any:
                break
        
        # 4. Update Best
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
    
    # Verify the best set is actually valid
    if best_S:
        verify_counts = [0] * (2 * max_val + 1)
        for i, a in enumerate(best_S):
            for j, b in enumerate(best_S):
                if i <= j:
                    verify_counts[a + b] += 1
        for c in verify_counts:
            if c > 2:
                return []
    
    return best_S
