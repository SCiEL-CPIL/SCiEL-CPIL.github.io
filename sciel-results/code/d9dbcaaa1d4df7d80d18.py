import itertools
import math

def score_of(U):
    U = sorted(set(U))
    if not U or U[0] != 0:
        return 0
    D = max(U)
    if D == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a+b)
            diffs.add(a-b)
    size_sums = len(sums)
    size_diffs = len(diffs)
    if size_sums == 0:
        return 0
    ratio = size_diffs / size_sums
    if ratio <= 1:
        return 1
    val = 1 + math.log(ratio) / math.log(2*D + 1)
    return val

# Search n=4 to 8
best_score = 0
best_set = [0,1,3]

# Precompute Golomb rulers or just random search?
# Random search with local improvement
import random

def local_search(U, max_iter=1000):
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    current_score = score_of(U)
    best_score_local = current_score
    best_set_local = U[:]
    
    for _ in range(max_iter):
        # Random move
        if random.random() < 0.5:
            # Add element
            max_val = max(U)
            new_val = random.randint(0, max_val + 10)
            if new_val not in U:
                new_U = U + [new_val]
                new_U = sorted(new_U)
                new_score = score_of(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best_set_local = U[:]
        else:
            # Remove element
            if len(U) > 1:
                idx = random.randint(1, len(U)-1) # Don't remove 0
                new_U = U[:idx] + U[idx+1:]
                new_score = score_of(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best_set_local = U[:]
        # Move element
        if len(U) > 1:
            idx = random.randint(1, len(U)-1)
            old_val = U[idx]
            max_val = max(U)
            new_val = random.randint(0, max_val + 5)
            if new_val not in U:
                new_U = U[:]
                new_U[idx] = new_val
                new_U = sorted(new_U)
                new_score = score_of(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best_set_local = U[:]
    return best_score_local, best_set_local

# Run search
random.seed(42)
for n in range(3, 9):
    for restart in range(20):
        # Initialize with geometric or random
        if restart % 2 == 0:
            U = [0]
            cur = 1
            for _ in range(n-1):
                U.append(cur)
                cur *= 3
        else:
            U = [0] + [random.randint(1, 50) for _ in range(n-1)]
            U = sorted(set(U))
        
        s, s_set = local_search(U, 200)
        if s > best_score:
            best_score = s
            best_set = s_set

print(f"Best Score: {best_score}")
print(f"Best Set: {best_set}")
