import time
import random
import numpy as np

def construct():
    deadline = time.time() + 82.0
    best_set = []
    best_len = 0
    
    def check_valid(S):
        """Verify all unordered triple sums are distinct."""
        if len(S) <= 2:
            return True
        S = sorted(S)
        n = len(S)
        seen = set()
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = S[i] + S[j] + S[k]
                    if s in seen:
                        return False
                    seen.add(s)
        return True
    
    def try_add(x, S, used_sums):
        """Try to add x to set S. Returns True if successful."""
        new_sums = set()
        s = 3 * x
        if s in used_sums:
            return False
        new_sums.add(s)
        
        for y in S:
            s1 = 2 * x + y
            if s1 in used_sums or s1 in new_sums:
                return False
            new_sums.add(s1)
            
            s2 = x + 2 * y
            if s2 in used_sums or s2 in new_sums:
                return False
            new_sums.add(s2)
        
        n = len(S)
        for i in range(n):
            y = S[i]
            for j in range(i + 1, n):
                z = S[j]
                s3 = x + y + z
                if s3 in used_sums or s3 in new_sums:
                    return False
                new_sums.add(s3)
        
        used_sums.update(new_sums)
        S.append(x)
        return True
    
    def greedy_fill(S, used_sums, order=None):
        """Greedy fill: try to add numbers in given order (or 1..4096)."""
        if order is None:
            order = range(1, 4097)
        for x in order:
            if time.time() > deadline:
                break
            if x in S:
                continue
            try_add(x, S, used_sums)
        return S
    
    # Strategy 1: Powers of 4 skeleton (known valid B_3[1])
    # 1, 4, 16, 64, 256, 1024 are all <= 4096
    skeleton1 = [1, 4, 16, 64, 256, 1024]
    S = list(skeleton1)
    used_sums = set()
    # Build used_sums for the skeleton
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                used_sums.add(S[i] + S[j] + S[k])
    greedy_fill(S, used_sums)
    if len(S) > best_len:
        best_len = len(S)
        best_set = list(S)
    
    # Strategy 2: Scaled powers of 4
    for scale in [1, 2, 3, 5, 7, 11]:
        if time.time() > deadline:
            break
        skel = [scale * (4**i) for i in range(6) if scale * (4**i) <= 4096]
        if len(skel) < 3:
            continue
        S = list(skel)
        used_sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    used_sums.add(S[i] + S[j] + S[k])
        greedy_fill(S, used_sums)
        if len(S) > best_len:
            best_len = len(S)
            best_set = list(S)
    
    # Strategy 3: Mixed geometric skeletons
    skeletons = [
        [1, 3, 12, 48, 192, 768],
        [1, 5, 25, 125, 625, 3125],
        [2, 8, 32, 128, 512, 2048],
        [1, 4, 16, 64, 256, 1024, 4096],
        [1, 2, 8, 32, 128, 512, 2048],
    ]
    for skel in skeletons:
        if time.time() > deadline:
            break
        # Verify skeleton is valid
        if not check_valid(skel):
            continue
        S = list(skel)
        used_sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    used_sums.add(S[i] + S[j] + S[k])
        greedy_fill(S, used_sums)
        if len(S) > best_len:
            best_len = len(S)
            best_set = list(S)
    
    # Strategy 4: Random restarts with geometric bias
    for seed in range(300):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        # Start with a small random geometric-like set
        S = []
        used_sums = set()
        
        # Generate a random geometric sequence
        base = random.choice([3, 4, 5, 6])
        scale = random.randint(1, 3)
        skel = []
        val = scale
        while val <= 4096 and len(skel) < 7:
            if try_add(val, S, used_sums):
                skel.append(val)
            val *= base
            if val > 4096:
                break
        
        # Also try adding some random small numbers
        for _ in range(random.randint(0, 3)):
            x = random.randint(1, 50)
            try_add(x, S, used_sums)
        
        # Greedy fill
        greedy_fill(S, used_sums)
        if len(S) > best_len:
            best_len = len(S)
            best_set = list(S)
    
    # Strategy 5: Numpy-accelerated conflict check for bulk operations
    for seed in range(100):
        if time.time() > deadline:
            break
        random.seed(seed + 50000)
        
        S = []
        used_sums = set()
        
        # Random start
        for _ in range(random.randint(2, 5)):
            x = random.randint(1, 4096)
            try_add(x, S, used_sums)
        
        # Greedy fill
        greedy_fill(S, used_sums)
        if len(S) > best_len:
            best_len = len(S)
            best_set = list(S)
    
    return best_set
