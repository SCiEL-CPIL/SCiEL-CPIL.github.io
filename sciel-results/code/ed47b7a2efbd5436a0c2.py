"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    N = 512
    rng = random.Random(42)
    
    def cantor_set(n):
        """Return set of numbers in [1,n] with no digit '2' in base 3."""
        result = set()
        for i in range(1, n+1):
            x = i
            ok = True
            while x > 0:
                if x % 3 == 2:
                    ok = False
                    break
                x //= 3
            if ok:
                result.add(i)
        return result
    
    def try_add(s, x, n):
        """Check if x can be added to s without creating a 4-AP.
        Returns True if x can be added safely."""
        if x in s or x > n or x < 1:
            return False
        # x as first term: x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x+d) in s and (x+2*d) in s and (x+3*d) in s:
                return False
        # x as second term: x-d, x, x+d, x+2d
        max_d = x // 2
        for d in range(1, max_d + 1):
            if (x-d) in s and (x+d) in s and (x+2*d) in s:
                return False
        # x as third term: x-2d, x-d, x, x+d
        max_d = x // 2
        for d in range(1, max_d + 1):
            if (x-2*d) in s and (x-d) in s and (x+d) in s:
                return False
        # x as fourth term: x-3d, x-2d, x-d, x
        max_d = x // 3
        for d in range(1, max_d + 1):
            if (x-3*d) in s and (x-2*d) in s and (x-d) in s:
                return False
        return True
    
    def greedy_fill(s, candidates, n, deadline):
        """Greedily add elements from candidates to s if they don't create 4-APs."""
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if x not in s and try_add(s, x, n):
                s.add(x)
        return s
    
    def local_search(s, n, deadline, max_iters=200):
        """Local search with 1-for-1 swaps and re-fills to improve the set."""
        best_s = set(s)
        best_size = len(best_s)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            
            # Try single additions
            for x in range(1, n+1):
                if time.time() > deadline - 3:
                    break
                if x not in best_s and try_add(best_s, x, n):
                    best_s.add(x)
                    improved = True
            
            # Try 1-for-1 swaps: remove y, add x
            for x in range(1, n+1):
                if time.time() > deadline - 3:
                    break
                if x in best_s:
                    continue
                for y in list(best_s):
                    if time.time() > deadline - 3:
                        break
                    s_temp = set(best_s)
                    s_temp.discard(y)
                    # Check if x can be added to s_temp
                    if try_add(s_temp, x, n):
                        s_temp.add(x)
                        best_s = s_temp
                        improved = True
                        break
            
            # Try 2-for-1 swaps: remove y1, y2, add x
            if time.time() > deadline - 10:
                for x in range(1, n+1):
                    if time.time() > deadline - 5:
                        break
                    if x in best_s:
                        continue
                    # Try all pairs to remove
                    for i, y1 in enumerate(best_s):
                        for y2 in list(best_s):
                            if y2 <= y1:
                                continue
                            s_temp = set(best_s)
                            s_temp.discard(y1)
                            s_temp.discard(y2)
                            if try_add(s_temp, x, n):
                                s_temp.add(x)
                                best_s = s_temp
                                improved = True
                                break
                        if time.time() > deadline - 5:
                            break
                        if improved:
                            break
                    if time.time() > deadline - 5:
                        break
            
            if len(best_s) > best_size:
                best_size = len(best_s)
                improved = True
            
            if not improved:
                break
        
        return best_s
    
    # Start with Cantor set as base
    best_set = cantor_set(N)
    best_size = len(best_set)
    
    # Multiple randomized greedy fills
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        s = set(best_set)
        
        # Randomize the order
        candidates = list(range(1, N+1))
        rng.shuffle(candidates)
        
        s = greedy_fill(s, candidates, N, deadline)
        
        # Local search
        s = local_search(s, N, deadline, max_iters=50)
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Final polish: more aggressive local search on the best set
    if time.time() < deadline - 5:
        best_set = local_search(best_set, N, deadline, max_iters=100)
        best_size = len(best_set)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
