# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 80  # seconds, safety margin before 90s kill
    
    # Generate 5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    n_smooth = len(smooth)
    
    # Precompute divisibility: div[i][j] = True if smooth[i] divides smooth[j]
    div = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                div[i][j] = True
    
    def is_antichain(indices):
        """Check if the set of indices forms a divisibility antichain."""
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                if div[a][b] or div[b][a]:
                    return False
        return True
    
    def find_for_target(n, max_k=30):
        """Find antichain representation of n with minimum terms."""
        # Try k = 1, 2, 3, ... up to max_k
        for k in range(1, max_k + 1):
            # Filter smooth numbers that could be part of a sum to n
            # Each term must be <= n
            candidates = [i for i in range(n_smooth) if smooth[i] <= n]
            
            if len(candidates) < k:
                break
            
            # Backtracking search
            best = None
            
            def backtrack(start, remaining, chosen):
                nonlocal best
                if time.time() - start_time > DEADLINE:
                    return
                if remaining == 0:
                    if is_antichain(chosen):
                        best = list(chosen)
                    return
                if len(chosen) + remaining > max_k:
                    return
                
                # Pruning: if we can't fill remaining with smallest available
                # The minimum possible sum of 'remaining' smallest unused candidates
                # is a lower bound; if sum(chosen) + min_possible > n, prune
                # Also max possible: sum of 'remaining' largest candidates
                
                for i in range(start, len(candidates)):
                    val = smooth[candidates[i]]
                    if val > remaining:
                        break
                    chosen.append(candidates[i])
                    backtrack(i + 1, remaining - val, chosen)
                    chosen.pop()
                    if best is not None:
                        return
            
            backtrack(0, n, [])
            if best is not None:
                return [smooth[i] for i in best]
        
        return None
    
    answers = []
    for n in range(1, 129):
        if time.time() - start_time > DEADLINE - 5:
            # Fallback: use a simple valid representation
            # For safety, use n itself if it's smooth, or find any valid one
            answer = None
            if n in smooth:
                answer = [n]
            else:
                # Try to find any valid representation quickly
                for k in range(2, 31):
                    candidates = [i for i in range(n_smooth) if smooth[i] <= n]
                    found = None
                    def bt2(start, rem, chosen):
                        nonlocal found
                        if found:
                            return
                        if rem == 0:
                            if is_antichain(chosen):
                                found = list(chosen)
                            return
                        if len(chosen) >= 30:
                            return
                        for i in range(start, len(candidates)):
                            val = smooth[candidates[i]]
                            if val > rem:
                                break
                            chosen.append(candidates[i])
                            bt2(i + 1, rem - val, chosen)
                            chosen.pop()
                            if found:
                                return
                    bt2(0, n, [])
                    if found:
                        answer = [smooth[i] for i in found]
                        break
                if answer is None:
                    # Last resort: should not happen, but provide something
                    answer = [1] * n if n <= 30 else [1] * 29 + [n - 29]
            answers.append(answer)
        else:
            answer = find_for_target(n)
            if answer is None:
                # Fallback
                answer = [n] if n in smooth else [1]
            answers.append(answer)
    
    return answers
# EVOLVE-BLOCK-END
