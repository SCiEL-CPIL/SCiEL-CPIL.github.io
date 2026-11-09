# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Precompute 5-smooth numbers up to 128
    smooths = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0: y //= 2
        while y % 3 == 0: y //= 3
        while y % 5 == 0: y //= 5
        if y == 1:
            smooths.append(x)
    
    M = len(smooths)
    
    # For each target, find the minimal antichain representation using iterative deepening
    answers = []
    
    def find_min_antichain(target, max_k=30):
        """Find antichain representation of target with minimum number of terms."""
        # Try k = 1, 2, 3, ... up to max_k
        for k in range(1, max_k + 1):
            if time.time() > deadline:
                return None
            # Find antichain of exactly k terms summing to target
            result = _find_antichain_k(target, k, smooths, M, deadline)
            if result is not None:
                return result
        return None
    
    def _find_antichain_k(target, k, smooths, M, deadline):
        """Find antichain of exactly k terms from smooths summing to target."""
        # Use backtracking with pruning
        def backtrack(start_idx, remaining, chosen):
            if time.time() > deadline:
                return None
            if len(chosen) == k:
                if remaining == 0:
                    return list(chosen)
                return None
            if remaining < 0:
                return None
            
            # Pruning: if we need 'need' more terms, the minimum possible sum is sum of 'need' smallest available
            need = k - len(chosen)
            if need == 0:
                return None if remaining != 0 else list(chosen)
            
            # Minimum possible: pick 'need' smallest from smooths[start_idx:]
            # Maximum possible: pick 'need' largest from smooths[start_idx:] that don't violate antichain
            # Simple pruning: if remaining < need * (smooths[start_idx] if start_idx < M else 1):
            if start_idx < M and remaining < need * smooths[start_idx]:
                return None
            
            for i in range(start_idx, M):
                val = smooths[i]
                if val > remaining:
                    break
                # Check antichain condition
                valid = True
                for x in chosen:
                    if val % x == 0 or x % val == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                chosen.append(val)
                result = backtrack(i + 1, remaining - val, chosen)
                if result is not None:
                    return result
                chosen.pop()
            
            return None
        
        return backtrack(0, target, [])
    
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: use {n} if n is smooth, else try a simple decomposition
            if n in smooths:
                answers.append([n])
            else:
                # Greedy fallback
                remaining = n
                chosen = []
                for s in sorted(smooths, reverse=True):
                    if s <= remaining and all(s % x != 0 and x % s != 0 for x in chosen):
                        chosen.append(s)
                        remaining -= s
                        if remaining == 0:
                            break
                if remaining == 0:
                    answers.append(chosen)
                else:
                    answers.append([n] if n in smooths else [])
            continue
        
        result = find_min_antichain(n)
        if result is not None:
            answers.append(result)
        else:
            # Fallback
            if n in smooths:
                answers.append([n])
            else:
                answers.append([])
    
    return answers
# EVOLVE-BLOCK-END
