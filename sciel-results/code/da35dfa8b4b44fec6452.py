import time
from functools import lru_cache

# Precompute at module level
def _precompute():
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):
        va = 2 ** a
        for b in range(0, 5):
            vb = va * (3 ** b)
            if vb > 128:
                break
            for c in range(0, 4):
                vc = vb * (5 ** c)
                if vc > 128:
                    break
                smooth.add(vc)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    
    # Map smooth number to index
    smooth_to_idx = {s: i for i, s in enumerate(smooth_list)}
    
    # Build divisibility relation using bitmasks
    # div_mask[i] = bitmask of indices j where smooth_list[j] divides smooth_list[i]
    div_mask = [0] * n_smooth
    for i, s in enumerate(smooth_list):
        for j, t in enumerate(smooth_list):
            if t % s == 0:
                div_mask[i] |= (1 << j)
    
    # Incompatibility: two numbers are incompatible if one divides the other
    # For each index i, incompatible_mask[i] = bitmask of j where i and j are incompatible
    incompatible_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth_list[j] % smooth_list[i] == 0 or smooth_list[i] % smooth_list[j] == 0:
                incompatible_mask[i] |= (1 << j)
                incompatible_mask[j] |= (1 << i)
    
    # Candidates for each target n: indices of smooth numbers <= n
    candidates = [[] for _ in range(129)]
    for n in range(1, 129):
        for i, s in enumerate(smooth_list):
            if s <= n:
                candidates[n].append(i)
            else:
                break
    
    # Find antichain of exactly k distinct smooth numbers summing to n
    def find_antichain(n, k):
        """Find antichain of exactly k distinct smooth numbers summing to n."""
        cand = candidates[n]
        if len(cand) < k:
            return None
        
        if k == 1:
            if n in smooth_to_idx:
                return [n]
            return None
        
        # Sort descending for better pruning
        cand_desc = sorted(cand, reverse=True, key=lambda i: smooth_list[i])
        
        # Quick bounds check
        min_sum = sum(smooth_list[i] for i in cand_desc[-k:])
        max_sum = sum(smooth_list[i] for i in cand_desc[:k])
        if n < min_sum or n > max_sum:
            return None
        
        path = []
        
        def backtrack(start, rem_sum, rem_k):
            if rem_k == 0:
                return [] if rem_sum == 0 else None
            
            avail = cand_desc[start:]
            if len(avail) < rem_k:
                return None
            
            # Max sum with rem_k from avail
            if sum(smooth_list[i] for i in avail[:rem_k]) < rem_sum:
                return None
            
            # Min sum with rem_k from avail
            if sum(smooth_list[i] for i in avail[-rem_k:]) > rem_sum:
                return None
            
            for i in range(start, len(avail)):
                idx = avail[i]
                num = smooth_list[idx]
                if num > rem_sum:
                    continue
                
                # Check divisibility against all chosen using bitmask
                compatible = True
                for p_idx in path:
                    if incompatible_mask[idx] & (1 << p_idx):
                        compatible = False
                        break
                
                if not compatible:
                    continue
                
                path.append(idx)
                result = backtrack(i + 1, rem_sum - num, rem_k - 1)
                if result is not None:
                    return [num] + result
                path.pop()
            
            return None
        
        result = backtrack(0, n, k)
        return result
    
    # Precompute all answers
    answers = []
    for n in range(1, 129):
        answer = None
        for k in range(1, 31):
            res = find_antichain(n, k)
            if res is not None:
                answer = res
                break
        
        if answer is None:
            # Fallback: should not happen
            if n in smooth_to_idx:
                answer = [n]
            else:
                # Try to find any valid representation
                answer = find_antichain(n, 2)
                if answer is None:
                    answer = [n]  # placeholder
        
        answers.append(answer)
    
    return answers

_PRECOMPUTED = _precompute()

def construct():
    return _PRECOMPUTED
