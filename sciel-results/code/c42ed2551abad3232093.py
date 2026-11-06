"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_smooth(n):
    y = n
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _divides(a, b):
    return b % a == 0

def _precompute_minimal_representations():
    """
    Pre-compute minimal antichain representations for all targets 1..128.
    Returns a dict mapping target -> list of terms (or None if not found).
    """
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = [x for x in range(1, 129) if _is_smooth(x)]
    smooth_set = set(smooth)
    n_smooth = len(smooth)
    
    # Pre-compute divisibility relation
    # divides[i] contains j if smooth[i] divides smooth[j]
    divides = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth[j] % smooth[i] == 0:
                divides[i].add(j)
    
    def is_antichain_pair(i, j):
        """Check if smooth[i] and smooth[j] form an antichain pair."""
        if i == j:
            return False
        if j in divides[i]:
            return False
        if i in divides[j]:
            return False
        return True
    
    # For each target, find minimal antichain representation
    answers = [None] * 129
    
    # Try 1-term representations
    for n in range(1, 129):
        if n in smooth_set:
            answers[n] = [n]
    
    # Try 2-term representations
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if not is_antichain_pair(i, j):
                continue
            s = smooth[i] + smooth[j]
            if s <= 128 and answers[s] is None:
                answers[s] = [smooth[i], smooth[j]]
    
    # Try 3-term representations
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            if not is_antichain_pair(i, j):
                continue
            sij = si + smooth[j]
            if sij > 128:
                continue
            for k in range(j + 1, n_smooth):
                if not is_antichain_pair(i, k):
                    continue
                if not is_antichain_pair(j, k):
                    continue
                s = sij + smooth[k]
                if s <= 128 and answers[s] is None:
                    answers[s] = [smooth[i], smooth[j], smooth[k]]
    
    # Try 4-term representations
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            if not is_antichain_pair(i, j):
                continue
            sij = si + smooth[j]
            if sij > 128:
                continue
            for k in range(j + 1, n_smooth):
                if not is_antichain_pair(i, k):
                    continue
                if not is_antichain_pair(j, k):
                    continue
                sijk = sij + smooth[k]
                if sijk > 128:
                    continue
                for l in range(k + 1, n_smooth):
                    if not is_antichain_pair(i, l):
                        continue
                    if not is_antichain_pair(j, l):
                        continue
                    if not is_antichain_pair(k, l):
                        continue
                    s = sijk + smooth[l]
                    if s <= 128 and answers[s] is None:
                        answers[s] = [smooth[i], smooth[j], smooth[k], smooth[l]]
    
    # For remaining targets, use DFS to find representations with 5+ terms
    for n in range(1, 129):
        if answers[n] is not None:
            continue
        
        # DFS for 5+ terms
        found = False
        for count in range(5, 15):
            result = None
            stack = [(0, [], 0)]
            while stack:
                idx_start, chosen, cur_sum = stack.pop()
                if len(chosen) == count:
                    if cur_sum == n:
                        # Verify antichain (should already be valid by construction)
                        result = [smooth[i] for i in chosen]
                    continue
                
                remaining = count - len(chosen)
                for t in range(idx_start, n_smooth):
                    new_sum = cur_sum + smooth[t]
                    if new_sum > n:
                        break
                    
                    # Check antichain with all chosen
                    valid = True
                    for c in chosen:
                        if t in divides[c] or c in divides[t]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Pruning: can we still reach n?
                    if remaining - 1 > 0:
                        min_add = 0
                        for m in range(remaining - 1):
                            if t + 1 + m < n_smooth:
                                min_add += smooth[t + 1 + m]
                            else:
                                min_add = n + 1
                                break
                        if new_sum + min_add > n:
                            continue
                    
                    stack.append((t + 1, chosen + [t], new_sum))
            
            if result is not None:
                answers[n] = result
                found = True
                break
    
    return answers

# Pre-compute the lookup table once at module load time
_MINIMAL_REPRESENTATIONS = _precompute_minimal_representations()

def construct():
    """
    Return 128 rows corresponding to targets 1..128.
    Each row contains 1..30 distinct positive 2^a*3^b*5^c terms summing to its target,
    forming a divisibility antichain.
    """
    return _MINIMAL_REPRESENTATIONS[1:129]
# EVOLVE-BLOCK-END
