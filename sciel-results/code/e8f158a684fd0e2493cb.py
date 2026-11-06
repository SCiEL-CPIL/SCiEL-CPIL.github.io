"""
Divisibility-antichain representations by smooth numbers.
construct() -> 128 rows corresponding to targets 1..128.
Each row: list of 1..30 distinct positive 2^a*3^b*5^c terms summing to target.
No chosen term divides another. Minimize total summands.
"""
import time
from functools import lru_cache
from itertools import combinations

_START = time.time()
DEADLINE = 70  # leave 20s margin before 90s kill

def _is_smooth(x):
    """Check if x is of form 2^a * 3^b * 5^c with a,b,c >= 0."""
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def construct():
    t0 = time.time()
    
    # Generate all smooth numbers up to 128
    smooth_set = set()
    for a in range(0, 8):
        v2 = 1 << a
        for b in range(0, 6):
            v23 = v2 * (3 ** b)
            if v23 > 128:
                break
            for c in range(0, 5):
                v = v23 * (5 ** c)
                if v > 128:
                    break
                if v >= 1:
                    smooth_set.add(v)
    smooth = sorted(smooth_set)
    n_smooth = len(smooth)
    smooth_idx = {s: i for i, s in enumerate(smooth)}
    
    # Precompute smooth lookup
    is_sm = [False] * 129
    for s in smooth:
        is_sm[s] = True
    
    # Precompute divisibility using bitmasks for efficiency
    # div_mask[i] is a bitmask where bit j is set if smooth[i] divides smooth[j]
    div_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                div_mask[i] |= (1 << j)
    
    # Precompute pair antichain validity
    pair_ok = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if (div_mask[i] >> j) & 1 or (div_mask[j] >> i) & 1:
                pair_ok[i][j] = False
                pair_ok[j][i] = False
    
    # For each index, precompute the set of indices that form an antichain with it
    # compatible[i] = bitmask of j > i such that pair_ok[i][j]
    compatible = [0] * n_smooth
    for i in range(n_smooth):
        mask = 0
        for j in range(i + 1, n_smooth):
            if pair_ok[i][j]:
                mask |= (1 << j)
        compatible[i] = mask
    
    # Precompute triple antichain validity - store as dict mapping sum to list of triples
    sum_to_triples = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if not pair_ok[i][j]:
                continue
            # Use compatible mask to restrict k
            mask = compatible[j] & ~( (1 << (j+1)) - 1 )  # only k > j
            k_mask = mask
            while k_mask:
                k_bit = k_mask & (-k_mask)
                k = k_bit.bit_length() - 1
                k_mask ^= k_bit
                if pair_ok[i][k]:
                    s = smooth[i] + smooth[j] + smooth[k]
                    if s <= 128:
                        if s not in sum_to_triples:
                            sum_to_triples[s] = []
                        sum_to_triples[s].append((i, j, k))
    
    # Precompute quad antichain validity
    sum_to_quads = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if not pair_ok[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                if not pair_ok[i][k] or not pair_ok[j][k]:
                    continue
                # Use compatible mask for l
                mask = compatible[k] & ~( (1 << (k+1)) - 1 )  # only l > k
                l_mask = mask
                while l_mask:
                    l_bit = l_mask & (-l_mask)
                    l = l_bit.bit_length() - 1
                    l_mask ^= l_bit
                    if pair_ok[i][l] and pair_ok[j][l]:
                        s = smooth[i] + smooth[j] + smooth[k] + smooth[l]
                        if s <= 128:
                            if s not in sum_to_quads:
                                sum_to_quads[s] = []
                            sum_to_quads[s].append((i, j, k, l))
    
    # For 2-term: for each pair (i,j) that is antichain, record sum
    sum_to_pairs = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if pair_ok[i][j]:
                s = smooth[i] + smooth[j]
                if s <= 128:
                    if s not in sum_to_pairs:
                        sum_to_pairs[s] = []
                    sum_to_pairs[s].append((i, j))
    
    # Now for each target, pick minimum terms
    answers = []
    
    # Helper function for fast antichain check using bitmasks
    def _is_antichain_with_mask(chosen_mask, new_idx):
        """Check if adding smooth[new_idx] to chosen_mask maintains antichain."""
        # Check if new_idx divides any chosen or any chosen divides new_idx
        # Use bit operations for speed
        # div_mask[new_idx] has bits set for indices that new_idx divides
        # We need to check if any chosen index is in div_mask[new_idx] or vice versa
        # For each chosen idx, check if div_mask[new_idx] has bit idx set OR div_mask[idx] has bit new_idx set
        for i in range(n_smooth):
            if (chosen_mask >> i) & 1:
                if (div_mask[new_idx] >> i) & 1 or (div_mask[i] >> new_idx) & 1:
                    return False
        return True
    
    # Precompute for each chosen_mask, the set of compatible indices
    # This is expensive, so we'll compute on the fly but cache results
    compatible_cache = {}
    
    def _get_compatible_indices(chosen_mask):
        """Get bitmask of indices compatible with all chosen indices."""
        if chosen_mask in compatible_cache:
            return compatible_cache[chosen_mask]
        
        # Start with all indices
        result = (1 << n_smooth) - 1
        
        # For each chosen index, intersect with its compatible set
        for i in range(n_smooth):
            if (chosen_mask >> i) & 1:
                # Compatible with i means: not dividing i and not being divided by i
                # div_mask[i] has bits for j where i divides j
                # We need j such that not (div_mask[i] has bit j) and not (div_mask[j] has bit i)
                # The second condition is harder to precompute
                # Let's just check each candidate
                pass
        
        # Simpler approach: for each index j, check if it's compatible with all chosen
        result = 0
        for j in range(n_smooth):
            if (chosen_mask >> j) & 1:
                continue  # already chosen
            ok = True
            for i in range(n_smooth):
                if (chosen_mask >> i) & 1:
                    if (div_mask[j] >> i) & 1 or (div_mask[i] >> j) & 1:
                        ok = False
                        break
            if ok:
                result |= (1 << j)
        
        compatible_cache[chosen_mask] = result
        return result
    
    # Memoized search for 5+ terms
    search_cache = {}
    
    def _search(rem, min_idx, chosen_mask, max_t):
        if max_t == 0:
            return list(smooth[i] for i in range(n_smooth) if (chosen_mask >> i) & 1) if rem == 0 else None
        if rem <= 0:
            return None
        
        # Pruning: minimum possible sum with remaining terms
        if max_t > 0:
            min_possible = 0
            count = 0
            for idx in range(min_idx, n_smooth):
                if count >= max_t:
                    break
                min_possible += smooth[idx]
                count += 1
            if min_possible > rem:
                return None
        
        # Pruning: maximum possible sum with remaining terms
        if max_t > 0:
            max_possible = 0
            count = 0
            for idx in range(n_smooth - 1, min_idx - 1, -1):
                if idx < min_idx:
                    break
                if count >= max_t:
                    break
                max_possible += smooth[idx]
                count += 1
            if max_possible < rem:
                return None
        
        # Try to use memoization
        cache_key = (rem, min_idx, chosen_mask, max_t)
        if cache_key in search_cache:
            return search_cache[cache_key]
        
        result = None
        for idx in range(min_idx, n_smooth):
            s = smooth[idx]
            if s > rem:
                break
            
            # Check antichain with all chosen
            if not _is_antichain_with_mask(chosen_mask, idx):
                continue
            
            new_mask = chosen_mask | (1 << idx)
            sub_result = _search(rem - s, idx + 1, new_mask, max_t - 1)
            if sub_result is not None:
                result = [s] + sub_result
                break
        
        search_cache[cache_key] = result
        return result
    
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE:
            # Fallback: find any valid representation quickly
            if is_sm[n]:
                answers.append([n])
            else:
                # Try 2-term
                found = False
                for i in range(n_smooth):
                    s1 = smooth[i]
                    if s1 >= n:
                        break
                    s2 = n - s1
                    if s2 <= s1:
                        continue
                    if not is_sm[s2]:
                        continue
                    j = smooth_idx.get(s2, -1)
                    if j >= 0 and pair_ok[i][j]:
                        answers.append([s1, s2])
                        found = True
                        break
                if not found:
                    # Try 3-term
                    found = False
                    for i in range(n_smooth):
                        s1 = smooth[i]
                        if s1 * 3 >= n:
                            break
                        rem1 = n - s1
                        for j in range(i + 1, n_smooth):
                            s2 = smooth[j]
                            if s2 * 2 >= rem1:
                                break
                            s3 = rem1 - s2
                            if s3 <= s2:
                                break
                            if not is_sm[s3]:
                                continue
                            k = smooth_idx.get(s3, -1)
                            if k >= 0 and pair_ok[i][j] and pair_ok[i][k] and pair_ok[j][k]:
                                answers.append([s1, s2, s3])
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        # Try 4-term as last resort
                        found = False
                        for i in range(n_smooth):
                            s1 = smooth[i]
                            if s1 * 4 >= n:
                                break
                            rem1 = n - s1
                            for j in range(i + 1, n_smooth):
                                s2 = smooth[j]
                                if s2 * 3 >= rem1:
                                    break
                                rem2 = rem1 - s2
                                for k in range(j + 1, n_smooth):
                                    s3 = smooth[k]
                                    if s3 * 2 >= rem2:
                                        break
                                    s4 = rem2 - s3
                                    if s4 <= s3:
                                        break
                                    if not is_sm[s4]:
                                        continue
                                    l = smooth_idx.get(s4, -1)
                                    if (l >= 0 and pair_ok[i][j] and pair_ok[i][k] and 
                                        pair_ok[i][l] and pair_ok[j][k] and 
                                        pair_ok[j][l] and pair_ok[k][l]):
                                        answers.append([s1, s2, s3, s4])
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if not found:
                            answers.append([1, n - 1] if n > 1 else [1])
            continue
        
        # Try 1-term
        if is_sm[n]:
            answers.append([n])
            continue
        
        # Try 2-term
        if n in sum_to_pairs:
            i, j = sum_to_pairs[n][0]
            answers.append([smooth[i], smooth[j]])
            continue
        
        # Try 3-term
        if n in sum_to_triples:
            i, j, k = sum_to_triples[n][0]
            answers.append([smooth[i], smooth[j], smooth[k]])
            continue
        
        # Try 4-term
        if n in sum_to_quads:
            i, j, k, l = sum_to_quads[n][0]
            answers.append([smooth[i], smooth[j], smooth[k], smooth[l]])
            continue
        
        # Fallback: search for 5+ terms with memoization
        found = False
        for max_terms in range(5, 15):
            if time.time() - t0 > DEADLINE:
                break
            
            # Clear cache for each new target to avoid stale results
            search_cache.clear()
            
            result = _search(n, 0, 0, max_terms)
            if result is not None:
                answers.append(result)
                found = True
                break
        
        if not found:
            # Last resort: try harder with more terms
            for max_terms in range(15, 31):
                if time.time() - t0 > DEADLINE:
                    break
                
                search_cache.clear()
                result = _search(n, 0, 0, max_terms)
                if result is not None:
                    answers.append(result)
                    found = True
                    break
            
            if not found:
                # Absolute last resort
                if is_sm[n]:
                    answers.append([n])
                else:
                    # Find any valid pair
                    for i in range(n_smooth):
                        s1 = smooth[i]
                        if s1 >= n:
                            break
                        s2 = n - s1
                        if s2 <= s1:
                            continue
                        if not is_sm[s2]:
                            continue
                        j = smooth_idx.get(s2, -1)
                        if j >= 0 and pair_ok[i][j]:
                            answers.append([s1, s2])
                            found = True
                            break
                    if not found:
                        answers.append([1, n - 1] if n > 1 else [1])
    
    return answers
