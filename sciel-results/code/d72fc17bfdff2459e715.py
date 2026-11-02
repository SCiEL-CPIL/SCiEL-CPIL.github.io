import random
import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if n <= 1:
        return "1"
    
    symbols = list(range(1, n + 1))
    perms = list(itertools.permutations(symbols))
    num_perms = len(perms)
    
    # Precompute prefix maps
    prefix_to_indices = {}
    for k in range(1, n):
        prefix_to_indices[k] = {}
        for idx, p in enumerate(perms):
            prefix = p[:k]
            if prefix not in prefix_to_indices[k]:
                prefix_to_indices[k][prefix] = []
            prefix_to_indices[k][prefix].append(idx)
            
    # Precompute suffix-to-prefix matches? No, we match current_suffix with next_prefix.
    # We can also precompute for each permutation, the list of permutations it can transition to with overlap k.
    # But 'used' state changes, so static precomputation of "valid next" is hard.
    
    best_word = None
    best_len = float('inf')
    
    # We will run several restarts.
    # To speed up, we can limit the lookahead depth.
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        used = [False] * num_perms
        current_word = []
        
        start_idx = random.randint(0, num_perms - 1)
        used[start_idx] = True
        current_word.extend(perms[start_idx])
        
        success = True
        
        for step in range(1, num_perms):
            if time.time() > deadline:
                success = False
                break
                
            next_idx = -1
            best_overlap = 0
            best_candidate = None
            best_score = -1
            
            # Try overlaps from n-1 down to 1
            for k in range(n - 1, 0, -1):
                if len(current_word) < k:
                    continue
                suffix = tuple(current_word[-k:])
                candidates = prefix_to_indices[k].get(suffix, [])
                
                valid_candidates = [idx for idx in candidates if not used[idx]]
                
                if valid_candidates:
                    if len(valid_candidates) == 1:
                        next_idx = valid_candidates[0]
                        best_overlap = k
                        break
                    else:
                        # Heuristic: Pick the candidate that has the most "exit" options
                        # Check overlap n-1 and n-2 for each candidate
                        score = 0
                        for idx in valid_candidates:
                            p_next = perms[idx]
                            # How many unused perms can follow p_next?
                            # Check suffix of p_next of length n-1
                            sfx1 = tuple(p_next[-(n-1):])
                            c1 = prefix_to_indices[n-1].get(sfx1, [])
                            cnt1 = sum(1 for i in c1 if not used[i])
                            
                            sfx2 = tuple(p_next[-(n-2):])
                            c2 = prefix_to_indices[n-2].get(sfx2, [])
                            cnt2 = sum(1 for i in c2 if not used[i])
                            
                            # Score is sum of options
                            cand_score = cnt1 * 2 + cnt2
                            if cand_score > best_score:
                                best_score = cand_score
                                best_candidate = idx
                                best_overlap = k
                        if best_candidate is not None:
                            next_idx = best_candidate
                            break
                            
            if next_idx == -1:
                success = False
                break
            
            used[next_idx] = True
            p = perms[next_idx]
            current_word.extend(p[best_overlap:])
        
        if success:
            word_str = "".join(map(str, current_word))
            if len(current_word) < best_len:
                best_len = len(current_word)
                best_word = word_str
            # If we hit the known optimal 5905 for n=7, we can stop
            if n == 7 and best_len <= 5905:
                break

    if best_word is not None:
        return best_word
    else:
        return _classic_construct(n)

def _classic_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
