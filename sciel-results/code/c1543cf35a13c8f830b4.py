import itertools
import random
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, leave 40s margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate all permutations
    symbols = list(range(1, n + 1))
    perms = list(itertools.permutations(symbols))
    N = len(perms)
    
    # Precompute overlap matrix? 
    # N=5040, N^2 = 25M. 
    # Storing 25M bytes (if uint8) is 25MB. 
    # Calculating 25M overlaps: 25M * 6 checks * 7 ops ~ 1B ops. Too slow in Python.
    # So we calculate on the fly.
    
    def get_overlap(a, b):
        # a, b are tuples
        max_k = n - 1
        # Check from largest overlap down
        for k in range(max_k, 0, -1):
            if a[n - k:] == b[:k]:
                return k
        return 0 # Should not happen for perms? 
        # Actually, if no overlap, we add n symbols? 
        # In superperm, we assume we can always append? 
        # If overlap is 0, we add n symbols. 
        # But for perms of 1..n, is overlap always >= 1? 
        # Suffix of A length 1 is A[-1]. Prefix of B length 1 is B[0].
        # If A[-1] == B[0], overlap is 1.
        # Otherwise 0.
        
    # Greedy Construction
    # Start with a random permutation
    best_seq = None
    best_score = -1
    
    # Run multiple greedy restarts
    for _ in range(10):
        if time.time() > deadline - 30: break
        
        current = random.choice(perms)
        used = {current}
        seq = [current]
        
        while len(used) < N:
            best_next = None
            best_ov = -1
            # To speed up, we could pre-sort candidates? 
            # Just iterate all unused
            for p in perms:
                if p in used:
                    continue
                ov = get_overlap(current, p)
                if ov > best_ov:
                    best_ov = ov
                    best_next = p
                    if ov == n - 1:
                        break # Max possible
            if best_next is None:
                # Fallback: pick random unused
                remaining = [p for p in perms if p not in used]
                best_next = random.choice(remaining)
            seq.append(best_next)
            used.add(best_next)
            current = best_next
            
        # Calculate score (total overlap)
        score = 0
        for i in range(len(seq) - 1):
            score += get_overlap(seq[i], seq[i + 1])
            
        if score > best_score:
            best_score = score
            best_seq = seq
            
    # Local Search (Simulated Annealing) on best_seq
    if time.time() < deadline - 10 and best_seq:
        seq = best_seq[:]
        current_score = best_score
        T = 1.0
        cooling = 0.999
        
        # Precompute overlaps for current seq? 
        # No, calculate delta on the fly.
        
        while time.time() < deadline - 10:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            
            # Reverse seq[i:j+1]
            # Old links: (i-1, i) and (j, j+1)
            # New links: (i-1, j) and (i, j+1)
            
            ov_old_1 = get_overlap(seq[i - 1], seq[i]) if i > 0 else 0
            ov_old_2 = get_overlap(seq[j], seq[j + 1]) if j < N - 1 else 0
            
            # The internal links (i, i+1)...(j-1, j) are reversed but overlap is symmetric? 
            # No, overlap(A, B) != overlap(B, A).
            # So we need to recalculate internal overlaps?
            # That's expensive.
            
            # Alternative: Use "Move" operator.
            # Pick random i, j. Move seq[i] to position j.
            # This only affects 3 links: (i-1, i), (i, i+1), (j-1, j), (j, j+1).
            # Let's use Move.
            
            # Re-do logic for Move
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j: continue
            
            # Remove from i, insert at j
            # If i < j, indices shift
            # Let's just swap two random elements? 2-Swap.
            # Swap seq[i] and seq[j].
            
            # Links affected:
            # (i-1, i), (i, i+1), (j-1, j), (j, j+1)
            # And if i and j are adjacent, fewer links.
            
            # Let's stick to simple 2-Swap for simplicity and speed.
            
            ov_old_1 = get_overlap(seq[i - 1], seq[i]) if i > 0 else 0
            ov_old_2 = get_overlap(seq[i], seq[i + 1]) if i < N - 1 else 0
            ov_old_3 = get_overlap(seq[j - 1], seq[j]) if j > 0 else 0
            ov_old_4 = get_overlap(seq[j], seq[j + 1]) if j < N - 1 else 0
            
            # Perform swap
            seq[i], seq[j] = seq[j], seq[i]
            
            ov_new_1 = get_overlap(seq[i - 1], seq[i]) if i > 0 else 0
            ov_new_2 = get_overlap(seq[i], seq[i + 1]) if i < N - 1 else 0
            ov_new_3 = get_overlap(seq[j - 1], seq[j]) if j > 0 else 0
            ov_new_4 = get_overlap(seq[j], seq[j + 1]) if j < N - 1 else 0
            
            delta = (ov_new_1 + ov_new_2 + ov_new_3 + ov_new_4) - (ov_old_1 + ov_old_2 + ov_old_3 + ov_old_4)
            
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                current_score += delta
            else:
                # Revert
                seq[i], seq[j] = seq[j], seq[i]
            
            T *= cooling
            if T < 0.01:
                T = 0.01
                
    # Construct word from best_seq
    if not best_seq:
        return ""
        
    word = list(best_seq[0])
    for i in range(1, len(best_seq)):
        prev = best_seq[i - 1]
        curr = best_seq[i]
        ov = get_overlap(prev, curr)
        # Append curr[ov:]
        word.extend(curr[ov:])
        
    return "".join(map(str, word))
