import time
import random
import math

def construct(n):
    start_time = time.time()
    # Leave a 5-second safety margin before the 120s hard kill
    deadline = start_time + 115.0
    
    # Helper: Standard recursive superpermutation construction
    def standard_construct(n):
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
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    # Helper: Build counts of permutations in a word
    def build_counts(S, n):
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts

    # Helper: Check validity
    def is_valid(S, n):
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i+n]))
        return len(seen) == math.factorial(n)

    # Helper: Construct word from a specific order of permutations of size n-1
    def construct_from_order(perms_order, n):
        # perms_order is a list of tuples of length n-1
        # We assume perms_order contains all permutations of 1..n-1 exactly once
        # The new symbol is n
        new = []
        for p in perms_order:
            block = list(p) + [n] + list(p)
            if not new:
                new = block
            else:
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
        return new

    # If n is small, just use standard construction
    if n < 7:
        S = standard_construct(n)
        return "".join(map(str, S))

    # For n=7, we try to optimize the order of the 720 permutations of 1..6
    # to minimize the length of the resulting superpermutation of 1..7.
    
    # Step 1: Get standard SP(6) and its permutations
    word_6 = standard_construct(6)
    perms_6, seen = [], set()
    for i in range(len(word_6) - 6 + 2):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
    
    num_perms = len(perms_6) # 720
    
    # Precompute overlap matrix for these permutations
    # M[i][j] = overlap between perms_6[i] and perms_6[j]
    # Overlap is longest suffix of i matching prefix of j
    M = [[0]*num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            # Max overlap 6
            for k in range(6, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
                    
    # Initial order: the order they appear in standard SP(6)
    order = list(range(num_perms))
    
    # Calculate initial score (total overlap)
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score
        # Total length = sum(len(block)) - sum(overlaps)
        # len(block) = 2*(n-1) + 1 = 13 for n=7
        # Total length = 13 * 720 - score
        # Minimizing length == Maximizing score
        
    best_order = order[:]
    best_score = calc_score(order)
    
    # Hill Climbing to improve the order
    # We have about 110 seconds.
    # Each iteration involves a swap or move and a score calculation (O(N) = 720 ops).
    # We can do many iterations.
    
    random.seed(42)
    
    # Time budget for optimization: 60 seconds
    opt_deadline = start_time + 60.0
    
    while time.time() < opt_deadline:
        # Pick a random move
        move_type = random.random()
        if move_type < 0.5:
            # Swap adjacent
            i = random.randint(0, num_perms - 2)
            # Calculate delta score efficiently?
            # Old score contribution around i: M[ord[i-1]][ord[i]] + M[ord[i]][ord[i+1]] + M[ord[i+1]][ord[i+2]]
            # New score contribution: M[ord[i-1]][ord[i+1]] + M[ord[i+1]][ord[i]] + M[ord[i]][ord[i+2]]
            # This is O(1) if we handle boundaries.
            
            # For simplicity and speed, let's just recalc full score? 
            # 720 additions is very fast in Python if optimized? 
            # Actually, a loop of 720 is ~50 microseconds. 
            # We can do ~2000 iterations per second. 
            # 60 seconds -> 120,000 iterations. This is fine.
            
            new_order = order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_score = calc_score(new_order)
            
            if new_score > best_score:
                best_score = new_score
                best_order = new_order
            if new_score >= calc_score(order): # Accept if not worse (simulated annealing style? No, just hill climb with restarts?)
                order = new_order
        else:
            # Move element i to position j
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j: continue
            new_order = order[:]
            item = new_order.pop(i)
            new_order.insert(j, item)
            new_score = calc_score(new_order)
            
            if new_score > best_score:
                best_score = new_score
                best_order = new_order
            if new_score >= calc_score(order):
                order = new_order

    # Construct the optimized word
    S = construct_from_order(best_order, n)
    
    # Step 2: Apply local search deletion to the optimized word
    # This is the core of the current program's strength.
    
    counts = build_counts(S, n)
    best_S = S[:]
    best_len = len(S)
    
    # Local Search Strategy:
    # We repeatedly try to remove characters.
    failed_passes = 0
    max_failed_passes_before_perturb = 3
    
    # Remaining time for deletion search
    # We have ~55 seconds left.
    
    while time.time() < deadline:
        # Attempt a full pass of removals from right to left
        removed_any = False
        i = len(S) - 1
        
        while i >= 0:
            if time.time() > deadline:
                break
            
            # Identify windows affected by removing S[i]
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            valid_removal = True
            old_perms_to_dec = []
            
            # Check if any old window has count 1
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid_removal = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid_removal:
                i -= 1
                continue
            
            # If valid, perform the removal
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # Identify new windows created by the removal
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, (len(S) - 1) - n)
            
            for k in range(start_new, end_new + 1):
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            # Commit removal
            S.pop(i)
            removed_any = True
            best_S = S[:]
            best_len = len(S)
            
        if removed_any:
            failed_passes = 0
        else:
            failed_passes += 1
            if failed_passes >= max_failed_passes_before_perturb:
                # Perturbation: Random swap in S?
                # Risky for validity, but we can check.
                # Or just break if stuck.
                # Let's try a simple perturbation: swap two random elements.
                if time.time() > deadline - 5.0:
                    break
                
                # Try a few random swaps to escape local minimum
                improved = False
                for _ in range(10):
                    if time.time() > deadline - 5.0:
                        break
                    idx1 = random.randint(0, len(S)-1)
                    idx2 = random.randint(0, len(S)-1)
                    if idx1 == idx2: continue
                    
                    # Save state
                    old_S = S[:]
                    old_counts = counts.copy()
                    
                    # Swap
                    S[idx1], S[idx2] = S[idx2], S[idx1]
                    
                    # Check validity quickly?
                    # Instead of full rebuild, just check if we can delete more.
                    # But the swap might break validity (missing perms).
                    # We must ensure the string remains valid.
                    # A swap changes 2*(n-1) windows.
                    # It's complex to update counts safely for a swap without full rebuild.
                    # Given time constraints, let's just break if stuck.
                    # The initial optimization should have provided a good start.
                    S = old_S
                    counts = old_counts
                    break
                break

    # Final check
    if not is_valid(best_S, n):
        # Fallback to standard construction if something went wrong
        S_std = standard_construct(n)
        return "".join(map(str, S_std))
        
    return "".join(map(str, best_S))

