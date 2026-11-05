import itertools
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # --- 1. Generate Baseline Recursive Superpermutation ---
    def recursive_superperm(n_val):
        word = [1]
        for m in range(2, n_val + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    baseline_word = recursive_superperm(n)
    
    if n <= 4:
        return "".join(map(str, baseline_word))

    best_S = baseline_word[:]
    best_len = len(baseline_word)

    # --- 2. Local Search ---
    
    def get_counts(word, n_val):
        counts = {}
        for i in range(len(word) - n_val + 1):
            p = tuple(word[i:i+n_val])
            if len(set(p)) == n_val: # Only count valid permutations
                counts[p] = counts.get(p, 0) + 1
        return counts

    def get_critical_positions(word, n_val, counts):
        L = len(word)
        critical = [False] * L
        for i in range(L - n_val + 1):
            p = tuple(word[i:i+n_val])
            if counts.get(p, 0) == 1:
                for pos in range(i, i + n_val):
                    critical[pos] = True
        return critical

    def perform_removals(word, n_val, counts, critical):
        """
        Iteratively removes non-critical symbols until no more can be removed.
        Returns the final word, counts, critical, and total removals.
        """
        total_removed = 0
        while True:
            # Find indices to remove
            to_remove = [i for i in range(len(word)) if not critical[i]]
            if not to_remove:
                break
            
            # Rebuild word without these indices
            new_word = [word[i] for i in range(len(word)) if critical[i]]
            
            # Update counts and critical
            counts = get_counts(new_word, n_val)
            critical = get_critical_positions(new_word, n_val, counts)
            
            removed_in_step = len(word) - len(new_word)
            total_removed += removed_in_step
            word = new_word
            
            if removed_in_step == 0:
                break
                
        return word, counts, critical, total_removed

    # Main Search Loop
    current_S = best_S[:]
    current_counts = get_counts(current_S, n)
    current_critical = get_critical_positions(current_S, n, current_counts)
    current_len = len(current_S)
    
    stuck_iterations = 0
    max_stuck = 20000 # Increase threshold for perturbations
    
    while time.time() < deadline:
        # 1. Try to remove
        new_S, new_counts, new_critical, removed = perform_removals(current_S, n, current_counts, current_critical)
        
        if removed > 0:
            current_S = new_S
            current_counts = new_counts
            current_critical = new_critical
            current_len = len(current_S)
            
            if current_len < best_len:
                best_S = current_S[:]
                best_len = current_len
            stuck_iterations = 0
        else:
            # 2. Perturbation
            if stuck_iterations > max_stuck:
                break
            
            # Try a random swap
            if len(current_S) > 1:
                i = random.randint(0, len(current_S) - 2)
                j = i + 1
                # Swap
                current_S[i], current_S[j] = current_S[j], current_S[i]
                
                # Check if this swap allows removals
                temp_counts = get_counts(current_S, n)
                temp_critical = get_critical_positions(current_S, n, temp_counts)
                
                # Try removals on the swapped version
                test_S, test_counts, test_critical, test_removed = perform_removals(current_S, n, temp_counts, temp_critical)
                
                if test_removed > 0:
                    # Accept swap and removals
                    current_S = test_S
                    current_counts = test_counts
                    current_critical = test_critical
                    current_len = len(current_S)
                    if current_len < best_len:
                        best_S = current_S[:]
                        best_len = current_len
                    stuck_iterations = 0
                else:
                    # Revert swap
                    current_S[i], current_S[j] = current_S[j], current_S[i]
                    stuck_iterations += 1
            else:
                stuck_iterations += 1

    return "".join(map(str, best_S))
