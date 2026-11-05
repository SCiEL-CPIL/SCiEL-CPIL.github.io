import itertools
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin

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
                # Find max overlap between end of new and start of block
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    baseline_word = recursive_superperm(n)
    baseline_len = len(baseline_word)
    
    if n <= 4:
        return "".join(map(str, baseline_word))

    S = baseline_word[:]
    best_S = S[:]
    best_len = len(S)

    # --- 2. Local Search: Removal Passes with efficient updates ---
    
    def build_counts(word, n_val):
        counts = {}
        for i in range(len(word) - n_val + 1):
            p = tuple(word[i:i+n_val])
            counts[p] = counts.get(p, 0) + 1
        return counts

    def get_critical_indices(counts, word, n_val):
        crit = set()
        for i in range(len(word) - n_val + 1):
            p = tuple(word[i:i+n_val])
            if counts[p] == 1:
                crit.add(i)
        return crit

    counts = build_counts(S, n)
    critical_indices = get_critical_indices(counts, S, n)
    
    def build_coverage(critical_idx, word_len, n_val):
        cov = [0] * word_len
        for j in critical_idx:
            for pos in range(j, min(j + n_val, word_len)):
                cov[pos] += 1
        return cov

    coverage = build_coverage(critical_indices, len(S), n)
    
    stuck_passes = 0
    max_stuck_passes = 300
    
    while time.time() < deadline:
        removed_in_pass = 0
        L = len(S)
        
        for i in range(L - 1, -1, -1):
            if time.time() > deadline:
                break
                
            if coverage[i] > 0:
                continue
            
            # Valid removal
            old_start = max(0, i - n + 1)
            old_end = min(i, L - n)
            
            old_windows = []
            for j in range(old_start, old_end + 1):
                old_windows.append((j, tuple(S[j:j+n])))
            
            S.pop(i)
            L -= 1
            removed_in_pass += 1
            
            # Decrement old windows
            for j, p in old_windows:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # Increment new windows
            new_start = max(0, i - n)
            new_end = min(i - 1, L - n)
            for j in range(new_start, new_end + 1):
                p = tuple(S[j:j+n])
                counts[p] = counts.get(p, 0) + 1
            
            # Rebuild critical and coverage
            critical_indices = get_critical_indices(counts, S, n)
            coverage = build_coverage(critical_indices, L, n)
            
            best_S = S[:]
            best_len = L
            
        if removed_in_pass == 0:
            stuck_passes += 1
            if stuck_passes > max_stuck_passes:
                break
        else:
            stuck_passes = 0
            
    if best_len < baseline_len:
        return "".join(map(str, best_S))
    else:
        return "".join(map(str, baseline_word))
