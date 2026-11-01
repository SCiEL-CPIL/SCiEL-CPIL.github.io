import time
import random
import itertools
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Recursive superpermutation construction
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Precompute all permutations
    all_perms = set(itertools.permutations(range(1, n+1)))
    
    # Track permutation counts in current word
    perm_count = defaultdict(int)
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        perm_count[p] += 1
    
    # Set of "lost" permutations (count == 0)
    lost = set()
    for p in all_perms:
        if perm_count.get(p, 0) == 0:
            lost.add(p)
    
    def compute_old_windows(i):
        """Compute windows in old word that cover position i."""
        L = len(word)
        if L <= n:
            return []
        start_j = max(0, i - n + 1)
        end_j = min(i, L - n)
        return [tuple(word[j:j+n]) for j in range(start_j, end_j + 1)]
    
    def compute_new_windows(i):
        """Compute windows in new word (after deleting i) that cover the junction."""
        L = len(word)
        if L <= n:
            return []
        # New word length is L-1
        # Windows in new word covering the junction at index i
        # (in new word, elements 0..i-1 are same, elements i..L-2 are old i+1..L-1)
        start_j_new = max(0, i - n + 1)
        end_j_new = min(i, (L - 1) - n)
        new_windows = []
        for j in range(start_j_new, end_j_new + 1):
            new_w = []
            for k in range(n):
                idx = j + k
                if idx < i:
                    new_w.append(word[idx])
                else:
                    new_w.append(word[idx + 1])
            new_windows.append(tuple(new_w))
        return new_windows
    
    def try_delete(i):
        """Attempt to delete word[i]. Returns True if successful."""
        L = len(word)
        if L <= n:
            return False
        
        old_windows = compute_old_windows(i)
        new_windows = compute_new_windows(i)
        
        # Tentatively apply
        for p in old_windows:
            perm_count[p] -= 1
            if perm_count[p] == 0:
                lost.add(p)
        
        for p in new_windows:
            if p in lost:
                lost.remove(p)
            perm_count[p] += 1
        
        if len(lost) > 0:
            # Revert
            for p in new_windows:
                perm_count[p] -= 1
                if perm_count[p] == 0:
                    lost.add(p)
            for p in old_windows:
                if p in lost:
                    lost.remove(p)
                perm_count[p] += 1
            return False
        else:
            word.pop(i)
            return True
    
    # Main deletion loop with randomization to escape local minima
    best_len = len(word)
    best_word = list(word)
    
    while time.time() < deadline:
        progress = False
        L = len(word)
        
        # Scan backwards (deletions don't affect indices < i)
        for i in range(L - 1, -1, -1):
            if time.time() > deadline:
                break
            if try_delete(i):
                progress = True
                if len(word) < best_len:
                    best_len = len(word)
                    best_word = list(word)
        
        if not progress:
            # Try randomized restarts to escape local minima
            stuck = 0
            while time.time() < deadline and stuck < 200:
                i = random.randint(0, len(word) - 1)
                if try_delete(i):
                    stuck = 0
                    if len(word) < best_len:
                        best_len = len(word)
                        best_word = list(word)
                else:
                    stuck += 1
            if len(word) >= best_len:
                # Restore best word
                word = best_word
                # Rebuild counts
                perm_count = defaultdict(int)
                for i in range(len(word) - n + 1):
                    p = tuple(word[i:i+n])
                    perm_count[p] += 1
                lost = set()
                for p in all_perms:
                    if perm_count.get(p, 0) == 0:
                        lost.add(p)
            break
    
    # Final check
    if len(word) < best_len:
        best_word = list(word)
    
    return "".join(map(str, best_word))
