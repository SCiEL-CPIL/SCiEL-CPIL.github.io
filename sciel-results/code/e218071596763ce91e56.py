"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random
from math import factorial
from itertools import permutations

def construct(n):
    deadline = time.time() + 145  # 5-second safety margin
    
    if n <= 1:
        return "1" if n == 1 else ""
    if n <= 3:
        # Brute-force search for small n
        perms = list(permutations(range(1, n + 1)))
        best = None
        best_len = n * factorial(n) + 1
        for start in perms:
            s = list(start)
            remaining = set(perms)
            remaining.discard(start)
            while remaining:
                if time.time() > deadline:
                    break
                last = tuple(s[-(n - 1):])
                best_next = None
                best_count = -1
                for sym in range(1, n + 1):
                    new_win = last[1:] + (sym,)
                    cnt = 0
                    for p in remaining:
                        if p[:n - 1] == new_win:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_next = sym
                if best_next is None:
                    break
                s.append(best_next)
                if len(s) >= n:
                    win = tuple(s[-n:])
                    if len(set(win)) == n:
                        remaining.discard(win)
            if len(s) < best_len and len(s) >= n:
                if _verify_superperm(s, n):
                    best_len = len(s)
                    best = s
        if best is not None:
            return "".join(map(str, best))
        word = _classic_superperm(n)
        return "".join(map(str, word))
    
    # For n >= 4, use classic construction as baseline
    word = _classic_superperm(n)
    best_word = list(word)
    best_len = len(word)
    
    # Try to improve with local search / jumping
    if time.time() < deadline - 5:
        improved = _improve_by_jumping(word, n, deadline)
        if improved is not None and len(improved) < best_len:
            best_word = improved
            best_len = len(improved)
    
    # For n=5, try randomized greedy to find optimal 152
    if n == 5 and time.time() < deadline - 10:
        result = _try_greedy_n5(deadline)
        if result is not None and len(result) < best_len:
            best_word = result
            best_len = len(result)
    
    return "".join(map(str, best_word))

def _classic_superperm(n):
    word = [1]
    for m in range(2, n + 1):
        perms_list, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms_list.append(window)
        new = []
        for p in perms_list:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o > 0 and new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _verify_superperm(word, n):
    """Check if word contains all n! permutations of 1..n as consecutive windows."""
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == factorial(n)

def _improve_by_jumping(word, n, deadline):
    """Try to reduce length by finding and removing redundant insertions."""
    word = list(word)
    best_word = list(word)
    best_len = len(word)
    
    # Find all positions of symbol n
    n_positions = [i for i, x in enumerate(word) if x == n]
    
    # For each position of n, check if removing it still covers all perms
    for pos in reversed(n_positions):
        if time.time() > deadline - 2:
            break
        if pos < n - 1 or pos >= len(word) - (n - 1):
            continue
        
        # Try removing word[pos]
        test_word = word[:pos] + word[pos+1:]
        
        # Quick check: does the window around pos still form valid perms?
        removed_win = tuple(word[pos-n+1:pos+1]) if pos >= n-1 else None
        if removed_win is None:
            continue
        if len(set(removed_win)) != n:
            continue
        
        # Verify the test word still contains all perms
        if _verify_superperm(test_word, n):
            word = test_word
            best_word = list(word)
            best_len = len(word)
            # Update n_positions
            n_positions = [i for i, x in enumerate(word) if x == n]
    
    return best_word

def _try_greedy_n5(deadline):
    """Try to find a 152-length superpermutation for n=5 using randomized greedy."""
    all_perms = set(permutations(range(1, 6)))
    target_len = 152
    symbols = range(1, 6)
    
    for attempt in range(200):
        if time.time() >= deadline - 5:
            break
        
        remaining = set(all_perms)
        # Start with a random permutation
        initial = list(random.choice(list(remaining)))
        remaining.discard(tuple(initial))
        current_word = list(initial)
        
        success = True
        while remaining and len(current_word) < target_len:
            if time.time() >= deadline - 5:
                success = False
                break
            
            best_next = None
            best_score = -1
            
            for x in symbols:
                new_word = current_word + [x]
                if len(new_word) > target_len:
                    continue
                
                score = 0
                if len(new_word) >= 5:
                    for i in range(max(0, len(new_word) - 7), len(new_word) - 4):
                        w = tuple(new_word[i:i+5])
                        if len(set(w)) == 5:
                            if w in remaining:
                                score += 10
                            else:
                                score -= 1
                
                if score > best_score:
                    best_score = score
                    best_next = x
            
            if best_next is None:
                best_next = random.choice(symbols)
            
            current_word.append(best_next)
            
            if len(current_word) >= 5:
                last_window = tuple(current_word[-5:])
                if last_window in remaining:
                    remaining.discard(last_window)
        
        if not remaining and len(current_word) <= target_len:
            # Verify it's actually a valid superpermutation
            seen_windows = set()
            for i in range(len(current_word) - 4):
                w = tuple(current_word[i:i+5])
                if len(set(w)) == 5:
                    seen_windows.add(w)
            if seen_windows == all_perms:
                return current_word
    
    return None
# EVOLVE-BLOCK-END
