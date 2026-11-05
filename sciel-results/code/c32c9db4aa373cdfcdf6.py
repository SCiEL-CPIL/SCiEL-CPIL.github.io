import itertools
import time
import random
import heapq
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Step 1: Build standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            w = tuple(word[i:i + m - 1])
            if len(set(w)) == m - 1 and w not in seen:
                seen.add(w)
                perms.append(w)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_ov = min(len(new_word), len(block))
            overlap = 0
            for k in range(max_ov, -1, -1):
                if k == 0 or new_word[len(new_word) - k:] == block[:k]:
                    overlap = k
                    break
            new_word += block[overlap:]
        word = new_word

    if time.time() > deadline - 15:
        return "".join(map(str, word))

    # Step 2: Optimization via priority-queue-based Greedy Deletion with randomized restarts
    
    best_word = list(word)
    best_len = len(word)
    
    def try_optimize(word_copy, time_limit):
        """Try to optimize the word using heap-based greedy deletion. Returns the optimized word."""
        word = list(word_copy)
        L = len(word)
        
        # Initialize freq
        freq = defaultdict(int)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            freq[p] += 1
        
        # Build initial heap of all deletable indices
        # Priority: -sum of freqs of windows covering the index (higher sum = better to delete first)
        heap = []
        for i in range(1, L - 1):
            lo = max(0, i - n + 1)
            hi = min(i, L - n)
            total_freq = 0
            can_del = True
            for j in range(lo, hi + 1):
                p = tuple(word[j:j+n])
                if freq[p] <= 1:
                    can_del = False
                    break
                total_freq += freq[p]
            if can_del:
                heapq.heappush(heap, (-total_freq, i))
        
        deleted = set()
        
        while heap and time.time() < time_limit:
            _, i = heapq.heappop(heap)
            
            if i in deleted or i <= 0 or i >= L - 1:
                continue
            
            # Verify it's still deletable
            lo = max(0, i - n + 1)
            hi = min(i, L - n)
            can_del = True
            for j in range(lo, hi + 1):
                p = tuple(word[j:j+n])
                if freq[p] <= 1:
                    can_del = False
                    break
            if not can_del:
                continue
            
            # Perform deletion
            deleted.add(i)
            
            # Update freq for windows that are removed/changed
            lo = max(0, i - n + 1)
            hi = min(i, L - n)
            for j in range(lo, hi + 1):
                p = tuple(word[j:j+n])
                freq[p] -= 1
                if freq[p] == 0:
                    del freq[p]
            
            # Remove word[i]
            word.pop(i)
            L -= 1
            
            # Update freq for new merged windows
            new_lo = max(0, i - n + 1)
            new_hi = min(i - 1, L - n)
            
            for k in range(new_lo, new_hi + 1):
                p = tuple(word[k:k+n])
                freq[p] += 1
            
            # Update heap for affected indices (neighbors of the deleted position)
            for idx in range(max(1, i - n), min(L - 1, i + n)):
                if idx in deleted:
                    continue
                lo = max(0, idx - n + 1)
                hi = min(idx, L - n)
                total_freq = 0
                can_del = True
                for j in range(lo, hi + 1):
                    p = tuple(word[j:j+n])
                    if freq[p] <= 1:
                        can_del = False
                        break
                    total_freq += freq[p]
                if can_del:
                    heapq.heappush(heap, (-total_freq, idx))
        
        return word
    
    # Multiple randomized restarts
    num_restarts = 5
    restart_time = (deadline - 30) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        if r == 0:
            word_copy = list(word)
        else:
            # Add some randomization: shuffle small portions
            word_copy = list(best_word)
            if len(word_copy) > 2*n:
                for _ in range(min(50, len(word_copy) // 100)):
                    pos = random.randint(n, len(word_copy) - n - 1)
                    length = random.randint(2, min(5, len(word_copy) - 2*n))
                    segment = word_copy[pos:pos+length]
                    random.shuffle(segment)
                    word_copy[pos:pos+length] = segment
        
        time_limit = time.time() + restart_time
        optimized = try_optimize(word_copy, time_limit)
        
        if len(optimized) < best_len:
            best_word = optimized
            best_len = len(optimized)
    
    # Final sweep with sequential passes on the best result
    if time.time() < deadline - 15:
        word = list(best_word)
        L = len(word)
        freq = defaultdict(int)
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            freq[p] += 1
        
        passes = 0
        while time.time() < deadline - 10 and passes < 50:
            changed = False
            # Left to Right
            i = 1
            while i < L - 1:
                lo = max(0, i - n + 1)
                hi = min(i, L - n)
                can_del = True
                for j in range(lo, hi + 1):
                    p = tuple(word[j:j+n])
                    if freq[p] <= 1:
                        can_del = False
                        break
                if can_del:
                    # Delete
                    lo = max(0, i - n + 1)
                    hi = min(i, L - n)
                    for j in range(lo, hi + 1):
                        p = tuple(word[j:j+n])
                        freq[p] -= 1
                        if freq[p] == 0:
                            del freq[p]
                    word.pop(i)
                    L -= 1
                    new_lo = max(0, i - n + 1)
                    new_hi = min(i - 1, L - n)
                    for k in range(new_lo, new_hi + 1):
                        p = tuple(word[k:k+n])
                        freq[p] += 1
                    changed = True
                else:
                    i += 1
            
            if not changed:
                # Right to Left
                i = L - 2
                while i > 0:
                    lo = max(0, i - n + 1)
                    hi = min(i, L - n)
                    can_del = True
                    for j in range(lo, hi + 1):
                        p = tuple(word[j:j+n])
                        if freq[p] <= 1:
                            can_del = False
                            break
                    if can_del:
                        lo = max(0, i - n + 1)
                        hi = min(i, L - n)
                        for j in range(lo, hi + 1):
                            p = tuple(word[j:j+n])
                            freq[p] -= 1
                            if freq[p] == 0:
                                del freq[p]
                        word.pop(i)
                        L -= 1
                        new_lo = max(0, i - n + 1)
                        new_hi = min(i - 1, L - n)
                        for k in range(new_lo, new_hi + 1):
                            p = tuple(word[k:k+n])
                            freq[p] += 1
                        changed = True
                    else:
                        i -= 1
            
            if not changed:
                break
            passes += 1
        
        if len(word) < best_len:
            best_word = word
            best_len = len(word)

    # Final Verification
    final_perms = set()
    for i in range(len(best_word) - n + 1):
        final_perms.add(tuple(best_word[i:i+n]))
    
    required = set(itertools.permutations(range(1, n+1)))
    if final_perms != required:
        # Fallback to original if broken
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - (m - 1) + 1):
                w = tuple(word[i:i + m - 1])
                if len(set(w)) == m - 1 and w not in seen:
                    seen.add(w)
                    perms.append(w)
            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ov = min(len(new_word), len(block))
                overlap = 0
                for k in range(max_ov, -1, -1):
                    if k == 0 or new_word[len(new_word) - k:] == block[:k]:
                        overlap = k
                        break
                new_word += block[overlap:]
            word = new_word
        return "".join(map(str, word))

    return "".join(map(str, best_word))
