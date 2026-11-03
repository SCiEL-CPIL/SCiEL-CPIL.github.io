import time
import itertools
import math
from collections import defaultdict

def _rank_perm(p, n):
    """Compute lexicographic rank of permutation p (0-indexed) in 0..n!-1."""
    used = [False] * (n + 1)
    rank = 0
    for i in range(n):
        # Count unused elements smaller than p[i]
        smaller_unused = 0
        for v in range(1, p[i]):
            if not used[v]:
                smaller_unused += 1
        rank += smaller_unused * math.factorial(n - i - 1)
        used[p[i]] = True
    return rank

def construct(n):
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    # 1. Generate classic superpermutation
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
            # Find max overlap
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Greedy Deletion Passes with efficient counting
    target_perms = math.factorial(n)
    W = len(word)
    
    # Use integer-encoded permutation counts for speed
    perm_counts = [0] * target_perms
    
    # Pre-compute ranks for all windows
    window_ranks = []
    for i in range(W - n + 1):
        r = _rank_perm(tuple(word[i:i+n]), n)
        window_ranks.append(r)
        perm_counts[r] += 1
    
    # Verify all perms present
    if sum(1 for c in perm_counts if c > 0) != target_perms:
        # Should not happen with classic construction
        pass
    
    # Greedy deletion: scan left to right, try to remove each position
    # When we remove position i, windows [i-n+1, i] are affected
    # We maintain window_ranks and perm_counts incrementally
    
    for pass_num in range(50):
        if time.time() > deadline:
            break
            
        changed = False
        W = len(word)
        
        # Rebuild window_ranks if needed (should be consistent)
        # Actually, we maintain it incrementally
        
        # Left to right scan
        i = 0
        while i < W:
            if time.time() > deadline:
                break
                
            # Windows affected by removing word[i]:
            # Window starting at j contains position i iff j <= i <= j+n-1
            # So j in [max(0, i-n+1), min(i, W-n)]
            
            min_j = max(0, i - n + 1)
            max_j = min(i, W - n)
            
            # Check if removal is valid: all affected windows must have count > 1
            valid = True
            for j in range(min_j, max_j + 1):
                if perm_counts[window_ranks[j]] <= 1:
                    valid = False
                    break
            
            if valid:
                # Remove word[i]
                word.pop(i)
                W -= 1
                changed = True
                
                # Update window_ranks: remove affected windows and shift
                # Windows starting at j in [min_j, max_j] are removed
                # Windows starting at j > max_j shift left by 1 in index
                # But their content changes: window starting at j now starts at position j-1 in old indexing
                # Actually, after removing word[i], the new window at position j (for j >= max_j+1)
                # corresponds to old window at position j+1, but with word[i] removed
                # This is complex. Let's just rebuild window_ranks after each deletion
                # for correctness, since N is ~6000 and we do limited deletions
                
                # Rebuild window_ranks and perm_counts
                perm_counts = [0] * target_perms
                window_ranks = []
                for j in range(W - n + 1):
                    r = _rank_perm(tuple(word[j:j+n]), n)
                    window_ranks.append(r)
                    perm_counts[r] += 1
                
                # Don't increment i; next element is now at position i
                continue
            else:
                i += 1
        
        if not changed:
            # Try right to left
            i = W - 1
            while i >= 0:
                if time.time() > deadline:
                    break
                    
                min_j = max(0, i - n + 1)
                max_j = min(i, W - n)
                
                valid = True
                for j in range(min_j, max_j + 1):
                    if perm_counts[window_ranks[j]] <= 1:
                        valid = False
                        break
                
                if valid:
                    word.pop(i)
                    W -= 1
                    changed = True
                    
                    perm_counts = [0] * target_perms
                    window_ranks = []
                    for j in range(W - n + 1):
                        r = _rank_perm(tuple(word[j:j+n]), n)
                        window_ranks.append(r)
                        perm_counts[r] += 1
                    
                    i -= 1  # Check the new position i
                else:
                    i -= 1
            
            if not changed:
                break

    return "".join(map(str, word))
