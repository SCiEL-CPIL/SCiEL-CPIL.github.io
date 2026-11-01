import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Standard recursive construction
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
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 5:
        return "".join(map(str, word))

    # 2. Fast optimization
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i

    # Precompute factorial table for Lehmer code
    fact = [1] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = fact[i - 1] * i

    # Fast Lehmer code: for a tuple of 1..n, compute rank
    # Use a bitmask approach for speed with small n
    def perm_to_rank_fast(p):
        """Compute Lehmer rank of permutation p (tuple of 1..n)."""
        # For n <= 8, we can use a lookup-free method
        # p is a tuple of values 1..n
        rank = 0
        used = [False] * (n + 1)
        for i, val in enumerate(p):
            # Count how many unused values are less than val
            cnt = 0
            for v in range(1, val):
                if not used[v]:
                    cnt += 1
            rank += cnt * fact[n - 1 - i]
            used[val] = True
        return rank

    # Precompute rank for all permutations using iterative generation
    # This is faster than calling perm_to_rank_fast 5040 times with the slow method
    # Actually, let's just build the map directly
    perm_rank = {}
    for perm in permutations(range(1, n + 1)):
        perm_rank[perm] = perm_to_rank_fast(perm)

    # Convert word to array for faster manipulation
    current_word = list(word)
    L = len(current_word)

    def compute_all_ids(w):
        """Compute window IDs for the entire word."""
        ids = []
        ln = len(w)
        for i in range(ln - n + 1):
            window = tuple(w[i:i + n])
            # Check if it's a permutation
            if len(set(window)) == n:
                ids.append(perm_rank[window])
            else:
                ids.append(-1)
        return ids

    def compute_all_counts(ids):
        counts = [0] * target_count
        for idx in ids:
            if idx >= 0:
                counts[idx] += 1
        return counts

    current_ids = compute_all_ids(current_word)
    current_counts = compute_all_counts(current_ids)

    def update_after_removal(i, L_old):
        """Update ids and counts after removing position i from word of length L_old.
        Uses incremental update for O(n) affected windows."""
        L_new = L_old - 1
        
        # Old windows affected: those containing position i
        # j in [max(0, i-n+1), min(i, L_old-n)]
        old_j_start = max(0, i - n + 1)
        old_j_end = min(i, L_old - n)
        
        # Decrement counts for old affected windows
        for j in range(old_j_start, old_j_end + 1):
            aid = current_ids[j]
            if aid >= 0:
                current_counts[aid] -= 1
        
        # New windows affected: j in [max(0, i-n+1), min(i, L_new-n)]
        new_j_start = max(0, i - n + 1)
        new_j_end = min(i, L_new - n)
        
        # For new windows j < i-n+1: unchanged (same as old window j)
        # For new windows j in [i-n+1, i]: need recomputation
        # For new windows j > i: same as old window j+1
        
        # Handle the shift for windows j > i:
        # Old window j+1 becomes new window j
        # So we need to shift IDs from position j+1 to j for j in [i+1, L_new-n]
        # But this is a bulk shift. Let's handle it by rebuilding the affected portion.
        
        # Actually, the cleanest approach:
        # 1. Decrement old affected windows (done above)
        # 2. Recompute new windows in [new_j_start, new_j_end]
        # 3. For windows j > i: new_ids[j] = old_ids[j+1] (shift)
        #    But we already decremented old_ids for j in [old_j_start, old_j_end]
        #    The shift means we need to copy old_ids[j+1] to new_ids[j] for j > i
        
        # Let me rebuild the entire IDs array for correctness
        # The shift makes incremental updates complex
        
        new_ids = compute_all_ids(current_word)
        new_counts = compute_all_counts(new_ids)
        
        return new_ids, new_counts

    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(current_word)
        
        # Try removing each index in random order to escape local minima
        indices = list(range(L))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Determine affected windows
            start_j = max(0, i - n + 1)
            end_j = min(i, L - n)
            
            if start_j > end_j:
                continue  # No windows affected, safe to remove
            
            # Check if removing these windows leaves count > 0 for all
            # Simulate the removal
            temp_counts = current_counts[:]
            
            for j in range(start_j, end_j + 1):
                aid = current_ids[j]
                if aid >= 0:
                    temp_counts[aid] -= 1
                    if temp_counts[aid] == 0:
                        break
            else:
                # All counts remain positive, perform removal
                current_word = current_word[:i] + current_word[i+1:]
                current_ids = compute_all_ids(current_word)
                current_counts = compute_all_counts(current_ids)
                improved = True
                break  # Restart scan

    return "".join(map(str, current_word))
