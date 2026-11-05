import time
import random
from collections import Counter

def classic_superperm(n):
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
            # Find max overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def encode_perm(p):
    val = 0
    for i in range(len(p)):
        val = val * 10 + p[i]
    return val

def construct(n):
    if n < 7:
        return "".join(map(str, classic_superperm(n)))
    
    # Start with classic construction for n=7
    s = classic_superperm(n)
    
    # Precompute target count
    target_perms = 5040 # 7!
    
    # Initialize counts and set
    # Using a list of integers is faster than tuples for hashing?
    # Let's use integers encoded in base 8 or 10.
    # Since symbols are 1..7, base 10 is fine.
    
    def get_windows_values(s, start, end):
        vals = []
        for i in range(start, end + 1):
            if 0 <= i and i + n <= len(s):
                vals.append(encode_perm(s[i:i+n]))
        return vals

    # Build initial state
    counts = Counter()
    seen_set = set()
    for i in range(len(s) - n + 1):
        v = encode_perm(s[i:i+n])
        counts[v] += 1
        seen_set.add(v)
    
    current_len = len(s)
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    # Local Search
    # We want to minimize length.
    # Moves: Delete, Swap, Insert (Insert increases length, only if it enables deletions later? 
    # Usually we just delete and swap. Inserting is risky for time.
    # Let's stick to Delete and Swap.
    
    # To avoid getting stuck, we can use a simple restart or randomization.
    # But for n=7, the solution is close.
    
    while time.time() < deadline and current_len > 5905:
        # Try random deletions
        found_move = False
        # Sample a few random positions
        for _ in range(100):
            i = random.randint(0, len(s) - 1)
            
            # Check if deletion is possible
            # Affected windows: i-6 to i
            start_idx = max(0, i - n + 1)
            end_idx = min(i, len(s) - n)
            
            if start_idx > end_idx:
                continue
                
            old_vals = []
            for k in range(start_idx, end_idx + 1):
                old_vals.append(encode_perm(s[k:k+n]))
                
            # Remove old
            for v in old_vals:
                counts[v] -= 1
                if counts[v] == 0:
                    seen_set.discard(v)
                    
            # Perform deletion
            s.pop(i)
            
            # Add new
            # Windows now start at i-6 to i-1 (shifted)
            # The windows that were at i+1.. are now at i..
            # We only need to add windows that were "broken" and reformed.
            # The windows starting at i-6..i-1 are new/changed.
            # Note: windows starting at i+1.. in old s are now at i.. in new s.
            # But we already accounted for windows up to i in old s.
            # Let's re-evaluate the range.
            # Old s: len L.
            # Delete i. New s: len L-1.
            # Old windows: 0..L-n.
            # New windows: 0..L-1-n.
            # Windows 0..i-7 are unchanged.
            # Windows i-6..i-1 in NEW s correspond to windows i-6..i-1 in OLD s? No.
            # New s[i-6] = Old s[i-6].
            # New s[i-6 : i+1] = Old s[i-6 : i] + Old s[i+1 : i+7].
            # So we need to check windows starting at i-6, ..., i-1 in the NEW string.
            
            new_start_idx = max(0, i - n + 1)
            new_end_idx = min(i - 1, len(s) - n)
            
            for k in range(new_start_idx, new_end_idx + 1):
                v = encode_perm(s[k:k+n])
                counts[v] += 1
                if counts[v] == 1:
                    seen_set.add(v)
            
            if len(seen_set) < target_perms:
                # Revert
                s.insert(i, old_vals[0] % 10) # Wait, we need the actual char
                # We didn't save the char. Let's save it.
                # This logic is getting messy. Let's refactor.
                pass

    # The above inline logic is error prone. Let's write a cleaner helper.
    
    # Refactored approach:
    s = classic_superperm(n)
    counts = Counter()
    seen_set = set()
    for i in range(len(s) - n + 1):
        v = encode_perm(s[i:i+n])
        counts[v] += 1
        seen_set.add(v)
        
    current_len = len(s)
    start_time = time.time()
    deadline = start_time + 110
    
    def try_delete(i):
        nonlocal s, counts, seen_set, current_len
        if i < 0 or i >= len(s):
            return False
        # Save char
        char = s[i]
        
        # Affected old windows: start from i-n+1 to i
        old_starts = list(range(max(0, i - n + 1), min(i, len(s) - n) + 1))
        old_vals = [encode_perm(s[k:k+n]) for k in old_starts]
        
        # Remove old
        for v in old_vals:
            counts[v] -= 1
            if counts[v] == 0:
                seen_set.discard(v)
                
        # Delete
        s.pop(i)
        current_len -= 1
        
        # Affected new windows: start from i-n+1 to i-1
        new_starts = list(range(max(0, i - n + 1), min(i - 1, len(s) - n) + 1))
        new_vals = [encode_perm(s[k:k+n]) for k in new_starts]
        
        # Add new
        for v in new_vals:
            counts[v] += 1
            if counts[v] == 1:
                seen_set.add(v)
                
        if len(seen_set) < target_perms:
            # Revert
            s.insert(i, char)
            current_len += 1
            # Re-add old
            for v in old_vals:
                counts[v] += 1
                if counts[v] == 1:
                    seen_set.add(v)
            # Re-remove new
            for v in new_vals:
                counts[v] -= 1
                if counts[v] == 0:
                    seen_set.discard(v)
            return False
        return True

    def try_swap(i, j):
        nonlocal s, counts, seen_set
        if i == j:
            return False
        # Ensure i < j for simplicity, or just handle general
        if i > j:
            i, j = j, i
            
        # Affected windows: i-n+1 to j
        starts = list(range(max(0, i - n + 1), min(j, len(s) - n) + 1))
        old_vals = [encode_perm(s[k:k+n]) for k in starts]
        
        # Remove old
        for v in old_vals:
            counts[v] -= 1
            if counts[v] == 0:
                seen_set.discard(v)
                
        # Swap
        s[i], s[j] = s[j], s[i]
        
        # Add new
        new_vals = [encode_perm(s[k:k+n]) for k in starts]
        for v in new_vals:
            counts[v] += 1
            if counts[v] == 1:
                seen_set.add(v)
                
        if len(seen_set) < target_perms:
            # Revert
            s[i], s[j] = s[j], s[i]
            for v in old_vals:
                counts[v] += 1
                if counts[v] == 1:
                    seen_set.add(v)
            for v in new_vals:
                counts[v] -= 1
                if counts[v] == 0:
                    seen_set.discard(v)
            return False
        return True

    # Main loop
    while time.time() < deadline and current_len > 5905:
        # Try deletions
        for _ in range(50):
            i = random.randint(0, len(s) - 1)
            if try_delete(i):
                break
        else:
            # Try swaps to escape local minima
            for _ in range(50):
                i = random.randint(0, len(s) - 1)
                j = random.randint(0, len(s) - 1)
                if try_swap(i, j):
                    break

    return "".join(map(str, s))

