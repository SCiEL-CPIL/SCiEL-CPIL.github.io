"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    """
    Constructs the shortest superpermutation for n symbols.
    Uses the standard recursive construction followed by local search
    to remove redundant characters.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    if n == 1:
        return "1"
    
    # Step 1: Build the standard recursive construction
    word = _build_recursive(n)
    
    # Step 2: Verify it's a valid superpermutation
    if not _is_valid_superperm(word, n):
        # Fallback: return what we have
        return "".join(map(str, word))
    
    # Step 3: Local search to remove redundant characters
    word = _local_search(word, n, deadline)
    
    return "".join(map(str, word))


def _build_recursive(n):
    """
    Builds the standard recursive superpermutation:
    SP(1) = "1"
    SP(m) = concat over all (m-1)-perms p in order of appearance in SP(m-1):
             p + [m] + p
    with maximum overlap between consecutive blocks.
    """
    if n == 1:
        return [1]
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap
            max_possible_overlap = min(len(new_word), len(block), m - 1)
            overlap = 0
            for k in range(max_possible_overlap, -1, -1):
                if new_word[len(new_word) - k:] == block[:k]:
                    overlap = k
                    break
            
            new_word.extend(block[overlap:])
        
        word = new_word
    
    return word


def _is_valid_superperm(word, n):
    """Check if word contains every permutation of 1..n as a contiguous subsequence."""
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    
    # Check if we have all n! permutations
    from itertools import permutations
    all_perms = set(permutations(range(1, n + 1)))
    return seen == all_perms


def _local_search(word, n, deadline):
    """
    Local search to remove redundant characters from the superpermutation.
    A character can be removed if no permutation becomes uncovered.
    """
    random.seed(42)
    
    # Build index mapping from permutation tuple to index
    from itertools import permutations
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Build coverage counts
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            idx = perm_to_idx.get(window)
            if idx is not None:
                counts[idx] += 1
    
    best_word = list(word)
    best_len = len(word)
    
    # Multiple passes of removal
    for iteration in range(50000):
        if time.time() > deadline - 5.0:
            break
        
        # Try to remove a random character
        found_removal = False
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            if len(word) <= n:
                break
            
            pos = random.randint(0, len(word) - 1)
            
            # Find affected windows (those that include position pos)
            start_lo = max(0, pos - n + 1)
            start_hi = min(pos, len(word) - n)
            
            if start_lo > start_hi:
                continue
            
            affected_indices = []
            for s in range(start_lo, start_hi + 1):
                window = tuple(word[s:s + n])
                if len(set(window)) == n:
                    idx = perm_to_idx.get(window)
                    if idx is not None:
                        affected_indices.append(idx)
            
            # Check if any affected permutation would lose its last coverage
            safe = True
            for idx in affected_indices:
                if counts[idx] == 1:
                    safe = False
                    break
            
            if safe:
                # Remove the character
                word.pop(pos)
                for idx in affected_indices:
                    counts[idx] -= 1
                found_removal = True
                if len(word) < best_len:
                    best_word = list(word)
                    best_len = len(word)
                break
        
        if not found_removal:
            # Try a different strategy: swap two adjacent characters
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                
                if len(word) < 2:
                    break
                
                pos = random.randint(0, len(word) - 2)
                
                # Compute affected windows before swap
                start_lo = max(0, pos - n + 1)
                start_hi = min(pos + 1, len(word) - n)
                
                if start_lo > start_hi:
                    continue
                
                before_counts = {}
                for s in range(start_lo, start_hi + 1):
                    window = tuple(word[s:s + n])
                    if len(set(window)) == n:
                        idx = perm_to_idx.get(window)
                        if idx is not None:
                            before_counts[idx] = before_counts.get(idx, 0) + 1
                
                # Perform swap
                word[pos], word[pos + 1] = word[pos + 1], word[pos]
                
                # Compute affected windows after swap
                after_counts = {}
                for s in range(start_lo, start_hi + 1):
                    window = tuple(word[s:s + n])
                    if len(set(window)) == n:
                        idx = perm_to_idx.get(window)
                        if idx is not None:
                            after_counts[idx] = after_counts.get(idx, 0) + 1
                
                # Check if swap is safe: no permutation goes from 1 to 0
                safe = True
                for idx in before_counts:
                    if before_counts[idx] == 1 and after_counts.get(idx, 0) == 0:
                        safe = False
                        break
                
                if not safe:
                    # Revert swap
                    word[pos], word[pos + 1] = word[pos + 1], word[pos]
                else:
                    # Update counts
                    for idx in before_counts:
                        counts[idx] -= before_counts[idx]
                    for idx in after_counts:
                        counts[idx] += after_counts[idx]
                    improved = True
                    break
            
            if not improved:
                # No improvement possible in this iteration; break to avoid infinite loop
                # But first, try a few more random removals
                for _ in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    if len(word) <= n:
                        break
                    
                    pos = random.randint(0, len(word) - 1)
                    start_lo = max(0, pos - n + 1)
                    start_hi = min(pos, len(word) - n)
                    
                    if start_lo > start_hi:
                        continue
                    
                    affected_indices = []
                    for s in range(start_lo, start_hi + 1):
                        window = tuple(word[s:s + n])
                        if len(set(window)) == n:
                            idx = perm_to_idx.get(window)
                            if idx is not None:
                                affected_indices.append(idx)
                    
                    safe = True
                    for idx in affected_indices:
                        if counts[idx] == 1:
                            safe = False
                            break
                    
                    if safe:
                        word.pop(pos)
                        for idx in affected_indices:
                            counts[idx] -= 1
                        if len(word) < best_len:
                            best_word = list(word)
                            best_len = len(word)
                        break
                else:
                    # Truly stuck
                    break
    
    return best_word
# EVOLVE-BLOCK-END
