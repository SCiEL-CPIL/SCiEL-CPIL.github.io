import time
import random
import itertools
import sys
from collections import defaultdict

def _standard_superperm(n):
    """Standard recursive superpermutation construction."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    word = _standard_superperm(n - 1)
    # Extract all (n-1)-permutations from word
    perms = []
    seen = set()
    for i in range(len(word) - (n - 2)):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    new_word = []
    for p in perms:
        block = list(p) + [n] + list(p)
        # Find max overlap between end of new_word and start of block
        max_ov = 0
        for ov in range(min(len(new_word), len(block)), -1, -1):
            if ov == 0:
                max_ov = 0
                break
            if new_word[-ov:] == block[:ov]:
                max_ov = ov
                break
        new_word.extend(block[max_ov:])
    return new_word

def _find_superperm_5(time_limit=15):
    """Find a short superpermutation for n=5 using greedy random restarts."""
    n = 5
    symbols = list(range(1, n + 1))
    perms = [tuple(p) for p in itertools.permutations(symbols)]
    perm_set = set(perms)
    best_len = float('inf')
    best_word = None
    start_time = time.time()
    
    # Precompute: for each suffix (last k symbols), which perms start with that prefix?
    # We'll build a trie of permutations for faster overlap search
    # For n=5, we can use a dictionary mapping prefix -> set of perms starting with it
    
    def build_prefix_index(perms):
        prefix_map = defaultdict(set)
        for p in perms:
            for k in range(1, n + 1):
                prefix_map[p[:k]].add(p)
        return prefix_map
    
    prefix_map = build_prefix_index(perms)
    
    def greedy_start(start_perm):
        word = list(start_perm)
        seen = set()
        # Add perms in current word
        for i in range(len(word) - n + 1):
            seen.add(tuple(word[i:i+n]))
        remaining = perm_set - seen
        if not remaining:
            return word
            
        # For efficiency, maintain a list of remaining perms
        remaining_list = list(remaining)
        
        while remaining:
            # Find the permutation with maximum overlap using prefix_map
            best_overlap = -1
            best_next = None
            best_future_cost = float('inf')
            
            # Try overlaps from max to min
            max_possible_ov = min(len(word), n)
            for ov in range(max_possible_ov, 0, -1):
                suffix = tuple(word[-ov:])
                candidates = prefix_map.get(suffix, set())
                # Filter to only remaining perms
                candidates_remaining = [p for p in candidates if p in remaining]
                if not candidates_remaining:
                    continue
                
                # For each candidate, compute a lookahead score
                # We prefer candidates that leave fewer "easy" options for the future
                # Specifically, we want to minimize the number of remaining perms
                # that share the new suffix after appending this perm
                
                # For the best candidate with this overlap, we pick the one
                # that minimizes the branching factor for the next step
                best_cand_for_ov = None
                best_cand_score = float('inf')
                
                for p in candidates_remaining:
                    # After appending p with overlap ov, the new suffix will be
                    # the last (n-1) symbols of p (since we're building a superperm)
                    # Actually, the new suffix of length n-1 is p[n-ov:]
                    # But for lookahead, we care about what perms start with
                    # the new suffix of various lengths
                    
                    # Simplified lookahead: count how many remaining perms
                    # start with the last 3 symbols of p (a good balance)
                    if len(p) >= 3:
                        new_suffix3 = p[-3:]
                        future_count = len([q for q in remaining if q != p and q[:3] == new_suffix3])
                    else:
                        future_count = len(remaining)
                    
                    if future_count < best_cand_score:
                        best_cand_score = future_count
                        best_cand_for_ov = p
                
                if best_cand_for_ov is not None:
                    # This overlap is achievable, and we have a good candidate
                    best_overlap = ov
                    best_next = best_cand_for_ov
                    best_future_cost = best_cand_score
                    break  # We found the maximum overlap, so we're done
            
            if best_next is None:
                # No overlap found, pick a random remaining perm
                best_next = random.choice(remaining_list)
                best_overlap = 0
                
            # Append best_next with overlap
            word.extend(best_next[best_overlap:])
            remaining.remove(best_next)
            remaining_list.remove(best_next)
            
            # Update seen and remove newly seen from remaining
            # Only need to check the new windows
            new_windows = []
            for i in range(len(word) - n, len(word) - n + 1):
                if i >= 0:
                    w = tuple(word[i:i+n])
                    new_windows.append(w)
                    if w in remaining:
                        remaining.remove(w)
                        remaining_list.remove(w)
                    
        return word

    # Try a few deterministic starts first?
    # Or just random restarts.
    while time.time() - start_time < time_limit:
        start_perm = tuple(random.sample(symbols, n))
        word = greedy_start(start_perm)
        if len(word) < best_len:
            best_len = len(word)
            best_word = word
            if best_len == 152: # Known optimal
                break
                
    if best_word is None:
        # Fallback to standard
        return _standard_superperm(5)
        
    return best_word

def _extend_superperm(word, m, time_limit=10):
    """Extend a superpermutation of 1..m-1 to 1..m using greedy heuristic."""
    n = m
    start_time = time.time()
    
    # Extract all (n-1)-permutations from word
    perms = []
    seen = set()
    for i in range(len(word) - (n - 2)):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # For each (n-1)-perm, we need to insert the new symbol n
    # The standard construction adds: p + [n] + p
    # But we can be smarter about ordering and overlap
    
    # Build a trie of the (n-1)-perms for fast overlap checking
    # Actually, for extension, the key insight is:
    # We have a sequence of (n-1)-perms from the base superperm.
    # We need to create an n-superperm by "expanding" each (n-1)-perm
    # into a block that contains n.
    
    # The standard method: for each (n-1)-perm p in order, add p + [n] + p
    # with maximum overlap with current word.
    
    # Improved: use lookahead to order the blocks
    # We want to minimize the total length, which means maximizing overlaps.
    
    # Let's use a greedy approach with lookahead:
    # At each step, choose the next block that maximizes overlap with current suffix,
    # and among ties, prefer the one that leaves more "good" options for the future.
    
    remaining = list(perms)
    new_word = []
    
    # Precompute: for each suffix of length k, which blocks start with it?
    # A block is p + [n] + p, so its prefix is p
    # We can index by the (n-1)-perm prefix
    
    while remaining:
        if time.time() - start_time > time_limit:
            # Fallback to standard method for remaining
            for p in remaining:
                block = list(p) + [n] + list(p)
                max_ov = 0
                for ov in range(min(len(new_word), len(block)), 0, -1):
                    if new_word[-ov:] == block[:ov]:
                        max_ov = ov
                        break
                new_word.extend(block[max_ov:])
            break
        
        best_overlap = -1
        best_block = None
        best_future_cost = float('inf')
        
        # Try to find the block with maximum overlap
        # The block starts with p, so we check if new_word ends with p's prefix
        max_possible_ov = min(len(new_word), 2 * (n - 1) + 1)
        
        for ov in range(max_possible_ov, 0, -1):
            if ov > len(new_word):
                continue
            suffix = tuple(new_word[-ov:])
            
            # Check which blocks have this as a prefix
            candidates = []
            for p in remaining:
                block = list(p) + [n] + list(p)
                if tuple(block[:ov]) == suffix:
                    candidates.append(p)
            
            if not candidates:
                continue
            
            # Among candidates with this overlap, pick the one with best lookahead
            best_cand = None
            best_cand_score = float('inf')
            
            for p in candidates:
                # After appending this block, what's the new suffix?
                # The new suffix will be the last (n-1) symbols of the block
                # which is p (the second copy)
                # For lookahead, we care about how many remaining blocks
                # start with this new suffix
                
                new_suffix = p  # The block ends with p
                future_count = len([q for q in remaining if q != p and q == new_suffix])
                
                if future_count < best_cand_score:
                    best_cand_score = future_count
                    best_cand = p
            
            if best_cand is not None:
                best_overlap = ov
                best_block = best_cand
                best_future_cost = best_cand_score
                break
        
        if best_block is None:
            # No overlap found, pick a random remaining perm
            best_block = random.choice(remaining)
            best_overlap = 0
        
        # Append the block with overlap
        block = list(best_block) + [n] + list(best_block)
        new_word.extend(block[best_overlap:])
        remaining.remove(best_block)
    
    return new_word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    if n <= 4:
        return "".join(map(str, _standard_superperm(n)))
        
    if n == 5:
        word = _find_superperm_5(10)
        return "".join(map(str, word))
        
    if n == 6 or n == 7:
        # Use best S5 as base
        s5 = _find_superperm_5(10)
        word = list(s5)
        
        # Recursive extension with time budget
        for m in range(6, n + 1):
            remaining_time = deadline - time.time() - 5
            if remaining_time < 1:
                break
            word = _extend_superperm(word, m, time_limit=remaining_time)
        
        return "".join(map(str, word))
        
    # Fallback for n > 7
    word = _standard_superperm(n)
    return "".join(map(str, word))

if __name__ == "__main__":
    # Quick test
    print(construct(3))
    print(len(construct(3)))
    # Test n=5 to verify it works
    if len(sys.argv) > 1 and sys.argv[1] == "test5":
        word = construct(5)
        print("n=5 length:", len(word))
        # Verify all permutations are present
        perms = set()
        for i in range(len(word) - 4):
            perms.add(word[i:i+5])
        print("Unique perms found:", len(perms))
        print("Expected:", 120)
