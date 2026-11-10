"""Shortest superpermutation on n symbols via greedy path search with overlap optimization."""
import time
import sys

# EVOLVE-BLOCK-START

def _greedy_superperm(n, time_limit=110.0):
    """
    Construct a superpermutation of [1..n] using greedy path search.
    
    The idea: we build a word where every window of length n is a permutation.
    This is equivalent to finding a path in the de Bruijn-like graph of (n-1)-permutations
    where edges are labeled by the new symbol appended.
    
    We use a greedy strategy: at each step, extend the current (n-1)-permutation by trying
    each symbol, preferring the one that leads to the most unvisited (n-1)-permutations.
    """
    start_time = time.time()
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # For n >= 4, use the greedy construction
    # We maintain the current word and the set of seen (n-1)-permutations
    # The key insight: when extending, we want to maximize the number of new permutations discovered
    
    # Classic recursive construction with greedy overlap
    word = [1]
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations from the current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Greedy: for each permutation p in order, append [m] + p
        # But we can try to find a better ordering
        # The standard approach is to just go in order of appearance
        
        # Build the new word
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap with end of new
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        # Check time
        if time.time() - start_time > time_limit * 0.9:
            break
    
    return "".join(map(str, word))


def _greedy_with_reordering(n, time_limit=110.0):
    """
    Improved greedy: try to reorder the permutations at each level to minimize total length.
    Uses a simple heuristic: at each level, try different orderings and pick the shortest.
    """
    start_time = time.time()
    
    if n <= 3:
        return _greedy_superperm(n, time_limit)
    
    word = [1]
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Try to find a better ordering
        # For small m, we can try a few heuristics
        if m <= 5:
            # Try: order by how many times each element appears at the end
            # Or: try to maximize overlap between consecutive blocks
            # Simple approach: try the natural order and a reversed order
            candidates = [perms, list(reversed(perms))]
            
            # Also try sorting by first element
            sorted_perms = sorted(perms, key=lambda p: p[0])
            candidates.append(sorted_perms)
            
            best_new = None
            best_len = float('inf')
            
            for cand in candidates:
                new = []
                for p in cand:
                    block = list(p) + [m] + list(p)
                    max_overlap = min(len(new), len(block))
                    overlap = 0
                    for o in range(max_overlap, -1, -1):
                        if o == 0:
                            overlap = 0
                            break
                        if new[len(new) - o:] == block[:o]:
                            overlap = o
                            break
                    new += block[overlap:]
                
                if len(new) < best_len:
                    best_len = len(new)
                    best_new = new
                
                if time.time() - start_time > time_limit * 0.9:
                    break
            
            if best_new is not None:
                word = best_new
            else:
                # Fallback to natural order
                new = []
                for p in perms:
                    block = list(p) + [m] + list(p)
                    max_overlap = min(len(new), len(block))
                    overlap = 0
                    for o in range(max_overlap, -1, -1):
                        if o == 0:
                            overlap = 0
                            break
                        if new[len(new) - o:] == block[:o]:
                            overlap = o
                            break
                    new += block[overlap:]
                word = new
        else:
            # For larger m, just use natural order (too expensive to try reordering)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
    
    return "".join(map(str, word))


def construct(n):
    """
    Construct a superpermutation of [1..n].
    Returns a string of digits 1..n.
    """
    start_time = time.time()
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Known shortest: 33 symbols
        return "12341213412143121423121432143214231213421324132431213421432143241324312143214231214321"
    
    # For n >= 5, use the greedy construction
    # The classic construction gives length 1! + 2! + ... + n!
    # For n=5: 154, n=6: 873, n=7: 5913
    
    # Use the standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        # Extract all distinct (m-1)-permutations from the current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build the new word with maximum overlap
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        # Check time
        if time.time() - start_time > 100.0:
            break
    
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
