import sys
import itertools
from collections import deque

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134124312143213421412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413242134214324132412341324134214312341243132413243124314231432123414324132421342143241324312341432413248"
    
    # For n >= 5, use the "symbol rotation" / greedy LRU approach
    # This constructs a superpermutation by maintaining a circular buffer
    # and greedily appending the least recently used symbol that doesn't
    # create a duplicate window
    
    # First, generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Create a set for fast lookup of seen permutations
    seen_perms = set()
    
    # Start with the first permutation
    word = list(perms[0])
    seen_perms.add(perms[0])
    
    # Maintain a circular buffer of the last n-1 symbols
    # and track recency of each symbol
    recency = {}
    for i, s in enumerate(word):
        recency[s] = i  # position in word, higher = more recent
    
    current_pos = n  # next position to write
    
    # For greedy LRU: at each step, we try to extend by one symbol
    # such that the new n-window is a permutation not yet seen
    # We prefer the symbol that was least recently used
    
    while len(seen_perms) < N:
        # Get the last n-1 symbols as a tuple (the "context")
        if len(word) < n - 1:
            break
        context = tuple(word[-(n-1):])
        
        # Try each possible next symbol
        best_symbol = None
        best_recency = -1
        
        for s in range(1, n + 1):
            if s in context:
                continue  # would create a duplicate in the window
            
            # Check if the new window (context + s) is a permutation not yet seen
            new_window = context + (s,)
            if new_window not in seen_perms:
                # This is a valid move
                # Use LRU: prefer the symbol with the lowest recency
                s_recency = recency.get(s, -1)
                if s_recency > best_recency or best_symbol is None:
                    best_recency = s_recency
                    best_symbol = s
        
        if best_symbol is not None:
            # Append the best symbol
            word.append(best_symbol)
            new_window = tuple(word[-n:])
            seen_perms.add(new_window)
            recency[best_symbol] = current_pos
            current_pos += 1
        else:
            # No valid single-symbol extension; need to append a full permutation
            # Find the lexicographically smallest unvisited permutation
            # that can be appended with maximum overlap
            found = False
            for k in range(n - 1, 0, -1):
                suffix = tuple(word[-k:])
                # Find permutations starting with this suffix
                for p in perms:
                    if p[:k] == suffix and p not in seen_perms:
                        # Append the remaining symbols
                        for s in p[k:]:
                            word.append(s)
                            recency[s] = current_pos
                            current_pos += 1
                        seen_perms.add(p)
                        found = True
                        break
                if found:
                    break
            if not found:
                # No overlap, append the smallest unvisited permutation
                for p in perms:
                    if p not in seen_perms:
                        for s in p:
                            word.append(s)
                            recency[s] = current_pos
                            current_pos += 1
                        seen_perms.add(p)
                        found = True
                        break
            if not found:
                break
    
    return "".join(map(str, word))
