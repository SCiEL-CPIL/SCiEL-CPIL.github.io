"""Improved program: multi-start greedy with local search for Property P."""
import time
import random
from itertools import combinations

N = 256
DEADLINE = 80.0  # seconds, safety margin from 90s limit

def _valid_subset(s):
    """Check if subset s satisfies: for all a<b<=c in s, a does not divide b+c."""
    s = sorted(s)
    n = len(s)
    for i in range(n):
        a = s[i]
        for j in range(i+1, n):
            b = s[j]
            for k in range(j, n):
                c = s[k]
                if (b + c) % a == 0:
                    return False
    return True

def _conflict(a, b, c):
    """Check if a divides b+c (the violation condition)."""
    return (b + c) % a == 0

def _is_valid_with(s, x):
    """Check if adding x to sorted set s keeps it valid."""
    # x must be placed in sorted order
    # For every a < b <= c in the new set, a must not divide b+c
    # New violations involve x as a, b, or c
    s = sorted(s)
    n = len(s)
    
    # Find position of x in sorted order
    # Case 1: x is the smallest element (x < s[0])
    # Case 2: x is in the middle
    # Case 3: x is the largest element (x > s[-1])
    
    # We need to check all triples (a, b, c) with a < b <= c where at least one is x.
    # Let's just rebuild the sorted list and check only triples involving x.
    
    new_s = sorted(s + [x])
    # Find index of x in new_s
    # Since all elements are distinct (it's a set), x appears exactly once
    idx = new_s.index(x)
    
    # Triples involving x:
    # 1. x is 'a': x < b <= c, both b,c from s, and x < b
    #    Need: x does not divide b+c for all b,c in s with x < b <= c
    # 2. x is 'b': a < x <= c, a from s with a < x, c from s with x <= c (c can be x? no, b<=c and b=x, so c>=x, but c from s, and x not in s yet... wait c can be x only if x is in s, but x is being added. Actually b=x, c can be any element >= x in new_s. Since x is the new element, c can be x (if we allow b=c=x) or c > x from s.)
    #    Wait, the condition is a<b<=c. If b=x, then c>=x. c can be x (b=c=x) or c>x.
    #    If c=x: need a < x and x does not divide x+x=2x, i.e., x does not divide 2x. x divides 2x always! So if a<x and b=c=x, then a divides 2x? We need a NOT to divide 2x.
    #    Hmm, let me re-read: "a must NOT divide b+c". So if a<x and b=c=x, then we need a NOT divide 2x.
    # 3. x is 'c': a < b <= x, a,b from s with a<b<=x. Need a not divide b+x.
    
    # Let's just check all triples involving x by iterating.
    for i in range(len(new_s)):
        for j in range(i+1, len(new_s)):
            for k in range(j, len(new_s)):
                a, b, c = new_s[i], new_s[j], new_s[k]
                if a == x or b == x or c == x:
                    if (b + c) % a == 0:
                        return False
    return True

def _greedy_extend(base, candidates):
    """Greedily add elements from candidates to base, maintaining validity."""
    s = list(base)
    s_set = set(s)
    for x in candidates:
        if x in s_set:
            continue
        if _is_valid_with(s, x):
            s.append(x)
            s_set.add(x)
    return s

def _local_search(s, time_limit):
    """Try to improve s by removing one element and adding another."""
    s = sorted(s)
    s_set = set(s)
    best = s
    best_len = len(s)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        for i in range(len(s)):
            for x in range(1, N+1):
                if x in s_set:
                    continue
                # Try removing s[i] and adding x
                new_s = [y for y in s if y != s[i]]
                if _is_valid_with(new_s, x):
                    new_s.append(x)
                    if len(new_s) > best_len:
                        best = sorted(new_s)
                        best_len = len(new_s)
                        s = best
                        s_set = set(s)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    
    # Start with the known valid upper tail
    base = list(range(192, 257))
    
    # Verify base is valid
    if not _valid_subset(base):
        # Fallback: just use a small valid set
        return list(range(200, 257))
    
    best = base
    best_len = len(base)
    
    # Try multiple strategies to extend
    random.seed(42)
    
    # Strategy 1: Try adding elements in descending order from 191 down to 1
    s = list(base)
    for x in range(191, 0, -1):
        if _is_valid_with(s, x):
            s.append(x)
    if len(s) > best_len:
        best = s
        best_len = len(s)
    
    # Strategy 2: Try adding elements in ascending order
    s = list(base)
    for x in range(1, 192):
        if _is_valid_with(s, x):
            s.append(x)
    if len(s) > best_len:
        best = s
        best_len = len(s)
    
    # Strategy 3: Random order
    candidates = list(range(1, 192))
    for trial in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        random.shuffle(candidates)
        s = list(base)
        for x in candidates:
            if _is_valid_with(s, x):
                s.append(x)
        if len(s) > best_len:
            best = s
            best_len = len(s)
    
    # Strategy 4: Local search on the best found so far
    if time.time() - start_time < DEADLINE - 10:
        best = _local_search(best, min(10, DEADLINE - (time.time() - start_time) - 5))
        best_len = len(best)
    
    # Final verification
    best = sorted(best)
    if not _valid_subset(best):
        # Fallback to a definitely valid subset
        return list(range(200, 257))
    
    return best

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
