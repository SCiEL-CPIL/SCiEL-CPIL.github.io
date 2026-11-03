import itertools
from collections import deque

def construct(n):
    if n <= 4:
        # Classic construction is optimal for small n
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n >= 5, use a greedy construction with a priority queue
    # to find the shortest superpermutation
    
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Precompute suffix/prefix matching for faster overlap calculation
    # For each permutation, store its suffixes and prefixes
    suffixes = {}
    prefixes = {}
    for i, ps in enumerate(perm_strs):
        for k in range(1, n):
            suffixes.setdefault(ps[n-k:], set()).add(i)
            prefixes.setdefault(ps[:k], set()).add(i)
    
    def get_overlap(a, b):
        """Return the maximum k such that suffix of a of length k equals prefix of b of length k"""
        for k in range(n - 1, 0, -1):
            if a[n-k:] == b[:k]:
                return k
        return 0
    
    # Use a greedy algorithm: start with the first permutation, then repeatedly
    # add the unvisited permutation that gives the maximum overlap
    # Use a heap for efficiency
    
    import heapq
    
    # State: (negative_overlap, counter, perm_index)
    # We'll use a different approach: maintain a set of unvisited perms
    # and for each, compute overlap with current, pick max
    
    # To speed up, use the suffix/prefix dictionaries
    # For current perm c, we want to find unvisited perm j that maximizes overlap(c, j)
    # overlap(c, j) = k if c[n-k:] == j[:k]
    # So for each k from n-1 down to 1, check if there's an unvisited perm with prefix c[n-k:]
    
    unvisited = set(range(N))
    unvisited.remove(0)
    path = [0]
    current = 0
    current_str = perm_strs[0]
    
    for _ in range(N - 1):
        best_next = -1
        best_ov = -1
        for k in range(n - 1, 0, -1):
            suffix = current_str[n-k:]
            candidates = prefixes.get(suffix, set())
            # Find unvisited candidates
            for j in candidates:
                if j in unvisited:
                    best_next = j
                    best_ov = k
                    break
            if best_next != -1:
                break
        
        if best_next == -1:
            # No overlap found, pick any unvisited
            best_next = next(iter(unvisited))
            best_ov = 0
        
        path.append(best_next)
        unvisited.remove(best_next)
        current = best_next
        current_str = perm_strs[best_next]
    
    # Build the result string
    res = perm_strs[path[0]]
    for i in range(1, N):
        ov = get_overlap(perm_strs[path[i-1]], perm_strs[path[i]])
        res += perm_strs[path[i]][ov:]
    
    return res
