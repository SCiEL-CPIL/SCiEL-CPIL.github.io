def construct(n):
    if n <= 1:
        return "1"
    
    import random
    from collections import defaultdict
    
    # Start with a permutation
    word = list(range(1, n+1))
    seen = {tuple(word)}
    
    # Greedy: at each step, try all possible next characters,
    # pick the one that adds the most new permutations
    # To avoid getting stuck, we can use a heuristic
    
    # This is O(L * n * n!) which might be too slow for n=7
    # L is ~5900, n=7, n!=5040, so 5900*7*5040 ~ 200M operations - too slow
    
    # Let's use the classic construction instead
    word = [1]
    for m in range(2, n + 1):
        perms, seen_set = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen_set:
                seen_set.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    return "".join(map(str, word))
