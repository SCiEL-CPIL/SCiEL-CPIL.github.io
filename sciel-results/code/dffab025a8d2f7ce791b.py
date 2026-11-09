import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s buffer
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Classic superpermutation as baseline
    def classic_superperm(n):
        if n == 1:
            return [1]
        prev = classic_superperm(n - 1)
        perms = []
        seen = set()
        for i in range(len(prev) - (n - 2)):
            window = tuple(prev[i:i + n - 1])
            if len(set(window)) == n - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [n] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new
    
    # Encode a permutation as an integer for fast comparison
    def perm_to_code(p):
        code = 0
        for x in p:
            code = code * n + (x - 1)
        return code
    
    # Generate all permutations and their codes
    all_perms = list(permutations(range(1, n + 1)))
    perm_codes = [perm_to_code(p) for p in all_perms]
    code_to_idx = {code: i for i, code in enumerate(perm_codes)}
    
    # For each permutation, compute:
    # - suffix: last n-1 elements
    # - prefix: first n-1 elements
    # - which permutations can follow it (those with prefix == its suffix)
    
    N = len(all_perms)  # 5040 for n=7
    
    suffixes = [tuple(p[n-1:]) for p in all_perms]  # last n-1
    prefixes = [tuple(p[:n-1]) for p in all_perms]  # first n-1
    
    # Build adjacency: for each perm i, list of perms j where prefix[j] == suffix[i]
    # Each perm has exactly (n-1)! successors in the de Bruijn sense, but we need
    # to be more careful. Actually in the superpermutation graph, from perm p1...pn,
    # the next window must start with p2...pn and end with some new symbol.
    # So the next perm is p2...pn + x where x is the remaining symbol.
    
    # Better: from permutation [a1, a2, ..., an], the next permutation in the sequence
    # must be [a2, a3, ..., an, x] where x is the symbol not in {a2,...,an}.
    # So each node has exactly 1 outgoing edge in the "tight" superpermutation graph!
    # Wait no, that's for the specific construction. Let me reconsider.
    
    # Actually, a superpermutation is a sequence where consecutive n-windows are
    # all distinct permutations. The transition from window i to window i+1 means
    # window[i+1:] of window i == window[:n-1] of window i+1.
    # So if window i = [a1, a2, ..., an], window i+1 = [a2, a3, ..., an, x]
    # where x is any symbol (but for it to be a permutation, x must be the one
    # missing from {a2,...,an}).
    
    # So from permutation p = (a1,...,an), the next permutation must be
    # (a2, a3, ..., an, x) where x is the unique symbol not in {a2,...,an}.
    # This means each node has EXACTLY ONE outgoing edge!
    
    # This makes the problem a directed graph where each node has out-degree 1,
    # so we're looking for a Hamiltonian path in a functional graph.
    # But wait, that would mean there's only one possible superpermutation structure...
    
    # Let me reconsider. The issue is that we don't HAVE to use every possible transition.
    # We can skip windows. The word doesn't need to have every consecutive pair be
    # a valid tight transition. We can insert extra characters.
    
    # So the graph model: nodes are permutations. An edge from A to B exists if
    # suffix(A) == prefix(B). Cost is 1 (we add one character). We want a path
    # visiting all nodes with minimum total cost.
    
    # Let me build this properly.
    
    N = len(all_perms)
    
    # suffix[i] = last n-1 of perm i, prefix[i] = first n-1 of perm i
    # Edge i->j exists if suffix[i] == prefix[j]
    
    # For n=7, suffix is 6 elements, prefix is 6 elements.
    # Number of distinct suffixes: at most (n-1)! * n / n = ... let's just compute.
    
    suffix_list = []
    prefix_list = []
    for p in all_perms:
        prefix_list.append(tuple(p[:n-1]))
        suffix_list.append(tuple(p[n-1:]))
    
    # Group permutations by their prefix
    prefix_to_perms = {}
    for i, p in enumerate(prefix_list):
        if p not in prefix_to_perms:
            prefix_to_perms[p] = []
        prefix_to_perms[p].append(i)
    
    # adjacency[i] = list of j such that suffix[i] == prefix[j]
    adjacency = [[] for _ in range(N)]
    for i in range(N):
        s = suffix_list[i]
        if s in prefix_to_perms:
            adjacency[i] = prefix_to_perms[s]
    
    # Each node should have exactly (n-1)! / (n-1)! ... let me check for n=3
    # For n=3, suffix is 2 elements. How many perms have a given 2-element prefix?
    # Perms of 1,2,3 with prefix (1,2): only (1,2,3). So 1 successor.
    # With prefix (1,3): only (1,3,2). So 1 successor.
    # So out-degree is 1 for n=3? That can't be right for a useful graph.
    
    # Wait, I think I'm confusing myself. Let me re-examine.
    
    # A superpermutation word w of length L has windows w[i:i+n] for i=0..L-n.
    # These windows must include all n! permutations.
    # Consecutive windows overlap in n-1 positions.
    # So if window i = (a1,...,an), window i+1 = (a2,...,an, x) for some x.
    # For window i+1 to be a permutation, x must be the symbol not in {a2,...,an}.
    # So YES, each node has exactly 1 valid successor in the "consecutive windows" sense.
    
    # This means any superpermutation corresponds to a path in this functional graph
    # that visits all nodes. But since each node has out-degree 1, the graph consists
    # of cycles with trees feeding into them. A Hamiltonian path can only exist if
    # the graph is a single path (no branching, no cycles except possibly at the end).
    
    # For n=3: perms are 123,132,213,231,312,321
    # suffix->prefix edges:
    # 123: suffix=23, prefix of ? with prefix=23: 231. So 123->231
    # 231: suffix=31, prefix of ? with prefix=31: 312. So 231->312
    # 312: suffix=12, prefix of ? with prefix=12: 123. So 312->123 (cycle!)
    # 132: suffix=32, prefix of ? with prefix=32: 321. So 132->321
    # 321: suffix=21, prefix of ? with prefix=21: 213. So 321->213
    # 213: suffix=13, prefix of ? with prefix=13: 132. So 213->132 (cycle!)
    
    # So for n=3, we have two cycles: 123->231->312->123 and 132->321->213->132.
    # No Hamiltonian path exists in this graph!
    
    # This means the "tight" model (consecutive windows must be valid transitions)
    # is too restrictive. The actual superpermutation can have gaps - not every
    # consecutive pair of n-windows in the word needs to be a "tight" transition.
    
    # I see the issue. In a superpermutation, we don't require that EVERY consecutive
    # pair of n-windows forms a valid permutation transition. We just require that
    # ALL n! permutations appear as SOME n-windows. The windows that are permutations
    # can be non-consecutive in the word.
    
    # So the correct model: we have a word, and we look at all length-n substrings.
    # The set of length-n substrings that are permutations must include all n! perms.
    # Other length-n substrings may be non-permutations (repeated symbols).
    
    # This is a much harder problem. The greedy pruning approach in the seed is
    # reasonable but too slow. Let me try a different strategy:
    
    # 1. Start with classic superpermutation (5913 for n=7)
    # 2. Try to remove characters while maintaining validity
    # 3. Use a smarter removal strategy
    
    # Actually, let me try a completely different approach: construct the word
    # directly by choosing transitions carefully.
    
    # For n=7, the classic construction gives 5913. The optimal is 5905.
    # The difference is 8 characters. Let me try to find a better construction.
    
    # Alternative: use the known result that optimal superpermutations for n<=4 are:
    # n=1: 1 (len 1)
    # n=2: 121 (len 3)
    # n=3: 123121321 (len 9)
    # n=4: 12341231423124312 (len 33)
    
    # For n=7, let me try a greedy approach with backtracking.
    
    # Given time constraints, let me focus on making the pruning faster.
    
    word = classic_superperm(n)
    
    # Encode windows for fast lookup
    def encode_window(w):
        code = 0
        for x in w:
            code = code * n + (x - 1)
        return code
    
    all_perm_codes = set(perm_codes)
    
    # Build initial window codes
    L = len(word)
    window_codes = [0] * (L - n + 1)
    for i in range(L - n + 1):
        window_codes[i] = encode_window(word[i:i+n])
    
    # Count occurrences of each permutation
    counts = {}
    for code in window_codes:
        if code in all_perm_codes:
            counts[code] = counts.get(code, 0) + 1
    
    # Fast prune: try removing each position, check if safe
    def prune_pass(word, window_codes, counts, n, deadline):
        i = 0
        L = len(word)
        removed = 0
        
        while i < L and time.time() < deadline:
            L = len(word)
            if L < n:
                break
            
            # Check windows affected by removing word[i]
            # Windows starting from max(0, i-n+1) to min(i, L-n)
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            safe = True
            for s in range(start_min, start_max + 1):
                code = window_codes[s]
                if code in all_perm_codes and counts.get(code, 0) == 1:
                    safe = False
                    break
            
            if not safe:
                i += 1
                continue
            
            # Safe to remove. Update counts.
            for s in range(start_min, start_max + 1):
                code = window_codes[s]
                if code in all_perm_codes:
                    counts[code] -= 1
                    if counts[code] == 0:
                        del counts[code]
            
            # New windows after removal
            # After removing word[i], windows shift.
            # New window at position s (s < i) becomes word[s:i] + word[i+1:s+n+1]
            # New window at position s (s >= i) becomes word[i+1:i+n+1] shifted
            
            new_L = L - 1
            new_wc = [0] * (new_L - n + 1)
            
            for s in range(new_L - n + 1):
                if s < i:
                    w_new = word[s:i] + word[i+1:s+n+1]
                else:
                    w_new = word[i+1:i+n+1]
                code = encode_window(w_new)
                new_wc[s] = code
                if code in all_perm_codes:
                    counts[code] = counts.get(code, 0) + 1
            
            word.pop(i)
            window_codes = new_wc
            removed += 1
            # Don't increment i
        
        return word, window_codes, counts, removed
    
    # Multiple passes
    for _ in range(100):
        if time.time() >= deadline:
            break
        word, window_codes, counts, removed = prune_pass(word, window_codes, counts, n, deadline)
        if removed == 0:
            break
    
    # Verify
    final_L = len(word)
    seen = set()
    for i in range(final_L - n + 1):
        code = encode_window(word[i:i+n])
        if code in all_perm_codes:
            seen.add(code)
    
    if len(seen) < N:
        word = classic_superperm(n)
    
    return "".join(map(str, word))

if __name__ == "__main__":
    res = construct(7)
    print(len(res))
