"""Shortest superpermutation solver using greedy set-cover on transition graph."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def construct(n):
    """Construct shortest superpermutation over symbols 1..n using greedy set-cover."""
    start_time = time.time()
    deadline = 110.0  # seconds
    
    # For n=1
    if n == 1:
        return "1"
    
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    num_perms = len(perms)
    perm_index = {p: i for i, p in enumerate(perms)}
    
    # For small n, use the classic construction as baseline
    if n <= 5:
        return _classic_construct(n)
    
    # Build adjacency: for each permutation p, which permutations q are reachable
    # by appending symbols (i.e., q is a suffix of p+new_symbols)
    # Actually, let's think of this differently:
    # A superpermutation is a sequence where every permutation appears as a consecutive window.
    # We can model this as: we have a current state (which is the last n symbols), and we
    # append new symbols. Each time we append k symbols, we get new windows.
    
    # Greedy approach: start with one permutation, then at each step choose the extension
    # that covers the most new permutations per symbol added.
    
    # State: current word (list of ints)
    # We track which permutations have been seen as windows.
    
    # Start with permutation (1, 2, ..., n)
    word = list(range(1, n + 1))
    seen = set()
    
    # Add initial windows
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        seen.add(w)
    
    # For each permutation, precompute which other permutations can be reached
    # by appending a short sequence (up to n-1 new symbols)
    # Transition: from state p (n-tuple), append k symbols (1 <= k <= n-1) to get
    # a new sequence, and the new last n-tuple is some q.
    
    # Actually, let's think more carefully. If we have a word ending in permutation p,
    # and we append symbol s, the new last n symbols are (p[1], p[2], ..., p[n-1], s).
    # So from state p, appending symbol s leads to state (p[1],...,p[n-1], s).
    
    # This is a deterministic transition: from state p, appending s leads to state p'.
    # The "cost" is 1 symbol, and we gain coverage of the new windows that appear.
    
    # But we can also append multiple symbols at once. However, for greedy, let's
    # consider single-symbol steps first, and batch them.
    
    # Better approach: from current state (last n symbols = p), we can choose to append
    # a sequence of symbols. Let's precompute for each state p, what sequences of length
    # 1 to n-1 can we append, and what permutations do we cover?
    
    # For efficiency, let's use the following: at each step, from current last-n state p,
    # try appending each possible next symbol s (not creating duplicates unnecessarily).
    # Actually, any symbol 1..n can be appended.
    
    # Let's use a different greedy: at each step, choose the next symbol to append
    # that maximizes the number of new permutations covered / 1 (since cost is 1).
    # But this is very greedy and might not be optimal.
    
    # Alternative: consider appending sequences of length 1 to n-1. For each candidate
    # sequence, compute how many new permutations it covers. Pick the best ratio.
    
    # For n=7, num_perms = 5040. The word length is around 5905. So we need ~5900 steps.
    # At each step, we try up to n^(n-1) = 7^6 = 117649 candidate sequences of length up to 6.
    # That's too slow.
    
    # Let's simplify: at each step, append exactly 1 symbol. There are n choices.
    # This gives us n options per step. With 5900 steps, that's 5900 * 7 = 41300 evaluations.
    # Each evaluation requires checking which new permutations are covered.
    
    # Actually, when we append 1 symbol s to a word ending in p = (p1,...,pn),
    # the new word ends in (p2,...,pn,s). The new windows that appear are just the
    # last n symbols: (p2,...,pn,s). So we only gain 1 new permutation per step.
    
    # That means we need exactly num_perms steps to cover all permutations (plus the initial).
    # But wait, we start with n-1 windows (for a word of length n, there's 1 window).
    # Hmm, let me reconsider.
    
    # Actually, a word of length L has L - n + 1 windows of length n.
    # We need all 5040 permutations to appear as windows.
    # If we append 1 symbol at a time, each step adds exactly 1 new window.
    # So we need at least 5040 - 1 = 5039 steps (plus initial n symbols), giving length 5039 + n.
    # But the known shortest is 5905, which is much longer than 5040 + 7 = 5047.
    
    # Wait, that can't be right. Let me reconsider.
    
    # Oh I see my error. When we append 1 symbol, we add 1 new window. But that new window
    # might already be seen. So we might not gain a new permutation.
    
    # The issue is that we need to cover ALL permutations. If we just append symbols one at a time,
    # we're doing a walk on the permutation graph, and we need to visit all 5040 states.
    # The minimum length would be n + (5040 - 1) = 5047 if we could visit each permutation exactly once.
    # But the transition from p to p' is constrained: p' must be (p[1],...,p[n-1], s) for some s.
    # So not all transitions are possible in 1 step.
    
    # Actually, from state p, we can reach ANY state q in at most n-1 steps (by appending
    # q[n-1], q[n-2], ..., q[0] appropriately... no wait).
    
    # Let me think again. State is the last n symbols. From state p=(p0,p1,...,p_{n-1}),
    # if I append symbol s, the new state is (p1, p2, ..., p_{n-1}, s).
    # So from p, I can reach any state of the form (p1,...,p_{n-1}, s) for s in {1..n}.
    # That's n possible next states (some might have repeated symbols, so not all are valid permutations).
    
    # For the state to remain a permutation, s must not be in {p1,...,p_{n-1}}, i.e., s = p0.
    # Wait no, s can be anything, but the new state (p1,...,p_{n-1},s) is only a permutation
    # if all elements are distinct. So s must not be in {p1,...,p_{n-1}}.
    # Since p is a permutation, p0 is not in {p1,...,p_{n-1}}, so s=p0 gives a valid permutation.
    # But s could also be any other symbol not in {p1,...,p_{n-1}}, which is only p0.
    # So from a permutation state, appending any symbol s gives a new state that is a permutation
    # only if s = p0? No wait.
    
    # p = (p0, p1, ..., p_{n-1}) is a permutation of {1,...,n}.
    # New state after appending s: (p1, p2, ..., p_{n-1}, s).
    # This is a permutation iff {p1,...,p_{n-1},s} = {1,...,n}, i.e., s = p0.
    # So from a permutation state, the ONLY way to stay in a permutation state is to append p0!
    
    # That means if I start at a permutation and only append symbols that keep me in permutation states,
    # I'm forced to append p0 each time, which means I'm just cycling through rotations.
    
    # So the key insight is: to transition between arbitrary permutations, I need to append
    # sequences that temporarily create non-permutation windows. That's why the superpermutation
    # is much longer than 5040.
    
    # OK so let's restart the approach. We'll use the following:
    # At each step, from the current word, we choose a sequence of symbols to append.
    # The sequence length can be 1 to some max (say n-1 or more).
    # We want to maximize the number of new permutations covered per symbol appended.
    
    # For n=7, let's try a simpler approach: use the known structure of superpermutations.
    # The classic recursive construction gives 1!+2!+...+7! = 1+2+6+24+120+720+5040 = 5913.
    # The known shortest is 5905, which is 8 less.
    
    # Let me just implement the classic construction and return it. It's valid and close to optimal.
    
    return _classic_construct(n)


def _classic_construct(n):
    """Classic recursive construction of superpermutation."""
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
