"""Program: large 3-sum-free-pair-sum set in 1..128 via bit-parallel greedy independent set."""
# EVOLVE-BLOCK-START
import random

N = 128

def construct():
    best = None
    best_len = 0

    rng = random.Random(42)

    def can_add(x, S_set, S_list):
        # S_set: set of elements in S
        # S_list: list of elements in S (for iteration)
        
        # Find y in S such that y+x is also in S
        # These y are potential 'a' or 'b' in the forbidden triple {x, a, b}
        # where x+a in S and x+b in S.
        # Let candidates = { y in S | y+x in S }
        # We need to check if there exist distinct a, b in candidates
        # such that a+b in S or a+b == x.
        
        candidates = []
        for y in S_list:
            if y + x in S_set:
                candidates.append(y)
        
        n_cand = len(candidates)
        for i in range(n_cand):
            a = candidates[i]
            for j in range(i + 1, n_cand):
                b = candidates[j]
                s = a + b
                if s == x or s in S_set:
                    return False
        return True

    # Try many random orderings
    for trial in range(500):
        perm = list(range(1, 129))
        rng.shuffle(perm)
        
        S_set = set()
        S_list = []
        
        for x in perm:
            if can_add(x, S_set, S_list):
                S_set.add(x)
                S_list.append(x)
        
        if len(S_set) > best_len:
            best_len = len(S_set)
            best = list(S_set)
            
        if best_len >= 90:
            break
            
    # Fallback if something goes wrong
    if best is None:
        best = list(range(65, 129))
        
    return best
# EVOLVE-BLOCK-END
