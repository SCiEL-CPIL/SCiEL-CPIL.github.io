import itertools
import random

def max_sidon_size(S):
    """
    Computes the size of the largest subset of S with all unordered pair sums unique.
    S is a list of 12 distinct integers.
    """
    n = len(S)
    
    def is_sidon(subset):
        sums = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon(list(subset)):
                return k
    return 1

def construct():
    random.seed(123) # Fixed seed for reproducibility
    
    best_set = None
    best_score = float('inf')
    
    # Number of restarts
    num_restarts = 5
    # Iterations per restart
    max_iters = 3000
    
    for r in range(num_restarts):
        # Initialize with random distinct integers
        S = list(random.sample(range(129), 12))
        current_score = max_sidon_size(S)
        
        if current_score < best_score:
            best_score = current_score
            best_set = S[:]
            
        # If we found a very good score, reduce iterations or break
        if best_score <= 4:
            # 4 is likely optimal or near-optimal. Let's try to find 3?
            # But 3 is very hard. Let's just keep searching for a bit.
            pass

        for _ in range(max_iters):
            # Pick a random index to replace
            i = random.randint(0, 11)
            # Pick a random new value not in S
            while True:
                v = random.randint(0, 128)
                if v not in S:
                    break
            
            S_new = S[:]
            S_new[i] = v
            
            new_score = max_sidon_size(S_new)
            
            if new_score < current_score:
                S = S_new
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = S[:]
            elif new_score == current_score:
                # Accept with probability 0.3 to explore
                if random.random() < 0.3:
                    S = S_new
            # else reject
        
        # If best score is 4, we can try to optimize further or break
        # Given 60s limit, 5 restarts * 3000 iters * ~0.02s/iter = 300s. Too slow.
        # Need to estimate time per iter more carefully.
        # max_sidon_size:
        # k=12: 1 comb, ~78 sums
        # k=11: 12 combs, ~66 sums
        # k=10: 66 combs, ~55 sums
        # k=9: 220 combs, ~45 sums
        # k=8: 495 combs, ~36 sums
        # k=7: 792 combs, ~28 sums
        # k=6: 924 combs, ~21 sums
        # Total subsets if max size is 6: 2510.
        # Average sums per check ~ 30.
        # Total ops ~ 75,000.
        # Python overhead is high. Let's say 1ms per max_sidon_size call?
        # 3000 iters * 1ms = 3s per restart.
        # 5 restarts = 15s. This is safe.
        
    return best_set
