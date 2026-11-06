import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Candidates 2..256
    all_candidates = list(range(2, 257))
    N = len(all_candidates)
    
    # Precompute reciprocal values
    inv = [0.0] * 257
    for i in range(2, 257):
        inv[i] = 1.0 / i
        
    # Precompute conflict matrix
    # conflict[a][b] = True if lcm(a,b) <= 256
    # We only need this for a < b to save memory/time, but a flat list or set is faster for lookup?
    # Let's use a list of sets for neighbors in the conflict graph.
    # neighbors[i] = set of j != i such that lcm(i,j) <= 256
    neighbors = [set() for _ in range(257)]
    
    for i in range(2, 257):
        for j in range(i + 1, 257):
            # Check if conflict
            # lcm(i, j) = i * j // gcd(i, j)
            # If i * j > 256 * gcd(i, j), then lcm > 256 (compatible)
            # If i * j <= 256 * gcd(i, j), then lcm <= 256 (conflict)
            g = gcd(i, j)
            if i * j <= 256 * g:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # Helper to check if a new item x can be added to a set represented by a boolean array
    # Or just iterate. Since N=255, iterating is fast enough.
    
    def is_valid_set(S):
        """Check if S is a valid independent set."""
        # S is a list or set of ints
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if b in neighbors[a]:
                    return False
        return True

    def score_set(S):
        return sum(inv[x] for x in S)

    best_score = 0.0
    best_set = []
    
    # Strategy: Multiple random restarts with Local Search
    # Initial solution: Greedy by value (smallest first) or Random?
    # Greedy by value (ascending order) is a good baseline.
    
    def greedy_init():
        S = []
        for x in all_candidates:
            # Check if x is compatible with all in S
            ok = True
            for y in S:
                if x in neighbors[y]:
                    ok = False
                    break
            if ok:
                S.append(x)
        return S

    def local_search(S, max_iters=10000):
        """
        1-Swap Local Search:
        Try replacing x in S with y not in S.
        Improvement if inv[y] > inv[x] and y is compatible with S\{x}.
        Also try 1-Add if possible (though S should be maximal if we start greedy, 
        but swaps might create space? No, 1-swap keeps size same. 
        If we remove x, we might be able to add y AND z? That's 1-out, 2-in.
        Let's stick to 1-out, 1-in for speed, then try 1-out, 2-in if time permits?
        Actually, 1-out, 1-in is sufficient for many cases. 
        To be more robust, let's also try adding any compatible item if S is not maximal.
        """
        S_set = set(S)
        current_score = score_set(S)
        
        # Ensure maximality: try to add any compatible item
        changed = True
        while changed:
            changed = False
            for x in all_candidates:
                if x not in S_set:
                    ok = True
                    for y in S_set:
                        if x in neighbors[y]:
                            ok = False
                            break
                    if ok:
                        S_set.add(x)
                        S.append(x)
                        current_score += inv[x]
                        changed = True
                        # Restart check? No, just continue adding
        
        # Now do swaps
        improved = True
        while improved:
            improved = False
            # Shuffle S to avoid order bias?
            # For deterministic behavior in local search steps, we might want to be careful.
            # But randomization helps escape local minima.
            for i in range(len(S)):
                x = S[i]
                # Try removing x
                S_set.remove(x)
                temp_score = current_score - inv[x]
                
                # Find best y to add
                best_y = -1
                best_gain = 0
                for y in all_candidates:
                    if y in S_set:
                        continue
                    # Check compatibility with S_set
                    ok = True
                    for z in S_set:
                        if y in neighbors[z]:
                            ok = False
                            break
                    if ok:
                        gain = inv[y]
                        if gain > best_gain:
                            best_gain = gain
                            best_y = y
                
                if best_y != -1:
                    # Add best_y
                    S_set.add(best_y)
                    S.append(best_y)
                    current_score += best_gain
                    improved = True
                    # Break and restart? Or continue? 
                    # Continuing might lead to a sequence of swaps.
                    # But we removed x, so we need to update S list indices?
                    # It's safer to break and restart the loop over S.
                    break
                else:
                    # Put x back
                    S_set.add(x)
        
        return S, current_score

    # Run multiple restarts
    # 1. Greedy baseline
    S0 = greedy_init()
    S0, score0 = local_search(S0)
    if score0 > best_score:
        best_score = score0
        best_set = list(S0)
        
    # 2. Random restarts
    # Time budget for random restarts
    num_restarts = 0
    while time.time() < deadline:
        # Random initialization
        # Pick a random subset? Or random order greedy?
        # Random order greedy: shuffle candidates, then greedy pick.
        random_candidates = all_candidates[:]
        random.shuffle(random_candidates)
        
        S_rand = []
        for x in random_candidates:
            ok = True
            for y in S_rand:
                if x in neighbors[y]:
                    ok = False
                    break
            if ok:
                S_rand.append(x)
        
        S_rand, score_rand = local_search(S_rand)
        
        if score_rand > best_score:
            best_score = score_rand
            best_set = list(S_rand)
            
        num_restarts += 1
        
        # Check time
        if time.time() > deadline - 5.0:
            break

    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Result: {res}")
    print(f"Sum: {sum(1/x for x in res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if (res[i]*res[j])//gcd(res[i], res[j]) <= 256:
                valid = False
                print(f"Invalid pair: {res[i]}, {res[j]}")
    print(f"Valid: {valid}")
