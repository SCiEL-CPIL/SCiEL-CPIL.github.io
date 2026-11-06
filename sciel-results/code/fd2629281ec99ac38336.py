import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s to be safe
    
    N = 32
    MAX_VAL = 512
    
    def score_set(S):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in S.
        S is a list of distinct integers.
        """
        S_set = set(S)
        diffs = set()
        # Optimization: sort S to potentially break early? No, just iterate pairs.
        # We need to check if a+d and a+2d are in S.
        # For each pair (x, y) with x < y, d = y - x.
        # Check if y + d is in S.
        # If yes, d is a valid difference.
        # Note: This counts d if there EXISTS any a.
        # The pair (x, y) corresponds to a=x, a+d=y.
        # We need a+2d = y+d to be in S.
        
        # We can just iterate all pairs.
        for i in range(N):
            x = S[i]
            for j in range(i + 1, N):
                y = S[j]
                d = y - x
                if y + d in S_set:
                    diffs.add(d)
        return len(diffs)

    def initial_random():
        return random.sample(range(1, MAX_VAL + 1), N)
    
    def initial_ap():
        # Arithmetic progression based
        # Try different starts and steps
        start = random.randint(1, 50)
        step = random.randint(1, 10)
        S = []
        for i in range(N):
            val = start + i * step
            if val <= MAX_VAL:
                S.append(val)
        # Fill rest randomly if needed
        while len(S) < N:
            val = random.randint(1, MAX_VAL)
            if val not in S:
                S.append(val)
        return S[:N]

    best_set = None
    best_score = -1

    # Generate a few diverse initial candidates
    candidates = []
    for _ in range(5):
        candidates.append(initial_random())
    for _ in range(5):
        candidates.append(initial_ap())
    
    # Evaluate initial candidates
    for S in candidates:
        s = score_set(S)
        if s > best_score:
            best_score = s
            best_set = list(S)

    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    
    current_set = list(best_set)
    current_score = best_score
    
    while time.time() < deadline:
        # Pick a random element to remove
        idx_remove = random.randint(0, N - 1)
        x = current_set[idx_remove]
        
        # Pick a random element to add
        y = random.randint(1, MAX_VAL)
        
        # Ensure y is not already in the set (except x which is being removed)
        # We will create a new set
        # To be efficient, we can check if y == x, then skip
        if y == x:
            continue
            
        # Check if y is in current_set
        # Since N is small, linear search is fine, but set is better.
        # Let's keep a set representation for O(1) lookup
        # But we are modifying the list.
        
        # Create new set
        # Remove x, add y
        # If y is in current_set and y != x, then we are not changing the set size?
        # No, we must maintain size N.
        # If y is in current_set (and y != x), then adding y doesn't change the set.
        # We should pick y not in current_set \ {x}.
        
        # Simple approach:
        # new_set_list = current_set.copy()
        # new_set_list[idx_remove] = y
        # But we need to ensure distinctness.
        # If y is in new_set_list (other than at idx_remove), it's invalid.
        
        # Efficient check:
        # Is y in current_set?
        # If y == x, skip.
        # If y in current_set (and y != x), then replacing x with y results in duplicate y.
        # So we must ensure y not in current_set or y == x.
        
        # Let's just use a set for membership check.
        # This is O(N) per step if we rebuild set, or O(1) if we maintain it.
        # Rebuilding set of 32 elements is fast.
        
        # Check if y is in current_set
        # We can use a set for current_set
        # But updating the set is O(1).
        
        # Let's maintain current_set as a list and a set.
        # Actually, let's just do it simply.
        
        # Check if y is in current_set
        # Since N=32, `y in current_set` is O(32).
        # If y in current_set and y != x:
        #    We are replacing x with y, but y is already there.
        #    So the new set would have two y's and no x.
        #    This is invalid for a set of distinct integers.
        #    So we skip this move.
        
        if y in current_set:
            # If y == x, we skipped earlier.
            # So y != x.
            # Thus y is a duplicate. Skip.
            continue
        
        # Perform move
        new_set = current_set.copy()
        new_set[idx_remove] = y
        
        # Calculate new score
        new_score = score_set(new_set)
        
        # Acceptance
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # Note: new_score <= current_score, so delta <= 0
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_set = new_set
                    current_score = new_score
            # If delta == 0, we might accept to explore?
            # Usually yes, but let's be strict to converge.
            # Or accept with small prob.
            # Let's accept if delta == 0 with prob 0.1
            elif delta == 0:
                if random.random() < 0.1:
                    current_set = new_set
                    current_score = new_score
        
        # Cool down
        T *= cooling
        
        # If T is too small, restart with a new random start?
        # Or just keep going.
        # Let's restart if T < 0.1
        if T < 0.1:
            # Try a few random restarts
            for _ in range(10):
                if time.time() > deadline:
                    break
                S = initial_random()
                s = score_set(S)
                if s > current_score:
                    current_set = S
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
            T = 10.0 # Reset temperature

    return best_set

# Run a quick self-test if executed directly
if __name__ == "__main__":
    res = construct()
    print(f"Result: {res}")
    print(f"Score: {score_set(res)}")
