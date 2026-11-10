import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_set = []
    best_size = 0
    best_counts = {}
    
    # Helper to check if x can be added to S with given counts
    def can_add(S, counts, x):
        # Check 2x
        if counts.get(2*x, 0) >= 2:
            return False
        # Check x + y for y in S, y >= x
        for y in S:
            if y >= x:
                s = x + y
                if counts.get(s, 0) >= 2:
                    return False
        return True

    def do_add(S, counts, x):
        # Add x to S and update counts
        S.add(x)
        # Update counts for 2x
        counts[2*x] = counts.get(2*x, 0) + 1
        # Update counts for x + y
        for y in S:
            if y >= x and y != x: # y != x already handled by 2x? No, 2x is x+x.
                # Wait, if y == x, we already did 2x.
                # So iterate y in S where y > x
                pass 
        # Better: iterate all y in S, if y >= x, sum = x+y.
        # But we already incremented 2x. So for y in S:
        # if y > x: counts[x+y] += 1
        for y in S:
            if y > x:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
        return S, counts

    def do_remove(S, counts, x):
        S.remove(x)
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for y in S:
            if y > x:
                s = x + y
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
            # What if y < x? Then the pair is (y, x) which is sum x+y.
            # In can_add, we checked y >= x.
            # In do_add, we added counts for x+y where y > x.
            # But what about pairs (y, x) where y < x?
            # The representation is unordered {a,b} with a<=b.
            # When we added x, we considered pairs (x, y) for y in S.
            # If y < x, the pair is (y, x). Sum is x+y.
            # Did we count it?
            # In can_add(S, counts, x), we checked `for y in S: if y >= x`.
            # This misses y < x.
            # The condition is "every sum ... with a <= b".
            # If we add x, new pairs involving x are {x, y} for all y in S.
            # If y < x, the pair is (y, x). Sum x+y.
            # If y > x, the pair is (x, y). Sum x+y.
            # If y == x, the pair is (x, x). Sum 2x.
            # So we must check ALL y in S.
            
            # Let's fix can_add and do_add.
            pass
        return S, counts

    # Corrected Helpers
    def can_add(S, counts, x):
        # Check 2x
        if counts.get(2*x, 0) >= 2:
            return False
        # Check x + y for all y in S
        for y in S:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def do_add(S, counts, x):
        S.add(x)
        counts[2*x] = counts.get(2*x, 0) + 1
        for y in S:
            if y != x:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
        return S, counts

    def do_remove(S, counts, x):
        S.remove(x)
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for y in S:
            if y != x:
                s = x + y
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
        return S, counts

    # Initial Greedy
    S = set()
    counts = {}
    candidates = list(range(1, 1001))
    random.shuffle(candidates)
    for x in candidates:
        if can_add(S, counts, x):
            do_add(S, counts, x)
            
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
        best_counts = dict(counts)

    # Local Search Loop
    while time.time() < deadline:
        # Try to improve current S
        # 1. Greedy fill with random order
        improved = True
        while improved:
            improved = False
            # Try a few random candidates from complement
            # To be efficient, we can just pick random numbers from 1..1000 not in S
            for _ in range(100): # Try 100 random adds
                x = random.randint(1, 1000)
                if x not in S:
                    if can_add(S, counts, x):
                        do_add(S, counts, x)
                        improved = True
                        break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            best_counts = dict(counts)
        
        # If stuck, kick
        # Remove a random element
        if S:
            x_to_remove = random.choice(list(S))
            do_remove(S, counts, x_to_remove)
            
            # Try to add a random element
            added = False
            for _ in range(100):
                x = random.randint(1, 1000)
                if x not in S:
                    if can_add(S, counts, x):
                        do_add(S, counts, x)
                        added = True
                        break
            
            # If we didn't add anything, we are smaller. 
            # We can try to re-greedy to fill it back up.
            # But the loop above does greedy fill.
            
    return list(best_set)

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
