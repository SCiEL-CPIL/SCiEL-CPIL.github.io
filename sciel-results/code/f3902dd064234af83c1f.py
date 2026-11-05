import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10-second safety margin
    
    best_set = set()
    
    # Helper function to check if adding x to current_set is valid
    def can_add(x, current_set, counts):
        # We need to check sums x+y for y in current_set where y >= x? 
        # No, the condition is a+b with a<=b.
        # If we add x, new sums are x+y for y in current_set.
        # We must ensure a<=b, so if x > y, the pair is (y, x).
        # The sum is the same. The representation is unordered pair {a,b}.
        # So for each y in current_set, the new representation is for sum x+y.
        # We also have the pair (x, x) for sum 2x.
        
        # Check 2x
        if counts.get(2*x, 0) >= 2:
            return False
            
        # Check x+y for y in current_set
        # We only need to check if the count for sum x+y would become 3.
        # Current count is counts.get(x+y, 0).
        # If it is 2, adding one more makes it 3 -> invalid.
        for y in current_set:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def update_counts(x, current_set, counts):
        counts[2*x] = counts.get(2*x, 0) + 1
        for y in current_set:
            s = x + y
            counts[s] = counts.get(s, 0) + 1

    def revert_counts(x, current_set, counts):
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for y in current_set:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]

    candidates = list(range(1, 1001))
    
    # Try multiple random restarts
    while time.time() < deadline:
        random.shuffle(candidates)
        current_set = set()
        counts = {}
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, current_set, counts):
                current_set.add(x)
                update_counts(x, current_set, counts)
                
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    return list(best_set)
