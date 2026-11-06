import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    n = 256
    # Initialize with greedy construction
    selected = []
    for x in range(1, n + 1):
        # Check if x forms a 3-AP with any pair in selected
        valid = True
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a, b = selected[i], selected[j]
                # Check if a, b, x is a 3-AP
                # Cases:
                # 1. x is middle: a + b = 2x
                # 2. x is end: a + x = 2b or b + x = 2a
                if a + b == 2 * x or a + x == 2 * b or b + x == 2 * a:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
    
    # Calculate current score
    def get_score(S):
        return sum(Fraction(1, x) for x in S)
    
    current_score = get_score(selected)
    best_set = set(selected)
    best_score = current_score
    
    # Helper to check if a set is valid (for safety, though logic above should ensure it)
    def is_valid(S):
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                # Check if there exists c in S such that a,b,c form 3-AP
                # c = 2b - a
                if 2 * b - a in S and 2 * b - a != a and 2 * b - a != b:
                    return False
                # c = 2a - b
                if 2 * a - b in S and 2 * a - b != a and 2 * a - b != b:
                    return False
        return True

    # Local Search: Try adding x and removing minimal conflicts
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_move = None
        best_delta = 0
        best_new_set = None
        
        current_set = list(best_set)
        
        for x in range(1, n + 1):
            if x in best_set:
                continue
            
            # Find all pairs (a, b) in best_set such that a + b = 2x
            # These are the conflicts. We must remove one from each pair.
            # The pairs are disjoint.
            to_remove = []
            for i in range(len(current_set)):
                a = current_set[i]
                # b = 2x - a
                b = 2 * x - a
                if b in best_set and b != a and b != x:
                    # We have a conflict pair {a, b}
                    # To minimize loss, remove the one with smaller reciprocal (larger value)
                    # Wait, we want to KEEP the larger reciprocal.
                    # Reciprocal of a is 1/a. Reciprocal of b is 1/b.
                    # If a < b, then 1/a > 1/b. We want to keep a (smaller index, larger weight) and remove b.
                    # So we remove the larger number.
                    # Let's verify:
                    # Pair {2, 8}, x=5. 1/2 = 0.5, 1/8 = 0.125. Keep 2, remove 8.
                    # So remove max(a, b).
                    to_remove.append(max(a, b))
            
            # Remove duplicates in to_remove?
            # Since pairs are disjoint, a number can only be in one pair.
            # Proof: If a is in pair (a, b) with a+b=2x, and a is in pair (a, c) with a+c=2x, then b=c.
            # So to_remove contains unique elements.
            
            # Calculate delta
            loss = sum(Fraction(1, r) for r in to_remove)
            gain = Fraction(1, x)
            delta = gain - loss
            
            if delta > best_delta:
                best_delta = delta
                best_move = x
                # Construct new set
                new_set = best_set.copy()
                for r in to_remove:
                    new_set.discard(r)
                new_set.add(x)
                best_new_set = new_set
        
        if best_move is not None:
            best_set = best_new_set
            best_score = get_score(best_set)
            improved = True
            
    return sorted(best_set)
# EVOLVE-BLOCK-END
