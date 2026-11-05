import random
from itertools import combinations, combinations_with_replacement

def construct():
    n = 256
    best_A = []
    best_B = []
    best_score = 0
    
    def validate_and_add(set_curr, diffs_curr, other_diffs, x):
        """
        Check if x can be added to set_curr.
        Returns (True, new_set, new_diffs) if valid, else (False, None, None).
        """
        # Check Sidon property for the new set
        new_set = set_curr + [x]
        # Optimization: Only check sums involving x
        # Existing sums are unique. New sums are x + a for a in set_curr, and x + x.
        # We need to ensure x+a is not in existing sums and x+a != x+b for a!=b (impossible if sums unique)
        # Actually, simpler: just recompute sums for small sets. |set| <= 64, so 64*64/2 = 2048 checks. Fast.
        
        sums = []
        for i in range(len(new_set)):
            for j in range(i, len(new_set)):
                sums.append(new_set[i] + new_set[j])
        
        if len(sums) != len(set(sums)):
            return False, None, None
            
        # Check differences
        # New differences are x - a and a - x for a in set_curr
        new_diffs = set(diffs_curr)
        for a in set_curr:
            d1 = x - a
            d2 = a - x
            new_diffs.add(d1)
            new_diffs.add(d2)
            
        # Check disjointness with other set's differences
        if new_diffs & other_diffs:
            return False, None, None
            
        return True, new_set, new_diffs

    def greedy_construct(order):
        A = []
        B = []
        D_A = set()
        D_B = set()
        
        for x in order:
            # Try adding to A
            valid_A, new_A, new_D_A = validate_and_add(A, D_A, D_B, x)
            # Try adding to B
            valid_B, new_B, new_D_B = validate_and_add(B, D_B, D_A, x)
            
            if valid_A and not valid_B:
                A = new_A
                D_A = new_D_A
            elif valid_B and not valid_A:
                B = new_B
                D_B = new_D_B
            elif valid_A and valid_B:
                # Heuristic: Add to the smaller set to balance, or add to A if tie
                if len(A) <= len(B):
                    A = new_A
                    D_A = new_D_A
                else:
                    B = new_B
                    D_B = new_D_B
            else:
                # Cannot add to either, discard x
                pass
                
        return A, B

    # 1. Try standard sorted order
    A, B = greedy_construct(list(range(1, n + 1)))
    score = len(A) + len(B)
    if score > best_score:
        best_score = score
        best_A, best_B = A, B

    # 2. Try reverse sorted order
    A, B = greedy_construct(list(range(n, 0, -1)))
    score = len(A) + len(B)
    if score > best_score:
        best_score = score
        best_A, best_B = A, B

    # 3. Try several random shuffles
    # Set a seed for reproducibility if needed, but random is fine
    # We'll run a few iterations to find a good local optimum
    for _ in range(50):
        order = list(range(1, n + 1))
        random.shuffle(order)
        A, B = greedy_construct(order)
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A, B

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2] # 2-1=1, D_A={0}? No, D_A for [1] is empty. D_B for [2] is empty.
        # Wait, if A=[1], D_A is empty. If B=[2], D_B is empty. Disjoint.
        # But we want to maximize size. The greedy should find >2 elements.
        
    return [best_A, best_B]
