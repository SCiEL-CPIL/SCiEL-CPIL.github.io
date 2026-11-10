import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for safety (limit is 90s, but task says 60s evaluation? "Each evaluation must finish within 60 s". The prompt says "evaluator KILLS the program after 90s wall-clock". I will use 55s to be safe for both).
    
    n = 256
    candidates = list(range(1, n + 1))
    
    best_size = 0
    best_A = []
    best_B = []
    
    def can_add(x, S, sums_S, diffs_S, diffs_other):
        # Check Sidon property for new sums in S
        # New sums are x+a for a in S, and 2x
        # We need to ensure these are not in sums_S and are unique among themselves.
        
        # 1. Check 2x
        if 2 * x in sums_S:
            return False
        
        # 2. Check x+a
        # Since S is Sidon, sums x+a are distinct for distinct a?
        # x+a1 = x+a2 => a1=a2. Yes.
        # So we just need to check if any x+a is in sums_S.
        # And if 2x is in {x+a}. 2x = x+a => x=a. But x is new, so x not in S.
        # So 2x is distinct from x+a.
        
        for a in S:
            if x + a in sums_S:
                return False
        
        # Check disjoint differences with other set
        # New differences are |x-a| for a in S.
        # We need to ensure none of these are in diffs_other.
        for a in S:
            d = abs(x - a)
            if d in diffs_other:
                return False
        
        return True

    def add_to_set(x, S, sums_S, diffs_S):
        # Update sums
        sums_S.add(2 * x)
        for a in S:
            sums_S.add(x + a)
        
        # Update diffs
        for a in S:
            diffs_S.add(abs(x - a))
            
        S.append(x)

    max_restarts = 100000
    restart_count = 0
    
    # Pre-shuffle candidates to avoid sorting overhead in loop
    while time.time() - start_time < time_limit and restart_count < max_restarts:
        restart_count += 1
        random.shuffle(candidates)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Optional: Randomly decide which set to try first to break symmetry
        first_set_A = random.random() < 0.5
        
        for x in candidates:
            # Try to add to A
            success = False
            if first_set_A:
                if can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_to_set(x, A, sums_A, diffs_A)
                    success = True
                elif can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
                    success = True
            else:
                if can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
                    success = True
                elif can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_to_set(x, A, sums_A, diffs_A)
                    success = True
            
            # If neither works, skip x
            
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
            
            # If we hit the theoretical max, we can stop early?
            # Bound is 32.
            if best_size >= 32:
                # We might want to keep searching for a while to see if 33 is possible (it's not, but just in case my bound is loose)
                # But 32 is the max derived from diff count.
                # Let's run a few more to be sure it's stable.
                pass

    if best_size == 0:
        return [[1], [2]]
        
    return [best_A, best_B]
