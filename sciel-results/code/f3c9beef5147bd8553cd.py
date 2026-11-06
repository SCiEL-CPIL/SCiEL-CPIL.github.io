import random
import math
import time

def construct():
    n = 256
    max_time = 55.0  # Leave 5s buffer
    start_time = time.time()
    
    # Helper to compute sums and diffs for a set
    def get_sums_diffs(S):
        sums = set()
        diffs = set()
        for i, a in enumerate(S):
            sums.add(2*a)
            for b in S[i+1:]:
                sums.add(a+b)
                diffs.add(a-b)
                diffs.add(b-a)
        return sums, diffs

    # Greedy initialization
    def greedy_init():
        A = []
        B = []
        sums_A, diffs_A = set(), set()
        sums_B, diffs_B = set(), set()
        
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        
        for x in candidates:
            can_A = True
            if 2*x in sums_A: can_A = False
            if can_A:
                for a in A:
                    if a+x in sums_A:
                        can_A = False
                        break
            if can_A:
                for a in A:
                    if x-a in diffs_B or a-x in diffs_B:
                        can_A = False
                        break
            
            can_B = True
            if 2*x in sums_B: can_B = False
            if can_B:
                for b in B:
                    if b+x in sums_B:
                        can_B = False
                        break
            if can_B:
                for b in B:
                    if x-b in diffs_A or b-x in diffs_A:
                        can_B = False
                        break
            
            if can_A and can_B:
                if len(A) <= len(B):
                    A.append(x)
                    sums_A.add(2*x)
                    for a in A[:-1]:
                        sums_A.add(a+x)
                        diffs_A.add(x-a)
                        diffs_A.add(a-x)
                else:
                    B.append(x)
                    sums_B.add(2*x)
                    for b in B[:-1]:
                        sums_B.add(b+x)
                        diffs_B.add(x-b)
                        diffs_B.add(b-x)
            elif can_A:
                A.append(x)
                sums_A.add(2*x)
                for a in A[:-1]:
                    sums_A.add(a+x)
                    diffs_A.add(x-a)
                    diffs_A.add(a-x)
            elif can_B:
                B.append(x)
                sums_B.add(2*x)
                for b in B[:-1]:
                    sums_B.add(b+x)
                    diffs_B.add(x-b)
                    diffs_B.add(b-x)
                    
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    # Initial state
    A, B, sums_A, diffs_A, sums_B, diffs_B = greedy_init()
    best_A, best_B = A[:], B[:]
    best_score = len(A) + len(B)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    def valid_add(x, S, sums_S, diffs_S, diffs_other):
        if 2*x in sums_S:
            return False
        for s in S:
            if s + x in sums_S:
                return False
            if x - s in diffs_other or s - x in diffs_other:
                return False
        return True

    def do_add(x, S, sums_S, diffs_S, diffs_other, is_A):
        # S is the list to append to
        # We need to update sums_S and diffs_S
        # And check against diffs_other (which is the other set's diffs)
        # Note: diffs_S is the set of diffs for S. diffs_other is diffs of the OTHER set.
        # Wait, the constraint is: diffs_A and diffs_B must be disjoint.
        # So if adding to A, we check new diffs against diffs_B.
        # The function valid_add above does this.
        
        # Perform the add
        S.append(x)
        sums_S.add(2*x)
        for s in S[:-1]:
            sums_S.add(s + x)
            diffs_S.add(x - s)
            diffs_S.add(s - x)

    def do_remove(x, S, sums_S, diffs_S):
        # Remove x from S
        S.remove(x)
        sums_S.discard(2*x)
        for s in S:
            sums_S.discard(s + x)
            diffs_S.discard(x - s)
            diffs_S.discard(s - x)

    while time.time() - start_time < max_time:
        # Choose a random move
        # 1. Add to A
        # 2. Add to B
        # 3. Remove from A
        # 4. Remove from B
        # 5. Move from A to B
        # 6. Move from B to A
        
        move_type = random.randint(0, 5)
        old_score = len(A) + len(B)
        
        # We need to be careful with state changes.
        # It's safer to simulate the move, check validity, and if accepted, commit.
        # Or commit and rollback.
        
        if move_type == 0: # Add to A
            candidates = [x for x in range(1, n+1) if x not in A]
            if not candidates: continue
            x = random.choice(candidates)
            if valid_add(x, A, sums_A, diffs_A, diffs_B):
                # Commit
                do_add(x, A, sums_A, diffs_A, diffs_B, True)
                new_score = len(A) + len(B)
                if new_score > best_score:
                    best_score = new_score
                    best_A = A[:]
                    best_B = B[:]
                # Always accept valid add? No, SA logic.
                # But add increases score, so delta > 0. Always accept.
                # Wait, if we accept a bad add that blocks future adds, we might get stuck?
                # But we can remove later.
                # Standard SA: accept if better, or with prob.
                # Here, adding always improves score (by 1). So always accept.
            else:
                # Invalid, do nothing
                pass
        elif move_type == 1: # Add to B
            candidates = [x for x in range(1, n+1) if x not in B]
            if not candidates: continue
            x = random.choice(candidates)
            if valid_add(x, B, sums_B, diffs_B, diffs_A):
                do_add(x, B, sums_B, diffs_B, diffs_A, False)
                new_score = len(A) + len(B)
                if new_score > best_score:
                    best_score = new_score
                    best_A = A[:]
                    best_B = B[:]
        elif move_type == 2: # Remove from A
            if A:
                x = random.choice(A)
                # Removing is always valid (it relaxes constraints)
                # Score decreases by 1.
                # Accept with probability exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    do_remove(x, A, sums_A, diffs_A)
                    new_score = len(A) + len(B)
                    # Update best? No, score decreased.
        elif move_type == 3: # Remove from B
            if B:
                x = random.choice(B)
                if random.random() < math.exp(-1.0 / T):
                    do_remove(x, B, sums_B, diffs_B)
                    new_score = len(A) + len(B)
        elif move_type == 4: # Move A -> B
            if A:
                x = random.choice(A)
                # Check if x can be added to B given A without x
                # Temporarily remove x from A
                do_remove(x, A, sums_A, diffs_A)
                # Check valid add to B
                if valid_add(x, B, sums_B, diffs_B, diffs_A):
                    do_add(x, B, sums_B, diffs_B, diffs_A, False)
                    # Score stays same
                    # Accept with prob 1? Or SA?
                    # Since score is same, we can accept to explore.
                    # Let's accept with prob 0.5 or similar to mix.
                    if random.random() < 0.5:
                        new_score = len(A) + len(B)
                        if new_score > best_score:
                            best_score = new_score
                            best_A = A[:]
                            best_B = B[:]
                    else:
                        # Rollback
                        do_remove(x, B, sums_B, diffs_B)
                        do_add(x, A, sums_A, diffs_A, diffs_B, True)
                else:
                    # Rollback
                    do_add(x, A, sums_A, diffs_A, diffs_B, True)
        elif move_type == 5: # Move B -> A
            if B:
                x = random.choice(B)
                do_remove(x, B, sums_B, diffs_B)
                if valid_add(x, A, sums_A, diffs_A, diffs_B):
                    do_add(x, A, sums_A, diffs_A, diffs_B, True)
                    if random.random() < 0.5:
                        new_score = len(A) + len(B)
                        if new_score > best_score:
                            best_score = new_score
                            best_A = A[:]
                            best_B = B[:]
                    else:
                        do_remove(x, A, sums_A, diffs_A)
                        do_add(x, B, sums_B, diffs_B, diffs_A, False)
                else:
                    do_add(x, B, sums_B, diffs_B, diffs_A, False)
        
        T *= cooling
        if T < T_min:
            T = T_min

    if not best_A: best_A = [1]
    if not best_B: best_B = [2]
    return [best_A, best_B]
