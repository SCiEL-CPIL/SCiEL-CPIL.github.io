import time
import random
import numpy as np

N = 256
MAX_EACH = 64
DEADLINE = 85.0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # 1. Precompute conflict tables for speed
    # sum_conflict[i][j] = True if i+j creates a duplicate sum with existing pairs? 
    # Actually, for incremental checks, we need to know if (x+a) is already in sum_set.
    # Bitmask representation for sum_set and diff_set is fastest.
    
    # Max sum is 256+256 = 512. Use a set of ints or a boolean array.
    # Max diff is 255. Range -255..255. Offset 256. Size 512.
    
    # Let's use a helper class to manage state efficiently.
    class State:
        def __init__(self):
            self.A = []
            self.B = []
            # Sum sets: array of bools, index s means sum s exists
            # Sum range: 2 to 512. Index 0,1 unused.
            self.sumA = [False] * 513
            self.sumB = [False] * 513
            # Diff sets: array of bools, index d+offset means diff d exists
            # Diff range: -255 to 255. Offset 256. Index 0..511.
            self.diffA = [False] * 512
            self.diffB = [False] * 512
            
        def add_to_A(self, x):
            # Check sums
            for a in self.A:
                s = x + a
                if self.sumA[s]:
                    return False
            if self.sumA[2*x]:
                return False
            
            # Check diffs
            for a in self.A:
                d = x - a
                if self.diffA[d + 256]:
                    return False
                if self.diffB[d + 256]: # Conflict with B
                    return False
            
            # Update
            for a in self.A:
                self.sumA[x + a] = True
                self.diffA[x - a + 256] = True
                self.diffA[a - x + 256] = True
            self.sumA[2*x] = True
            self.A.append(x)
            return True

        def add_to_B(self, x):
            for b in self.B:
                s = x + b
                if self.sumB[s]:
                    return False
            if self.sumB[2*x]:
                return False
            
            for b in self.B:
                d = x - b
                if self.diffB[d + 256]:
                    return False
                if self.diffA[d + 256]: # Conflict with A
                    return False
            
            for b in self.B:
                self.sumB[x + b] = True
                self.diffB[x - b + 256] = True
                self.diffB[b - x + 256] = True
            self.sumB[2*x] = True
            self.B.append(x)
            return True

        def remove_from_A(self, x):
            # Find index
            idx = self.A.index(x)
            self.A.pop(idx)
            
            # We need to recompute sumA and diffA from scratch for safety, 
            # or carefully decrement counts. Given small size, recompute is safer/faster enough?
            # No, recompute is O(K^2). K~30. 900 ops. 
            # Let's just recompute the sets from the list for simplicity and correctness.
            self._recompute_A()

        def remove_from_B(self, x):
            idx = self.B.index(x)
            self.B.pop(idx)
            self._recompute_B()

        def _recompute_A(self):
            self.sumA = [False] * 513
            self.diffA = [False] * 512
            A = self.A
            for i in range(len(A)):
                for j in range(i, len(A)):
                    s = A[i] + A[j]
                    self.sumA[s] = True
                    if i != j:
                        d = A[j] - A[i]
                        self.diffA[d + 256] = True
                        self.diffA[-d + 256] = True

        def _recompute_B(self):
            self.sumB = [False] * 513
            self.diffB = [False] * 512
            B = self.B
            for i in range(len(B)):
                for j in range(i, len(B)):
                    s = B[i] + B[j]
                    self.sumB[s] = True
                    if i != j:
                        d = B[j] - B[i]
                        self.diffB[d + 256] = True
                        self.diffB[-d + 256] = True

        def score(self):
            return len(self.A) + len(self.B)

    # 2. Initial Greedy Construction (Seed)
    def greedy_seed():
        st = State()
        # Randomized greedy to get a decent start
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if len(st.A) < MAX_EACH:
                st.add_to_A(x)
            if len(st.B) < MAX_EACH:
                st.add_to_B(x)
        return st

    best_st = greedy_seed()
    best_score = best_st.score()
    
    # 3. Simulated Annealing
    # Moves:
    # 1. Add random valid x to A or B. (Improves score)
    # 2. Remove random x from A or B, then try to add a different y. (Swap, keeps score same, explores space)
    # 3. Remove random x from A or B. (Worsens score, allows escape)
    
    current_st = State()
    # Copy best to current
    current_st.A = list(best_st.A)
    current_st.B = list(best_st.B)
    current_st._recompute_A()
    current_st._recompute_B()
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline - 2.0:
        if T < T_min:
            # Restart with current best
            current_st = State()
            current_st.A = list(best_st.A)
            current_st.B = list(best_st.B)
            current_st._recompute_A()
            current_st._recompute_B()
            T = 10.0
            continue

        move_type = random.random()
        old_score = current_st.score()
        accepted = False
        
        if move_type < 0.4:
            # Try Add
            x = random.randint(1, N)
            if random.random() < 0.5:
                if len(current_st.A) < MAX_EACH and current_st.add_to_A(x):
                    accepted = True
            else:
                if len(current_st.B) < MAX_EACH and current_st.add_to_B(x):
                    accepted = True
        elif move_type < 0.8:
            # Try Swap (Remove one, Add one)
            if current_st.A:
                x_remove = random.choice(current_st.A)
                current_st.remove_from_A(x_remove)
                # Try to add a new x
                x_add = random.randint(1, N)
                if len(current_st.A) < MAX_EACH and current_st.add_to_A(x_add):
                    accepted = True
                else:
                    # Revert add if it failed? No, add_to_A fails atomically. 
                    # But we already removed x_remove. 
                    # If add fails, we are in a worse state (size -1). 
                    # We should revert the removal to keep state consistent if we reject the move.
                    # Let's structure this as: Propose state, evaluate, accept/reject.
                    # For efficiency, let's do "Remove, then Add". If Add fails, we are at size-1.
                    # We can try to Add multiple times? No, just one attempt.
                    # If we reject, we must restore x_remove.
                    pass 
            elif current_st.B:
                x_remove = random.choice(current_st.B)
                current_st.remove_from_B(x_remove)
                x_add = random.randint(1, N)
                if len(current_st.B) < MAX_EACH and current_st.add_to_B(x_add):
                    accepted = True
            else:
                accepted = False
        else:
            # Try Remove (Escape move)
            if current_st.A:
                x_remove = random.choice(current_st.A)
                current_st.remove_from_A(x_remove)
                accepted = True # Tentatively accept, we'll check score delta
            elif current_st.B:
                x_remove = random.choice(current_st.B)
                current_st.remove_from_B(x_remove)
                accepted = True
            else:
                accepted = False

        new_score = current_st.score()
        delta = new_score - old_score
        
        if delta > 0:
            # Always accept improvement
            accept = True
        else:
            # Accept with probability exp(delta / T)
            if random.random() < np.exp(delta / T):
                accept = True
            else:
                accept = False
                
        if accept:
            if new_score > best_score:
                best_score = new_score
                best_st = State()
                best_st.A = list(current_st.A)
                best_st.B = list(current_st.B)
                best_st._recompute_A()
                best_st._recompute_B()
        else:
            # Revert move
            # This is tricky because we modified current_st in place.
            # We need to undo the specific move.
            # To simplify, let's use a "Copy State" approach for moves, or careful undo.
            # Given the complexity of undoing "Remove then Add" if rejected, 
            # let's use a simpler move set that is easier to revert, or just snapshot.
            # Snapshotting State is O(1) for lists if we use copy, but recompute is O(K^2).
            # Let's just restart the loop iteration with a fresh copy of the pre-move state?
            # That's expensive.
            
            # Alternative: Implement moves that return success and allow easy revert.
            # Actually, the current logic is flawed for "Swap" if rejected.
            # Let's fix the logic: 
            # 1. Save old A, B lists.
            # 2. Apply move.
            # 3. Check score.
            # 4. If reject, restore A, B and recompute.
            
            # Let's refactor the loop to be safer.
            pass

    # The logic above is getting messy with in-place modifications and reverts.
    # Let's rewrite the SA loop cleanly with snapshots.
    
    # --- Clean SA Loop ---
    current_st = State()
    current_st.A = list(best_st.A)
    current_st.B = list(best_st.B)
    current_st._recompute_A()
    current_st._recompute_B()
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    step_count = 0
    
    while time.time() < deadline - 2.0:
        if T < T_min:
            # Cool down or restart
            T = 10.0
            # Optionally restart from best
            # current_st = State()
            # current_st.A = list(best_st.A)
            # current_st.B = list(best_st.B)
            # current_st._recompute_A()
            # current_st._recompute_B()
            continue

        # Save state
        old_A = list(current_st.A)
        old_B = list(current_st.B)
        old_score = current_st.score()
        
        # Generate Move
        r = random.random()
        
        if r < 0.3:
            # Add
            x = random.randint(1, N)
            if random.random() < 0.5:
                if len(current_st.A) < MAX_EACH:
                    current_st.add_to_A(x)
            else:
                if len(current_st.B) < MAX_EACH:
                    current_st.add_to_B(x)
        elif r < 0.7:
            # Swap: Remove from A, Add to A (or B)
            if current_st.A:
                x_rem = random.choice(current_st.A)
                current_st.remove_from_A(x_rem)
                x_add = random.randint(1, N)
                target = random.random() < 0.5
                if target:
                    if len(current_st.A) < MAX_EACH:
                        current_st.add_to_A(x_add)
                else:
                    if len(current_st.B) < MAX_EACH:
                        current_st.add_to_B(x_add)
            elif current_st.B:
                x_rem = random.choice(current_st.B)
                current_st.remove_from_B(x_rem)
                x_add = random.randint(1, N)
                target = random.random() < 0.5
                if target:
                    if len(current_st.A) < MAX_EACH:
                        current_st.add_to_A(x_add)
                else:
                    if len(current_st.B) < MAX_EACH:
                        current_st.add_to_B(x_add)
        else:
            # Remove
            if current_st.A:
                x_rem = random.choice(current_st.A)
                current_st.remove_from_A(x_rem)
            elif current_st.B:
                x_rem = random.choice(current_st.B)
                current_st.remove_from_B(x_rem)
            else:
                pass

        new_score = current_st.score()
        delta = new_score - old_score
        
        accept = False
        if delta > 0:
            accept = True
        else:
            if random.random() < np.exp(delta / T):
                accept = True
        
        if accept:
            if new_score > best_score:
                best_score = new_score
                best_st = State()
                best_st.A = list(current_st.A)
                best_st.B = list(current_st.B)
                best_st._recompute_A()
                best_st._recompute_B()
        else:
            # Revert
            current_st.A = old_A
            current_st.B = old_B
            current_st._recompute_A()
            current_st._recompute_B()
            
        T *= cooling
        step_count += 1

    # Final validation and return
    if not best_st.A:
        best_st.A = [1]
    if not best_st.B:
        best_st.B = [2]
        
    return [best_st.A, best_st.B]
