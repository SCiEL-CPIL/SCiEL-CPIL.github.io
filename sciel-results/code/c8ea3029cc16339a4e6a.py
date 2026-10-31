import math
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s buffer for 120s limit
    
    # Helper to calculate score
    def calc_score(U):
        if not U:
            return 0.0
        M = max(U)
        if M == 0:
            return 1.0
        
        # Calculate U+U and U-U sizes
        # Using sets for simplicity first, then optimize if needed
        # But for speed in search, we need fast updates.
        # For now, let's use a helper that takes the current bitsets and list
        
        # This function is for initial checks or if we don't maintain state
        sums = set()
        diffs = set()
        for u in U:
            for v in U:
                sums.add(u + v)
                diffs.add(u - v)
        
        sum_count = len(sums)
        diff_count = len(diffs)
        
        if sum_count == 0:
            return 0.0
            
        ratio = diff_count / sum_count
        # log base 2 or 10 doesn't matter for the ratio part, but denominator is log(2M+1)
        # Formula: 1 + log(ratio) / log(2M+1)
        # If ratio <= 1, log(ratio) <= 0, score <= 1.
        # We want ratio > 1.
        
        try:
            score = 1.0 + math.log(ratio) / math.log(2 * M + 1)
        except ValueError:
            return 1.0
            
        return score

    # Optimized State Class for Hill Climbing
    class SetState:
        def __init__(self, U):
            self.U = list(U)
            self.M = max(self.U) if self.U else 0
            self.sums_bits = 0
            self.pos_diffs_bits = 0
            self._recompute()
            
        def _recompute(self):
            self.sums_bits = 0
            self.pos_diffs_bits = 0
            U_bits = 0
            for u in self.U:
                U_bits |= (1 << u)
            # Sums
            for u in self.U:
                self.sums_bits |= (U_bits << u)
            # Pos Diffs
            for u in self.U:
                for v in self.U:
                    if u > v:
                        self.pos_diffs_bits |= (1 << (u - v))
                        
        def add(self, x):
            if x in self.U:
                return False
            # Check if adding x improves score
            # We need to calculate new sums and diffs
            # New sums: old_sums | (U_bits << x) | (1 << 2x)
            # New pos_diffs: old_pos_diffs | (diffs involving x)
            
            U_bits = 0
            for u in self.U:
                U_bits |= (1 << u)
                
            new_sums = self.sums_bits | (U_bits << x) | (1 << (2 * x))
            
            new_pos_diffs = self.pos_diffs_bits
            for u in self.U:
                if x > u:
                    new_pos_diffs |= (1 << (x - u))
                else:
                    new_pos_diffs |= (1 << (u - x))
                    
            new_M = max(self.M, x)
            
            sum_count = new_sums.bit_count()
            pos_diff_count = new_pos_diffs.bit_count()
            diff_count = 1 + 2 * pos_diff_count
            
            if sum_count == 0:
                return False
                
            ratio = diff_count / sum_count
            score = 1.0 + math.log(ratio) / math.log(2 * new_M + 1)
            
            # Store temporary state to check improvement
            self._temp_sums = new_sums
            self._temp_pos_diffs = new_pos_diffs
            self._temp_M = new_M
            self._temp_score = score
            self._temp_U = self.U + [x]
            
            return score
            
        def commit_add(self):
            self.U = self._temp_U
            self.M = self._temp_M
            self.sums_bits = self._temp_sums
            self.pos_diffs_bits = self._temp_pos_diffs
            
        def remove(self, x):
            if x not in self.U:
                return False
            # Recalculate from scratch is easier for remove
            # Or track counts, but bitsets don't support decrement easily without recompute
            # For n < 100, recompute is fast enough for remove?
            # Let's just recompute for simplicity in remove, or use a multiset of bits?
            # Given time limit, recompute for remove is O(n^2).
            # If n=50, 2500 ops. Fast.
            
            new_U = [u for u in self.U if u != x]
            if not new_U:
                return 0.0
                
            # Recompute bits
            new_sums = 0
            new_pos_diffs = 0
            U_bits = 0
            for u in new_U:
                U_bits |= (1 << u)
            for u in new_U:
                new_sums |= (U_bits << u)
            for u in new_U:
                for v in new_U:
                    if u > v:
                        new_pos_diffs |= (1 << (u - v))
                        
            new_M = max(new_U)
            
            sum_count = new_sums.bit_count()
            pos_diff_count = new_pos_diffs.bit_count()
            diff_count = 1 + 2 * pos_diff_count
            
            if sum_count == 0:
                return 0.0
                
            ratio = diff_count / sum_count
            score = 1.0 + math.log(ratio) / math.log(2 * new_M + 1)
            
            self._temp_sums = new_sums
            self._temp_pos_diffs = new_pos_diffs
            self._temp_M = new_M
            self._temp_score = score
            self._temp_U = new_U
            
            return score

        def commit_remove(self):
            self.U = self._temp_U
            self.M = self._temp_M
            self.sums_bits = self._temp_sums
            self.pos_diffs_bits = self._temp_pos_diffs

    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Initial best check
    initial_state = SetState(best_set)
    initial_score = calc_score(best_set)
    if initial_score > best_score:
        best_score = initial_score

    # Hill Climbing
    # Try different initial sizes and ranges
    for n_init in [5, 10, 20, 30]:
        for M_init in [20, 50, 100, 200]:
            # Generate random set
            candidates = random.sample(range(M_init + 1), n_init)
            if 0 not in candidates:
                candidates[0] = 0
            state = SetState(candidates)
            current_score = calc_score(state.U)
            
            if current_score > best_score:
                best_score = current_score
                best_set = state.U[:]
            
            # Local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try adding
                for _ in range(100): # Limit attempts per step
                    x = random.randint(0, M_init + 20)
                    if x in state.U:
                        continue
                    score = state.add(x)
                    if score is not None and score > current_score:
                        state.commit_add()
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = state.U[:]
                            M_init = max(M_init, state.M) # Expand range if needed
                        break
                        
                # Try removing
                if state.U and time.time() < deadline:
                    for _ in range(10):
                        idx = random.randint(0, len(state.U) - 1)
                        x = state.U[idx]
                        score = state.remove(x)
                        if score > current_score:
                            state.commit_remove()
                            current_score = score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = state.U[:]
                            break

    # Final polish: try to swap elements
    # (Omitted for brevity, add/remove is usually sufficient)
    
    return best_set
