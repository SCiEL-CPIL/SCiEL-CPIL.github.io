import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0
    
    prescribed = [1, 2, 5, 11]
    pres_set = set(prescribed)
    N = 284
    
    def remaining():
        return DEADLINE - time.time()

    def is_valid_set(s):
        if len(s) != len(set(s)):
            return False
        if not pres_set.issubset(set(s)):
            return False
        for v in s:
            if v < 1 or v > N:
                return False
        seen = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                v = s[i] + s[j]
                if v in seen:
                    return False
                seen.add(v)
        return True

    # Greedy to get a quick baseline
    def greedy_build():
        arr = list(prescribed)
        used = set(arr)
        sums = set()
        for a in arr:
            for b in arr:
                if a <= b:
                    sums.add(a + b)
        
        # Sort candidates to try to fit more
        # Simple greedy: add smallest possible
        for x in range(1, N + 1):
            if x in used:
                continue
            ok = True
            new_sums = []
            for a in arr:
                s = a + x
                if s in sums:
                    ok = False
                    break
                new_sums.append(s)
            if ok and (x + x) in sums:
                ok = False
            if ok:
                sums.update(new_sums)
                sums.add(x + x)
                used.add(x)
                arr.append(x)
        return arr

    # State for SA
    class State:
        __slots__ = ('arr', 'n', 'counts', 'conflict', 'val_counts', 'fixed', 'sums_map')

        def __init__(self, arr, fixed_mask):
            self.arr = arr
            self.n = len(arr)
            self.fixed = fixed_mask
            self.counts = {}
            self.conflict = 0
            self.val_counts = {}
            
            # Build sum counts
            for i in range(self.n):
                vi = arr[i]
                self.val_counts[vi] = self.val_counts.get(vi, 0) + 1
                for j in range(i, self.n):
                    s = vi + arr[j]
                    c = self.counts.get(s, 0)
                    if c == 1:
                        self.conflict += 1
                    self.counts[s] = c + 1

        def _inc_sum(self, s):
            c = self.counts.get(s, 0)
            if c == 1:
                self.conflict += 1
            self.counts[s] = c + 1

        def _dec_sum(self, s):
            c = self.counts[s]
            if c == 2:
                self.conflict -= 1
            if c == 1:
                del self.counts[s]
            else:
                self.counts[s] = c - 1

        def propose_move(self, i, new_val):
            """Move element i to new_val. Returns new conflict count or None if invalid."""
            old = self.arr[i]
            if old == new_val:
                return self.conflict
            if self.val_counts.get(new_val, 0) > 0:
                return None # Collision on value itself
            
            # Remove old sums
            for j in range(self.n):
                self._dec_sum(old + self.arr[j])
            self.val_counts[old] -= 1
            if self.val_counts[old] == 0:
                del self.val_counts[old]
            
            # Set new value
            self.arr[i] = new_val
            for j in range(self.n):
                self._inc_sum(new_val + self.arr[j])
            self.val_counts[new_val] = 1
            
            return self.conflict

        def revert_move(self, i, old_val, new_val):
            """Revert move of i from new_val back to old_val."""
            # Remove new sums
            for j in range(self.n):
                self._dec_sum(new_val + self.arr[j])
            self.val_counts[new_val] -= 1
            if self.val_counts[new_val] == 0:
                del self.val_counts[new_val]
            
            # Restore old value
            self.arr[i] = old_val
            for j in range(self.n):
                self._inc_sum(old_val + self.arr[j])
            self.val_counts[old_val] = 1

    def run_sa(target_size, init_arr, budget):
        """
        Runs SA to minimize conflicts for a set of target_size.
        Returns a valid list if conflicts reach 0, else None.
        """
        end_time = min(time.time() + budget, DEADLINE - 1.0)
        
        # Prepare fixed mask
        fixed_mask = [v in pres_set for v in init_arr]
        st = State(init_arr, fixed_mask)
        
        best_conflict = st.conflict
        best_arr = list(init_arr)
        
        T = 1.0
        T_end = 0.01
        cool = 0.9999
        
        no_improve = 0
        improve_limit = 500
        
        while time.time() < end_time and T > T_end:
            # Choose a non-fixed element to move
            # Weighted towards elements with high conflict contribution?
            # For now, random non-fixed
            i = random.randrange(target_size)
            if fixed_mask[i]:
                continue
                
            old_val = st.arr[i]
            cur_conflict = st.conflict
            
            # Choose new value
            # Strategy: random in [1, N]
            new_val = random.randint(1, N)
            
            nc = st.propose_move(i, new_val)
            if nc is None:
                continue
                
            if nc <= cur_conflict:
                if nc < best_conflict:
                    best_conflict = nc
                    best_arr = list(st.arr)
                    no_improve = 0
                    if nc == 0:
                        return best_arr
                else:
                    no_improve += 1
            else:
                # Accept if random() < exp((cur - new) / T)
                if random.random() < math.exp((cur_conflict - nc) / T):
                    no_improve += 1
                else:
                    st.revert_move(i, old_val, new_val)
                    no_improve += 1
            
            T *= cool
            
            if no_improve > improve_limit:
                # Restart or break?
                # Let's try a few restarts with higher T
                if random.random() < 0.1:
                    T = 1.0
                    no_improve = 0
                else:
                    break
        
        if best_conflict == 0:
            return best_arr
        return None

    # Initial greedy solution
    best_set = greedy_build()
    best_size = len(best_set)
    
    # Try to improve size using SA growth
    # We start from best_size and try to go up to 20
    for target in range(best_size + 1, 21):
        if remaining() < 10:
            break
            
        # Try multiple times to grow to this target
        success = False
        attempts = 0
        max_attempts = 20
        while not success and attempts < max_attempts and remaining() > 5:
            attempts += 1
            # Create initial state: take best_set, add a random element
            # Ensure the added element is not in best_set
            pool = [x for x in range(1, N + 1) if x not in set(best_set)]
            if not pool:
                break
            random_add = random.choice(pool)
            init_arr = list(best_set) + [random_add]
            
            # Run SA with a time budget
            # Allocate time based on remaining time
            time_per_attempt = min(5.0, (remaining() - 2.0) / (max_attempts - attempts + 1))
            res = run_sa(target, init_arr, time_per_attempt)
            
            if res is not None and is_valid_set(res):
                best_set = res
                best_size = target
                success = True
                break
        
        if not success:
            # If we failed to grow to this target, we might still try to polish current best?
            # Or just break, as larger targets are harder.
            # But maybe we can find a better configuration for current size?
            # The task asks to maximize size. So if we can't grow, we stop trying to grow.
            break

    # Final polish: Try to improve the current best set's quality (not size, just validity/robustness)
    # Actually, if is_valid_set(best_set) is True, we are good.
    # If we have time, we can try to see if we can grow further with a more aggressive search?
    # The loop above already tries to grow.
    
    # Ensure we return a valid set
    if not is_valid_set(best_set):
        # Fallback to greedy if something went wrong
        best_set = greedy_build()
        
    return best_set

# Test
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}, Set: {res}")
    print(f"Valid: {is_valid_set(res)}")
