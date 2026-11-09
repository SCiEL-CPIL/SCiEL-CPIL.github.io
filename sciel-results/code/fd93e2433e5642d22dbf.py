"""
Strategy: Simulated Annealing on a Discrete State Space.
State: x -> {0 (unused), 1 (in A), 2 (in B)} for x in 1..256.
Objective: Maximize |A| + |B| subject to Sidon and disjoint diff constraints.
Energy function penalizes violations heavily to ensure validity, then maximizes size.
This is a global optimization approach, distinct from the local greedy construction of the parent.
"""

import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE_SECONDS = 85  # Safety margin for 90s limit

def _check_sidon(s):
    """Return True if s is a Sidon set (pair sums including doubles are unique)."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _diffs(s):
    """Return set of nonzero absolute differences for sorted s."""
    d = set()
    n = len(s)
    for i in range(n):
        for j in range(i):
            d.add(s[i] - s[j])
    return d

def _valid_pair(A, B):
    """Check both are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A) or not _check_sidon(B):
        return False
    dA = _diffs(A)
    dB = _diffs(B)
    return not (dA & dB)

def _score(A, B):
    return len(A) + len(B)

class SAState:
    def __init__(self, n):
        self.n = n
        # state[x] = 0 (unused), 1 (in A), 2 (in B)
        self.state = [0] * (n + 1)
        self.A = []
        self.B = []
        self.sums_A = set()
        self.sums_B = set()
        self.diffs_A = set()
        self.diffs_B = set()
        
    def copy(self):
        new_state = SAState(self.n)
        new_state.state = self.state[:]
        new_state.A = self.A[:]
        new_state.B = self.B[:]
        new_state.sums_A = set(self.sums_A)
        new_state.sums_B = set(self.sums_B)
        new_state.diffs_A = set(self.diffs_A)
        new_state.diffs_B = set(self.diffs_B)
        return new_state

    def _add_to_A(self, x):
        self.A.append(x)
        self.sums_A.add(x + x)
        for a in self.A[:-1]:
            self.sums_A.add(x + a)
            self.diffs_A.add(abs(x - a))
            
    def _remove_from_A(self, x):
        self.A.remove(x)
        # Recalculate sums and diffs is safer than trying to subtract
        self.sums_A = set()
        self.diffs_A = set()
        n = len(self.A)
        for i in range(n):
            ai = self.A[i]
            self.sums_A.add(ai + ai)
            for j in range(i):
                aj = self.A[j]
                self.sums_A.add(ai + aj)
                self.diffs_A.add(ai - aj)

    def _add_to_B(self, x):
        self.B.append(x)
        self.sums_B.add(x + x)
        for b in self.B[:-1]:
            self.sums_B.add(x + b)
            self.diffs_B.add(abs(x - b))
            
    def _remove_from_B(self, x):
        self.B.remove(x)
        self.sums_B = set()
        self.diffs_B = set()
        n = len(self.B)
        for i in range(n):
            bi = self.B[i]
            self.sums_B.add(bi + bi)
            for j in range(i):
                bj = self.B[j]
                self.sums_B.add(bi + bj)
                self.diffs_B.add(bi - bj)

    def get_size(self):
        return len(self.A) + len(self.B)

    def is_valid(self):
        # Check Sidon
        if len(self.sums_A) != len(self.A) * (len(self.A) + 1) // 2:
            return False
        if len(self.sums_B) != len(self.B) * (len(self.B) + 1) // 2:
            return False
        # Check disjoint diffs
        if self.diffs_A & self.diffs_B:
            return False
        # Check max size
        if len(self.A) > MAX_EACH or len(self.B) > MAX_EACH:
            return False
        return True

    def get_penalty(self):
        """
        Returns a penalty score. Lower is better.
        Base score is negative size (to maximize size).
        Large penalties for violations.
        """
        penalty = 0.0
        size = self.get_size()
        penalty -= size  # Maximize size
        
        # Sidon violations
        expected_sums_A = len(self.A) * (len(self.A) + 1) // 2
        if len(self.sums_A) < expected_sums_A:
            penalty += 100 * (expected_sums_A - len(self.sums_A))
            
        expected_sums_B = len(self.B) * (len(self.B) + 1) // 2
        if len(self.sums_B) < expected_sums_B:
            penalty += 100 * (expected_sums_B - len(self.sums_B))
            
        # Disjoint diff violations
        common_diffs = self.diffs_A & self.diffs_B
        if common_diffs:
            penalty += 100 * len(common_diffs)
            
        # Max size violations
        if len(self.A) > MAX_EACH:
            penalty += 100 * (len(self.A) - MAX_EACH)
        if len(self.B) > MAX_EACH:
            penalty += 100 * (len(self.B) - MAX_EACH)
            
        return penalty

    def try_move(self, x, new_state_val):
        """
        Attempt to change state of x to new_state_val.
        Returns (old_penalty, new_penalty, success_flag)
        We don't actually commit here, we just calculate the delta.
        However, for simplicity and speed, we will do a "what-if" calculation.
        Given the small N, we can afford to copy the state if we want to be safe,
        but modifying in place and reverting is faster.
        """
        old_val = self.state[x]
        if old_val == new_state_val:
            return 0.0, True
        
        old_penalty = self.get_penalty()
        
        # Apply move
        if old_val == 1:
            self._remove_from_A(x)
        elif old_val == 2:
            self._remove_from_B(x)
        
        self.state[x] = new_state_val
        
        if new_state_val == 1:
            self._add_to_A(x)
        elif new_state_val == 2:
            self._add_to_B(x)
            
        new_penalty = self.get_penalty()
        
        return (new_penalty - old_penalty, True)

    def revert_move(self, x, old_val):
        """Revert the last move on x back to old_val."""
        new_val = self.state[x]
        if new_val == 1:
            self._remove_from_A(x)
        elif new_val == 2:
            self._remove_from_B(x)
            
        self.state[x] = old_val
        
        if old_val == 1:
            self._add_to_A(x)
        elif old_val == 2:
            self._add_to_B(x)

def _simulated_annealing(initial_state, start_time, deadline):
    """
    Run SA on the state.
    """
    current_state = initial_state.copy()
    current_penalty = current_state.get_penalty()
    
    best_state = current_state.copy()
    best_penalty = current_penalty
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    # Number of steps is dynamic based on time
    # We'll do iterations until deadline - 2s
    
    # Precompute possible moves? No, random selection is fine.
    
    # To make SA effective, we need to explore the space.
    # Moves: pick random x, pick random new state in {0,1,2}.
    
    # We can speed up by only considering moves that might improve or are valid.
    # But SA requires accepting worse moves.
    
    # Optimization: 
    # get_penalty is O(N^2) due to set sizes and intersection.
    # With N=256, N^2 is ~65k. 10k steps * 65k = 650M ops. Too slow in Python.
    
    # We need a faster penalty evaluation or fewer steps.
    # Let's optimize the penalty calculation.
    # Instead of recalculating sums/diffs from scratch in _add/_remove, 
    # we can maintain them incrementally.
    # _add_to_A already does incremental addition.
    # _remove_from_A does a full recalculation. This is the bottleneck.
    
    # Let's rewrite remove to be incremental.
    # Removing x from A:
    # sums_A: remove all sums involving x.
    # diffs_A: remove all diffs involving x.
    
    def _remove_from_A_fast(x):
        self.A.remove(x)
        # Remove sums
        for a in self.A:
            self.sums_A.discard(x + a)
        self.sums_A.discard(x + x)
        # Remove diffs
        for a in self.A:
            self.diffs_A.discard(abs(x - a))
            
    def _remove_from_B_fast(x):
        self.B.remove(x)
        for b in self.B:
            self.sums_B.discard(x + b)
        self.sums_B.discard(x + x)
        for b in self.B:
            self.diffs_B.discard(abs(x - b))
            
    # Patch the state object methods for speed
    current_state._remove_from_A = lambda x: _remove_from_A_fast(x)
    current_state._remove_from_B = lambda x: _remove_from_B_fast(x)
    best_state._remove_from_A = lambda x: _remove_from_A_fast(x)
    best_state._remove_from_B = lambda x: _remove_from_B_fast(x)
    
    # Re-initialize penalty with fast methods? No, logic is same.
    
    # Let's estimate speed.
    # _add is O(K) where K is size of set.
    # _remove is O(K).
    # get_penalty is O(1) if we track counts?
    # Let's track:
    # num_sums_A, num_diffs_A, etc.
    # Actually, len(set) is O(1) in Python.
    # Intersection is O(min(len(A), len(B))).
    
    # So get_penalty is fast enough if K is small (<=64).
    # 10k steps * (64+64) ops = ~1.2M ops. Very fast.
    
    steps = 0
    T = T_start
    cooling_rate = 0.999
    
    while time.time() < deadline - 2:
        # Pick random x
        x = random.randint(1, self.n)
        # Pick random new state
        new_val = random.randint(0, 2)
        
        delta, _ = current_state.try_move(x, new_val)
        
        if delta <= 0 or random.random() < pow(2.71828, -delta / T):
            # Accept move
            # current_state is already updated
            current_penalty += delta
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_state = current_state.copy()
        else:
            # Revert
            current_state.revert_move(x, current_state.state[x] if False else 0) # Wait, revert needs old val
            # My try_move doesn't return old val. Let's fix try_move to return old val.
            
        steps += 1
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return best_state

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Initialize state
    init_state = SAState(N)
    
    # Run SA for a few restarts
    best_A = []
    best_B = []
    best_score = 0
    
    # We will run SA for a fixed number of steps or time per restart
    # Since SA is stochastic, multiple restarts help.
    
    # To ensure we find *something*, let's also run a quick greedy first
    # to get a decent starting point for SA.
    
    def _greedy_init():
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        active = list(range(1, N + 1))
        random.shuffle(active)
        
        for x in active:
            # Try A
            ok_A = True
            if len(A) < MAX_EACH:
                if x + x in sums_A:
                    ok_A = False
                else:
                    for a in A:
                        if x + a in sums_A:
                            ok_A = False
                            break
                    if ok_A:
                        for a in A:
                            if abs(x - a) in diffs_A or abs(x - a) in diffs_B:
                                ok_A = False
                                break
            else:
                ok_A = False
                
            # Try B
            ok_B = True
            if len(B) < MAX_EACH:
                if x + x in sums_B:
                    ok_B = False
                else:
                    for b in B:
                        if x + b in sums_B:
                            ok_B = False
                            break
                    if ok_B:
                        for b in B:
                            if abs(x - b) in diffs_B or abs(x - b) in diffs_A:
                                ok_B = False
                                break
            else:
                ok_B = False
                
            if ok_A and not ok_B:
                A.append(x)
                sums_A.add(x + x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(abs(x - a))
            elif ok_B and not ok_A:
                B.append(x)
                sums_B.add(x + x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(abs(x - b))
            elif ok_A and ok_B:
                if random.random() < 0.5:
                    A.append(x)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        diffs_A.add(abs(x - a))
                else:
                    B.append(x)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(abs(x - b))
                        
        return A, B

    # Run multiple restarts
    for i in range(20):
        if time.time() > deadline - 5:
            break
            
        # Greedy init
        A_init, B_init = _greedy_init()
        
        # Build SA state from greedy result
        sa_state = SAState(N)
        for x in A_init:
            sa_state.state[x] = 1
            sa_state.A.append(x)
        for x in B_init:
            sa_state.state[x] = 2
            sa_state.B.append(x)
        
        # Recalc sums/diffs
        sa_state.sums_A = set()
        sa_state.diffs_A = set()
        for i in range(len(sa_state.A)):
            ai = sa_state.A[i]
            sa_state.sums_A.add(ai + ai)
            for j in range(i):
                aj = sa_state.A[j]
                sa_state.sums_A.add(ai + aj)
                sa_state.diffs_A.add(ai - aj)
                
        sa_state.sums_B = set()
        sa_state.diffs_B = set()
        for i in range(len(sa_state.B)):
            bi = sa_state.B[i]
            sa_state.sums_B.add(bi + bi)
            for j in range(i):
                bj = sa_state.B[j]
                sa_state.sums_B.add(bi + bj)
                sa_state.diffs_B.add(bi - bj)
                
        # Run SA
        # We need to fix the revert logic in SAState
        # Let's implement a simpler SA that doesn't rely on complex revert logic
        # but just copies state. Copy is O(N). 10k steps * 256 = 2.5M ops. Fine.
        
        def _run_sa_simple(state, time_limit):
            current = state.copy()
            current_penalty = current.get_penalty()
            best = current.copy()
            best_penalty = current_penalty
            
            T = 5.0
            end_time = time.time() + time_limit
            
            while time.time() < end_time:
                x = random.randint(1, N)
                new_val = random.randint(0, 2)
                
                # Try move
                old_val = current.state[x]
                if old_val == new_val:
                    continue
                    
                # Apply
                if old_val == 1:
                    current._remove_from_A(x)
                elif old_val == 2:
                    current._remove_from_B(x)
                    
                current.state[x] = new_val
                
                if new_val == 1:
                    current._add_to_A(x)
                elif new_val == 2:
                    current._add_to_B(x)
                    
                new_penalty = current.get_penalty()
                delta = new_penalty - current_penalty
                
                if delta <= 0 or random.random() < pow(2.71828, -delta / T):
                    current_penalty = new_penalty
                    if new_penalty < best_penalty:
                        best_penalty = new_penalty
                        best = current.copy()
                else:
                    # Revert
                    current.state[x] = old_val
                    if new_val == 1:
                        current._remove_from_A(x)
                    elif new_val == 2:
                        current._remove_from_B(x)
                        
                    if old_val == 1:
                        current._add_to_A(x)
                    elif old_val == 2:
                        current._add_to_B(x)
                        
                T *= 0.999
                if T < 0.01:
                    T = 0.01
                    
            return best

        # Allocate time to SA
        # If we have 80s total, and 20 restarts, that's 4s per restart.
        # But greedy is fast. Let's do 10 restarts of 5s each.
        time_per_sa = 4.0
        
        sa_result = _run_sa_simple(sa_state, time_per_sa)
        
        # Extract result
        A_res = sa_result.A
        B_res = sa_result.B
        
        if _valid_pair(A_res, B_res):
            score = _score(A_res, B_res)
            if score > best_score:
                best_score = score
                best_A = A_res
                best_B = B_res
                
    # Fallback
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [3]
        
    if not _valid_pair(best_A, best_B):
        best_A = [1]
        best_B = [3]
        
    return [best_A, best_B]

if __name__ == "__main__":
    A, B = construct()
    print(f"A: {len(A)}, B: {len(B)}, Total: {len(A)+len(B)}")
    print(_valid_pair(A, B))
