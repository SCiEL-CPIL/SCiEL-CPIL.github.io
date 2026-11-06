"""Two Sidon sets with disjoint difference sets.
Uses a structured construction based on modular arithmetic plus aggressive
local search with bitmask-based collision detection."""

import time
import random
import numpy as np

N = 256
MAX_SUM = 2 * N  # 512
DIFF_OFFSET = N  # 256, so diffs in [-255, 255] map to [1, 511]

def _check_sidon(els):
    """Check all pair sums a+b (a<=b) are unique."""
    n = len(els)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _nonzero_diffs(els):
    """Set of all nonzero differences (positive and negative)."""
    d = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = els[j] - els[i]
                d.add(diff)
    return d

def _valid(A, B):
    """Check A,B are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A):
        return False
    if not _check_sidon(B):
        return False
    if _nonzero_diffs(A) & _nonzero_diffs(B):
        return False
    return True

class State:
    """State using Python integers as bitmasks for fast collision detection."""
    def __init__(self, A, B):
        self.A = sorted(A)
        self.B = sorted(B)
        self.sums_A = 0
        self.sums_B = 0
        self.diffs_A = 0
        self.diffs_B = 0
        self._rebuild()
    
    def _rebuild(self):
        self.sums_A = 0
        for i in range(len(self.A)):
            for j in range(i, len(self.A)):
                s = self.A[i] + self.A[j]
                self.sums_A |= (1 << s)
        self.sums_B = 0
        for i in range(len(self.B)):
            for j in range(i, len(self.B)):
                s = self.B[i] + self.B[j]
                self.sums_B |= (1 << s)
        self.diffs_A = 0
        for i in range(len(self.A)):
            for j in range(len(self.A)):
                if i != j:
                    d = self.A[j] - self.A[i] + DIFF_OFFSET
                    self.diffs_A |= (1 << d)
        self.diffs_B = 0
        for i in range(len(self.B)):
            for j in range(len(self.B)):
                if i != j:
                    d = self.B[j] - self.B[i] + DIFF_OFFSET
                    self.diffs_B |= (1 << d)
    
    def can_add_to_A(self, x):
        """Check if x can be added to A while maintaining all constraints."""
        # Sidon check for A
        if (self.sums_A >> (2*x)) & 1:
            return False
        for a in self.A:
            if (self.sums_A >> (x + a)) & 1:
                return False
        # Diff disjointness: new diffs from x must not be in diffs_B
        for a in self.A:
            d1 = x - a + DIFF_OFFSET
            d2 = a - x + DIFF_OFFSET
            if d1 != DIFF_OFFSET and (self.diffs_B >> d1) & 1:
                return False
            if d2 != DIFF_OFFSET and (self.diffs_B >> d2) & 1:
                return False
        return True
    
    def can_add_to_B(self, x):
        """Check if x can be added to B while maintaining all constraints."""
        if (self.sums_B >> (2*x)) & 1:
            return False
        for b in self.B:
            if (self.sums_B >> (x + b)) & 1:
                return False
        for b in self.B:
            d1 = x - b + DIFF_OFFSET
            d2 = b - x + DIFF_OFFSET
            if d1 != DIFF_OFFSET and (self.diffs_A >> d1) & 1:
                return False
            if d2 != DIFF_OFFSET and (self.diffs_A >> d2) & 1:
                return False
        return True
    
    def add_to_A(self, x):
        self.A.append(x)
        self.A.sort()
        self.sums_A |= (1 << (2*x))
        for a in self.A:
            if a != x:
                self.sums_A |= (1 << (a + x))
        for a in self.A:
            if a != x:
                self.diffs_A |= (1 << (x - a + DIFF_OFFSET))
                self.diffs_A |= (1 << (a - x + DIFF_OFFSET))
    
    def add_to_B(self, x):
        self.B.append(x)
        self.B.sort()
        self.sums_B |= (1 << (2*x))
        for b in self.B:
            if b != x:
                self.sums_B |= (1 << (b + x))
        for b in self.B:
            if b != x:
                self.diffs_B |= (1 << (x - b + DIFF_OFFSET))
                self.diffs_B |= (1 << (b - x + DIFF_OFFSET))
    
    def remove_from_A(self, x):
        self.A.remove(x)
        self.A.sort()
        self._rebuild()
    
    def remove_from_B(self, x):
        self.B.remove(x)
        self.B.sort()
        self._rebuild()
    
    def score(self):
        return len(self.A) + len(self.B)

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    n = 256
    max_size = 64

    best_A = [1]
    best_B = [2]
    best_score = 2

    rng = random.Random(12345)

    def _greedy_build(A0, B0, candidates_order=None):
        """Build A and B from seeds using greedy additions."""
        A = list(A0) if A0 else []
        B = list(B0) if B0 else []
        
        if not A:
            A.append(rng.randint(1, n))
        if not B:
            B.append(rng.randint(1, n))
        
        A = list(set(A))
        B = list(set(B))
        
        if not _valid(A, B):
            return None, None, 0
        
        state = State(A, B)
        
        if candidates_order is None:
            candidates = list(range(1, n + 1))
            rng.shuffle(candidates)
        else:
            candidates = candidates_order
        
        # Greedy: alternate adding to A and B
        for _round in range(500):
            if time.time() > deadline - 5:
                break
            progress = False
            
            if len(state.A) < max_size:
                for x in candidates:
                    if x in state.A:
                        continue
                    if state.can_add_to_A(x):
                        state.add_to_A(x)
                        progress = True
                        break
            
            if len(state.B) < max_size:
                for x in candidates:
                    if x in state.B:
                        continue
                    if state.can_add_to_B(x):
                        state.add_to_B(x)
                        progress = True
                        break
            
            if not progress:
                break
        
        if _valid(state.A, state.B):
            return state.A, state.B, len(state.A) + len(state.B)
        return None, None, 0

    def _local_search(state, time_limit=3.0):
        """Local search with add, swap, and move operations."""
        end_time = time.time() + time_limit
        score = state.score()
        candidates = list(range(1, n + 1))
        
        while time.time() < end_time:
            improved = False
            
            # Try adding to A
            if len(state.A) < max_size:
                rng.shuffle(candidates)
                for x in candidates:
                    if x in state.A:
                        continue
                    if state.can_add_to_A(x):
                        state.add_to_A(x)
                        score += 1
                        improved = True
                        break
            
            # Try adding to B
            if len(state.B) < max_size:
                rng.shuffle(candidates)
                for x in candidates:
                    if x in state.B:
                        continue
                    if state.can_add_to_B(x):
                        state.add_to_B(x)
                        score += 1
                        improved = True
                        break
            
            # Try 1-for-1 swap in A
            if len(state.A) >= 1 and time.time() < end_time:
                for a in state.A[:]:
                    if time.time() > end_time:
                        break
                    state.remove_from_A(a)
                    found = False
                    for x in candidates:
                        if x == a or x in state.A:
                            continue
                        if state.can_add_to_A(x):
                            state.add_to_A(x)
                            improved = True
                            found = True
                            break
                    if not found:
                        state.add_to_A(a)
                        state._rebuild()
            
            # Try 1-for-1 swap in B
            if len(state.B) >= 1 and time.time() < end_time:
                for b in state.B[:]:
                    if time.time() > end_time:
                        break
                    state.remove_from_B(b)
                    found = False
                    for x in candidates:
                        if x == b or x in state.B:
                            continue
                        if state.can_add_to_B(x):
                            state.add_to_B(x)
                            improved = True
                            found = True
                            break
                    if not found:
                        state.add_to_B(b)
                        state._rebuild()
            
            # Try moving an element from A to B or vice versa
            if time.time() < end_time and len(state.A) > 1:
                for a in state.A[:]:
                    if time.time() > end_time:
                        break
                    state.remove_from_A(a)
                    if a not in state.B and state.can_add_to_B(a):
                        state.add_to_B(a)
                        # Net zero in size, but may enable further additions
                        improved = True
                        break
                    else:
                        state.add_to_A(a)
                        state._rebuild()
            
            if time.time() < end_time and len(state.B) > 1:
                for b in state.B[:]:
                    if time.time() > end_time:
                        break
                    state.remove_from_B(b)
                    if b not in state.A and state.can_add_to_A(b):
                        state.add_to_A(b)
                        improved = True
                        break
                    else:
                        state.add_to_B(b)
                        state._rebuild()
            
            if not improved:
                # Try a few random restarts of local moves
                for _ in range(3):
                    if time.time() > end_time:
                        break
                    if len(state.A) > 1:
                        a = rng.choice(state.A)
                        state.remove_from_A(a)
                        for x in rng.sample(candidates, min(10, len(candidates))):
                            if x not in state.A and state.can_add_to_A(x):
                                state.add_to_A(x)
                                break
                        else:
                            state.add_to_A(a)
                            state._rebuild()
                    if len(state.B) > 1:
                        b = rng.choice(state.B)
                        state.remove_from_B(b)
                        for x in rng.sample(candidates, min(10, len(candidates))):
                            if x not in state.B and state.can_add_to_B(x):
                                state.add_to_B(x)
                                break
                        else:
                            state.add_to_B(b)
                            state._rebuild()
            
            if not improved:
                break
        
        return state, score

    # Main search: multiple strategies
    strategies = []
    
    # Strategy 1: Random seeds
    for _ in range(30):
        A0 = [rng.randint(1, n)]
        B0 = [rng.randint(1, n)]
        strategies.append(('random', A0, B0))
    
    # Strategy 2: Structured seeds based on modular arithmetic
    # Use sets with controlled difference patterns
    for mod in [4, 5, 6, 7, 8]:
        A0 = [1 + i * mod for i in range(min(3, (n - 1) // mod))]
        B0 = [2 + i * mod for i in range(min(3, (n - 2) // mod))]
        if len(A0) > 0 and len(B0) > 0 and A0[0] != B0[0]:
            strategies.append(('modular', A0[:3], B0[:3]))
    
    # Strategy 3: Disjoint ranges
    for _ in range(10):
        A0 = [rng.randint(1, 128)]
        B0 = [rng.randint(129, 256)]
        strategies.append(('ranges', A0, B0))
    
    # Strategy 4: Small structured sets
    for a_seed in range(1, 10):
        A0 = [a_seed]
        B0 = [a_seed + 1] if a_seed + 1 <= n else [a_seed - 1]
        strategies.append(('small', A0, B0))
    
    rng.shuffle(strategies)
    
    for strat_name, A0, B0 in strategies:
        if time.time() > deadline - 3:
            break
        
        A, B, score = _greedy_build(A0, B0)
        
        if A is not None and score > best_score:
            best_score = score
            best_A = list(A)
            best_B = list(B)
            
            # Local search
            if time.time() < deadline - 8:
                state = State(best_A, best_B)
                tl = min(3.0, max(0.5, deadline - 5 - time.time()))
                state, new_score = _local_search(state, time_limit=tl)
                if new_score > best_score:
                    best_score = new_score
                    best_A = list(state.A)
                    best_B = list(state.B)
    
    # Additional random restarts with local search
    remaining = deadline - 5 - time.time()
    if remaining > 2:
        n_restarts = max(1, int(remaining / 3))
        for _ in range(n_restarts):
            if time.time() > deadline - 3:
                break
            A0 = [rng.randint(1, n)]
            B0 = [rng.randint(1, n)]
            A, B, score = _greedy_build(A0, B0)
            if A is not None:
                state = State(A, B)
                tl = min(2.0, max(0.5, deadline - 5 - time.time()))
                state, new_score = _local_search(state, time_limit=tl)
                if new_score > best_score:
                    best_score = new_score
                    best_A = list(state.A)
                    best_B = list(state.B)
    
    # Final validation
    if not _valid(best_A, best_B):
        best_A = [1]
        best_B = [2]
        best_score = 2
    
    return [best_A, best_B]
