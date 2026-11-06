"""Rewritten program for two Sidon sets with disjoint difference sets.
Uses incremental state tracking and local search with time budget."""

import time
import random

DEADLINE = 85.0  # safety margin before 90s kill
N = 256
MAX_SIZE = 64

class SidonState:
    """Tracks pair sums and nonzero differences for a set, with O(1) add/remove."""
    def __init__(self):
        self.els = []
        self.sums = set()      # all a+b for a<=b
        self.diffs = set()     # all a-b for a!=b (nonzero)
    
    def can_add(self, x, other_diffs=None):
        """Check if x can be added maintaining Sidon property and disjoint diffs."""
        for a in self.els:
            s = a + x
            if s in self.sums:
                return False
            d1 = a - x
            d2 = x - a
            if d1 in self.diffs or d2 in self.diffs:
                return False
            if other_diffs is not None:
                if d1 in other_diffs or d2 in other_diffs:
                    return False
        # also check x+x
        if (x + x) in self.sums:
            return False
        return True
    
    def add(self, x, other_diffs=None):
        """Add x to the set. Assumes can_add returned True."""
        for a in self.els:
            self.sums.add(a + x)
            d1 = a - x
            d2 = x - a
            self.diffs.add(d1)
            self.diffs.add(d2)
        self.sums.add(x + x)
        self.els.append(x)
        self.els.sort()
    
    def remove(self, x):
        """Remove x from the set."""
        self.els.remove(x)
        # Remove all sums and diffs involving x
        to_remove_sums = set()
        to_remove_diffs = set()
        for a in self.els:
            to_remove_sums.add(a + x)
            d1 = a - x
            d2 = x - a
            to_remove_diffs.add(d1)
            to_remove_diffs.add(d2)
        to_remove_sums.add(x + x)
        self.sums -= to_remove_sums
        self.diffs -= to_remove_diffs
        self.els.sort()
    
    def size(self):
        return len(self.els)

def greedy_build_A(deadline, rng):
    """Greedily build set A."""
    state = SidonState()
    candidates = list(range(1, N+1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if state.size() >= MAX_SIZE:
            break
        if time.time() > deadline:
            break
        if state.can_add(x):
            state.add(x)
    
    return state

def local_search(state_a, state_b, deadline, rng):
    """Perform local search to improve combined size."""
    best_size = state_a.size() + state_b.size()
    best_a_els = list(state_a.els)
    best_b_els = list(state_b.els)
    
    # Try to add elements
    while time.time() < deadline:
        progress = False
        
        # Try adding to A
        for x in range(1, N+1):
            if x in state_a.els:
                continue
            if state_a.size() >= MAX_SIZE:
                break
            if state_a.can_add(x, state_b.diffs):
                state_a.add(x, state_b.diffs)
                progress = True
                break
        
        # Try adding to B
        for x in range(1, N+1):
            if x in state_b.els:
                continue
            if state_b.size() >= MAX_SIZE:
                break
            if state_b.can_add(x, state_a.diffs):
                state_b.add(x, state_a.diffs)
                progress = True
                break
        
        if not progress:
            # Try swaps: remove one element, try to add two
            break
        
        new_size = state_a.size() + state_b.size()
        if new_size > best_size:
            best_size = new_size
            best_a_els = list(state_a.els)
            best_b_els = list(state_b.els)
    
    return state_a, state_b, best_size, best_a_els, best_b_els

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    
    best_total = 0
    best_a = []
    best_b = []
    
    # Multiple restarts with different random seeds
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Greedy build A
        state_a = SidonState()
        candidates_a = list(range(1, N+1))
        rng.shuffle(candidates_a)
        
        for x in candidates_a:
            if state_a.size() >= MAX_SIZE:
                break
            if time.time() > deadline - 5.0:
                break
            if state_a.can_add(x):
                state_a.add(x)
        
        # Greedy build B, avoiding A's diffs
        state_b = SidonState()
        candidates_b = list(range(1, N+1))
        rng.shuffle(candidates_b)
        
        for x in candidates_b:
            if state_b.size() >= MAX_SIZE:
                break
            if time.time() > deadline - 5.0:
                break
            if state_b.can_add(x, state_a.diffs):
                state_b.add(x, state_a.diffs)
        
        total = state_a.size() + state_b.size()
        
        # Local search
        state_a, state_b, ls_size, _, _ = local_search(state_a, state_b, deadline - 5.0, rng)
        total = state_a.size() + state_b.size()
        
        if total > best_total:
            best_total = total
            best_a = list(state_a.els)
            best_b = list(state_b.els)
    
    # Ensure nonempty
    if not best_a:
        best_a = [1]
    if not best_b:
        best_b = [2]
    
    return [best_a, best_b]
