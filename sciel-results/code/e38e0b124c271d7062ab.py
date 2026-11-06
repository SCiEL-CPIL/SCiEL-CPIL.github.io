"""
Two Sidon sets with disjoint difference sets.
Combined approach: bitset fast checks + randomized local search with proper swaps.
"""

import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 85.0  # Safety margin before 90s kill


class SidonState:
    """Fast Sidon state using bitsets for sums and diffs."""
    
    def __init__(self):
        self.els = []
        self.els_set = set()
        self.sum_bits = 0   # bit i set means sum i is used (i = a+b for a<=b)
        self.diff_bits = 0  # bit i set means diff i is used (i = |a-b| for a!=b)
    
    def size(self):
        return len(self.els)
    
    def can_add(self, x, other_diff_bits):
        """Check if x can be added. other_diff_bits: bitset of other set's diffs."""
        # Check sum collisions
        # x + x
        if (self.sum_bits >> (x + x)) & 1:
            return False
        # x + a for each a in els
        for a in self.els:
            s = x + a
            if (self.sum_bits >> s) & 1:
                return False
            d = abs(x - a)
            if (self.diff_bits >> d) & 1:
                return False
            if (other_diff_bits >> d) & 1:
                return False
        return True
    
    def add(self, x):
        """Add x to the set."""
        self.els.append(x)
        self.els_set.add(x)
        self.sum_bits |= (1 << (x + x))
        for a in self.els[:-1]:
            self.sum_bits |= (1 << (x + a))
            d = abs(x - a)
            self.diff_bits |= (1 << d)
    
    def remove(self, x):
        """Remove x from the set. Rebuilds sums and diffs."""
        if x not in self.els_set:
            return
        self.els_set.discard(x)
        self.els.remove(x)
        # Rebuild
        self.sum_bits = 0
        self.diff_bits = 0
        for i in range(len(self.els)):
            for j in range(i, len(self.els)):
                self.sum_bits |= (1 << (self.els[i] + self.els[j]))
            for j in range(len(self.els)):
                if i != j:
                    d = abs(self.els[i] - self.els[j])
                    self.diff_bits |= (1 << d)


def greedy_build_interleaved(rng, deadline):
    """Build two Sidon sets by interleaving element placement."""
    state_a = SidonState()
    state_b = SidonState()
    
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if state_a.size() >= MAX_SIZE and state_b.size() >= MAX_SIZE:
            break
        
        # Try A first (if not full)
        if state_a.size() < MAX_SIZE:
            if state_a.can_add(x, state_b.diff_bits):
                state_a.add(x)
                continue
        
        # Try B (if not full)
        if state_b.size() < MAX_SIZE:
            if state_b.can_add(x, state_a.diff_bits):
                state_b.add(x)
                continue
    
    return state_a, state_b


def local_search(state_a, state_b, rng, deadline):
    """
    Local search: try adds, then 1-for-2 swaps.
    Returns (total_size, list_a, list_b).
    """
    best_total = state_a.size() + state_b.size()
    best_a = list(state_a.els)
    best_b = list(state_b.els)
    
    all_vals = list(range(1, N + 1))
    
    while time.time() < deadline:
        progress = False
        
        # --- Try adds ---
        if state_a.size() < MAX_SIZE:
            cands = [x for x in all_vals if x not in state_a.els_set]
            rng.shuffle(cands)
            for x in cands:
                if state_a.can_add(x, state_b.diff_bits):
                    state_a.add(x)
                    progress = True
                    break
        
        if not progress and state_b.size() < MAX_SIZE:
            cands = [x for x in all_vals if x not in state_b.els_set]
            rng.shuffle(cands)
            for x in cands:
                if state_b.can_add(x, state_a.diff_bits):
                    state_b.add(x)
                    progress = True
                    break
        
        # --- Try 1-for-2 swaps ---
        if not progress:
            # Try removing from A and adding 2 elements (to A and/or B)
            if state_a.size() > 1:
                a_cands = list(state_a.els)
                rng.shuffle(a_cands)
                for x_rem in a_cands:
                    # Save
                    old_a_els = list(state_a.els)
                    old_a_set = set(state_a.els)
                    old_a_sums = state_a.sum_bits
                    old_a_diffs = state_a.diff_bits
                    
                    state_a.remove(x_rem)
                    
                    added = 0
                    added_list = []
                    
                    # Try to add one to A
                    if state_a.size() < MAX_SIZE:
                        cands_a = [x for x in all_vals if x not in state_a.els_set and x != x_rem]
                        rng.shuffle(cands_a)
                        for x_add in cands_a:
                            if state_a.can_add(x_add, state_b.diff_bits):
                                state_a.add(x_add)
                                added += 1
                                added_list.append(('A', x_add))
                                break
                    
                    # Try to add one to B
                    if state_b.size() < MAX_SIZE:
                        cands_b = [x for x in all_vals if x not in state_b.els_set]
                        rng.shuffle(cands_b)
                        for x_add in cands_b:
                            if state_b.can_add(x_add, state_a.diff_bits):
                                state_b.add(x_add)
                                added += 1
                                added_list.append(('B', x_add))
                                break
                    
                    new_total = state_a.size() + state_b.size()
                    if added >= 2:
                        progress = True
                        if new_total > best_total:
                            best_total = new_total
                            best_a = list(state_a.els)
                            best_b = list(state_b.els)
                    elif added == 1:
                        # Size unchanged, but different composition. Accept with prob to explore.
                        if rng.random() < 0.3:
                            progress = True
                            if new_total > best_total:
                                best_total = new_total
                                best_a = list(state_a.els)
                                best_b = list(state_b.els)
                        else:
                            # Revert
                            state_a.els = old_a_els
                            state_a.els_set = old_a_set
                            state_a.sum_bits = old_a_sums
                            state_a.diff_bits = old_a_diffs
                    else:
                        # Revert
                        state_a.els = old_a_els
                        state_a.els_set = old_a_set
                        state_a.sum_bits = old_a_sums
                        state_a.diff_bits = old_a_diffs
                    
                    if progress:
                        break
            
            # Try removing from B and adding 2 elements
            if not progress and state_b.size() > 1:
                b_cands = list(state_b.els)
                rng.shuffle(b_cands)
                for x_rem in b_cands:
                    old_b_els = list(state_b.els)
                    old_b_set = set(state_b.els)
                    old_b_sums = state_b.sum_bits
                    old_b_diffs = state_b.diff_bits
                    
                    state_b.remove(x_rem)
                    
                    added = 0
                    
                    # Try to add one to B
                    if state_b.size() < MAX_SIZE:
                        cands_b = [x for x in all_vals if x not in state_b.els_set and x != x_rem]
                        rng.shuffle(cands_b)
                        for x_add in cands_b:
                            if state_b.can_add(x_add, state_a.diff_bits):
                                state_b.add(x_add)
                                added += 1
                                break
                    
                    # Try to add one to A
                    if state_a.size() < MAX_SIZE:
                        cands_a = [x for x in all_vals if x not in state_a.els_set]
                        rng.shuffle(cands_a)
                        for x_add in cands_a:
                            if state_a.can_add(x_add, state_b.diff_bits):
                                state_a.add(x_add)
                                added += 1
                                break
                    
                    new_total = state_a.size() + state_b.size()
                    if added >= 2:
                        progress = True
                        if new_total > best_total:
                            best_total = new_total
                            best_a = list(state_a.els)
                            best_b = list(state_b.els)
                    elif added == 1:
                        if rng.random() < 0.3:
                            progress = True
                            if new_total > best_total:
                                best_total = new_total
                                best_a = list(state_a.els)
                                best_b = list(state_b.els)
                        else:
                            state_b.els = old_b_els
                            state_b.els_set = old_b_set
                            state_b.sum_bits = old_b_sums
                            state_b.diff_bits = old_b_diffs
                    else:
                        state_b.els = old_b_els
                        state_b.els_set = old_b_set
                        state_b.sum_bits = old_b_sums
                        state_b.diff_bits = old_b_diffs
                    
                    if progress:
                        break
        
        if not progress:
            break
    
    return best_total, best_a, best_b


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(12345)
    
    best_total = 0
    best_a = []
    best_b = []
    
    # Multiple restarts
    while time.time() < deadline - 3.0:
        state_a, state_b = greedy_build_interleaved(rng, deadline - 3.0)
        cur_total = state_a.size() + state_b.size()
        
        if cur_total > best_total:
            best_total = cur_total
            best_a = list(state_a.els)
            best_b = list(state_b.els)
        
        # Local search if we have time
        if time.time() < deadline - 5.0:
            ls_total, ls_a, ls_b = local_search(state_a, state_b, rng, deadline - 5.0)
            if ls_total > best_total:
                best_total = ls_total
                best_a = ls_a
                best_b = ls_b
    
    # Ensure nonempty
    if not best_a:
        best_a = [1]
    if not best_b:
        best_b = [2]
    
    return [best_a, best_b]


if __name__ == "__main__":
    pass
