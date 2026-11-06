"""Two Sidon sets with disjoint difference sets – Fast incremental search with local improvement."""

import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 55.0  # Safety margin before 90s limit (evaluator kills at 90s, but let's be safe)


class SidonState:
    """Incremental state for checking Sidon property and difference disjointness.
    Uses bitsets for fast collision checking."""
    
    def __init__(self, elements):
        self.elements = list(elements)
        self.sums = set()
        self.diffs = set()
        # Bitset for sums: index s means sum s is present. Max sum = 2*256=512
        self.sum_bits = 0
        # Bitset for diffs: we store abs(d) as index, but need to track sign separately
        # Actually, let's use a dict or set for diffs since they can be negative
        # For sums, bitset is efficient since range is [2, 512]
        self._rebuild()
    
    def _rebuild(self):
        self.sums = set()
        self.diffs = set()
        self.sum_bits = 0
        for i, a in enumerate(self.elements):
            for b in self.elements[i:]:
                s = a + b
                self.sums.add(s)
                self.sum_bits |= (1 << s)
            for b in self.elements:
                d = b - a
                if d != 0:
                    self.diffs.add(d)
    
    def can_add(self, x, other_diffs):
        """Check if x can be added to this set. Uses bitset for fast sum check."""
        # Check sum collisions using bitset:
        # For each existing a, check if (a+x) is in sums
        # We can check: for all a in elements, (a+x) not in sums
        # Equivalent: no a in elements such that (a+x) is a sum
        # i.e., no s in sums such that (s - x) is in elements
        # Let's just iterate but use bitset for the sum check
        for a in self.elements:
            s = a + x
            if self.sum_bits & (1 << s):
                return False
        # Check x+x
        s2 = x + x
        if self.sum_bits & (1 << s2):
            return False
        # Check difference collisions with other set
        for a in self.elements:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
        return True
    
    def add(self, x):
        """Add x to the set, updating sums and diffs."""
        self.elements.append(x)
        for a in self.elements[:-1]:  # all previous elements
            s = a + x
            self.sums.add(s)
            self.sum_bits |= (1 << s)
            d1 = x - a
            d2 = a - x
            self.diffs.add(d1)
            self.diffs.add(d2)
        s2 = x + x
        self.sums.add(s2)
        self.sum_bits |= (1 << s2)
    
    def remove(self, x):
        """Remove x from the set, updating sums and diffs."""
        self.elements.remove(x)
        # Rebuild is simpler and safer for correctness
        self._rebuild()
    
    def can_swap(self, x_out, x_in, other_diffs):
        """Check if we can swap x_out for x_in in this set."""
        if x_out not in self.elements:
            return False
        # Temporarily remove x_out
        temp_elements = [e for e in self.elements if e != x_out]
        temp_sums = set()
        temp_diffs = set()
        temp_sum_bits = 0
        for i, a in enumerate(temp_elements):
            for b in temp_elements[i:]:
                s = a + b
                temp_sums.add(s)
                temp_sum_bits |= (1 << s)
            for b in temp_elements:
                d = b - a
                if d != 0:
                    temp_diffs.add(d)
        # Now check if x_in can be added to temp set
        for a in temp_elements:
            s = a + x_in
            if temp_sum_bits & (1 << s):
                return False
        s2 = x_in + x_in
        if temp_sum_bits & (1 << s2):
            return False
        for a in temp_elements:
            d1 = x_in - a
            d2 = a - x_in
            if d1 in other_diffs or d2 in other_diffs:
                return False
        return True
    
    def do_swap(self, x_out, x_in):
        """Perform a swap: remove x_out, add x_in."""
        self.remove(x_out)
        self.add(x_in)


def greedy_build(order_a, order_b, deadline):
    """Build two Sidon sets greedily following given element orders."""
    state_a = SidonState([])
    state_b = SidonState([])
    
    for x in order_a:
        if time.time() > deadline:
            break
        if len(state_a.elements) >= MAX_SIZE:
            break
        if state_a.can_add(x, state_b.diffs):
            state_a.add(x)
    
    for x in order_b:
        if time.time() > deadline:
            break
        if len(state_b.elements) >= MAX_SIZE:
            break
        if state_b.can_add(x, state_a.diffs):
            state_b.add(x)
    
    return state_a.elements, state_b.elements


def greedy_build_interleaved(order_a, order_b, deadline):
    """Build two Sidon sets greedily, interleaving additions to A and B."""
    state_a = SidonState([])
    state_b = SidonState([])
    
    ia, ib = 0, 0
    while (ia < len(order_a) or ib < len(order_b)) and time.time() < deadline:
        # Try to add to A
        if ia < len(order_a) and len(state_a.elements) < MAX_SIZE:
            x = order_a[ia]
            ia += 1
            if state_a.can_add(x, state_b.diffs):
                state_a.add(x)
        # Try to add to B
        if ib < len(order_b) and len(state_b.elements) < MAX_SIZE:
            x = order_b[ib]
            ib += 1
            if state_b.can_add(x, state_a.diffs):
                state_b.add(x)
    
    return state_a.elements, state_b.elements


def local_improve(sets_a, sets_b, deadline):
    """Try to improve by adding or swapping elements.
    Uses a set for O(1) membership checks and tries 2-swap moves when 1-swap stalls."""
    a = list(sets_a)
    b = list(sets_b)
    state_a = SidonState(a)
    state_b = SidonState(b)
    a_set = set(a)
    b_set = set(b)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add new elements to A
        if len(a) < MAX_SIZE:
            for x in range(1, N + 1):
                if x in a_set:
                    continue
                if state_a.can_add(x, state_b.diffs):
                    state_a.add(x)
                    a.append(x)
                    a_set.add(x)
                    improved = True
                    break
        
        # Try to add new elements to B
        if not improved and len(b) < MAX_SIZE:
            for x in range(1, N + 1):
                if x in b_set:
                    continue
                if state_b.can_add(x, state_a.diffs):
                    state_b.add(x)
                    b.append(x)
                    b_set.add(x)
                    improved = True
                    break
        
        # Try swaps in A: remove one element, add another
        if not improved and len(a) > 0 and len(a) < MAX_SIZE:
            for x_out in list(a):
                if time.time() > deadline:
                    break
                for x_in in range(1, N + 1):
                    if x_in in a_set or x_in == x_out:
                        continue
                    if state_a.can_swap(x_out, x_in, state_b.diffs):
                        state_a.do_swap(x_out, x_in)
                        a.remove(x_out)
                        a_set.discard(x_out)
                        a.append(x_in)
                        a_set.add(x_in)
                        improved = True
                        break
                if improved:
                    break
        
        # Try swaps in B: remove one element, add another
        if not improved and len(b) > 0 and len(b) < MAX_SIZE:
            for x_out in list(b):
                if time.time() > deadline:
                    break
                for x_in in range(1, N + 1):
                    if x_in in b_set or x_in == x_out:
                        continue
                    if state_b.can_swap(x_out, x_in, state_a.diffs):
                        state_b.do_swap(x_out, x_in)
                        b.remove(x_out)
                        b_set.discard(x_out)
                        b.append(x_in)
                        b_set.add(x_in)
                        improved = True
                        break
                if improved:
                    break
        
        # Try 2-swap in A: remove two elements, add two new ones
        if not improved and len(a) >= 2 and len(a) < MAX_SIZE:
            # Sample a few pairs to try
            if time.time() < deadline - 1.0:
                import itertools
                # Try a limited number of 2-swap combinations
                a_list = list(a)
                tried = 0
                for i in range(min(len(a_list), 10)):
                    if time.time() > deadline - 1.0:
                        break
                    for j in range(i + 1, min(len(a_list), 10)):
                        x_out1, x_out2 = a_list[i], a_list[j]
                        # Try to find two replacements
                        for x_in1 in range(1, N + 1):
                            if x_in1 in a_set or x_in1 == x_out1 or x_in1 == x_out2:
                                continue
                            # Check if x_in1 can replace x_out1 (with x_out2 still present)
                            temp_a = [e for e in a if e != x_out1]
                            temp_state = SidonState(temp_a)
                            if temp_state.can_add(x_in1, state_b.diffs):
                                # Now try to add x_in2 to replace x_out2
                                for x_in2 in range(1, N + 1):
                                    if x_in2 in a_set or x_in2 == x_out1 or x_in2 == x_out2 or x_in2 == x_in1:
                                        continue
                                    temp_a2 = [e for e in temp_a if e != x_out2]
                                    temp_state2 = SidonState(temp_a2)
                                    if temp_state2.can_add(x_in1, state_b.diffs) and temp_state2.can_add(x_in2, state_b.diffs):
                                        # Perform the double swap
                                        state_a.remove(x_out1)
                                        state_a.remove(x_out2)
                                        state_a.add(x_in1)
                                        state_a.add(x_in2)
                                        a = list(state_a.elements)
                                        a_set = set(a)
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                        tried += 1
                        if tried >= 20:
                            break
                if improved:
                    a = list(state_a.elements)
                    a_set = set(a)
        
        # Try 2-swap in B: remove two elements, add two new ones
        if not improved and len(b) >= 2 and len(b) < MAX_SIZE:
            if time.time() < deadline - 1.0:
                b_list = list(b)
                tried = 0
                for i in range(min(len(b_list), 10)):
                    if time.time() > deadline - 1.0:
                        break
                    for j in range(i + 1, min(len(b_list), 10)):
                        x_out1, x_out2 = b_list[i], b_list[j]
                        for x_in1 in range(1, N + 1):
                            if x_in1 in b_set or x_in1 == x_out1 or x_in1 == x_out2:
                                continue
                            temp_b = [e for e in b if e != x_out1]
                            temp_state = SidonState(temp_b)
                            if temp_state.can_add(x_in1, state_a.diffs):
                                for x_in2 in range(1, N + 1):
                                    if x_in2 in b_set or x_in2 == x_out1 or x_in2 == x_out2 or x_in2 == x_in1:
                                        continue
                                    temp_b2 = [e for e in temp_b if e != x_out2]
                                    temp_state2 = SidonState(temp_b2)
                                    if temp_state2.can_add(x_in1, state_a.diffs) and temp_state2.can_add(x_in2, state_a.diffs):
                                        state_b.remove(x_out1)
                                        state_b.remove(x_out2)
                                        state_b.add(x_in1)
                                        state_b.add(x_in2)
                                        b = list(state_b.elements)
                                        b_set = set(b)
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                        tried += 1
                        if tried >= 20:
                            break
                if improved:
                    b = list(state_b.elements)
                    b_set = set(b)
    
    return a, b


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_a, best_b = [], []
    best_size = 0
    
    # Multiple restarts with different random orders
    random.seed(42)
    
    # Try both sequential and interleaved greedy strategies
    use_interleaved = False
    restart_count = 0
    
    while time.time() < deadline - 2.0:  # Leave 2s margin
        restart_count += 1
        
        # Create orders with some structure:
        # For early restarts, use spread-out orders to reduce collisions
        # For later restarts, use pure random
        if restart_count <= 5:
            # Spread-out order: interleave low and high numbers
            base = list(range(1, N + 1))
            # Create a spread order by taking from both ends
            order_a = []
            left, right = 0, N - 1
            while left <= right:
                order_a.append(base[left])
                left += 1
                if left <= right:
                    order_a.append(base[right])
                    right -= 1
            random.shuffle(order_a)
            
            order_b = []
            left, right = 0, N - 1
            while left <= right:
                order_b.append(base[left])
                left += 1
                if left <= right:
                    order_b.append(base[right])
                    right -= 1
            random.shuffle(order_b)
        else:
            order_a = list(range(1, N + 1))
            random.shuffle(order_a)
            order_b = list(range(1, N + 1))
            random.shuffle(order_b)
        
        if use_interleaved:
            a, b = greedy_build_interleaved(order_a, order_b, deadline - 1.0)
        else:
            a, b = greedy_build(order_a, order_b, deadline - 1.0)
        
        current_size = len(a) + len(b)
        
        if current_size > best_size:
            best_size = current_size
            best_a, best_b = a, b
        
        # Try local improvement if we have time
        if time.time() < deadline - 3.0 and current_size >= best_size:
            a_imp, b_imp = local_improve(a, b, deadline - 1.0)
            imp_size = len(a_imp) + len(b_imp)
            if imp_size > best_size:
                best_size = imp_size
                best_a, best_b = a_imp, b_imp
        
        # Alternate between sequential and interleaved
        use_interleaved = not use_interleaved
    
    # Ensure nonempty
    if not best_a:
        best_a = [1]
    if not best_b:
        best_b = [2]
    
    return [best_a, best_b]
