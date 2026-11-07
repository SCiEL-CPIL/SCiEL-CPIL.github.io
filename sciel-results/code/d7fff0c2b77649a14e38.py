import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a large subset of {1..128} with no three distinct members a,b,c
    such that a+b, a+c, b+c are all also in the set.
    Uses incremental constraint tracking for fast O(N^2)-ish per-move checks.
    """
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s limit

    # ---------- Precompute pair data ----------
    # pairs_by_sum[s] = list of (i, j) with i<j, i+j = s (values, 1-indexed)
    pairs_by_sum = [[] for _ in range(257)]
    for a in range(1, 129):
        for b in range(a + 1, 129):
            s = a + b
            if s <= 256:
                pairs_by_sum[s].append((a, b))

    # ---------- Incremental constraint state ----------
    # For each sum s, track how many pairs (a,b) with a+b=s have both in S.
    pair_counts = [0] * 257
    # blocked[c] = sum over s in 1..128 of (pair_counts[s] >= 2) where c is in pairs_by_sum[s]
    # (a pair (a,b) with a+b=s "blocks" c if s,c,s-c are all in S)
    # We use counts: for each s, block contribution = max(0, pair_counts[s] - 1)
    # blocked[c] = sum over s containing c of max(0, pair_counts[s]-1)
    blocked = [0] * 257

    # Which sums does each value c participate in (as the third element)?
    # c participates in sum s if c < s and s-c >= 1 and (s-c) != c (distinct)
    # i.e., c in (1, s) and s-c != c, and s-c >= 1, c <= 128, s-c <= 128
    c_sums = [[] for _ in range(129)]  # c_sums[c] = list of sums s where c is a valid third element
    for s in range(3, 257):
        for c in range(1, min(129, s - 1)):
            d = s - c
            if d >= 1 and d != c and d <= 128:
                c_sums[c].append(s)

    def check_validity():
        """Full O(N^3) validation for safety."""
        S = set()
        for v in range(1, 129):
            if selected[v]:
                S.add(v)
        for a in range(1, 129):
            if a not in S:
                continue
            for b in range(a + 1, 129):
                if b not in S:
                    continue
                ab = a + b
                if ab not in S or ab > 128:
                    continue
                # Need c distinct from a,b with a+c in S and b+c in S
                # c must satisfy: a+c <= 128, b+c <= 128, a+c in S, b+c in S, c not in {a,b}
                for c in range(1, 129):
                    if c == a or c == b:
                        continue
                    ac = a + c
                    bc = b + c
                    if ac > 128 or bc > 128:
                        continue
                    if ac in S and bc in S:
                        return False
        return True

    def add_elem(v):
        """Add element v to S, update constraints. Returns True if valid."""
        if selected[v]:
            return True
        if v > 128:
            return False
        # Check if adding v would create a violation
        # v participates as third element in sums s in c_sums[v]
        for s in c_sums[v]:
            if pair_counts[s] >= 2:
                # Would create a triple: there are >=2 pairs with sum s,
                # and v completes a triple
                return False
        # Also check: v as part of a pair (a,v) where a+v = s
        # If pair_counts[s] >= 2 already, then adding (a,v) makes it 3,
        # which means for any c in that sum with pair_counts... 
        # Actually the constraint is: for each s, if pair_counts[s] >= 2,
        # then no element c with s-c and c both valid can be in S.
        # Adding v creates pairs (a,v) for a in S with a+v <= 256.
        # For each such pair, it increments pair_counts[a+v].
        # If pair_counts[a+v] becomes >= 2, then all c in that sum are blocked.
        # But we need: for every new pair (a,v), if pair_counts[a+v] was already >= 1,
        # then after increment it's >= 2, meaning c=v might be blocked if v is in that sum.
        # More precisely: after adding v, for each sum s, pair_counts[s] increases by
        # the number of a in S (a != v) such that a+v = s, i.e., a = s-v.
        # So for each a in S with a != v and a+v <= 256, pair_counts[a+v] += 1.
        # The new blocked count for c: blocked[c] increases by sum over s in c_sums[c] of
        # max(0, new_pair_counts[s] - 1) - max(0, old_pair_counts[s] - 1)
        
        # Let's just check: is v currently blocked?
        # blocked[v] = sum over s in c_sums[v] of max(0, pair_counts[s] - 1)
        # If this is > 0, adding v creates a violation.
        bv = 0
        for s in c_sums[v]:
            if pair_counts[s] >= 2:
                bv += pair_counts[s] - 1
        if bv > 0:
            return False

        # Safe to add. Update state.
        selected[v] = True
        size += 1  # will use nonlocal
        
        # Update pair_counts and blocked
        for a in range(1, v):
            if selected[a]:
                s = a + v
                if s <= 256:
                    old_pc = pair_counts[s]
                    new_pc = old_pc + 1
                    pair_counts[s] = new_pc
                    # Update blocked for all c in this sum
                    # delta_blocked = max(0, new_pc - 1) - max(0, old_pc - 1)
                    if new_pc >= 2 and old_pc < 2:
                        # This sum now blocks all its elements
                        for c in range(1, min(129, s - 1)):
                            d = s - c
                            if d >= 1 and d != c and d <= 128:
                                blocked[c] += 1
                    elif new_pc > old_pc and new_pc >= 2:
                        # Each c gains 1 more block
                        for c in range(1, min(129, s - 1)):
                            d = s - c
                            if d >= 1 and d != c and d <= 128:
                                blocked[c] += 1
        for a in range(v + 1, 129):
            if selected[a]:
                s = v + a
                if s <= 256:
                    old_pc = pair_counts[s]
                    new_pc = old_pc + 1
                    pair_counts[s] = new_pc
                    if new_pc >= 2 and old_pc < 2:
                        for c in range(1, min(129, s - 1)):
                            d = s - c
                            if d >= 1 and d != c and d <= 128:
                                blocked[c] += 1
                    elif new_pc > old_pc and new_pc >= 2:
                        for c in range(1, min(129, s - 1)):
                            d = s - c
                            if d >= 1 and d != c and d <= 128:
                                blocked[c] += 1
        return True

    def remove_elem(v):
        """Remove element v from S, update constraints."""
        if not selected[v]:
            return
        selected[v] = False
        size -= 1
        
        for a in range(1, v):
            if selected[a]:
                s = a + v
                if s <= 256:
                    old_pc = pair_counts[s]
                    new_pc = old_pc - 1
                    pair_counts[s] = new_pc
                    if old_pc >= 2 and new_pc < 2:
                        for c in range(1, min(129, s - 1)):
                            d = s - c
                            if d >= 1 and d != c and d <= 128:
                                blocked[c] -= 1
                    elif old_pc > new_pc and old_pc >= 2:
                        for c in range(1, min(129, s - 1)):
                            d = s - c
                            if d >= 1 and d != c and d <= 128:
                                blocked[c] -= 1
        for a in range(v + 1, 129):
            if selected[a]:
                s = v + a
                if s <= 256:
                    old_pc = pair_counts[s]
                    new_pc = old_pc - 1
                    pair_counts[s] = new_pc
                    if old_pc >= 2 and new_pc < 2:
                        for c in range(1, min(129, s - 1)):
                            d = s - c
                            if d >= 1 and d != c and d <= 128:
                                blocked[c] -= 1
                    elif old_pc > new_pc and old_pc >= 2:
                        for c in range(1, min(129, s - 1)):
                            d = s - c
                            if d >= 1 and d != c and d <= 128:
                                blocked[c] -= 1

    # ---------- Strategy 1: Build from scratch with greedy + local search ----------
    def build_greedy_order(order, dl):
        """Build set by trying elements in given order, adding if valid."""
        nonlocal selected, size, pair_counts, blocked
        # Reset state
        selected = [False] * 129
        pair_counts = [0] * 257
        blocked = [0] * 257
        size = 0
        
        for v in order:
            if time.time() > dl:
                break
            if not selected[v] and blocked[v] == 0:
                if add_elem(v):
                    pass
        return list(range(1, 129)) if False else None  # dummy, we track selected
    
    # Actually, let me restructure to use a cleaner state management approach.
    # I'll use a class-like structure with explicit state.
    
    class State:
        def __init__(self):
            self.selected = [False] * 129
            self.pair_counts = [0] * 257
            self.blocked = [0] * 257
            self.size = 0
        
        def copy(self):
            st = State()
            st.selected = self.selected[:]
            st.pair_counts = self.pair_counts[:]
            st.blocked = self.blocked[:]
            st.size = self.size
            return st
        
        def can_add(self, v):
            if self.selected[v] or v > 128:
                return False
            if self.blocked[v] > 0:
                return False
            # Double check: is v actually blocked?
            # blocked[v] should already account for this
            return True
        
        def add(self, v):
            if not self.can_add(v):
                return False
            self.selected[v] = True
            self.size += 1
            # Update pair counts and blocked
            for a in range(1, v):
                if self.selected[a]:
                    s = a + v
                    if s <= 256:
                        old = self.pair_counts[s]
                        new = old + 1
                        self.pair_counts[s] = new
                        if new >= 2:
                            if old < 2:
                                delta = 1
                            else:
                                delta = 1
                            for c in range(1, min(129, s - 1)):
                                d = s - c
                                if d >= 1 and d != c and d <= 128:
                                    self.blocked[c] += delta
            for a in range(v + 1, 129):
                if self.selected[a]:
                    s = v + a
                    if s <= 256:
                        old = self.pair_counts[s]
                        new = old + 1
                        self.pair_counts[s] = new
                        if new >= 2:
                            if old < 2:
                                delta = 1
                            else:
                                delta = 1
                            for c in range(1, min(129, s - 1)):
                                d = s - c
                                if d >= 1 and d != c and d <= 128:
                                    self.blocked[c] += delta
            return True
        
        def remove(self, v):
            if not self.selected[v]:
                return
            self.selected[v] = False
            self.size -= 1
            for a in range(1, v):
                if self.selected[a]:
                    s = a + v
                    if s <= 256:
                        old = self.pair_counts[s]
                        new = old - 1
                        self.pair_counts[s] = new
                        if old >= 2:
                            for c in range(1, min(129, s - 1)):
                                d = s - c
                                if d >= 1 and d != c and d <= 128:
                                    self.blocked[c] -= 1
            for a in range(v + 1, 129):
                if self.selected[a]:
                    s = v + a
                    if s <= 256:
                        old = self.pair_counts[s]
                        new = old - 1
                        self.pair_counts[s] = new
                        if old >= 2:
                            for c in range(1, min(129, s - 1)):
                                d = s - c
                                if d >= 1 and d != c and d <= 128:
                                    self.blocked[c] -= 1
        
        def full_check(self):
            """O(N^3) validation."""
            S = set(i for i in range(1, 129) if self.selected[i])
            for a in S:
                for b in S:
                    if b <= a:
                        continue
                    ab = a + b
                    if ab not in S or ab > 128:
                        continue
                    for c in S:
                        if c == a or c == b:
                            continue
                        if (a + c) in S and (b + c) in S:
                            return False
            return True
        
        def to_list(self):
            return [i for i in range(1, 129) if self.selected[i]]

    def local_search(st, dl):
        """Improve state by trying to add any valid element."""
        improved = True
        while improved and time.time() < dl:
            improved = False
            for v in range(1, 129):
                if time.time() > dl:
                    break
                if not st.selected[v] and st.blocked[v] == 0:
                    if st.add(v):
                        improved = True
        return st

    def local_search_swap(st, dl):
        """Try removing one element and adding up to two."""
        improved = True
        while improved and time.time() < dl:
            improved = False
            for rem in range(1, 129):
                if time.time() > dl:
                    break
                if not st.selected[rem]:
                    continue
                trial = st.copy()
                trial.remove(rem)
                added = 0
                for add in range(1, 129):
                    if time.time() > dl:
                        break
                    if add == rem:
                        continue
                    if not trial.selected[add] and trial.blocked[add] == 0:
                        if trial.add(add):
                            added += 1
                if trial.size > st.size:
                    st = trial
                    improved = True
        return st

    # ---------- Phase 1: Greedy build ----------
    # Start with 65..128 (size 64, always valid)
    def phase1():
        st = State()
        for v in range(65, 129):
            st.add(v)
        # Try to add 1..64
        for v in range(1, 65):
            if time.time() > deadline - 30:
                break
            if not st.selected[v] and st.blocked[v] == 0:
                st.add(v)
        # Local search
        st = local_search(st, deadline - 25)
        return st

    # ---------- Phase 2: Multiple restarts with different orders ----------
    def phase2(best_st, dl):
        best = best_st
        best_size = best_st.size
        rng = random.Random(42)
        
        for restart in range(30):
            if time.time() > dl:
                break
            
            # Perturb: start from best, remove some random elements
            st = best.copy()
            set_elems = [v for v in range(1, 129) if st.selected[v]]
            if not set_elems:
                break
            # Remove 1-3 random elements
            n_remove = min(len(set_elems), rng.randint(1, 3))
            to_remove = rng.sample(set_elems, n_remove)
            for v in to_remove:
                st.remove(v)
            
            # Try to add elements in random order
            unset = [v for v in range(1, 129) if not st.selected[v]]
            rng.shuffle(unset)
            for v in unset:
                if time.time() > dl:
                    break
                if not st.selected[v] and st.blocked[v] == 0:
                    st.add(v)
            
            # Local search
            st = local_search(st, dl)
            
            if st.size > best_size:
                best = st
                best_size = st.size
            
            # Also try swap-based local search
            st2 = best.copy()
            st2 = local_search_swap(st2, dl)
            if st2.size > best_size:
                best = st2
                best_size = st2.size
        
        return best

    # ---------- Phase 3: Targeted improvements ----------
    def phase3(best_st, dl):
        best = best_st
        best_size = best_st.size
        rng = random.Random(123)
        
        # Try to find elements that can be added by removing just one
        for _ in range(200):
            if time.time() > dl:
                break
            
            st = best.copy()
            set_elems = [v for v in range(1, 129) if st.selected[v]]
            unset = [v for v in range(1, 129) if not st.selected[v]]
            
            if not set_elems or not unset:
                break
            
            # Try random swap: remove one, try to add one or two
            rem = rng.choice(set_elems)
            st.remove(rem)
            
            # Try to add up to 2 elements
            added = 0
            candidates = unset[:]
            rng.shuffle(candidates)
            for v in candidates:
                if time.time() > dl:
                    break
                if added >= 2:
                    break
                if not st.selected[v] and st.blocked[v] == 0:
                    if st.add(v):
                        added += 1
            
            if st.size > best_size:
                best = st
                best_size = st.size
            
            # Local search after improvement
            st = local_search(best, dl)
            if st.size > best_size:
                best = st
                best_size = st.size
        
        return best

    # ---------- Main flow ----------
    # Phase 1: Greedy build
    best = phase1()
    best_size = best.size
    
    # Phase 2: Multiple restarts
    best = phase2(best, deadline - 10)
    best_size = best.size
    
    # Phase 3: Targeted improvements
    best = phase3(best, deadline - 2)
    best_size = best.size
    
    # Final local search
    best = local_search(best, deadline - 1)
    best_size = best.size
    
    # Validate
    if not best.full_check():
        # Fallback: 65..128
        fb = State()
        for v in range(65, 129):
            fb.add(v)
        return [v for v in range(65, 129)]
    
    return best.to_list()

# EVOLVE-BLOCK-END
