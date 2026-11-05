"""Rewritten: dense upper-half seed + repair + backfill with O(N^2) residue checks."""

import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    N = 192
    MAX_K = 48

    # Precompute: for each value v, list of all 3-element sums from other values in [1..N]
    # that are divisible by v. We'll use this for fast violation counting.
    # conflict[v][x] = number of pairs (a,b) with a<b, a!=v, b!=v, a!=b, such that (a+b+x) % v == 0
    # Actually we need: for a set S, v in S violates if there exist distinct a,b in S\{v} with (a+b) % v == 0
    # So precompute for each v: for each x in [1..N], x != v: count how many y in [1..N], y != v, y != x, with (x+y) % v == 0
    # But that's O(N^3). Instead, for each v, precompute a bitmask or count array.
    
    # For each v in 1..N, precompute: for each x in 1..N, x != v:
    #   pair_count[v][x] = number of y in 1..N, y != v, y != x, such that (x + y) % v == 0
    # Then if S contains v and x, the number of violating triples involving v and x is pair_count[v][x] restricted to S.
    # But we need: does there exist ANY pair (a,b) in S\{v} with (a+b)%v==0?
    # Equivalent: for v in S, check if any x in S\{v} has a partner y in S\{v,x} with (x+y)%v==0.
    
    # Simpler: for each v, precompute a set of "bad pairs" (x,y) where x<y, x!=v, y!=v, (x+y)%v==0.
    # Then v is violated in S iff S contains v AND S contains both elements of some bad pair for v.
    
    # For efficiency, precompute for each v a list of bad pairs as bitmasks or sorted lists.
    # With N=192, for each v, number of bad pairs is at most C(191,2) ~ 18k, but typically much less.
    
    # Precompute bad_pairs[v] = list of (x, y) with x < y, x != v, y != v, (x+y) % v == 0
    bad_pairs = [[] for _ in range(N + 1)]
    for v in range(1, N + 1):
        pairs = []
        for x in range(1, N + 1):
            if x == v:
                continue
            for y in range(x + 1, N + 1):
                if y == v:
                    continue
                if (x + y) % v == 0:
                    pairs.append((x, y))
        bad_pairs[v] = pairs

    # Precompute for each v, a set of elements that appear in any bad pair for v
    # And for fast checking: given a set S (as a boolean array or set), check if v is violated.
    
    # For the repair approach:
    # Start with a dense set of large numbers (upper half), which are less likely to divide sums.
    # Then repair by removing elements that cause violations, and backfill with new elements.
    
    # Let's use a bitset representation: state as a Python int (192 bits)
    # Precompute for each v: a bitmask of all elements that can form a bad pair with some other element for v.
    # Actually, for checking if v is violated in S: 
    #   v is violated iff there exists (x,y) in bad_pairs[v] such that both x and y are in S.
    # We can precompute for each v a list of bitmasks: for each bad pair (x,y), mask = (1<<x)|(1<<y)
    # Then v is violated iff any of these masks has both bits set in S.
    # Even faster: for each v, precompute a 2D structure or just iterate bad_pairs[v].
    
    # Given N=192 and typical bad_pairs[v] being small, iterating is fine.
    
    def is_valid(state):
        """Check if state (int bitmask) is a valid configuration."""
        # Extract elements
        elems = [i for i in range(1, N + 1) if (state >> i) & 1]
        if len(elems) > MAX_K:
            return False
        elem_set = set(elems)
        for v in elems:
            for (x, y) in bad_pairs[v]:
                if x in elem_set and y in elem_set:
                    return False
        return True
    
    def count_violations(state):
        """Count number of violating elements in state."""
        elems = [i for i in range(1, N + 1) if (state >> i) & 1]
        elem_set = set(elems)
        count = 0
        for v in elems:
            violated = False
            for (x, y) in bad_pairs[v]:
                if x in elem_set and y in elem_set:
                    violated = True
                    break
            if violated:
                count += 1
        return count
    
    # Start with dense upper-half set
    # Large numbers are less likely to divide sums of smaller numbers
    # Start with numbers from 97 to 192 (96 numbers), but we need at most 48
    # Let's start with the largest 48 numbers and repair
    
    best_state = 0
    best_count = 0
    
    # Initial seed: largest 48 numbers
    initial_elems = list(range(145, 193))  # 145..192 = 48 numbers
    state = 0
    for e in initial_elems:
        state |= (1 << e)
    
    # Repair: iteratively remove the most problematic element
    for _ in range(1000):
        if time.time() > deadline:
            break
        violations = count_violations(state)
        if violations == 0:
            break
        # Find the element with the most conflicts
        elems = [i for i in range(1, N + 1) if (state >> i) & 1]
        elem_set = set(elems)
        worst_elem = None
        worst_conflicts = 0
        for v in elems:
            conflicts = 0
            for (x, y) in bad_pairs[v]:
                if x in elem_set and y in elem_set:
                    conflicts += 1
            if conflicts > worst_conflicts:
                worst_conflicts = conflicts
                worst_elem = v
        if worst_elem is None:
            break
        # Remove the worst element
        state &= ~(1 << worst_elem)
    
    # Backfill: try adding elements from the remaining pool
    current_count = bin(state).count('1')
    if current_count > best_count:
        best_state = state
        best_count = current_count
    
    # Now do local search: try adding random elements, removing if needed
    remaining = [i for i in range(1, N + 1) if not (state >> i) & 1]
    random.shuffle(remaining)
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        if not remaining:
            break
        # Try adding a random element from remaining
        idx = random.randint(0, len(remaining) - 1)
        e = remaining[idx]
        new_state = state | (1 << e)
        new_count = bin(new_state).count('1')
        if new_count > MAX_K:
            continue
        # Check if adding e creates violations
        # Only need to check: does e violate any existing element, or does it create new violations for others?
        # e is violated if there's a bad pair for e where both elements are in new_state
        # Existing elements v might become violated if e participates in a bad pair for v
        # Check if new_state is valid
        if is_valid(new_state):
            state = new_state
            remaining.remove(e)
            if bin(state).count('1') > best_count:
                best_state = state
                best_count = bin(state).count('1')
    
    # Try multiple random restarts with different initial seeds
    for restart in range(50):
        if time.time() > deadline:
            break
        # Random subset of size 48 from upper 2/3
        pool = list(range(65, 193))  # 65..192
        random.shuffle(pool)
        seed_elems = pool[:48]
        state = 0
        for e in seed_elems:
            state |= (1 << e)
        
        # Repair
        for _ in range(500):
            if time.time() > deadline:
                break
            violations = count_violations(state)
            if violations == 0:
                break
            elems = [i for i in range(1, N + 1) if (state >> i) & 1]
            elem_set = set(elems)
            worst_elem = None
            worst_conflicts = 0
            for v in elems:
                conflicts = 0
                for (x, y) in bad_pairs[v]:
                    if x in elem_set and y in elem_set:
                        conflicts += 1
                if conflicts > worst_conflicts:
                    worst_conflicts = conflicts
                    worst_elem = v
            if worst_elem is None:
                break
            state &= ~(1 << worst_elem)
        
        # Backfill
        current_count = bin(state).count('1')
        remaining = [i for i in range(1, N + 1) if not (state >> i) & 1]
        random.shuffle(remaining)
        for e in remaining:
            if time.time() > deadline:
                break
            if bin(state).count('1') >= MAX_K:
                break
            new_state = state | (1 << e)
            if is_valid(new_state):
                state = new_state
        
        cnt = bin(state).count('1')
        if cnt > best_count:
            best_state = state
            best_count = cnt
    
    # Final: try to add any remaining elements that fit
    for e in range(1, N + 1):
        if time.time() > deadline:
            break
        if (best_state >> e) & 1:
            continue
        if bin(best_state).count('1') >= MAX_K:
            break
        new_state = best_state | (1 << e)
        if is_valid(new_state):
            best_state = new_state
    
    # Extract result
    result = [i for i in range(1, N + 1) if (best_state >> i) & 1]
    return result
