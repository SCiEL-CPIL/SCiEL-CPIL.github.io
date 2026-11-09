"""Difference triangle set (7,5) – signature-based genetic algorithm with reconstruction."""
import time
import random
import sys
import math

_START = time.time()
_DEADLINE = _START + 145.0


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    """Return frozenset of all positive differences in a row."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


def _row_from_diffs(diffs, k):
    """
    Reconstruct a row from a set of 15 differences.
    A row [0, a1, a2, ..., ak] has differences:
    a1, a2, ..., ak (from 0)
    a2-a1, a3-a1, ..., ak-a1
    a3-a2, ..., ak-a2
    ...
    ak-a(k-1)
    
    We need to find a consistent assignment. Use the fact that:
    - The smallest diff must be a1 (distance from 0 to first element)
    - Actually, the diffs include all pairwise differences.
    
    Strategy: The row has k+1 elements. The differences form a "difference set".
    We can reconstruct by noting that if we sort the differences, the structure
    is constrained.
    
    Key insight: For a row [0, x1, x2, ..., xk], the set of differences is:
    {xj - xi : 0 <= i < j <= k}
    
    This is a "Sidon-like" structure. To reconstruct:
    - The largest difference is xk (from 0 to last)
    - The second largest could be xk-x1 or x(k-1)
    
    Actually, let's use a different approach: 
    The differences determine the row up to translation (fixed at 0).
    We can reconstruct by:
    1. The set of differences from 0 to each element: {x1, x2, ..., xk}
       These are exactly the differences that appear as "first element to some element"
       But we don't know which differences are from 0.
    
    Better approach: Use the fact that in a valid difference set from a sorted sequence,
    if we know all pairwise differences, we can reconstruct the sequence.
    
    Algorithm:
    - Let D be the sorted set of differences.
    - The row must contain 0. The other elements are some subset of D.
    - Specifically, if the row is [0, a1, a2, ..., ak], then:
      * a1, a2, ..., ak are in D
      * aj - ai for all i<j are in D
      * D has exactly C(k+1,2) = 15 elements
    
    Reconstruction:
    - The largest element in the row is max(row) = xk, and xk must be in D.
    - xk is the largest element in D? Not necessarily, but xk is the maximum of the row.
    - Actually, the largest difference in D is xk - 0 = xk (if 0 is the min).
      Wait, no: the differences are all positive, and the largest is xk - 0 = xk.
      Yes! Since 0 is the smallest element, the largest difference is xk - 0 = xk.
    
    So: xk = max(D).
    
    Now, xk-1 is either:
    - xk - x1 (difference between last and first non-zero)
    - x(k-1) (the second-to-last element, difference from 0)
    
    This is complex. Let me use a simpler heuristic reconstruction:
    
    Given D (set of 15 diffs), reconstruct row:
    1. xk = max(D)
    2. The row is [0, a1, a2, ..., ak] with ak = xk
    3. For each element aj (j=1..k-1), aj must be in D, and xk - aj must be in D.
    4. Also, for any i<j, aj-ai must be in D.
    
    We can try: the elements a1,...,a(k-1) are a subset of D\{xk} of size k-1,
    such that all pairwise differences are in D, and xk-ai in D for all i.
    
    This is still combinatorial. For k=5, we need to choose 4 elements from up to 14 candidates.
    
    Simpler: use a greedy reconstruction.
    """
    if len(diffs) != k * (k + 1) // 2:
        return None
    
    D = sorted(diffs)
    xk = D[-1]  # Largest element in row
    
    # The row is [0, a1, a2, a3, a4, a5] with a5 = xk
    # a1, a2, a3, a4 must be chosen from D\{xk}
    # Constraints:
    # - All pairwise differences among {0, a1, a2, a3, a4, xk} must be in D
    # - Specifically: a1, a2, a3, a4, xk-a1, xk-a2, xk-a3, xk-a4,
    #   a2-a1, a3-a1, a4-a1, xk-a1 (already), a3-a2, a4-a2, xk-a2 (already),
    #   a4-a3, xk-a3 (already), xk-a4 (already)
    #   Total: 15 differences
    
    # Greedy: try to build the row from left to right
    # a1 must be in D, and xk-a1 must be in D
    # a2 must be in D, xk-a2 in D, a2-a1 in D
    # etc.
    
    candidates = [d for d in D if d < xk]
    
    # Try all combinations? For k=5, C(14,4) = 1001, manageable
    # But let's be smarter: use backtracking
    
    row = [0]
    
    def backtrack(pos, current_row, remaining_diffs):
        if pos == k:
            # Verify all differences
            all_diffs = set()
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    all_diffs.add(current_row[j] - current_row[i])
            return all_diffs == diffs
        
        start = current_row[-1] + 1
        for x in range(start, xk):
            if x not in remaining_diffs:
                continue
            # Check if x - last is in remaining_diffs (it's the diff from previous element)
            # Actually, we need to check that all new diffs are available
            new_diffs = set()
            for prev in current_row:
                nd = x - prev
                if nd not in remaining_diffs:
                    break
                new_diffs.add(nd)
            else:
                # All new diffs are available
                new_remaining = set(remaining_diffs)
                for nd in new_diffs:
                    new_remaining.discard(nd)
                if backtrack(pos + 1, current_row + [x], new_remaining):
                    return True
        return False
    
    # Start with xk as the last element
    # We need to find a1, a2, a3, a4 such that all diffs work
    # The diffs from 0 to each element must be in D: a1, a2, a3, a4, xk
    # The diffs between elements must also be in D
    
    # Let's try: the row elements (excluding 0) are a subset of D that includes xk
    # and is closed under the difference constraint
    
    # Simpler: just try to find any valid row
    # For speed, use a heuristic: the elements should be "spread out"
    
    # Try greedy: pick smallest valid x at each step
    current_row = [0]
    used_in_row = set()
    
    for pos in range(k):
        found = False
        for x in sorted(candidates):
            if x in used_in_row:
                continue
            # Check all new differences
            valid = True
            new_diffs = set()
            for prev in current_row:
                nd = x - prev
                if nd not in diffs:
                    valid = False
                    break
                new_diffs.add(nd)
            if valid and len(new_diffs) > 0:
                # Check no conflicts with already-used diffs in this row
                # (all diffs in the row must be distinct, which is guaranteed by the set)
                current_row.append(x)
                used_in_row.add(x)
                found = True
                break
        if not found:
            return None
    
    # Verify
    all_diffs = set()
    for i in range(k + 1):
        for j in range(i + 1, k + 1):
            all_diffs.add(current_row[j] - current_row[i])
    if all_diffs == diffs:
        return current_row
    return None


def _greedy_row_fast(tracker, k, max_mark, rnd=None):
    """Greedy: build a row of k+1 entries starting at 0, smallest valid x at each step."""
    row = [0]
    for _ in range(k):
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = {x - a for a in row}
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None
    return row


def _interleaved_greedy(n_rows, k, max_mark, rnd=None):
    """Interleaved greedy: extend the row with smallest current max at each step."""
    tracker = DiffTracker(max_mark + 1)
    rows = [[0] for _ in range(n_rows)]

    for _ in range(n_rows * k):
        candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not candidates:
            break
        i = min(candidates, key=lambda idx: rows[idx][-1])
        row = rows[i]
        x = row[-1] + 1
        found = False
        while x <= max_mark:
            new_diffs = {x - a for a in row}
            if tracker.check(new_diffs):
                row.append(x)
                tracker.add(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None, None

    if all(len(r) == k + 1 for r in rows):
        return rows, tracker
    return None, None


class DiffTracker:
    """Fast difference tracking using a boolean array for O(1) membership."""
    def __init__(self, max_diff):
        self.max_diff = max_diff
        self.used = bytearray(max_diff + 1)
        self.count = 0

    def add(self, diffs):
        for d in diffs:
            if d <= self.max_diff and not self.used[d]:
                self.used[d] = 1
                self.count += 1

    def remove(self, diffs):
        for d in diffs:
            if d <= self.max_diff and self.used[d]:
                self.used[d] = 0
                self.count -= 1

    def check(self, diffs):
        """Return True if none of the diffs conflict with current used set."""
        for d in diffs:
            if d <= self.max_diff and self.used[d]:
                return False
        return True

    def copy(self):
        nt = DiffTracker(self.max_diff)
        nt.used = bytearray(self.used)
        nt.count = self.count
        return nt


def _genetic_signature_search(n_rows, k, max_mark, rnd, time_budget=60.0):
    """
    Genetic algorithm over difference signatures.
    
    Each individual is a tuple of n_rows sets, each set containing k*(k+1)//2 differences.
    Fitness: number of successfully reconstructed rows + penalty for conflicts.
    """
    deadline = time.time() + time_budget
    num_diffs = k * (k + 1) // 2  # 15 for k=5
    
    # Initial population: random disjoint signature sets
    def random_signature():
        """Generate a random set of num_diffs values from [1, max_mark]."""
        return set(rnd.randint(1, max_mark) for _ in range(num_diffs))
    
    def random_individual():
        """Generate a random individual with disjoint signatures."""
        sigs = []
        used = set()
        for _ in range(n_rows):
            sig = set()
            while len(sig) < num_diffs:
                d = rnd.randint(1, max_mark)
                if d not in used:
                    sig.add(d)
                    used.add(d)
            sigs.append(sig)
        return sigs
    
    def fitness(sigs):
        """
        Fitness: count of rows that can be reconstructed + 0.1 * (1 - conflict_ratio).
        """
        score = 0
        for sig in sigs:
            row = _row_from_diffs(sig, k)
            if row is not None and max(row) <= max_mark:
                score += 1.0
        # Penalty for overlapping signatures (should be 0 by construction, but check)
        all_diffs = set()
        for sig in sigs:
            all_diffs |= sig
        overlap_penalty = (n_rows * num_diffs - len(all_diffs)) * 0.05
        return score - overlap_penalty
    
    # Initialize population
    pop_size = 50
    population = [random_individual() for _ in range(pop_size)]
    best_individual = max(population, key=fitness)
    best_fit = fitness(best_individual)
    
    # Evolution
    while time.time() < deadline:
        new_pop = []
        
        # Elitism: keep top 10%
        sorted_pop = sorted(population, key=fitness, reverse=True)
        elite_count = max(1, pop_size // 10)
        new_pop.extend(sorted_pop[:elite_count])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline:
                break
            
            # Crossover: pick two parents, combine signatures
            p1_idx = rnd.randint(0, len(population) - 1)
            p2_idx = rnd.randint(0, len(population) - 1)
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            child = []
            used = set()
            for i in range(n_rows):
                # With 50% probability, take from p1 or p2
                if rnd.random() < 0.5:
                    sig = set(p1[i])
                else:
                    sig = set(p2[i])
                
                # Mutation: with probability 0.3, replace 1-3 differences
                if rnd.random() < 0.3:
                    num_mut = rnd.randint(1, 3)
                    for _ in range(num_mut):
                        if sig:
                            # Remove a random element
                            to_remove = rnd.choice(list(sig))
                            sig.discard(to_remove)
                            used.discard(to_remove)
                            # Add a new random element
                            new_d = rnd.randint(1, max_mark)
                            attempts = 0
                            while new_d in used and attempts < 20:
                                new_d = rnd.randint(1, max_mark)
                                attempts += 1
                            if new_d not in used:
                                sig.add(new_d)
                                used.add(new_d)
                
                child.append(sig)
            
            # Ensure disjointness: if there are conflicts, resolve by replacing
            all_used = set()
            for sig in child:
                for d in sig:
                    if d in all_used:
                        # Replace with a new random value
                        new_d = rnd.randint(1, max_mark)
                        attempts = 0
                        while new_d in all_used and attempts < 20:
                            new_d = rnd.randint(1, max_mark)
                            attempts += 1
                        sig.discard(d)
                        if new_d not in all_used:
                            sig.add(new_d)
                            all_used.add(new_d)
                    else:
                        all_used.add(d)
            
            new_pop.append(child)
        
        population = new_pop
        
        # Update best
        for ind in population:
            f = fitness(ind)
            if f > best_fit:
                best_fit = f
                best_individual = [set(s) for s in ind]
        
        # If we have a perfect solution, extract it
        if best_fit >= n_rows:
            rows = []
            for sig in best_individual:
                row = _row_from_diffs(sig, k)
                if row is not None:
                    rows.append(row)
                else:
                    rows = None
                    break
            if rows is not None and len(rows) == n_rows:
                return rows
    
    # Return best found (may not be complete)
    rows = []
    for sig in best_individual:
        row = _row_from_diffs(sig, k)
        if row is not None:
            rows.append(row)
    if len(rows) == n_rows:
        return rows
    return None


def _targeted_reduction(rows, n_rows, k, time_budget):
    """Systematically try to reduce each entry of each row from its current value downward."""
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    max_mark = best_scope

    deadline = time.time() + time_budget
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(n_rows):
            row = best[ri]
            for pi in range(len(row) - 1, 0, -1):  # From right to left
                old_val = row[pi]
                # Compute all other diffs
                other_diffs = set()
                for r2 in best:
                    if r2 is not row:
                        other_diffs |= _diffs_of_row(r2)
                # Also diffs within this row not involving position pi
                row_diffs_other = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        if i != pi and j != pi:
                            row_diffs_other.add(row[j] - row[i])
                forbidden = other_diffs | row_diffs_other

                for new_val in range(old_val - 1, row[pi - 1], -1):
                    if pi < len(row) - 1 and new_val >= row[pi + 1]:
                        continue
                    # Check new diffs involving pi
                    new_diffs_pi = set()
                    for j in range(len(row)):
                        if j != pi:
                            if j < pi:
                                new_diffs_pi.add(new_val - row[j])
                            else:
                                new_diffs_pi.add(row[j] - new_val)
                    if new_diffs_pi & forbidden:
                        continue
                    # Valid!
                    row[pi] = new_val
                    new_scope = _scope(best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    improved = True
                    break
    return best, best_scope


def construct(n_rows, k):
    rnd = random.Random(12345)
    best_rows = None
    best_scope = float('inf')

    lb = n_rows * k * (k + 1) // 2

    # Phase 1: Quick interleaved greedy to get a starting point
    rows, _ = _interleaved_greedy(n_rows, k, 200)
    if rows is not None:
        best_rows = rows
        best_scope = _scope(rows)

    if _dead():
        return best_rows

    # Phase 2: Targeted reduction to tighten the solution
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 10.0
        if remaining > 2.0:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # Phase 3: Genetic algorithm on difference signatures
    remaining = _DEADLINE - time.time() - 10.0
    if remaining > 5.0 and best_rows is not None:
        ga_result = _genetic_signature_search(n_rows, k, best_scope, rnd, time_budget=remaining)
        if ga_result is not None:
            sc = _scope(ga_result)
            if sc < best_scope:
                best_scope = sc
                best_rows = ga_result

    if _dead():
        return best_rows

    # Phase 4: Final targeted reduction
    if best_rows is not None:
        remaining = _DEADLINE - time.time() - 3.0
        if remaining > 1.0:
            best_rows, best_scope = _targeted_reduction(best_rows, n_rows, k, remaining)

    if _dead():
        return best_rows

    # Final validation
    if best_rows is not None and not _valid(best_rows, n_rows, k):
        # Fallback: re-run greedy
        rows, _ = _interleaved_greedy(n_rows, k, 200)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)

    return best_rows
