"""Two Sidon sets with disjoint difference sets.
Uses bitmask-based fast collision detection combined with randomized greedy
construction and aggressive 1-for-1 / 1-for-2 swap local search."""

import time
import random
import sys

def _check_sidon_unique_sums(els):
    """Check that all pair sums a+b (a<=b) are unique."""
    sums = set()
    n = len(els)
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_nonzero_diffs(els):
    """Get set of nonzero differences (positive and negative)."""
    diffs = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = els[j] - els[i]
                if d != 0:
                    diffs.add(d)
    return diffs

def _is_valid_pair(A, B):
    """Check that A and B are Sidon and their nonzero difference sets are disjoint."""
    if not A or not B:
        return False
    if not _check_sidon_unique_sums(A):
        return False
    if not _check_sidon_unique_sums(B):
        return False
    diffsA = _get_nonzero_diffs(A)
    diffsB = _get_nonzero_diffs(B)
    if diffsA & diffsB:
        return False
    return True

class _SidonBuilder:
    """Fast Sidon set builder using bitmasks for collision detection."""
    
    def __init__(self, n=256, max_size=64):
        self.n = n
        self.max_size = max_size
        # Sum bitmask: bits 0..511 (sums range 2..512, but we use 0..511)
        # Actually sums range from 1+1=2 to 256+256=512. Use 513 bits.
        # Let's use a Python int as a bitmask for sums (up to 512)
        self.sum_mask = 0
        # Difference mask: differences range from -(255) to 255.
        # We need to track which differences are used by this set.
        # For disjointness with the other set, we track forbidden diffs.
        self.diff_mask = 0  # bitmask where bit d (shifted) indicates diff d is present
        # We'll use a dict or set for diffs for clarity, but bitmask for speed
        # Actually, let's just use sets for diffs since the range is small (-255..255)
        # But for speed in checking, a set of ints is fine.
        self.diffs = set()
        self.els = []
    
    def add(self, x):
        """Try to add x to the set. Returns True if successful."""
        if len(self.els) >= self.max_size:
            return False
        if x in self.els:
            return False
        
        # Check sum collisions
        # New sums: x + a for all a in self.els, and x + x
        # Check if any of these are already in self.sum_mask
        new_sums = set()
        for a in self.els:
            s = a + x
            if (self.sum_mask >> s) & 1:
                return False
            new_sums.add(s)
        s2 = 2 * x
        if (self.sum_mask >> s2) & 1:
            return False
        new_sums.add(s2)
        
        # Check diff collisions with forbidden diffs (handled externally)
        # and internal diff uniqueness isn't required for Sidon, only sums.
        # Wait, Sidon only requires unique sums. Differences don't need to be unique within a set.
        # The constraint is that diffs of A and B are disjoint.
        
        # Update state
        self.els.append(x)
        self.els.sort()
        for s in new_sums:
            self.sum_mask |= (1 << s)
        for a in self.els[:-1]:
            d = a - x
            if d != 0:
                self.diffs.add(d)
                self.diffs.add(-d)
        return True
    
    def remove(self, x):
        """Remove x from the set."""
        if x not in self.els:
            return
        idx = self.els.index(x)
        self.els.pop(idx)
        # Rebuild masks from scratch for simplicity (sets are small)
        self.sum_mask = 0
        self.diffs = set()
        for i in range(len(self.els)):
            for j in range(i, len(self.els)):
                s = self.els[i] + self.els[j]
                self.sum_mask |= (1 << s)
        for i in range(len(self.els)):
            for j in range(len(self.els)):
                if i != j:
                    d = self.els[j] - self.els[i]
                    if d != 0:
                        self.diffs.add(d)
    
    def can_add(self, x, forbidden_diffs):
        """Check if x can be added without violating constraints."""
        if len(self.els) >= self.max_size:
            return False
        if x in self.els:
            return False
        
        # Check sums
        for a in self.els:
            s = a + x
            if (self.sum_mask >> s) & 1:
                return False
        s2 = 2 * x
        if (self.sum_mask >> s2) & 1:
            return False
        
        # Check diffs against forbidden
        for a in self.els:
            d1 = a - x
            d2 = x - a
            if d1 != 0 and d1 in forbidden_diffs:
                return False
            if d2 != 0 and d2 in forbidden_diffs:
                return False
        return True

def _greedy_build(builder, other_diffs, n, rng, max_size=64):
    """Greedily build a set using the builder, given forbidden diffs from other set."""
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(builder.els) >= max_size:
            break
        if builder.can_add(x, other_diffs):
            builder.add(x)
    
    return builder.els

def _local_search_improve(A, B, n=256, max_iters=50, rng=None):
    """Local search with 1-for-1 and 1-for-2 swaps."""
    if rng is None:
        rng = random.Random()
    
    best_A = list(A)
    best_B = list(B)
    best_size = len(A) + len(B)
    
    for _ in range(max_iters):
        improved = False
        
        # Try 1-for-1 swaps: replace one element in A with another
        if len(best_A) < 64:
            B_diffs = _get_nonzero_diffs(best_B)
            # Try adding
            candidates = list(range(1, n + 1))
            rng.shuffle(candidates)
            for x in candidates:
                if x in best_A:
                    continue
                # Check if x can be added to A
                valid = True
                A_sums = set()
                for i in range(len(best_A)):
                    for j in range(i, len(best_A)):
                        A_sums.add(best_A[i] + best_A[j])
                for a in best_A:
                    s = a + x
                    if s in A_sums:
                        valid = False
                        break
                if valid:
                    s2 = 2 * x
                    if s2 in A_sums:
                        valid = False
                if valid:
                    for a in best_A:
                        d1 = a - x
                        d2 = x - a
                        if d1 != 0 and d1 in B_diffs:
                            valid = False
                            break
                        if d2 != 0 and d2 in B_diffs:
                            valid = False
                            break
                if valid:
                    best_A.append(x)
                    best_A.sort()
                    improved = True
                    break
        
        # Try 1-for-1 swap in A
        if len(best_A) > 1:
            B_diffs = _get_nonzero_diffs(best_B)
            for i in range(len(best_A)):
                old = best_A[i]
                candidates = list(range(1, n + 1))
                rng.shuffle(candidates)
                for x in candidates:
                    if x in best_A:
                        continue
                    new_A = best_A[:i] + best_A[i+1:] + [x]
                    new_A.sort()
                    if _is_valid_pair(new_A, best_B):
                        best_A = new_A
                        improved = True
                        break
                if improved:
                    break
        
        # Try adding to B
        if len(best_B) < 64:
            A_diffs = _get_nonzero_diffs(best_A)
            candidates = list(range(1, n + 1))
            rng.shuffle(candidates)
            for x in candidates:
                if x in best_B:
                    continue
                valid = True
                B_sums = set()
                for i in range(len(best_B)):
                    for j in range(i, len(best_B)):
                        B_sums.add(best_B[i] + best_B[j])
                for b in best_B:
                    s = b + x
                    if s in B_sums:
                        valid = False
                        break
                if valid:
                    s2 = 2 * x
                    if s2 in B_sums:
                        valid = False
                if valid:
                    for b in best_B:
                        d1 = b - x
                        d2 = x - b
                        if d1 != 0 and d1 in A_diffs:
                            valid = False
                            break
                        if d2 != 0 and d2 in A_diffs:
                            valid = False
                            break
                if valid:
                    best_B.append(x)
                    best_B.sort()
                    improved = True
                    break
        
        # Try 1-for-1 swap in B
        if len(best_B) > 1:
            A_diffs = _get_nonzero_diffs(best_A)
            for i in range(len(best_B)):
                old = best_B[i]
                candidates = list(range(1, n + 1))
                rng.shuffle(candidates)
                for x in candidates:
                    if x in best_B:
                        continue
                    new_B = best_B[:i] + best_B[i+1:] + [x]
                    new_B.sort()
                    if _is_valid_pair(best_A, new_B):
                        best_B = new_B
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_A, best_B

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin within 90s limit
    n = 256
    max_size = 64
    
    best_A = [1]
    best_B = [2]
    best_score = 2
    
    rng = random.Random()
    
    for restart in range(20000):
        if time.time() > deadline:
            break
        
        # Strategy 1: Build A first, then B
        if restart % 3 == 0:
            A_builder = _SidonBuilder(n, max_size)
            A = _greedy_build(A_builder, set(), n, rng, max_size)
            if not A:
                A = [rng.randint(1, n)]
                A_builder = _SidonBuilder(n, max_size)
                A_builder.add(A[0])
            
            B_diffs_forbidden = A_builder.diffs
            B_builder = _SidonBuilder(n, max_size)
            B = _greedy_build(B_builder, B_diffs_forbidden, n, rng, max_size)
            
            # Now try to add more to A using B's diffs
            A_builder2 = _SidonBuilder(n, max_size)
            for a in A:
                A_builder2.add(a)
            A = _greedy_build(A_builder2, B_builder.diffs, n, rng, max_size)
        
        # Strategy 2: Build B first, then A
        elif restart % 3 == 1:
            B_builder = _SidonBuilder(n, max_size)
            B = _greedy_build(B_builder, set(), n, rng, max_size)
            if not B:
                B = [rng.randint(1, n)]
                B_builder = _SidonBuilder(n, max_size)
                B_builder.add(B[0])
            
            A_diffs_forbidden = B_builder.diffs
            A_builder = _SidonBuilder(n, max_size)
            A = _greedy_build(A_builder, A_diffs_forbidden, n, rng, max_size)
            
            B_builder2 = _SidonBuilder(n, max_size)
            for b in B:
                B_builder2.add(b)
            B = _greedy_build(B_builder2, A_builder.diffs, n, rng, max_size)
        
        # Strategy 3: Random small sets, then extend
        else:
            size_a = rng.randint(1, 5)
            size_b = rng.randint(1, 5)
            A_init = sorted(rng.sample(range(1, n + 1), size_a))
            B_init = sorted(rng.sample(range(1, n + 1), size_b))
            if not _is_valid_pair(A_init, B_init):
                continue
            
            A_builder = _SidonBuilder(n, max_size)
            for a in A_init:
                A_builder.add(a)
            A = _greedy_build(A_builder, _get_nonzero_diffs(B_init), n, rng, max_size)
            
            B_builder = _SidonBuilder(n, max_size)
            for b in B_init:
                B_builder.add(b)
            B = _greedy_build(B_builder, _get_nonzero_diffs(A), n, rng, max_size)
        
        if not A:
            A = [1]
        if not B:
            B = [2]
        
        if _is_valid_pair(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
        
        # Local search
        if time.time() < deadline - 3:
            A_ls, B_ls = _local_search_improve(A, B, n, max_iters=30, rng=rng)
            if _is_valid_pair(A_ls, B_ls):
                score_ls = len(A_ls) + len(B_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_A = list(A_ls)
                    best_B = list(B_ls)
    
    if not _is_valid_pair(best_A, best_B):
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
