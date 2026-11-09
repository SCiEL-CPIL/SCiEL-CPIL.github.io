<<<<<<< SEARCH
"""Rewritten program with a stronger algorithm for integer sum-product extremal sets."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 70.0  # Safety margin before 90s kill
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using vectorized numpy."""
        a = np.array(sorted(A), dtype=np.int64)
        # Sumset A+A (with equal summands allowed): all a[i]+a[j] for i<=j
        # Use broadcasting for upper triangle
        sumset = np.unique(a[:, None] + a[None, :])
        # Product set A*A (with equal factors allowed)
        prodset = np.unique(a[:, None] * a[None, :])
        return max(len(sumset), len(prodset))

    def evaluate_partial(A, changed_idx):
        """Faster evaluation when only one element changes. Still uses numpy but we can
        skip recomputing sums/products that don't involve the changed element.
        For simplicity we just call evaluate; the vectorized version is fast enough."""
        return evaluate(A)
=======
"""Rewritten program with a stronger algorithm for integer sum-product extremal sets."""
import time
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 70.0  # Safety margin before 90s kill
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using vectorized numpy."""
        a = np.array(sorted(A), dtype=np.int64)
        sumset = np.unique(a[:, None] + a[None, :])
        prodset = np.unique(a[:, None] * a[None, :])
        return max(len(sumset), len(prodset))

    class FastEvaluator:
        """Incremental evaluator that tracks sum and product multisets."""
        
        def __init__(self, A):
            self.n = len(A)
            self.A = sorted(A)
            self._rebuild()
        
        def _rebuild(self):
            """Rebuild all internal state from scratch."""
            a = self.A
            n = self.n
            # Build sum multiset: count of each sum value
            self.sum_counts = Counter()
            self.prod_counts = Counter()
            for i in range(n):
                for j in range(i, n):
                    self.sum_counts[a[i] + a[j]] += 1
                    self.prod_counts[a[i] * a[j]] += 1
            self.sum_size = len(self.sum_counts)
            self.prod_size = len(self.prod_counts)
            self.score = max(self.sum_size, self.prod_size)
        
        def update(self, old_val, new_val):
            """Update state when one element changes from old_val to new_val."""
            a = self.A
            n = self.n
            
            # Find index of old_val
            idx = a.index(old_val)
            
            # Remove contributions involving old_val
            for j in range(n):
                if j == idx:
                    continue
                s = old_val + a[j]
                p = old_val * a[j]
                self.sum_counts[s] -= 1
                if self.sum_counts[s] == 0:
                    del self.sum_counts[s]
                self.prod_counts[p] -= 1
                if self.prod_counts[p] == 0:
                    del self.prod_counts[p]
            
            # Also remove the self-pair (old_val, old_val)
            s_self = old_val + old_val
            p_self = old_val * old_val
            self.sum_counts[s_self] -= 1
            if self.sum_counts[s_self] == 0:
                del self.sum_counts[s_self]
            self.prod_counts[p_self] -= 1
            if self.prod_counts[p_self] == 0:
                del self.prod_counts[p_self]
            
            # Add new value
            a[idx] = new_val
            a.sort()
            self.A = a
            
            # Add contributions involving new_val
            for j in range(n):
                if j == idx:
                    continue
                s = new_val + a[j]
                p = new_val * a[j]
                self.sum_counts[s] += 1
                self.prod_counts[p] += 1
            
            # Add self-pair
            s_self = new_val + new_val
            p_self = new_val * new_val
            self.sum_counts[s_self] += 1
            self.prod_counts[p_self] += 1
            
            self.sum_size = len(self.sum_counts)
            self.prod_size = len(self.prod_counts)
            self.score = max(self.sum_size, self.prod_size)
>>>>>>> REPLACE

