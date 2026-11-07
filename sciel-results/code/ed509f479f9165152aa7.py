"""
Strategy: Complete Backtracking Search with Bitmasks.

The previous approach used a greedy heuristic (primes first, then composites)
which can get stuck in local optima. This new approach uses a systematic
backtracking search over the candidate set {2, ..., 40}.

Key differences:
1. Representation: Uses Python integers as bitmasks to represent sets of
   candidates. This is faster for set operations (intersection, union) than
   manipulating lists of products.
2. Search: Exhaustively explores adding/removing candidates to find the
   maximum size valid set. It doesn't rely on a fixed ordering (like primes
   first) but tries to maximize count.
3. Pruning: 
   - If the current size + remaining candidates <= best found, prune.
   - If adding a candidate causes a collision in subset products, prune.
   - Time budgeting ensures we return the best valid set found within the limit.

This addresses the stagnation by potentially finding larger sets that the
greedy heuristic missed due to early bad choices (e.g., picking a composite
that blocks two other beneficial composites later).
"""

import time
import sys

_START_TIME = time.time()
# Safety margin: 60s limit, leave 5s for overhead
_DEADLINE = _START_TIME + 55.0

# Candidate range 2..40
CANDIDATES = list(range(2, 41))
N_CANDS = len(CANDIDATES)

# Precompute the index of each candidate for bitmask operations
CAND_IDX = {val: i for i, val in enumerate(CANDIDATES)}

# Precompute the "bit" for each candidate
CAND_BITS = [1 << i for i in range(N_CANDS)]

# Global best result
_best_set = []
_best_size = 0

def _check_collision(current_prods, new_val):
    """
    Check if adding new_val to a set with subset products 'current_prods'
    results in any duplicate products.
    
    The new products are {p * new_val for p in current_prods}.
    We need to ensure:
    1. No new product equals an existing product.
    2. No two new products are equal (impossible if current_prods are distinct and new_val != 0).
    
    So we just check: for any p in current_prods, is (p * new_val) in current_prods?
    """
    for p in current_prods:
        if p * new_val in current_prods:
            return True
    return False

def _update_prods(current_prods, new_val):
    """
    Compute the new set of subset products after adding new_val.
    """
    new_prods = set(current_prods)
    for p in current_prods:
        new_prods.add(p * new_val)
    return new_prods

def _backtrack(index, current_mask, current_prods, current_size):
    """
    Recursive backtracking search.
    
    index: current index in CANDIDATES to consider
    current_mask: bitmask of selected candidates
    current_prods: set of subset products for the current selection
    current_size: number of selected candidates
    """
    global _best_set, _best_size
    
    # Time check
    if time.time() > _DEADLINE:
        return

    # Update best if we found a larger set
    if current_size > _best_size:
        _best_size = current_size
        # Reconstruct the set from the mask
        best_vals = [CANDIDATES[i] for i in range(N_CANDS) if current_mask & CAND_BITS[i]]
        _best_set = best_vals

    # Base case: no more candidates to consider
    if index >= N_CANDS:
        return

    # Pruning: even if we pick all remaining candidates, can we beat the best?
    remaining = N_CANDS - index
    if current_size + remaining <= _best_size:
        return

    # Option 1: Skip current candidate
    _backtrack(index + 1, current_mask, current_prods, current_size)

    # Option 2: Include current candidate
    val = CANDIDATES[index]
    
    # Check if adding val causes a collision
    if not _check_collision(current_prods, val):
        new_mask = current_mask | CAND_BITS[index]
        new_prods = _update_prods(current_prods, val)
        _backtrack(index + 1, new_mask, new_prods, current_size + 1)

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    """
    global _best_set, _best_size, _DEADLINE
    
    # Reset globals
    _best_set = []
    _best_size = 0
    _DEADLINE = time.time() + 55.0

    # Initial state: empty set, product {1}
    initial_prods = {1}
    
    # Start backtracking
    # We can optimize by sorting candidates. 
    # Heuristic: Try primes first? Or small numbers first?
    # Actually, for backtracking, the order affects pruning efficiency.
    # If we put "risky" (composite) numbers later, we might explore more "safe" (prime) branches first.
    # However, the search space is 2^39, which is huge. 
    # But the constraint is very tight. 
    # Let's try to order by "complexity" or just standard order.
    # Standard order 2..40 is fine.
    
    # To improve pruning, we can sort candidates such that those likely to be excluded
    # are considered later? Or those likely to be included first?
    # Usually, trying to include items that are "harder" to fit (composites) earlier 
    # might fail faster. But primes are "safe".
    # Let's stick to the natural order 2..40. The bitmask approach handles the set logic.
    
    _backtrack(0, 0, initial_prods, 0)

    # Ensure types are correct
    # _best_set contains ints from CANDIDATES which are already ints
    return [int(x) for x in _best_set]

# To prevent recursion limit issues, although depth is only 39, 
# Python's default recursion limit is 1000, so we are safe.
# sys.setrecursionlimit(10000) # Not strictly needed for depth 39, but safe.

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found {len(res)} elements: {res}")
