"""
Stronger covering design algorithm for C(v, k, t).

Approach:
- Use a priority-queue based greedy set cover where the 'cost' of each candidate block
  is the number of currently-uncovered t-subsets it covers.
- To keep this tractable within 120s, we use a randomized local-search / greedy hybrid:
  1. Start with a random partial cover.
  2. Repeatedly pick the block (from a candidate pool) that covers the most uncovered
     t-subsets, add it, and update.
  3. Candidate pool is generated on-the-fly using a heuristic: for each uncovered
     t-subset, generate k-combinations that contain it (sampled or full if small).
  4. Use numpy for fast set operations where possible.

Key optimizations:
- Represent uncovered t-subsets as a set of frozensets or sorted tuples.
- For each candidate block, compute how many uncovered t-subsets it covers via
  precomputed mapping or direct iteration (since t is small, C(k,t) is manageable).
- Use a max-heap keyed by coverage count, but since coverage changes dynamically,
  we use a "lazy" heap: push all candidates initially, then when popping, recompute
  current coverage and skip if stale.

Given v=19, k=8, t=5:
- Total t-subsets: C(19,5) = 11628
- Total k-subsets: C(19,8) = 75582
- C(k,t) = C(8,5) = 56 t-subsets per block

We can afford to iterate over all k-subsets if we do it smartly.
Strategy:
1. Enumerate all C(19,5) = 11628 t-subsets.
2. For each k-subset, precompute which t-subsets it covers (56 per block).
3. Use greedy: repeatedly pick the k-subset that covers the most *currently uncovered* t-subsets.
   To do this efficiently, maintain a count for each t-subset of how many times it's covered
   by selected blocks (but we just need uncovered vs not).
   
   Better: maintain a set of uncovered t-subsets. For each candidate k-subset, compute
   intersection with uncovered. Pick max. Add to cover. Remove from uncovered.
   
   Doing this naively: 11628 * 75582 * 56 operations ≈ too slow in pure Python.

Alternative: Use the "greedy with candidate generation" approach:
- Maintain uncovered set.
- At each step, for each uncovered t-subset S, the blocks that can cover it are
  all k-subsets containing S. There are C(v-t, k-t) = C(14, 3) = 364 such blocks per S.
- We can sample or enumerate these.
- Use a counter: for each candidate block, how many uncovered t-subsets does it cover?
  We can update this incrementally.

Given time constraints, let's use a simpler but effective approach:
- Randomized greedy: at each step, sample a few candidate blocks from the pool of
  blocks that cover a random uncovered t-subset, and pick the one with best coverage.
- Or: use the "first-fit" style but with better block construction.

Actually, for v=19, k=8, t=5, the total number of k-subsets is 75582, which is manageable.
Let's precompute for each k-subset the list of t-subsets it covers.
Then use a priority queue approach:
- Initialize: for each k-subset, coverage = 56 (all its t-subsets are uncovered initially).
- Use a max-heap by coverage.
- Pop the best, add to cover, then for each t-subset in that block, if it was uncovered,
  mark as covered, and for every other k-subset that also covers that t-subset, decrement
  its coverage count. Push updated coverage back.

The issue: updating all k-subsets that share a t-subset. Each t-subset is in C(v-t, k-t)=364
k-subsets. So per t-subset covered, we update 364 k-subsets. Total updates: 11628 * 364 ≈ 4.2M,
which is fine.

Let's implement this.
"""

import itertools
import heapq
import random
from collections import defaultdict


def construct(v, k, t):
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # For each k-subset, compute the list of t-subset indices it covers
    k_to_t_indices = []
    for k_sub in all_k_subsets:
        t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            t_indices.append(t_subset_to_idx[t_sub])
        k_to_t_indices.append(t_indices)
    
    # Reverse mapping: for each t-subset index, which k-subsets cover it
    t_to_k_indices = [[] for _ in range(num_t)]
    for ki, t_indices in enumerate(k_to_t_indices):
        for ti in t_indices:
            t_to_k_indices[ti].append(ki)
    
    # uncovered: set of t-subset indices that are still uncovered
    uncovered = set(range(num_t))
    
    # coverage[ki] = number of uncovered t-subsets covered by k-subset ki
    coverage = [len(k_to_t_indices[ki]) for ki in range(num_k)]
    
    # Build max-heap: (-coverage, ki)
    heap = [(-coverage[ki], ki) for ki in range(num_k)]
    heapq.heapify(heap)
    
    # To handle stale entries, we keep a "version" or just recompute on pop
    # Actually, we'll use a lazy heap: when we pop, we check if coverage[ki] matches -neg_cov
    # If not, we push the updated value and continue.
    
    blocks = []
    selected = set()  # selected k-subset indices
    
    while uncovered:
        # Pop from heap until we find a valid entry
        while heap:
            neg_cov, ki = heapq.heappop(heap)
            if ki in selected:
                continue
            if -neg_cov != coverage[ki]:
                # Stale entry, push updated
                heapq.heappush(heap, (-coverage[ki], ki))
                continue
            if coverage[ki] == 0:
                continue
            break
        else:
            # Heap exhausted but uncovered not empty? Shouldn't happen.
            break
        
        # Select this k-subset
        selected.add(ki)
        blocks.append(list(all_k_subsets[ki]))
        
        # Update: for each t-subset in this k-subset, if uncovered, mark covered
        # and decrement coverage of all other k-subsets that cover that t-subset
        for ti in k_to_t_indices[ki]:
            if ti not in uncovered:
                continue
            uncovered.discard(ti)
            # For each k-subset that covers ti, decrement its coverage
            for other_ki in t_to_k_indices[ti]:
                if other_ki in selected:
                    continue
                coverage[other_ki] -= 1
                heapq.heappush(heap, (-coverage[other_ki], other_ki))
    
    return blocks
