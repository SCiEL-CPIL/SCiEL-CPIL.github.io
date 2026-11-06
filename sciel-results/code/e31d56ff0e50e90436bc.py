"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

# Global state for the cap set solver
_start_time = None
_DEADLINE = 325.0  # Safety margin before 330s kill

def _init_state(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _is_deadline_exceeded():
    return _elapsed() > _DEADLINE

def priority(vector, n):
    # This function is called repeatedly by the evaluator.
    # We need to maintain state across calls to build the cap set greedily.
    # However, the contract says "vectors of {0,1,2}^n are added greedily in descending priority".
    # This implies the evaluator sorts all 3^n vectors by priority (descending) and adds them if valid.
    # Wait, if the evaluator does the greedy addition, it calls priority for each vector to sort them.
    # If so, priority must be deterministic and fast.
    # But the task says "Interface: priority(vector: tuple, n: int) -> float".
    # And "Score: cap set size".
    # Usually in FunSearch, the code *is* the algorithm.
    # Let's re-read carefully: "vectors ... are added greedily in descending priority when they keep the set a cap set".
    # This suggests the *evaluator* performs the greedy insertion based on the priorities returned by `priority`.
    # If the evaluator does the greedy insertion, then `priority` is just a ranking function.
    # The size of the resulting cap set depends on the ordering defined by `priority`.
    # To maximize the cap set size, we need to find an ordering of {0,1,2}^n such that the greedy cap set algorithm produces a large set.
    
    # Since `priority` is called for each vector, we can pre-compute a global ordering or use a heuristic.
    # However, `n` is passed, so we can cache results for specific `n`.
    
    # Strategy:
    # 1. Use a heuristic priority that tends to keep vectors "far apart" in some metric or structure.
    # 2. Or, since we have time, we can't do heavy computation inside `priority` if it's called 3^n times.
    #    For n=10, 3^10 = 59049 calls. That's fine.
    #    For n=11, 3^11 = 177147 calls. Still fine.
    #    For n=12, 3^12 = 531441 calls. Might be tight if priority is slow.
    #    For n=13, 3^13 = 1.5M calls. Too slow for complex logic.
    
    # Let's assume n is small enough (likely <= 10 or 11 for standard benchmarks, but could be higher).
    # A known good heuristic for cap sets is to prioritize vectors with low "energy" or specific structural properties.
    
    # One effective heuristic:
    # Maximize the distance to already selected vectors? No, priority is static per vector (usually).
    # If priority can depend on the current set, the interface would need to be different.
    # The interface is `priority(vector, n)`. It does NOT take the current set.
    # This means the priority is a STATIC ranking of all vectors.
    # The greedy algorithm then picks the highest priority vector, then the next highest that doesn't form a line, etc.
    
    # So the problem reduces to: Find a permutation of {0,1,2}^n that maximizes the size of the greedy cap set.
    
    # Heuristics for static ranking:
    # 1. Lexicographic (current seed, score 1.0)
    # 2. Random (score ~1.0 usually, maybe slightly better or worse)
    # 3. Structured: e.g., sort by sum of elements, or by number of 0s, or by some hash.
    
    # A known result: The greedy algorithm with a "random" order often produces cap sets of size ~ c * n^1.5 or similar?
    # Actually, for {0,1,2}^n, the max cap set size is ~ 3^n / (n * log n) or similar? No, it's exponential but smaller base.
    # Known bounds: c_n ~ 3^n / (n^{1+o(1)}).
    
    # Let's try a heuristic that groups vectors by their "complexity" or "distance from 0".
    # Vectors with many 0s might be "safer" to include early? Or later?
    
    # Let's try sorting by the sum of elements.
    # Or, better: sort by the number of non-zero elements.
    
    # Another idea: Use a deterministic pseudo-random hash to shuffle the order.
    # A fixed random permutation often outperforms lexicographic.
    
    # Let's implement a fixed random permutation based on a seed.
    # Since `priority` is called for each vector, we can return a pre-computed rank.
    
    # Optimization:
    # We can cache the priorities for a given n.
    # Generate a random permutation of all 3^n vectors, assign ranks 0..3^n-1.
    # Return the rank as the priority (higher is better, so invert or use negative).
    
    # But wait, if we just use a random permutation, is that optimal?
    # The "inspiration" mentioned "directed hill climbing on the vector ordering".
    # This implies we can *optimize* the ordering.
    # But `priority` is a pure function (vector, n) -> float. It cannot maintain state between calls unless we use global variables.
    # We CAN use global variables to store the optimized ordering.
    
    # Plan:
    # 1. On first call for a given n, generate an initial ordering (e.g., random).
    # 2. Use the remaining time to improve the ordering via local search (hill climbing).
    #    - Swap two vectors in the ordering.
    #    - Re-evaluate the greedy cap set size.
    #    - If size increases, keep the swap.
    # 3. Store the final ordering in a global dict: {n: {vector: priority_rank}}.
    # 4. `priority(vector, n)` just looks up the rank.
    
    # This fits the "directed hill climbing" inspiration.
    
    global _orderings
    if '_orderings' not in globals():
        _orderings = {}
    
    if n not in _orderings:
        # Initialize ordering for this n
        _orderings[n] = _build_initial_ordering(n)
    
    # Return the priority (rank). Higher rank = higher priority.
    # We want the vector with the highest priority to be picked first.
    # So if rank is 0 for first, 1 for second, etc., we should return (total - rank) or just -rank.
    # Let's return the index in the sorted list, where index 0 is the most preferred.
    # So priority = -index. Or just return the index and let the evaluator sort descending?
    # "descending priority" -> highest float first.
    # So if we want vector v to be first, it should have the highest float.
    # Let's assign priority = (3^n - 1) - index.
    
    # But we need to do the hill climbing *somewhere*.
    # We can do it lazily on the first call for a given n, but we must respect the deadline.
    # Actually, we can do the hill climbing in a background-like manner? No, Python is single-threaded here.
    # We can do a quick hill climbing session on the first call for n, limited by time.
    
    if _is_deadline_exceeded():
        # Just return a simple heuristic to avoid timeout
        return sum(vector)
        
    # Perform hill climbing for this n if not already done
    if 'done_hill_climb' not in _orderings or not _orderings['done_hill_climb'].get(n, False):
        _hill_climb_ordering(n)
        _orderings['done_hill_climb'] = _orderings.get('done_hill_climb', {})
        _orderings['done_hill_climb'][n] = True

    # Get the rank
    rank = _orderings[n][vector]
    # Convert rank to priority (higher is better)
    total = 3**n
    return float(total - 1 - rank)

def _build_initial_ordering(n):
    """Generate an initial random ordering for {0,1,2}^n."""
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(vectors)
    # Create a dict: vector -> rank
    ordering = {v: i for i, v in enumerate(vectors)}
    return ordering

def _greedy_cap_set_size(n, ordering):
    """
    Simulate the greedy cap set algorithm with the given ordering.
    Returns the size of the resulting cap set.
    ordering: dict {vector: rank}
    We need to process vectors in order of increasing rank (0 is first).
    """
    # Get vectors sorted by rank
    # Since n can be up to ~10-12, 3^n is manageable.
    # For efficiency, we can just create a list of vectors sorted by rank.
    # But doing this every time in hill climbing might be slow.
    # Let's keep a list `order` where order[i] is the vector with rank i.
    # Actually, let's change _build_initial_ordering to return a list `order` and a dict `rank_map`.
    
    # For now, let's assume we have a list `order` of vectors.
    # This function will be refactored to take `order` directly.
    pass

def _hill_climb_ordering(n):
    """
    Perform hill climbing on the ordering for dimension n.
    """
    # Get current ordering
    # We need to store the order as a list for easy swapping.
    if n not in _orderings:
        return
        
    # Re-initialize to have a list
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    random.seed(42)
    random.shuffle(vectors)
    order = vectors[:]  # list of vectors in priority order (index 0 is highest priority)
    
    def evaluate(order):
        """
        Evaluate the greedy cap set size for the given order.
        Returns (size, set_of_vectors).
        """
        cap_set = set()
        # For efficiency, we need a fast way to check if adding a vector creates a line.
        # A line is {a, b, c} where a+b+c = (0,0,...,0) mod 3.
        # Equivalently, for any two vectors x, y in the set, the third point z = -x-y mod 3 must not be in the set.
        # When adding a new vector v, we need to check if there exist x, y in cap_set such that x+y+v = 0 mod 3.
        # This is equivalent to checking if v is in the "sumset" of cap_set.
        # Alternatively, for each x in cap_set, check if (-x - v) mod 3 is in cap_set.
        
        # This is O(|cap_set|) per insertion.
        # Total O(N^2) where N is the number of vectors considered.
        # For 3^n vectors, this is 9^n, which is too slow for n=10 (9^10 = 3.4e9).
        
        # We need a faster check.
        # We can maintain a set of "forbidden" vectors.
        # When we add v, the new forbidden vectors are those w such that v+w+u=0 for some u in cap_set.
        # i.e., w = -v-u mod 3.
        # So for each u in cap_set, w = (-v - u) mod 3 is forbidden.
        # We can maintain a set `forbidden` of all vectors that would complete a line with two existing vectors.
        # Actually, `forbidden` should be the set of vectors v such that adding v creates a line.
        # If we add v, we must check if v is in `forbidden`.
        # After adding v, we update `forbidden`:
        # For each u in cap_set (including v), the new forbidden vectors are (-v - u) mod 3.
        # Wait, if we add v, then for any u in old_cap_set, the vector w = -v-u mod 3 is now forbidden (because {v, u, w} is a line).
        # Also, for any u in old_cap_set, the vector w = -u-v mod 3 is the same.
        
        # So algorithm:
        # cap_set = set()
        # forbidden = set()
        # for v in order:
        #     if v not in forbidden:
        #         cap_set.add(v)
        #         # Update forbidden
        #         for u in cap_set:  # including v
        #             w = tuple((-v - u[i]) % 3 for i in range(n))
        #             forbidden.add(w)
        
        # This is still O(|cap_set|) per insertion.
        # But |cap_set| is much smaller than 3^n.
        # If the cap set size is S, the total work is O(S * 3^n) in the worst case? No.
        # We iterate through all 3^n vectors. For each vector, we check if it's in forbidden (O(1)).
        # If we add it, we do O(|cap_set|) work to update forbidden.
        # Total work: O(3^n + S * S) ? No.
        # We iterate 3^n times.
        # For each v:
        #   Check forbidden: O(1)
        #   If add:
        #     Loop over current cap_set (size k): O(k)
        #     Add k new vectors to forbidden.
        # Total additions: S.
        # Total update cost: Sum_{k=1}^{S} k = O(S^2).
        # Total iteration cost: O(3^n).
        # So total time is O(3^n + S^2).
        # For n=10, 3^10 = 59049. S is maybe ~100-200? S^2 = 40000.
        # This is very fast!
        
        cap_set = set()
        forbidden = set()
        for v in order:
            if v not in forbidden:
                cap_set.add(v)
                # Update forbidden
                # For each u in cap_set, w = -v - u mod 3 is forbidden
                for u in cap_set:
                    w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                    forbidden.add(w)
        return len(cap_set)

    # Initial evaluation
    current_size = evaluate(order)
    
    # Hill climbing
    while not _is_deadline_exceeded():
        improved = False
        # Try random swaps
        for _ in range(100):  # Fixed number of attempts per iteration
            if _is_deadline_exceeded():
                break
            i, j = random.sample(range(len(order)), 2)
            order[i], order[j] = order[j], order[i]
            new_size = evaluate(order)
            if new_size > current_size:
                current_size = new_size
                improved = True
            else:
                # Revert
                order[i], order[j] = order[j], order[i]
        
        if not improved:
            # Try a few more rounds or break
            # To avoid getting stuck, we can do a random restart or just break
            break

    # Update the global ordering dict
    _orderings[n] = {v: i for i, v in enumerate(order)}

# EVOLVE-BLOCK-END
