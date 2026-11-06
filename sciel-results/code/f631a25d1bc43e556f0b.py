import time
import random
from itertools import combinations

def _precompute_k222s(n):
    """
    Precompute all K(2,2,2) configurations and triple-to-K mapping.
    Returns:
    - all_triples: list of tuples
    - triple_to_k222s: list of lists (for each triple index, list of K indices it belongs to)
    - k222_list: list of frozensets of triple indices
    """
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    triple_to_k222s = [[] for _ in range(len(all_triples))]
    k222_list = []
    
    # Optimization: Precompute matchings for any set of 6 points?
    # Since n=16, there are C(16,6) = 8008 sets of 6 points.
    # For each set, there are 15 matchings.
    # Total K(2,2,2)s = 8008 * 15 = 120120.
    # This is manageable.
    
    for points in combinations(range(n), 6):
        # Generate all perfect matchings of these 6 points
        # There are 15 perfect matchings for 6 points
        def gen_matchings(pts):
            if not pts:
                yield []
                return
            first = pts[0]
            rest = pts[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for m in gen_matchings(new_rest):
                    yield [(first, second)] + m
        
        for matching in gen_matchings(points):
            # matching is list of 3 tuples: (a,a'), (b,b'), (c,c')
            k_triples = []
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        t = tuple(sorted((matching[0][x], matching[1][y], matching[2][z])))
                        k_triples.append(t)
            
            k_indices = set()
            for t in k_triples:
                k_indices.add(triple_to_idx[t])
            
            k222_idx = len(k222_list)
            k222_list.append(frozenset(k_indices))
            
            for idx in k_indices:
                triple_to_k222s[idx].append(k222_idx)
    
    return all_triples, triple_to_k222s, k222_list

class WeightedGreedyMIS:
    def __init__(self, all_triples, triple_to_k222s, k222_list):
        self.all_triples = all_triples
        self.n_triples = len(all_triples)
        self.triple_to_k222s = triple_to_k222s
        self.k222_list = k222_list
        self.n_k222s = len(k222_list)
        
        # Precompute for each triple, the list of K indices it belongs to
        # And for each K, the list of triples it contains
        self.k_to_triples = [list(k) for k in k222_list]
        
        # We need to quickly check if a triple can be added to a set.
        # A triple t can be added to set S if no K(2,2,2) containing t is fully contained in S ∪ {t}.
        # i.e., for all K in triple_to_k222s[t], |K ∩ S| < |K| - 1 (since t is not in S yet, we need K\{t} not subset of S)
        # Actually, simpler: K is a set of 8 triples. If we add t, K becomes complete if all other 7 are in S.
        
        # For fast local search, we maintain:
        # present: set of triple indices
        # k_count: array of length n_k222s, where k_count[k] = number of triples in K_k that are in present
        
    def _can_add(self, t_idx, present, k_count):
        for k_idx in self.triple_to_k222s[t_idx]:
            # If all other members of K_k are already in present, we cannot add t_idx
            # The size of K_k is 8.
            # If k_count[k_idx] == 7, then all other 7 are present.
            # Wait, k_count counts present elements in K.
            # If t_idx is NOT in present, and k_count[k_idx] == 7, then all 7 others are present.
            # Adding t_idx would make k_count 8 -> Complete K.
            if k_count[k_idx] == 7:
                return False
        return True

    def _add(self, t_idx, present, k_count):
        present.add(t_idx)
        for k_idx in self.triple_to_k222s[t_idx]:
            k_count[k_idx] += 1

    def _remove(self, t_idx, present, k_count):
        present.discard(t_idx)
        for k_idx in self.triple_to_k222s[t_idx]:
            k_count[k_idx] -= 1

    def run_greedy(self, weights, time_limit):
        """
        Run weighted greedy selection.
        weights: array of length n_triples.
        """
        present = set()
        k_count = [0] * self.n_k222s
        
        # Create a list of (weight, idx) and sort
        # To make it adaptive, we don't just sort once. 
        # But for a "Weighted Greedy" heuristic, we typically sort by weight and pick if valid.
        # To allow for "Adaptive Conflict Penalties", we can update weights during the process?
        # The prompt suggests: "dynamic vertex weights updated by the frequency of conflicts in previous failed attempts".
        # This implies an iterative process: 
        # 1. Select set S.
        # 2. Identify "conflict" patterns (e.g., K's that are almost full).
        # 3. Penalize vertices in those K's.
        # 4. Repeat.
        
        # Let's implement a simple version:
        # 1. Sort all triples by current weight.
        # 2. Iterate through sorted list. If valid, add.
        # 3. After one pass, analyze which K's are "close" to being complete (count 7).
        #    The 8th vertex that *would* complete them is "bad".
        #    Actually, if we didn't add it, it's because it was invalid or not reached?
        #    If it was invalid, it means 7 others were present.
        #    So the 7 present vertices are "conflicting" with each other in a way that blocks others.
        #    We should penalize the present vertices that are part of dense conflicts?
        #    Or penalize the *unselected* vertices that were blocked?
        #    Usually, in MIS, if a vertex is blocked, it's because its neighbors were picked.
        #    So we should penalize the *picked* neighbors? No, that would remove them.
        #    We want to find a *different* set.
        #    
        #    Alternative: "Weighted Greedy" usually means:
        #    Repeat:
        #      Pick highest weight vertex v.
        #      Add v to solution.
        #      Remove v and all its neighbors from consideration.
        #      Update weights.
        #    
        #    This is a standard heuristic for MIS. Let's use this.
        #    It produces a maximal independent set.
        #    To improve, we run it multiple times with different weight initializations/updates.
        
        # Let's implement the standard Weighted Greedy MIS:
        # 1. Initialize weights.
        # 2. While vertices remain:
        #    a. Pick v with max weight.
        #    b. Add v to solution.
        #    c. Remove v and all neighbors of v from the candidate pool.
        #    d. (Optional) Update weights of remaining vertices?
        #       A common update: w(u) = w(u) * (1 - 1/deg(u)) or similar.
        #       Or simply, since we remove neighbors, the "effective" graph shrinks.
        
        # However, the prompt specifically mentions "updated by frequency of conflicts in previous failed attempts".
        # This sounds more like a "No-Ad" or "Tabu" style update, or a penalty method.
        
        # Let's try a hybrid: 
        # 1. Start with random weights.
        # 2. Run a fast greedy: sort by weight, add if valid.
        # 3. Identify "critical" K's (those with count 7 or 8 in the final set? No, valid set has no 8s. Count 7s are "almost there").
        # 4. The vertices in those "almost complete" K's are "dangerous". 
        #    If we keep them, we block many others.
        #    So, in the next iteration, we penalize the weights of the vertices that are *present* in these almost-complete K's?
        #    No, if we penalize them, we might remove them.
        #    If we remove one vertex from an almost-complete K, we free up the other 7? No, the other 7 are still there.
        #    Removing one vertex from a K of size 8 reduces the count to 6. This unblocks the 8th vertex.
        #    So, we should *encourage* removing one of the 7?
        #    Or, we should *penalize* the 8th vertex? No, the 8th was blocked.
        
        # Let's stick to a robust Local Search on top of a good Greedy seed.
        # The "Weighted" part will be used to generate diverse seeds.
        
        pass # Placeholder, logic moved to construct

def _greedy_with_weights(weights, all_triples, triple_to_k222s, k222_list, time_limit):
    n_triples = len(all_triples)
    n_k222s = len(k222_list)
    k_to_triples = [list(k) for k in k222_list]
    
    present = set()
    k_count = [0] * n_k222s
    
    # Sort indices by weight descending
    # To make it fast, we don't re-sort every step. We just iterate through a pre-sorted list.
    # This is a "Static Weighted Greedy".
    sorted_indices = sorted(range(n_triples), key=lambda i: weights[i], reverse=True)
    
    for idx in sorted_indices:
        if time.time() > time_limit:
            break
        # Check if idx can be added
        can_add = True
        for k_idx in triple_to_k222s[idx]:
            if k_count[k_idx] == 7:
                can_add = False
                break
        
        if can_add:
            present.add(idx)
            for k_idx in triple_to_k222s[idx]:
                k_count[k_idx] += 1
            # Optional: Early stop if we've added a lot? No, keep going.
            
    return present, k_count

def _local_search(present, k_count, all_triples, triple_to_k222s, k222_list, time_limit):
    """
    Local search: Try to improve the set by removing 1 and adding k.
    Moves:
    - Remove 1, Add 1 (Swap)
    - Remove 1, Add 2
    - Remove 2, Add 3
    """
    n_triples = len(all_triples)
    n_k222s = len(k222_list)
    best_set = set(present)
    best_k_count = list(k_count)
    best_size = len(present)
    
    start_time = time.time()
    
    # Precompute neighbors? 
    # For a given triple t, which other triples conflict with it?
    # t conflicts with u if there exists a K containing both t and u.
    # We can precompute a "conflict graph" adjacency list.
    # n_triples = 560.
    # Each triple is in C(13,3) * ... wait.
    # Let's just use the k222 lists.
    
    # To make local search fast, we need to quickly find candidates for "Add".
    # Candidates for Add are triples not in present that are currently "blocked" by exactly the removed set?
    # If we remove R, a triple u can be added if for all K in triple_to_k222s[u], 
    # |K ∩ (present \ R)| < |K| - 1.
    # Since K size is 8, we need |K ∩ (present \ R)| <= 6.
    # Originally |K ∩ present| <= 7.
    # If u was blocked, it meant there was a K_u such that |K_u ∩ present| = 7.
    # If we remove R, and R intersects K_u, then |K_u ∩ (present \ R)| = 7 - |K_u ∩ R|.
    # So if we remove at least 1 vertex from every "blocking" K for u, u becomes addable.
    
    # Strategy:
    # 1. Identify "blocked" triples: triples not in present that have at least one K with count 7.
    # 2. For each blocked triple u, find the set of "blockers" B_u = { v in present | v is in a K with u and |K ∩ present|=7 }.
    # 3. To add u, we must remove at least one vertex from B_u for *every* blocking K.
    #    Actually, we just need to reduce the count of *each* blocking K to <= 6.
    #    So we need to remove a set R such that for every K in Blockers(u), |R ∩ K| >= 1.
    #    This is a hitting set problem.
    #    Simple heuristic: Pick one random blocker from each blocking K, take the union.
    #    Or, pick the vertex that appears in the most blocking K's for u.
    
    # Let's implement a simple "Remove 1, Add k" search.
    # 1. Pick a random vertex v from present to remove.
    # 2. Find all triples u not in present that become addable if v is removed.
    #    u becomes addable if all its blocking K's contain v.
    #    i.e., for all K in triple_to_k222s[u] where k_count[K]==7, v in K.
    # 3. If we found a set of addable triples A, we can add up to |A| triples.
    #    But adding one might block another?
    #    If we add u1, it might increase counts in K's that block u2.
    #    So we should try to add them one by one, checking validity.
    
    # Optimization: Precompute for each v, the list of u's that are *only* blocked by v?
    # That's hard to maintain dynamically.
    
    # Let's do a simpler local search:
    # Repeat:
    #   1. Randomly pick v in present.
    #   2. Remove v.
    #   3. Scan all u not in present. If u is valid, add it.
    #   4. If size increased, keep. Else, revert.
    
    # This is O(N^2) per step. N=560. 560^2 = 313,600.
    # With 120s, we can do millions of steps.
    
    current_set = set(present)
    current_k_count = list(k_count)
    current_size = len(current_set)
    
    # Precompute for each triple u, the list of K indices.
    # We already have triple_to_k222s.
    
    while time.time() - start_time < time_limit:
        if not current_set:
            break
            
        # Pick a random vertex to remove
        v = random.choice(list(current_set))
        
        # Remove v
        current_set.discard(v)
        for k_idx in triple_to_k222s[v]:
            current_k_count[k_idx] -= 1
            
        # Try to add as many as possible
        added_any = False
        # To avoid bias, we can shuffle the order of candidates
        # Candidates are all indices not in current_set
        # We iterate through all indices? 560 is small.
        
        # We need to check validity for each u.
        # Validity: for all k in triple_to_k222s[u], current_k_count[k] < 7
        
        # We can pre-sort candidates by "potential" (e.g., number of blocking K's)?
        # For now, just iterate.
        
        # Optimization: Only check u if it was previously blocked?
        # We don't track "previously blocked" easily.
        # Just check all u not in current_set.
        
        # To speed up, we can break early if we've added enough?
        # No, we want max size.
        
        # Let's try adding in a random order to avoid bias
        # But randomizing 560 items every step is expensive.
        # Let's just iterate 0..N-1.
        
        added_indices = []
        for u in range(n_triples):
            if u in current_set:
                continue
                
            # Check if u can be added
            can_add = True
            for k_idx in triple_to_k222s[u]:
                if current_k_count[k_idx] == 7:
                    can_add = False
                    break
                    
            if can_add:
                # Add u
                current_set.add(u)
                for k_idx in triple_to_k222s[u]:
                    current_k_count[k_idx] += 1
                added_indices.append(u)
                added_any = True
                
        # Check if we improved
        new_size = len(current_set)
        if new_size > current_size:
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
                best_k_count = list(current_k_count)
        else:
            # Revert
            # Remove added_indices
            for u in added_indices:
                current_set.discard(u)
                for k_idx in triple_to_k222s[u]:
                    current_k_count[k_idx] -= 1
            # Re-add v
            current_set.add(v)
            for k_idx in triple_to_k222s[v]:
                current_k_count[k_idx] += 1
            # current_size should be back to original
            # We don't need to update best if it didn't improve
            
    return best_set, best_k_count

def construct():
    start_time = time.time()
    deadline = 120.0
    n = 16
    
    all_triples, triple_to_k222s, k222_list = _precompute_k222s(n)
    n_triples = len(all_triples)
    
    best_set = set()
    best_k_count = [0] * len(k222_list)
    best_size = 0
    
    # Phase 1: Multiple Weighted Greedy Seeds
    # We will run several iterations of Weighted Greedy with different weight initializations.
    # Then refine the best one with Local Search.
    
    num_greedy_runs = 20
    time_per_greedy = (deadline - 30.0) / num_greedy_runs # Leave 30s for local search
    
    for i in range(num_greedy_runs):
        if time.time() > deadline - 30.0:
            break
            
        # Initialize weights
        # Strategy: Random weights, or based on degree?
        # Let's use random weights in [0, 1]
        weights = [random.random() for _ in range(n_triples)]
        
        # Run Greedy
        seed_set, seed_k_count = _greedy_with_weights(
            weights, all_triples, triple_to_k222s, k222_list, time.time() + time_per_greedy * 0.5
        )
        
        if len(seed_set) > best_size:
            best_size = len(seed_set)
            best_set = set(seed_set)
            best_k_count = list(seed_k_count)
            
    # Phase 2: Local Search on the Best Seed
    # Allocate remaining time for local search
    remaining_time = deadline - 5.0 - (time.time() - start_time)
    if remaining_time > 1.0 and best_set:
        # Run local search
        # We can run multiple local search passes if time permits
        # But local search is a loop, so just give it the time.
        final_set, final_k_count = _local_search(
            best_set, best_k_count, all_triples, triple_to_k222s, k222_list, remaining_time
        )
        
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = set(final_set)
            best_k_count = list(final_k_count)

    # Final validation
    valid = True
    for k_idx in range(len(k222_list)):
        if k222_list[k_idx].issubset(best_set):
            valid = False
            break
            
    if not valid:
        # Fallback: remove one triple from each violating K
        for k_idx in range(len(k222_list)):
            if k222_list[k_idx].issubset(best_set):
                to_remove = random.choice(list(k222_list[k_idx]))
                best_set.discard(to_remove)
        
        # Re-verify
        for k_idx in range(len(k222_list)):
            if k222_list[k_idx].issubset(best_set):
                best_set = set()
                break

    return [list(all_triples[i]) for i in sorted(best_set)]

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Found {len(res)} triples in {t1-t0:.2f}s")
