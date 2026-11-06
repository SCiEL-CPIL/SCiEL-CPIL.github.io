import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 180
    
    # Precompute all forbidden triples: distinct a,b,c in [1,N] with 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c => a = bc/(b+c)
    # WLOG let a < b < c (all distinct)
    # Then a = bc/(b+c) < b < c
    # c = ab/(b-a)
    triples = []
    triple_set = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            den = b - a
            if den <= 0:
                continue
            num = a * b
            if num % den != 0:
                continue
            c = num // den
            if c > b and c <= N:
                t = (a, b, c)
                triple_set.add(t)
                triples.append(t)
            elif c > a and c < b and c != a and c != b:
                # c is between a and b, reorder so a < c < b
                t = (a, c, b)
                triple_set.add(t)
                triples.append(t)
    
    # Build per-element conflict lists for fast checking
    # For each element x, store list of (other1, other2) pairs such that {x, other1, other2} is a forbidden triple
    conflict_pairs = {x: [] for x in range(1, N+1)}
    for (a, b, c) in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    # Also build a direct conflict partner set for degree-based heuristics
    conflict_partners = {x: set() for x in range(1, N+1)}
    for (a, b, c) in triples:
        conflict_partners[a].add(b)
        conflict_partners[a].add(c)
        conflict_partners[b].add(a)
        conflict_partners[b].add(c)
        conflict_partners[c].add(a)
        conflict_partners[c].add(b)
    
    def creates_triple(elem, current_set):
        """Check if adding elem to current_set creates a forbidden triple.
        Fast: only check pairs involving elem."""
        if elem in current_set:
            return False
        for (o1, o2) in conflict_pairs[elem]:
            if o1 in current_set and o2 in current_set:
                return True
        return False
    
    def greedy(order):
        s = set()
        for elem in order:
            if not creates_triple(elem, s):
                s.add(elem)
        return s
    
    def local_search(s, max_iters=20000):
        """Add missing elements that don't create conflicts."""
        s = set(s)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            missing = [x for x in range(1, N+1) if x not in s]
            random.shuffle(missing)
            for elem in missing:
                if not creates_triple(elem, s):
                    s.add(elem)
                    improved = True
        return s
    
    def remove_then_add(s, max_iters=20000):
        """Remove one element, try to add multiple others."""
        s = set(s)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            elems = list(s)
            random.shuffle(elems)
            for rem in elems:
                s2 = set(s)
                s2.discard(rem)
                missing = [x for x in range(1, N+1) if x not in s2]
                random.shuffle(missing)
                for elem in missing:
                    if not creates_triple(elem, s2):
                        s2.add(elem)
                if len(s2) > len(s):
                    s = s2
                    improved = True
                    break
        return s
    
    def two_swap(s, max_iters=30000):
        """Try swapping: remove one element, add two others (or remove two, add three)."""
        s = set(s)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            elems = list(s)
            random.shuffle(elems)
            missing = [x for x in range(1, N+1) if x not in s]
            random.shuffle(missing)
            for rem in elems[:20]:  # limit per iteration for speed
                s2 = set(s)
                s2.discard(rem)
                # Try to add up to 2 missing elements
                added = 0
                for elem in missing:
                    if added >= 2:
                        break
                    if not creates_triple(elem, s2):
                        s2.add(elem)
                        added += 1
                if len(s2) > len(s):
                    s = s2
                    improved = True
                    break
        return s
    
    def greedy_remove_most_conflicts():
        """Start with all elements, greedily remove elements with most conflicts in current set."""
        S = set(range(1, N + 1))
        while True:
            if time.time() > deadline - 5:
                break
            # Find element with highest number of conflict partners in S
            best_elem = None
            best_count = 0
            for x in S:
                partners_in_S = len(conflict_partners[x] & S)
                if partners_in_S > best_count:
                    best_count = partners_in_S
                    best_elem = x
            if best_elem is None or best_count == 0:
                break
            S.remove(best_elem)
        return S
    
    def weighted_greedy():
        """Greedy with weights favoring elements with fewer conflicts."""
        order = list(range(1, N+1))
        # Sort by number of conflict partners (fewer first)
        order.sort(key=lambda x: len(conflict_partners[x]))
        s = set()
        for elem in order:
            if not creates_triple(elem, s):
                s.add(elem)
        return s
    
    def spectral_community_pruning():
        """Use spectral clustering on conflict graph to identify dense communities,
        then apply targeted minimum vertex cover heuristic within dense clusters."""
        
        # Build adjacency matrix for conflict graph (edges between elements that appear in same triple)
        # Use a simplified spectral approach: compute eigenvectors of the Laplacian
        
        # Create index mapping
        elem_list = list(range(1, N+1))
        elem_to_idx = {x: i for i, x in enumerate(elem_list)}
        
        # Build degree matrix and adjacency
        degrees = [len(conflict_partners[x]) for x in elem_list]
        
        # Build sparse-like adjacency using only non-zero entries
        # For spectral clustering, we need the Laplacian L = D - A
        # We'll use a power iteration approach to find the Fiedler vector (2nd smallest eigenvector)
        
        def laplacian_vector_mult(v):
            """Compute L*v where L = D - A"""
            result = [0.0] * N
            for i, x in enumerate(elem_list):
                # D*v: degrees[i] * v[i]
                result[i] = degrees[i] * v[i]
                # Subtract A*v: sum of v[j] for j in neighbors of i
                for y in conflict_partners[x]:
                    j = elem_to_idx[y]
                    result[i] -= v[j]
            return result
        
        # Power iteration for Fiedler vector (2nd smallest eigenvalue)
        # Start with random vector orthogonal to all-ones
        random.seed(12345)
        v = [random.random() - 0.5 for _ in range(N)]
        
        # Make orthogonal to all-ones vector
        mean_v = sum(v) / N
        v = [x - mean_v for x in v]
        norm_v = math.sqrt(sum(x*x for x in v))
        if norm_v > 0:
            v = [x / norm_v for x in v]
        
        # Iterate: project out all-ones, normalize
        for iteration in range(100):
            if time.time() > deadline - 30:
                break
            v_new = laplacian_vector_mult(v)
            # Project out all-ones component
            mean_new = sum(v_new) / N
            v_new = [x - mean_new for x in v_new]
            norm_new = math.sqrt(sum(x*x for x in v_new))
            if norm_new < 1e-10:
                break
            v_new = [x / norm_new for x in v_new]
            v = v_new
        
        # Cluster based on sign of Fiedler vector
        # Elements with v[x] > 0 are in one community, v[x] < 0 in another
        # But we want multiple communities, so let's use a threshold-based approach
        
        # Sort elements by their Fiedler vector value
        sorted_elems = sorted(elem_list, key=lambda x: v[elem_to_idx[x]])
        
        # Identify dense communities: elements with high degree and similar Fiedler values
        # Small numbers tend to be hubs - group them
        threshold = 0.0
        dense_community = [x for x in elem_list if v[elem_to_idx[x]] < threshold and degrees[elem_to_idx[x]] > 5]
        sparse_region = [x for x in elem_list if x not in dense_community]
        
        # Within dense community, apply minimum vertex cover heuristic
        # Greedily remove elements that cover the most triples
        S = set(range(1, N+1))
        
        # First, handle dense community with aggressive pruning
        for x in sorted(dense_community, key=lambda y: -len(conflict_partners[y] & set(dense_community))):
            if time.time() > deadline - 25:
                break
            # Check if removing x would break triples
            # Count triples that x participates in with other elements in S
            covered_triples = 0
            for (o1, o2) in conflict_pairs[x]:
                if o1 in S and o2 in S:
                    covered_triples += 1
            if covered_triples > 0:
                S.remove(x)
        
        # Now add back sparse region elements that don't create conflicts
        for x in sorted(sparse_region, key=lambda y: -len(conflict_partners[y])):
            if time.time() > deadline - 25:
                break
            if x not in S and not creates_triple(x, S):
                S.add(x)
        
        return S
    
    # Multiple restarts with different strategies
    random.seed(42)
    
    best_set = set(range(91, 181))  # seed baseline: {91,...,180}
    best_size = len(best_set)
    
    # Strategy 1: Descending greedy (favor large elements - fewer constraints)
    s_desc = greedy(list(range(N, 0, -1)))
    s_desc = local_search(s_desc, max_iters=5000)
    if len(s_desc) > best_size:
        best_set = s_desc
        best_size = len(s_desc)
    
    # Strategy 2: Ascending greedy
    s_asc = greedy(list(range(1, N+1)))
    s_asc = local_search(s_asc, max_iters=5000)
    if len(s_asc) > best_size:
        best_set = s_asc
        best_size = len(s_asc)
    
    # Strategy 3: Weighted greedy (fewer conflicts first)
    s_weighted = weighted_greedy()
    s_weighted = local_search(s_weighted, max_iters=5000)
    if len(s_weighted) > best_size:
        best_set = s_weighted
        best_size = len(s_weighted)
    
    # Strategy 4: Greedy remove most conflicts
    s_greedy_remove = greedy_remove_most_conflicts()
    s_greedy_remove = local_search(s_greedy_remove, max_iters=5000)
    if len(s_greedy_remove) > best_size:
        best_set = s_greedy_remove
        best_size = len(s_greedy_remove)
    
    # Strategy 5: Spectral community pruning (NEW)
    if time.time() < deadline - 20:
        s_spectral = spectral_community_pruning()
        s_spectral = local_search(s_spectral, max_iters=5000)
        if len(s_spectral) > best_size:
            best_set = s_spectral
            best_size = len(s_spectral)
    
    # Strategy 6: Random restarts with biased ordering (favor larger elements)
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        # Create ordering with bias toward larger elements
        order = list(range(1, N+1))
        # Shuffle but with a bias: larger elements more likely to come first
        for i in range(N-1, 0, -1):
            # Probability of swapping decreases with position
            j = random.randint(0, i)
            if random.random() < 0.5:  # 50% chance to actually swap
                order[i], order[j] = order[j], order[i]
        
        s = greedy(order)
        s = local_search(s, max_iters=3000)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Strategy 7: Start from high half, add lower elements
    s_high = set(range(91, 181))
    s_high = local_search(s_high, max_iters=5000)
    if len(s_high) > best_size:
        best_set = s_high
        best_size = len(s_high)
    
    # Strategy 8: Start from top third
    s_top = set(range(121, 181))
    s_top = local_search(s_top, max_iters=5000)
    if len(s_top) > best_size:
        best_set = s_top
        best_size = len(s_top)
    
    # Strategy 9: Start from top quarter
    s_top4 = set(range(136, 181))
    s_top4 = local_search(s_top4, max_iters=5000)
    if len(s_top4) > best_size:
        best_set = s_top4
        best_size = len(s_top4)
    
    # Apply intensive local search to best
    if time.time() < deadline - 15:
        best_set = local_search(best_set, max_iters=50000)
        best_size = len(best_set)
    
    # Remove-then-add phase
    if time.time() < deadline - 15:
        best_set = remove_then_add(best_set, max_iters=30000)
        best_size = len(best_set)
    
    # Two-swap phase
    if time.time() < deadline - 10:
        best_set = two_swap(best_set, max_iters=20000)
        best_size = len(best_set)
    
    # More local search
    if time.time() < deadline - 8:
        best_set = local_search(best_set, max_iters=50000)
        best_size = len(best_set)
    
    # Another round of remove-then-add
    if time.time() < deadline - 8:
        best_set = remove_then_add(best_set, max_iters=30000)
        best_size = len(best_set)
    
    # Final polish
    if time.time() < deadline - 5:
        best_set = two_swap(best_set, max_iters=10000)
        best_size = len(best_set)
    
    if time.time() < deadline - 3:
        best_set = local_search(best_set, max_iters=50000)
        best_size = len(best_set)
    
    return list(best_set)
