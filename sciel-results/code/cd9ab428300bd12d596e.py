"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import product

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    
    # Build adjacency sets and masks
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    deadline = time.time() + 55  # safety margin before 90s kill

    # Order vertices by degree descending (more constrained first)
    order = sorted(range(n), key=lambda v: -len(adj[v]))

    # Find all maximal cliques and maximal independent sets using Bron-Kerbosch
    def find_maximal_cliques(vertices_mask, current_clique, excluded_mask):
        """Bron-Kerbosch with pivot to find maximal cliques."""
        if vertices_mask == 0 and excluded_mask == 0:
            if current_clique:
                yield current_clique
            return
        
        # Choose pivot u to minimize branching
        # We want a vertex in vertices_mask U excluded_mask that has max neighbors in vertices_mask
        candidates = vertices_mask | excluded_mask
        best_pivot = -1
        best_count = -1
        temp = candidates
        while temp:
            u = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            count = bin(adj_mask[u] & vertices_mask).count('1')
            if count > best_count:
                best_count = count
                best_pivot = u
        
        if best_pivot == -1:
            # No candidates, but vertices_mask is not 0? Should not happen if logic is right
            # Actually if candidates is 0, loop above doesn't run.
            # If vertices_mask != 0, we must process it.
            pass
        
        # Branch on vertices not adjacent to pivot
        # If pivot exists, branch on V \ N(pivot)
        if best_pivot != -1:
            candidates_to_branch = vertices_mask & ~(adj_mask[best_pivot] | (1 << best_pivot))
        else:
            candidates_to_branch = vertices_mask
            
        # Also include vertices in excluded_mask? No, standard BK uses V \ N(u) where u in V U X
        # The set to iterate is (vertices_mask U excluded_mask) \ N(pivot)
        # But we only add from vertices_mask to the clique.
        # Correct BK:
        # P, X
        # if P is empty and X is empty: report C
        # choose pivot u in P U X
        # for each v in (P U X) \ N(u):
        #   if v in P:
        #     C.add(v); BK(P \ N(v), X \ N(v), C); C.remove(v)
        #     P.remove(v); X.add(v)
        
        # Let's re-implement carefully
        def bk_rec(P, X, C):
            if P == 0 and X == 0:
                if C:
                    yield C
                return
            
            # Choose pivot
            PX = P | X
            pivot = -1
            max_neighbors = -1
            tmp = PX
            while tmp:
                u = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                cnt = bin(adj_mask[u] & P).count('1')
                if cnt > max_neighbors:
                    max_neighbors = cnt
                    pivot = u
            
            if pivot != -1:
                candidates = PX & ~(adj_mask[pivot] | (1 << pivot))
            else:
                candidates = PX
            
            tmp = candidates
            while tmp:
                v_bit = tmp & -tmp
                v = v_bit.bit_length() - 1
                tmp ^= v_bit
                
                if P & v_bit:
                    new_P = P & ~(adj_mask[v] | v_bit)
                    new_X = X & ~(adj_mask[v] | v_bit)
                    new_C = C | v_bit
                    yield from bk_rec(new_P, new_X, new_C)
                    P &= ~v_bit
                    X |= v_bit
                    # Note: P and X are modified in place in this recursive style?
                    # In Python, integers are immutable. So P and X are local copies.
                    # This is correct for BK.
        
        yield from bk_rec(vertices_mask, excluded_mask, current_clique)

    # Find maximal cliques
    all_cliques = []
    for c in find_maximal_cliques((1 << n) - 1, 0, 0):
        all_cliques.append(c)
        
    # Find maximal independent sets (cliques in complement graph)
    # Complement adjacency mask
    comp_adj_mask = [0] * n
    for i in range(n):
        # All vertices except i and its neighbors in original graph
        comp_adj_mask[i] = ((1 << n) - 1) ^ (1 << i) ^ adj_mask[i]
    
    def find_maximal_independent_sets(vertices_mask, current_is, excluded_mask):
        """Bron-Kerbosch on complement graph."""
        def bk_rec(P, X, C):
            if P == 0 and X == 0:
                if C:
                    yield C
                return
            
            PX = P | X
            pivot = -1
            max_neighbors = -1
            tmp = PX
            while tmp:
                u = (tmp & -tmp).bit_length() - 1
                tmp &= tmp - 1
                cnt = bin(comp_adj_mask[u] & P).count('1')
                if cnt > max_neighbors:
                    max_neighbors = cnt
                    pivot = u
            
            if pivot != -1:
                candidates = PX & ~(comp_adj_mask[pivot] | (1 << pivot))
            else:
                candidates = PX
            
            tmp = candidates
            while tmp:
                v_bit = tmp & -tmp
                v = v_bit.bit_length() - 1
                tmp ^= v_bit
                
                if P & v_bit:
                    new_P = P & ~(comp_adj_mask[v] | v_bit)
                    new_X = X & ~(comp_adj_mask[v] | v_bit)
                    new_C = C | v_bit
                    yield from bk_rec(new_P, new_X, new_C)
                    P &= ~v_bit
                    X |= v_bit
        
        yield from bk_rec(vertices_mask, excluded_mask, current_is)

    all_independent = []
    for c in find_maximal_independent_sets((1 << n) - 1, 0, 0):
        all_independent.append(c)

    # Combine candidate sets
    # Type 0: Clique, Type 1: Independent Set
    candidates = []
    for c in all_cliques:
        candidates.append((0, c))
    for c in all_independent:
        candidates.append((1, c))
        
    # Remove duplicates? Maximal sets are unique.
    # Sort candidates by size descending (larger sets cover more, likely better for set cover)
    candidates.sort(key=lambda x: -bin(x[1]).count('1'))
    
    m = len(candidates)
    
    # Precompute for each candidate, which vertices it covers (it's the mask itself)
    # And for each vertex, which candidates cover it
    vertex_to_candidates = [[] for _ in range(n)]
    for i, (t, mask) in enumerate(candidates):
        # mask is the set of vertices
        temp = mask
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            temp ^= v_bit
            vertex_to_candidates[v].append(i)

    best_solution = None
    best_count = n + 1
    best_masks = []

    # Branch and Bound for Set Cover
    # State: uncovered_mask, current_selected_indices, current_count
    # Lower bound: greedy estimate or simple bound
    # Since n=14, we can use a strong lower bound.
    # LB = ceil(popcount(uncovered) / max_candidate_size_in_uncovered) is weak.
    # Better: Solve LP relaxation? Too slow.
    # Simple bound: For each vertex in uncovered, it must be covered by at least one set.
    # If we pick a vertex v with fewest covering sets, we must pick at least one of them.
    
    def solve_set_cover(uncovered_mask, selected_indices, current_count):
        nonlocal best_solution, best_count, best_masks
        
        if time.time() > deadline:
            return
        
        if current_count >= best_count:
            return
        
        if uncovered_mask == 0:
            if current_count < best_count:
                best_count = current_count
                best_masks = list(selected_indices)
                best_solution = [candidates[i][1] for i in selected_indices]
            return
        
        # Lower bound
        # Find vertex with minimum number of available covering sets
        min_cover_options = n + 1
        temp = uncovered_mask
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            temp ^= v_bit
            
            # Count how many candidates in our current branch could cover v
            # We don't track "available" explicitly, but we can check if candidate covers v
            # and if it doesn't conflict with already selected?
            # Wait, Set Cover allows overlapping sets. There is no conflict.
            # So any candidate that covers v is a valid choice.
            # However, we should prioritize candidates that cover more uncovered vertices.
            
            # For LB, we just need to know if it's possible to cover v.
            # If no candidate covers v, impossible (but we have all maximal sets, so likely covered).
            
            # A stronger LB:
            # Let S be the set of uncovered vertices.
            # If we pick a set C, it covers C ∩ S.
            # Greedy LB: 
            #   rem = S
            #   lb = 0
            #   while rem != 0:
            #      find candidate C that covers max |C ∩ rem|
            #      if max == 0: break (infeasible)
            #      rem -= C
            #      lb += 1
            # This is expensive. Let's use a simpler one.
            
            # Simple LB: 1 if uncovered != 0
            # Better: ceil(|uncovered| / max_set_size)
            # Even better: For each v in uncovered, let d(v) be number of candidates covering v.
            # This doesn't directly give a count.
            
            # Let's use the greedy LB but optimized.
            # Since n=14, and we are in a search tree, we should be careful.
            
            pass # Placeholder for LB logic below

        # Greedy Lower Bound
        rem = uncovered_mask
        lb = 0
        while rem != 0:
            max_cov = 0
            best_cand_idx = -1
            for i in range(m):
                cov = candidates[i][1] & rem
                if cov:
                    cnt = bin(cov).count('1')
                    if cnt > max_cov:
                        max_cov = cnt
                        best_cand_idx = i
            if max_cov == 0:
                return # Infeasible
            rem &= ~candidates[best_cand_idx][1]
            lb += 1
            
        if current_count + lb >= best_count:
            return
            
        # Branching
        # Choose a vertex v in uncovered with fewest covering candidates (or most constrained)
        # Or choose a vertex and branch on all candidates covering it.
        
        # Pick vertex with minimum coverage options among uncovered?
        # Or just pick the first uncovered vertex and branch on candidates covering it.
        # To reduce symmetry, we can order candidates by how much they cover.
        
        v_bit = uncovered_mask & -uncovered_mask
        v = v_bit.bit_length() - 1
        
        # Get candidates covering v
        cands = vertex_to_candidates[v]
        
        # Sort candidates by coverage of current uncovered set (descending)
        # This helps find good solutions early
        cands.sort(key=lambda i: -bin(candidates[i][1] & uncovered_mask).count('1'))
        
        for i in cands:
            cand_mask = candidates[i][1]
            
            # If this candidate doesn't cover any new uncovered vertices, skip?
            # It must cover v, so it covers at least v.
            
            new_uncovered = uncovered_mask & ~cand_mask
            selected_indices.append(i)
            
            solve_set_cover(new_uncovered, selected_indices, current_count + 1)
            
            selected_indices.pop()
            
            if time.time() > deadline:
                return

    solve_set_cover((1 << n) - 1, [], 0)

    if best_masks:
        result = []
        for mask in best_masks:
            verts = [v for v in range(n) if (mask >> v) & 1]
            result.append(verts)
        return result
    else:
        # Fallback: each vertex its own part
        return [[v] for v in range(n)]

# EVOLVE-BLOCK-END
