import time
import random
from itertools import combinations


def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def largest_sidon_subset_size(S):
    S = sorted(S)
    n = len(S)
    # Try from largest k down
    for k in range(n, 0, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 1


def _build_collision_graph(S):
    """Build a graph where edge (i,j) exists if S[i]+S[j] collides with another pair sum.
    For Sidon: all pair sums (including diagonals 2*x) must be unique.
    An edge means removing either endpoint would be needed to break a collision."""
    n = len(S)
    # Map each sum to the list of (i,j) pairs that produce it
    sum_to_pairs = {}
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s not in sum_to_pairs:
                sum_to_pairs[s] = []
            sum_to_pairs[s].append((i, j))
    
    # Build adjacency: edge between any two indices that share a colliding sum
    adj = [[False] * n for _ in range(n)]
    for s, pairs in sum_to_pairs.items():
        if len(pairs) > 1:
            # All indices involved in this sum form a clique
            indices = set()
            for (i, j) in pairs:
                indices.add(i)
                indices.add(j)
            idx_list = list(indices)
            for a in range(len(idx_list)):
                for b in range(a + 1, len(idx_list)):
                    adj[idx_list[a]][idx_list[b]] = True
                    adj[idx_list[b]][idx_list[a]] = True
    return adj


def _min_vertex_cover_size(adj, n):
    """Compute minimum vertex cover size for a small graph using brute force (n<=12)."""
    best = n
    # Try all subsets as vertex covers
    for mask in range(1 << n):
        # Check if this is a vertex cover: every edge has at least one endpoint in the set
        is_cover = True
        for i in range(n):
            if not (mask >> i) & 1:
                continue
            for j in range(i + 1, n):
                if adj[i][j]:
                    if not ((mask >> j) & 1):
                        is_cover = False
                        break
            if not is_cover:
                break
        if is_cover:
            size = bin(mask).count('1')
            if size < best:
                best = size
    return best


def _graph_objective(S):
    """Objective: maximize minimum vertex cover size (which minimizes largest Sidon subset).
    largest_sidon = n - min_vertex_cover for the collision graph."""
    adj = _build_collision_graph(S)
    n = len(S)
    mvc = _min_vertex_cover_size(adj, n)
    return mvc  # Higher is better (means smaller largest Sidon subset)


def _generate_cluster_candidates():
    """Generate structured candidate sets with high additive collision density."""
    candidates = []
    # Two tight clusters of 6
    for base1 in range(0, 60, 4):
        for base2 in range(base1 + 12, 129, 4):
            c1 = [base1 + i for i in range(6)]
            c2 = [base2 + i for i in range(6)]
            if max(c2) <= 128 and len(set(c1 + c2)) == 12:
                candidates.append(c1 + c2)
    # Three clusters of 4
    for b1 in range(0, 40, 4):
        for b2 in range(b1 + 8, 70, 4):
            for b3 in range(b2 + 8, 129, 4):
                c1 = [b1 + i for i in range(4)]
                c2 = [b2 + i for i in range(4)]
                c3 = [b3 + i for i in range(4)]
                if max(c3) <= 128 and len(set(c1 + c2 + c3)) == 12:
                    candidates.append(c1 + c2 + c3)
    # Single tight cluster of 12
    for base in range(0, 117, 3):
        s = [base + i for i in range(12)]
        if max(s) <= 128:
            candidates.append(s)
    # Mixed: 8 tight + 4 spread
    for base in range(0, 80, 5):
        tight = [base + i for i in range(8)]
        spread = [base + 10 + i * 3 for i in range(4)]
        s = tight + spread
        if max(s) <= 128 and len(set(s)) == 12:
            candidates.append(s)
    return candidates


def _graph_local_search(S, deadline):
    """Improve S by maximizing the minimum vertex cover of the collision graph."""
    S = sorted(S)
    best_obj = _graph_objective(S)
    best_set = S[:]
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        improved = False
        
        for i in range(12):
            if time.time() > deadline:
                break
            old = best_set[i]
            best_new = None
            best_new_obj = best_obj
            
            # Try replacement values
            candidates = random.sample(range(0, 129), min(80, 129 - 11))
            for c in candidates:
                if c == old or c in best_set:
                    continue
                trial = best_set[:]
                trial[i] = c
                obj = _graph_objective(trial)
                if obj > best_new_obj:
                    best_new_obj = obj
                    best_new = c
            
            if best_new is not None:
                new_set = best_set[:]
                new_set[i] = best_new
                best_obj = best_new_obj
                best_set = new_set
                improved = True
                break  # Restart with new best
        
        if not improved:
            # Random jump to escape local optimum
            if random.random() < 0.2:
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val != best_set[i] and new_val not in best_set:
                    trial = best_set[:]
                    trial[i] = new_val
                    obj = _graph_objective(trial)
                    if obj > best_obj:
                        best_obj = obj
                        best_set = trial
    
    return best_set, best_obj


def construct():
    start_time = time.time()
    deadline = start_time + 55

    best_set = None
    best_obj = -1
    best_val = 99

    random.seed(12345)
    
    # Phase 1: Evaluate structured cluster candidates using graph objective
    cluster_candidates = _generate_cluster_candidates()
    random.shuffle(cluster_candidates)
    for cand in cluster_candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12:
            continue
        obj = _graph_objective(cand)
        if obj > best_obj:
            best_obj = obj
            best_set = cand[:]
            best_val = largest_sidon_subset_size(best_set)

    # Phase 2: Greedy construction maximizing graph objective
    for restart in range(80):
        if time.time() > deadline:
            break

        current = set()
        current.add(random.randint(0, 128))

        for step in range(11):
            if time.time() > deadline:
                break
            best_candidate = None
            best_candidate_obj = -1
            candidates = random.sample(range(0, 129), min(50, 129 - len(current)))
            for c in candidates:
                if c in current:
                    continue
                trial = list(current) + [c]
                obj = _graph_objective(trial)
                if obj > best_candidate_obj:
                    best_candidate_obj = obj
                    best_candidate = c
            if best_candidate is not None:
                current.add(best_candidate)

        if len(current) == 12:
            S = sorted(current)
            obj = _graph_objective(S)
            if obj > best_obj:
                best_obj = obj
                best_set = S[:]
                best_val = largest_sidon_subset_size(best_set)

    # Phase 3: Graph-based local search from best found
    if best_set is not None and time.time() < deadline - 5:
        improved_set, improved_obj = _graph_local_search(best_set, deadline)
        if improved_obj > best_obj:
            best_obj = improved_obj
            best_set = improved_set
            best_val = largest_sidon_subset_size(best_set)

    # Fallback
    if best_set is None:
        best_set = list(range(12))
        best_val = largest_sidon_subset_size(best_set)

    return best_set
