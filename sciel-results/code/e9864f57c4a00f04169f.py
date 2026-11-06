"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if a subset is Sidon (all pair sums including diagonals are unique)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _largest_sidon_size(arr):
    """Find the size of the largest Sidon subset of arr (12 elements)."""
    n = len(arr)
    # Try from largest possible down
    for k in range(n, 1, -1):
        for combo in combinations(arr, k):
            if _is_sidon(combo):
                return k
    return 1  # Any single element is Sidon

def _build_collision_graph():
    """
    Build a weighted collision graph on [0,128].
    Edge weight between a,b = number of distinct (c,d) pairs in [0,128] 
    such that a+b = c+d (including c=a,d=b etc.) minus 1 (exclude self).
    Higher weight means more additive conflicts.
    """
    N = 129  # 0 to 128
    # Precompute sum counts: count[s] = number of pairs (i,j) with i<=j, i+j=s
    sum_counts = [0] * (2*N + 1)
    for i in range(N):
        for j in range(i, N):
            sum_counts[i + j] += 1
    
    # For each pair (a,b), collision weight = sum_counts[a+b] - 1 (exclude the pair itself)
    # But we want to measure how many OTHER pairs share the same sum
    # weight[a][b] = sum_counts[a+b] - 1
    weights = [[0]*N for _ in range(N)]
    for a in range(N):
        for b in range(a, N):
            w = sum_counts[a + b] - 1
            weights[a][b] = w
            weights[b][a] = w
    
    return weights, sum_counts

def _spectral_cluster_select(weights, N, k=12, num_clusters=20):
    """
    Use spectral methods to find dense clusters in the collision graph.
    Returns a list of candidate sets (each of size k) from dense regions.
    """
    # Compute degree vector
    degrees = [sum(weights[i]) for i in range(N)]
    max_deg = max(degrees) if max(degrees) > 0 else 1
    
    # Normalize: use sqrt of degree for symmetric normalization
    sqrt_deg = [math.sqrt(d) if d > 0 else 1.0 for d in degrees]
    inv_sqrt = [1.0 / sd for sd in sqrt_deg]
    
    # Build normalized adjacency (sparse-like, but N=129 is small)
    # A_norm[i][j] = A[i][j] / sqrt(deg[i]*deg[j])
    A_norm = [[0.0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            if weights[i][j] > 0 and sqrt_deg[i] > 0 and sqrt_deg[j] > 0:
                val = weights[i][j] * inv_sqrt[i] * inv_sqrt[j]
                A_norm[i][j] = val
                A_norm[j][i] = val
    
    # Power iteration for top eigenvectors of A_norm
    # We want the smallest eigenvectors (Fiedler vectors) for clustering
    # Actually, for finding dense subgraphs, we want large eigenvalues of A
    # Let's use power iteration on A (unnormalized) to find high-degree regions
    
    # Simpler approach: just use the degree vector as a proxy for "collision density"
    # Nodes with high degree are in dense collision regions
    
    # Rank nodes by degree
    ranked = sorted(range(N), key=lambda i: degrees[i], reverse=True)
    
    # Take top nodes and try to form sets from them
    candidates = []
    
    # Strategy 1: Take contiguous blocks from the ranked list
    for start in range(0, len(ranked) - k + 1, 5):
        block = ranked[start:start+k]
        if len(block) == k:
            candidates.append(sorted(block))
    
    # Strategy 2: Random samples from top 50% of nodes
    top_half = ranked[:N//2]
    for _ in range(100):
        if len(top_half) >= k:
            sample = random.sample(top_half, k)
            candidates.append(sorted(sample))
    
    # Strategy 3: Mix top and mid nodes
    top_third = ranked[:N//3]
    mid_third = ranked[N//3:2*N//3]
    for _ in range(100):
        if len(top_third) >= k//2 and len(mid_third) >= k//2:
            s1 = random.sample(top_third, k//2)
            s2 = random.sample(mid_third, k - k//2)
            sample = s1 + s2
            candidates.append(sorted(sample))
    
    return candidates

def _local_search_improve(initial_set, deadline):
    """Local search to improve a candidate set."""
    current = initial_set[:]
    current_score = _largest_sidon_size(current)
    best_set = current[:]
    best_score = current_score
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 3 and no_improve < max_no_improve:
        improved = False
        # Try all single-element replacements
        for i in range(12):
            if time.time() >= deadline - 3:
                break
            old_val = current[i]
            for new_val in range(0, 129):
                if new_val == old_val or new_val in current:
                    continue
                current[i] = new_val
                new_score = _largest_sidon_size(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_set = current[:]
                        no_improve = 0
                else:
                    current[i] = old_val
                if time.time() >= deadline - 3:
                    break
            if time.time() >= deadline - 3:
                break
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return best_set, best_score

def construct():
    """Find 12 distinct integers in [0,128] minimizing the largest Sidon subset size."""
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    random.seed(42)
    
    # Build collision graph
    weights, sum_counts = _build_collision_graph()
    
    # Get candidate sets from spectral clustering
    candidates = _spectral_cluster_select(weights, 129, k=12)
    
    # Also add some random sets in small ranges as fallback
    for _ in range(50):
        max_range = random.choice([20, 25, 30, 35, 40])
        s = random.sample(range(max_range + 1), 12)
        candidates.append(sorted(s))
    
    # Evaluate candidates and keep the best
    best_set = None
    best_score = 12  # worst case
    
    # Sort candidates by a quick heuristic: prefer sets with more sum collisions
    def collision_heuristic(s):
        """Higher is better: more internal sum collisions means smaller Sidon subsets."""
        sums = []
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.append(s[i] + s[j])
        # Count duplicate sums
        from collections import Counter
        counts = Counter(sums)
        dup_count = sum(v - 1 for v in counts.values() if v > 1)
        return dup_count
    
    candidates.sort(key=collision_heuristic, reverse=True)
    
    # Evaluate top candidates
    for candidate in candidates[:200]:
        if time.time() > deadline - 15:
            break
        if len(set(candidate)) != 12:
            continue
        if not all(0 <= x <= 128 for x in candidate):
            continue
        
        score = _largest_sidon_size(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate[:]
        if best_score <= 4:
            break
    
    # If no good candidate found, use a default
    if best_set is None:
        best_set = list(range(12))
        best_score = _largest_sidon_size(best_set)
    
    # Local search improvement
    if time.time() < deadline - 10:
        improved_set, improved_score = _local_search_improve(best_set, deadline)
        if improved_score < best_score:
            best_set = improved_set
            best_score = improved_score
    
    # Additional local search rounds from different starting points
    if time.time() < deadline - 5:
        for start_idx in range(min(10, len(candidates))):
            if time.time() > deadline - 5:
                break
            candidate = candidates[start_idx]
            if len(set(candidate)) != 12:
                continue
            improved_set, improved_score = _local_search_improve(candidate, deadline)
            if improved_score < best_score:
                best_score = improved_score
                best_set = improved_set[:]
            if best_score <= 4:
                break
    
    # Final polish
    if time.time() < deadline - 2:
        for _ in range(5000):
            if time.time() >= deadline - 2:
                break
            candidate = best_set[:]
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in candidate:
                continue
            candidate[i] = new_val
            score = _largest_sidon_size(candidate)
            if score < best_score:
                best_score = score
                best_set = candidate[:]
    
    # Verify best_set is valid
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return best_set
# EVOLVE-BLOCK-END
