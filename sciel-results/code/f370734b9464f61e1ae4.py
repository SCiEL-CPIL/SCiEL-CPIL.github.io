import time
import random
from itertools import combinations
import numpy as np

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560
N_WORDS = (N_TRIPLES + 63) // 64  # 9

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

def _has_presence(presence, idx):
    """Check if triple idx is in the presence bitmask."""
    return (presence[idx >> 6] >> (idx & 63)) & 1

def _set_presence(presence, idx, val):
    """Set or clear bit for triple idx in presence."""
    if val:
        presence[idx >> 6] |= (1 << (idx & 63))
    else:
        presence[idx >> 6] &= ~(1 << (idx & 63))

def _find_witnesses(a, b, c, presence, max_witnesses=10):
    """Find up to max_witnesses K(2,2,2)s that block triple (a,b,c)."""
    witnesses = []
    excluded = {a, b, c}
    
    for a_p in range(V):
        if a_p in excluded:
            continue
        if not _has_presence(presence, _tidx(a_p, b, c)):
            continue
        for b_p in range(V):
            if b_p in excluded or b_p == a_p:
                continue
            if not _has_presence(presence, _tidx(a, b_p, c)):
                continue
            if not _has_presence(presence, _tidx(a_p, b_p, c)):
                continue
            for c_p in range(V):
                if c_p in excluded or c_p == a_p or c_p == b_p:
                    continue
                if not _has_presence(presence, _tidx(a, b, c_p)):
                    continue
                if not _has_presence(presence, _tidx(a_p, b, c_p)):
                    continue
                if not _has_presence(presence, _tidx(a, b_p, c_p)):
                    continue
                if not _has_presence(presence, _tidx(a_p, b_p, c_p)):
                    continue
                
                # Found a witness
                witness = [
                    _tidx(a_p, b, c),
                    _tidx(a, b_p, c),
                    _tidx(a_p, b_p, c),
                    _tidx(a, b, c_p),
                    _tidx(a_p, b, c_p),
                    _tidx(a, b_p, c_p),
                    _tidx(a_p, b_p, c_p)
                ]
                witnesses.append(witness)
                if len(witnesses) >= max_witnesses:
                    return witnesses
    return witnesses

def _build_conflict_graph(deadline):
    """Build the K(2,2,2) conflict graph as adjacency matrix."""
    conflict = np.zeros((N_TRIPLES, N_TRIPLES), dtype=np.int8)
    
    for i in range(N_TRIPLES):
        if time.time() > deadline:
            break
        a, b, c = ALL_TRIPLES[i]
        for j in range(i+1, N_TRIPLES):
            d, e, f = ALL_TRIPLES[j]
            shared = len({a,b,c} & {d,e,f})
            if shared <= 1:
                conflict[i, j] = 1
                conflict[j, i] = 1
    
    return conflict

def _spectral_clustering_simple(conflict, deadline, n_clusters=8):
    """Simple spectral clustering using eigenvectors."""
    if time.time() > deadline:
        return list(range(N_TRIPLES))
    
    degree = np.sum(conflict, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-10))
    L = np.eye(N_TRIPLES) - D_inv_sqrt @ conflict @ D_inv_sqrt
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        return list(range(N_TRIPLES))
    
    U = eigvecs[:, :n_clusters]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1
    U = U / norms
    
    centers = U[random.sample(range(N_TRIPLES), n_clusters)]
    labels = np.zeros(N_TRIPLES, dtype=int)
    
    for _ in range(20):
        changed = False
        for i in range(N_TRIPLES):
            dists = np.sum((centers - U[i]) ** 2, axis=1)
            best_k = np.argmin(dists)
            if labels[i] != best_k:
                labels[i] = best_k
                changed = True
        
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                centers[k] = np.mean(U[mask], axis=0)
        
        if not changed:
            break
    
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return [clusters[k] for k in sorted(clusters.keys())]

def _greedy_in_cluster(cluster_indices, deadline):
    """Greedy set packing within a cluster."""
    presence = [0] * N_WORDS
    count = 0
    shuffled = cluster_indices[:]
    random.shuffle(shuffled)
    
    for idx in shuffled:
        if time.time() > deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        witnesses = _find_witnesses(a, b, c, presence, 1)
        if witnesses:
            continue
        presence[idx >> 6] |= (1 << (idx & 63))
        count += 1
    
    return count, presence

def _merge_presences(presences, deadline):
    """Merge multiple presence bitmaps, resolving conflicts."""
    final_presence = [0] * N_WORDS
    final_count = 0
    
    for presence in presences:
        if time.time() > deadline:
            break
        for i in range(N_TRIPLES):
            if _has_presence(presence, i):
                if not _has_presence(final_presence, i):
                    a, b, c = ALL_TRIPLES[i]
                    witnesses = _find_witnesses(a, b, c, final_presence, 1)
                    if not witnesses:
                        final_presence[i >> 6] |= (1 << (i & 63))
                        final_count += 1
    
    return final_count, final_presence

def _greedy(order, deadline):
    """Greedy: add triples in order, skipping conflicts."""
    presence = [0] * N_WORDS
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        witnesses = _find_witnesses(a, b, c, presence, 1)
        if witnesses:
            continue
        presence[idx >> 6] |= (1 << (idx & 63))
        count += 1
    return count, presence

def _local_search(presence, count, deadline):
    """Local search with Add and Swap moves."""
    present_set = set()
    for i in range(N_TRIPLES):
        if (presence[i >> 6] >> (i & 63)) & 1:
            present_set.add(i)
    
    absent_list = [i for i in range(N_TRIPLES) if i not in present_set]
    
    improved = True
    iter_count = 0
    max_iters = 50
    while improved and iter_count < max_iters and time.time() < deadline:
        improved = False
        iter_count += 1
        
        # 1. Try to add any absent triple
        random.shuffle(absent_list)
        added = False
        for idx in absent_list:
            if time.time() >= deadline:
                return count, presence
            a, b, c = ALL_TRIPLES[idx]
            witnesses = _find_witnesses(a, b, c, presence, 1)
            if not witnesses:
                presence[idx >> 6] |= (1 << (idx & 63))
                present_set.add(idx)
                absent_list.remove(idx)
                count += 1
                improved = True
                added = True
                break
        
        if added:
            continue
        
        # 2. If no add possible, try swaps (remove 1, add 1)
        if not absent_list:
            break
            
        random.shuffle(absent_list)
        swapped = False
        for new_idx in absent_list:
            if time.time() >= deadline:
                return count, presence
            a, b, c = ALL_TRIPLES[new_idx]
            
            witnesses = _find_witnesses(a, b, c, presence, 10)
            if not witnesses:
                presence[new_idx >> 6] |= (1 << (new_idx & 63))
                present_set.add(new_idx)
                absent_list.remove(new_idx)
                count += 1
                improved = True
                swapped = True
                break

            candidates = set()
            for w in witnesses:
                for t in w:
                    candidates.add(t)
            
            for t_rem in candidates:
                if t_rem not in present_set:
                    continue
                
                presence[t_rem >> 6] &= ~(1 << (t_rem & 63))
                present_set.remove(t_rem)
                
                remaining_witnesses = _find_witnesses(a, b, c, presence, 1)
                
                if not remaining_witnesses:
                    presence[new_idx >> 6] |= (1 << (new_idx & 63))
                    present_set.add(new_idx)
                    absent_list.remove(new_idx)
                    swapped = True
                    improved = True
                    break
                else:
                    presence[t_rem >> 6] |= (1 << (t_rem & 63))
                    present_set.add(t_rem)
                    
            if swapped:
                break
        
        if not swapped:
            break

    return count, presence

def _pairwise_swap(presence, count, deadline):
    """Try swapping two present triples for two absent ones."""
    present_list = [i for i in range(N_TRIPLES) if _has_presence(presence, i)]
    absent_list = [i for i in range(N_TRIPLES) if not _has_presence(presence, i)]
    
    if not present_list or not absent_list:
        return count, presence
    
    random.shuffle(present_list)
    random.shuffle(absent_list)
    
    # Try a few random pairwise swaps
    for _ in range(50):
        if time.time() >= deadline:
            return count, presence
        
        # Pick two random present triples to remove
        i1, i2 = random.sample(present_list, 2)
        
        # Pick two random absent triples to add
        j1, j2 = random.sample(absent_list, 2)
        
        # Temporarily remove i1, i2
        presence[i1 >> 6] &= ~(1 << (i1 & 63))
        presence[i2 >> 6] &= ~(1 << (i2 & 63))
        
        # Check if j1, j2 can be added
        a1, b1, c1 = ALL_TRIPLES[j1]
        a2, b2, c2 = ALL_TRIPLES[j2]
        
        w1 = _find_witnesses(a1, b1, c1, presence, 1)
        if not w1:
            presence[j1 >> 6] |= (1 << (j1 & 63))
            w2 = _find_witnesses(a2, b2, c2, presence, 1)
            if not w2:
                presence[j2 >> 6] |= (1 << (j2 & 63))
                # Swap successful, count unchanged
                # Update lists
                present_list.remove(i1)
                present_list.remove(i2)
                present_list.append(j1)
                present_list.append(j2)
                absent_list.remove(j1)
                absent_list.remove(j2)
                absent_list.append(i1)
                absent_list.append(i2)
                return count, presence
            else:
                # Restore j1
                presence[j1 >> 6] &= ~(1 << (j1 & 63))
        
        # Restore i1, i2
        presence[i1 >> 6] |= (1 << (i1 & 63))
        presence[i2 >> 6] |= (1 << (i2 & 63))
    
    return count, presence

def construct():
    start = time.time()
    deadline = start + 115
    
    best_count = 0
    best_presence = [0] * N_WORDS
    
    # Phase 1: Spectral clustering
    conflict = _build_conflict_graph(deadline)
    clusters = _spectral_clustering_simple(conflict, deadline, n_clusters=8)
    
    cluster_presences = []
    for cluster in clusters:
        if time.time() > deadline - 20:
            break
        count, presence = _greedy_in_cluster(cluster, deadline)
        cluster_presences.append(presence)
    
    if cluster_presences:
        merged_count, merged_presence = _merge_presences(cluster_presences, deadline)
        if merged_count > best_count:
            best_count = merged_count
            best_presence = merged_presence[:]
    
    # Phase 2: Standard greedy + local search
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        count, presence = _greedy(order, deadline)
        count, presence = _local_search(presence, count, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            
        if time.time() > deadline - 15:
            break
    
    result = []
    for i in range(N_TRIPLES):
        if (best_presence[i >> 6] >> (i & 63)) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

random.seed(42)
