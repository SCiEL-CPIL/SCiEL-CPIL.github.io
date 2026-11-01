import numpy as np
import itertools
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    # 1. Generate all norm-2 integer vectors (squared norm 4)
    vectors = []
    
    # Type A: one +/- 2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type B: four +/- 1
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, val in zip(combo, signs):
                v[pos] = val
            vectors.append(v)
            
    N = len(vectors)
    
    # 2. Build conflict graph
    vec_arr = np.array(vectors, dtype=np.int8)
    dots = vec_arr @ vec_arr.T
    
    # Conflict if dot > 2
    rows, cols = np.where(dots > 2)
    valid = rows < cols
    rows = rows[valid]
    cols = cols[valid]
    
    adj = [[] for _ in range(N)]
    for r, c in zip(rows, cols):
        adj[r].append(c)
        adj[c].append(r)
        
    degrees = np.array([len(a) for a in adj], dtype=np.int32)
    
    # 3. Build sparse Laplacian
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    # 4. Compute Fiedler vector and additional eigenvectors
    eigvecs = []
    try:
        eigvals, eigvecs_all = eigsh(L, k=6, which='SM')
        idx = np.argsort(eigvals)
        for i in idx[1:6]:
            eigvecs.append(eigvecs_all[:, i])
    except Exception:
        eigvecs = [np.random.randn(N) for _ in range(5)]
    
    fiedler = eigvecs[0]
    
    def greedy_is_from_order(order):
        """Greedy independent set processing vertices in given order."""
        s = set()
        in_s = [False] * N
        for v in order:
            conflict = False
            for u in adj[v]:
                if in_s[u]:
                    conflict = True
                    break
            if not conflict:
                s.add(v)
                in_s[v] = True
        return s
    
    def kempe_chain_flip(S, max_iter=500):
        """
        Aggressive Kempe-chain local search with multi-vertex component swaps.
        """
        S = set(S)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            T = [v for v in range(N) if not in_S[v]]
            if not T:
                break
            
            # Find connected components in T
            visited = set()
            components = []
            for start in T:
                if start in visited:
                    continue
                comp = []
                stack = [start]
                visited.add(start)
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for v in adj[u]:
                        if not in_S[v] and v not in visited:
                            visited.add(v)
                            stack.append(v)
                components.append(comp)
            
            # Try to add any vertex that has no conflicts with S
            for v in T:
                can_add = True
                for u in adj[v]:
                    if in_S[u]:
                        can_add = False
                        break
                if can_add:
                    in_S[v] = True
                    S.add(v)
                    improved = True
            
            if improved:
                continue
            
            # For each component, evaluate swap gain
            best_gain = 0
            best_I = None
            best_R = None
            
            for comp in components:
                if len(comp) > N // 2:
                    continue
                
                R = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                
                sorted_comp = sorted(comp, key=lambda x: degrees[x], reverse=True)
                I = set()
                for v in sorted_comp:
                    conflict = False
                    for u in adj[v]:
                        if u in I:
                            conflict = True
                            break
                    if not conflict:
                        I.add(v)
                
                gain = len(I) - len(R)
                if gain > best_gain:
                    best_gain = gain
                    best_I = I
                    best_R = R
            
            # Try single-vertex swaps if no component swap helps
            if best_gain <= 0:
                S_list = sorted(S, key=lambda x: degrees[x])
                T_list = sorted(T, key=lambda x: degrees[x], reverse=True)
                
                for v_r in S_list[:20]:
                    for v_a in T_list[:20]:
                        can_add = True
                        for u in adj[v_a]:
                            if in_S[u] and u != v_r:
                                can_add = False
                                break
                        if can_add:
                            in_S[v_r] = False
                            S.discard(v_r)
                            in_S[v_a] = True
                            S.add(v_a)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved and best_I is not None and best_gain > 0:
                for v in best_R:
                    in_S[v] = False
                    S.discard(v)
                for v in best_I:
                    in_S[v] = True
                    S.add(v)
                improved = True
        
        return S
    
    def spectral_bisect(evec):
        """Bisect based on eigenvector sign."""
        P1 = [i for i in range(N) if evec[i] > 0]
        P2 = [i for i in range(N) if evec[i] <= 0]
        return [P1, P2]
    
    def merge_sets(S1, S2):
        """Merge two independent sets, keeping the larger as base."""
        if len(S1) < len(S2):
            S1, S2 = S2, S1
        S = set(S1)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
        for v in S2:
            if in_S[v]:
                continue
            conflict = False
            for u in adj[v]:
                if in_S[u]:
                    conflict = True
                    break
            if not conflict:
                in_S[v] = True
                S.add(v)
        return S
    
    # 5. Generate candidate partitions and find best IS
    best_set = set()
    best_size = 0
    
    all_candidates = []
    
    # Strategy 1: Fiedler-based 2-way split
    parts = spectral_bisect(fiedler)
    all_candidates.append(parts)
    
    # Strategy 2: Multi-eigenvector quadrants
    if len(eigvecs) >= 2:
        X = np.column_stack([eigvecs[0], eigvecs[1]])
        X = X - X.mean(axis=0)
        med = np.median(X, axis=0)
        quads = [[], [], [], []]
        for i in range(N):
            if X[i,0] >= med[0] and X[i,1] >= med[1]:
                quads[0].append(i)
            elif X[i,0] < med[0] and X[i,1] >= med[1]:
                quads[1].append(i)
            elif X[i,0] >= med[0] and X[i,1] < med[1]:
                quads[2].append(i)
            else:
                quads[3].append(i)
        all_candidates.append([quads[0]+quads[1], quads[2]+quads[3]])
        all_candidates.append([quads[0]+quads[2], quads[1]+quads[3]])
        all_candidates.append([quads[0]+quads[3], quads[1]+quads[2]])
    
    # Strategy 3: Perturbed Fiedler
    for restart in range(5):
        noise = np.random.randn(N) * 0.03
        perturbed = fiedler + noise
        all_candidates.append(spectral_bisect(perturbed))
    
    # Strategy 4: Higher eigenvectors
    for i in range(1, min(len(eigvecs), 4)):
        all_candidates.append(spectral_bisect(eigvecs[i]))
    
    # Process all candidates
    for parts in all_candidates:
        if len(parts) != 2:
            continue
        
        cleaned = []
        for part in parts:
            if not part:
                cleaned.append(set())
            else:
                order = sorted(part, key=lambda x: degrees[x], reverse=True)
                s = greedy_is_from_order(order)
                cleaned.append(s)
        
        merged = merge_sets(cleaned[0], cleaned[1])
        merged = kempe_chain_flip(merged)
        
        if len(merged) > best_size:
            best_set = merged
            best_size = len(merged)
    
    # Additional: Direct greedy with spectral ordering
    order = list(np.argsort(fiedler))
    greedy = greedy_is_from_order(order)
    greedy = kempe_chain_flip(greedy)
    if len(greedy) > best_size:
        best_set = greedy
        best_size = len(greedy)
    
    # Final repair
    in_best = [False] * N
    for v in best_set:
        in_best[v] = True
    for v in range(N):
        if in_best[v]:
            continue
        can_add = True
        for u in adj[v]:
            if in_best[u]:
                can_add = False
                break
        if can_add:
            in_best[v] = True
            best_set.add(v)
    
    final_vectors = []
    for i in range(N):
        if i in best_set:
            final_vectors.append(vectors[i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
