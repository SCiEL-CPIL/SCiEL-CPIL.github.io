import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    # Convert to int8 for faster matrix operations
    A = np.array(adjacency, dtype=np.int8)
    degrees = A.sum(axis=1)
    
    # Deadline: 120s limit, kill at 150s. 
    # Use 140s as soft deadline to allow safe return.
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_labels = None
    best_cut = -1
    
    def cut_size(labels):
        # Vectorized cut size calculation
        # diff[i,j] is 1 if labels[i] != labels[j]
        # Cut = sum(A * diff) / 2
        diff = (labels[:, None] != labels[None, :]).astype(np.int8)
        return int(np.sum(A * diff) // 2)

    def local_search(labels, deadline):
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        # Z0[u] = 1 if labels[u]==0 else 0
        # Z1[u] = 1 if labels[u]==1 else 0
        Z0 = (labels == 0).astype(np.int8)
        Z1 = (labels == 1).astype(np.int8)
        
        # V0 = A @ Z0 -> V0[v] is sum of A[v,u] for u with label 0
        # V1 = A @ Z1 -> V1[v] is sum of A[v,u] for u with label 1
        V0 = A @ Z0
        V1 = A @ Z1
        
        # S_all[v] is the count of neighbors of v with the SAME label as v
        # If labels[v]==0, S_all[v] = V0[v]
        # If labels[v]==1, S_all[v] = V1[v]
        S_all = np.where(labels == 0, V0, V1)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Gain for flipping v: 2 * S_all[v] - degrees[v]
            gains = 2 * S_all - degrees
            
            # Find the vertex with the maximum gain
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain > 0:
                v = max_gain_idx
                old_label = labels[v]
                new_label = 1 - old_label
                
                # Apply flip
                labels[v] = new_label
                current_cut += max_gain
                
                # Update Z0, Z1
                if old_label == 0:
                    Z0[v] = 0
                    Z1[v] = 1
                else:
                    Z0[v] = 1
                    Z1[v] = 0
                
                # Update V0, V1
                # V0 = A @ Z0. Z0 changed at v.
                # If old_label was 0, Z0[v] went 1->0. V0 -= A[:, v]
                # If old_label was 1, Z0[v] went 0->1. V0 += A[:, v]
                # Similarly for V1.
                
                if old_label == 0:
                    V0 -= A[:, v]
                    V1 += A[:, v]
                else:
                    V0 += A[:, v]
                    V1 -= A[:, v]
                
                # Update S_all
                # S_all depends on V0, V1 and labels.
                # Since V0, V1 changed for all indices, and labels changed at v,
                # we recompute S_all.
                S_all = np.where(labels == 0, V0, V1)
                
                improved = True
            else:
                break
                
        return labels, current_cut

    # 1. Initial alternating start
    init_labels = np.arange(n) % 2
    l, c = local_search(init_labels, deadline)
    if c > best_cut:
        best_cut = c
        best_labels = l
        
    # 2. Random restarts
    # We check time before starting a new restart to avoid exceeding deadline mid-search 
    # (though local_search also checks time, it's good practice).
    while time.time() < deadline - 2.0:
        rand_labels = np.random.randint(0, 2, n)
        l, c = local_search(rand_labels, deadline)
        if c > best_cut:
            best_cut = c
            best_labels = l
            
    if best_labels is None:
        return np.zeros(n, dtype=int)
        
    return best_labels
