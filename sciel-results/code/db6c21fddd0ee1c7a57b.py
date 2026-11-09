# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # labels is a 0/1 vector
    # cut = 0.5 * sum_{v,u} adj[v,u] * (labels[v] != labels[u])
    # Using matrix operations:
    # Let x = labels (0/1). Then (1-2x) maps 0->1, 1->-1.
    # cut = 0.25 * (1^T A 1 - (1-2x)^T A (1-2x)) ... no, that's for normalized.
    # Simpler: cut = sum_v (adj[v] @ (1 - labels)) where we only count once?
    # Actually, adj @ (1-labels) gives for each v, the number of neighbors NOT in same set.
    # Summing over all v counts each crossing edge twice (once from each endpoint).
    # So cut = 0.5 * sum(adj @ (1 - labels))
    # But wait, adj is symmetric.
    # Let's verify: if v in set 0, u in set 1. adj[v,u]=1.
    # (1-labels)[v]=1, (1-labels)[u]=0.
    # adj[v] @ (1-labels) includes adj[v,u]*(1-labels)[u] = 0.
    # adj[u] @ (1-labels) includes adj[u,v]*(1-labels)[v] = 1.
    # So sum is 1. We need to divide by 2?
    # Let's re-evaluate.
    # Let S0 = {v | labels[v]=0}, S1 = {v | labels[v]=1}.
    # Cut = sum_{v in S0, u in S1} adj[v,u].
    # Vector x = labels.
    # adj @ (1-x) for v in S0: sum_u adj[v,u]*(1-x[u]). Since x[v]=0, 1-x[v]=1.
    # This sums adj[v,u] for all u where x[u]=0? No, 1-x[u] is 1 if x[u]=0.
    # So adj @ (1-x) for v in S0 sums neighbors in S0. That's internal edges.
    # We want neighbors in S1. That is adj @ x.
    # So for v in S0, number of neighbors in S1 is adj[v] @ x.
    # For v in S1, number of neighbors in S0 is adj[v] @ (1-x).
    # Total cut = sum_{v in S0} (adj[v] @ x) + sum_{v in S1} (adj[v] @ (1-x)).
    # This is equivalent to:
    # cut = sum_v x[v] * (adj[v] @ (1-x)) + (1-x[v]) * (adj[v] @ x)
    # Note that adj[v] @ (1-x) = deg[v] - adj[v] @ x.
    # So term is x[v]*(deg[v] - adj[v]@x) + (1-x[v])*(adj[v]@x)
    # = x[v]*deg[v] - x[v]*adj[v]@x + adj[v]@x - x[v]*adj[v]@x
    # = x[v]*deg[v] + adj[v]@x - 2*x[v]*adj[v]@x
    # This is getting complicated.
    
    # Easier vectorized way:
    # cut = 0.5 * ( (1-2x)^T A (1-2x) )? No.
    # Let z = 1 - 2x. z is +1 for 0, -1 for 1.
    # z^T A z = sum_{v,u} z[v] z[u] A[v,u].
    # If v,u same side, z[v]z[u]=1. If diff, -1.
    # sum A[v,u] z[v]z[u] = sum_same A - sum_diff A.
    # Total edges E = sum_same A + sum_diff A.
    # sum_same A = E - cut.
    # So z^T A z = (E - cut) - cut = E - 2*cut.
    # So cut = (E - z^T A z) / 2.
    
    x = labels
    z = 1 - 2 * x
    e_total = np.sum(adj) / 2.0
    zAz = z @ adj @ z
    return (e_total - zAz) / 2.0

def _spectral_init(adj):
    """Initialize partition using the Fiedler vector."""
    n = len(adj)
    # Laplacian L = D - A
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    
    # Compute eigenvalues and eigenvectors
    # For n=100, this is fast.
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.random.randint(0, 2, n)
    
    # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue.
    # eigh returns sorted ascending.
    fiedler = eigvecs[:, 1]
    
    # Partition based on sign of Fiedler vector
    # Mean 0 is a good threshold
    labels = (fiedler > 0).astype(int)
    
    # Handle edge case where all are same side
    if np.all(labels == 0) or np.all(labels == 1):
        labels = np.random.randint(0, 2, n)
        
    return labels

def _greedy_flip(adj, labels):
    """Perform single-flip local search until convergence."""
    n = len(labels)
    improved = True
    while improved:
        improved = False
        for v in range(n):
            # Count neighbors in same and opposite sets
            # Same set: labels[u] == labels[v]
            # Opposite: labels[u] != labels[v]
            
            # Efficient update:
            # Let s = labels[v]
            # same_count = adj[v] @ labels  (if s=0, this is count of 1s? No.)
            # Let's use the property:
            # gain = (number of neighbors in opposite set) - (number of neighbors in same set)
            # If gain > 0, flipping increases cut.
            
            # Let x = labels.
            # Neighbors in same set: adj[v] @ x (if x[v]=0, this counts 1s? No, x is 0/1.
            # If x[v]=0, same set is x=0. Count is sum_{u: x[u]=0} adj[v,u].
            # This is not directly adj[v] @ x.
            
            # Let's stick to the reliable method for n=100.
            same = np.sum(adj[v, labels == labels[v]])
            other = np.sum(adj[v, labels != labels[v]])
            
            if other > same:
                labels[v] = 1 - labels[v]
                improved = True
    return labels

def _two_swap(adj, labels):
    """Perform 2-swap local search."""
    n = len(labels)
    improved = True
    while improved:
        improved = False
        # Try all pairs? O(n^2 * degree). For n=100, 5000 pairs.
        # We can be smarter, but let's start with random pairs or systematic.
        # Systematic is safer for determinism if needed, but random is faster to find improvement.
        # Let's do a few passes of random swaps.
        for _ in range(100): # Limit iterations per pass to avoid infinite loops if no improvement found quickly
             if not improved:
                 break
             v1, v2 = random.sample(range(n), 2)
             if labels[v1] == labels[v2]:
                 continue # Swapping same labels does nothing to partition structure? 
                 # Wait, if labels are same, swapping them doesn't change the partition.
                 # So we only care if labels[v1] != labels[v2].
            
            # Actually, a "2-swap" usually means moving two vertices from one side to the other?
            # Or swapping the labels of two vertices.
            # If we swap labels of v1 and v2, and they have different labels, it's equivalent to flipping both.
            # If they have same labels, it's a no-op.
            # So we only consider pairs with different labels.
            
            # Let's restructure.
            # We want to find v1 in S0, v2 in S1 such that flipping both increases cut.
            # Delta = gain(v1) + gain(v2) + interaction.
            # Interaction: if v1 and v2 are connected, the edge (v1,v2) was crossing (contributing 1).
            # After flip, v1 in S1, v2 in S0. Edge (v1,v2) is still crossing.
            # So the edge (v1,v2) contribution doesn't change?
            # Wait.
            # Before: v1 in 0, v2 in 1. Edge (1,2) crosses.
            # After: v1 in 1, v2 in 0. Edge (1,2) crosses.
            # So the direct edge between them doesn't change the cut count.
            # However, the edges to OTHER vertices change.
            
            # Let's just compute the cut size before and after the swap.
            # For n=100, this is fast enough.
            
            # We need to be careful with the loop.
            # Let's just do a limited number of random swaps.
        
        # Better approach: iterate through all pairs of different labels.
        # This is O(n^2).
        for v1 in range(n):
            for v2 in range(v1 + 1, n):
                if labels[v1] == labels[v2]:
                    continue
                
                # Calculate gain of flipping v1 and v2
                # Gain = Cut(new) - Cut(old)
                
                # We can compute this incrementally.
                # Let's just use _cut_size for simplicity and correctness first, 
                # but it's O(n^2) per check. Total O(n^4) which is 100^4 = 10^8. 
                # Might be too slow in Python if done many times.
                
                # Let's try to compute gain analytically.
                # Let S0, S1 be sets.
                # v1 in S0, v2 in S1.
                # Flip v1 to S1, v2 to S0.
                
                # Edges incident to v1:
                # Neighbors in S0 (excluding v2 if connected? v2 is in S1 so not in S0):
                #   Before: crossing. After: internal (v1 in S1, neighbor in S0 -> crossing? No, v1 moves to S1. Neighbor in S0. So crossing.)
                #   Wait.
                #   v1 in S0. Neighbor u in S0. Edge (v1,u) is internal.
                #   Flip v1 to S1. Edge (v1,u) becomes crossing. Gain +1.
                #   Neighbor u in S1. Edge (v1,u) is crossing.
                #   Flip v1 to S1. Edge (v1,u) becomes internal. Loss -1.
                #   Neighbor u = v2 (in S1). Edge (v1,v2) is crossing.
                #   Flip v1 to S1. v2 flips to S0. Edge (v1,v2) is still crossing. Change 0.
                
                # So for v1:
                # Gain = (neighbors in S0) - (neighbors in S1 \ {v2})
                # Note: neighbors in S1 includes v2 if connected.
                # Let n0(v1) = # neighbors in S0.
                # Let n1(v1) = # neighbors in S1.
                # If v1-v2 connected:
                #   n1(v1) includes v2.
                #   Neighbors in S1 \ {v2} = n1(v1) - 1.
                #   Gain1 = n0(v1) - (n1(v1) - 1) = n0(v1) - n1(v1) + 1.
                # If not connected:
                #   Gain1 = n0(v1) - n1(v1).
                
                # Similarly for v2 (moving S1 -> S0):
                #   Neighbors in S1 (excluding v1 if connected):
                #     Before: internal. After: crossing. Gain +1.
                #   Neighbors in S0:
                #     Before: crossing. After: internal. Loss -1.
                #   Neighbor v1 (in S0):
                #     Before: crossing. After: crossing. Change 0.
                
                # Gain2 = (neighbors in S1) - (neighbors in S0 \ {v1})
                # If connected:
                #   n0(v2) includes v1.
                #   Neighbors in S0 \ {v1} = n0(v2) - 1.
                #   Gain2 = n1(v2) - (n0(v2) - 1) = n1(v2) - n0(v2) + 1.
                # If not connected:
                #   Gain2 = n1(v2) - n0(v2).
                
                # Total Gain = Gain1 + Gain2.
                
                n0_v1 = np.sum(adj[v1, labels == 0])
                n1_v1 = np.sum(adj[v1, labels == 1])
                n0_v2 = np.sum(adj[v2, labels == 0])
                n1_v2 = np.sum(adj[v2, labels == 1])
                
                connected = adj[v1, v2]
                
                gain1 = n0_v1 - n1_v1 + (1 if connected else 0)
                gain2 = n1_v2 - n0_v2 + (1 if connected else 0)
                
                total_gain = gain1 + gain2
                
                if total_gain > 0:
                    labels[v1] = 1 - labels[v1]
                    labels[v2] = 1 - labels[v2]
                    improved = True
                    # Break inner loop? Or continue?
                    # If we continue, we might find another swap.
                    # But the state changed, so precomputed counts are invalid.
                    # We should break and restart the outer loops or just break and let the while loop restart.
                    break
            if improved:
                break
    return labels

def partition(adjacency):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    adj = np.array(adjacency, dtype=int)
    n = len(adj)
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts
    # We can try spectral init, random init, etc.
    
    # 1. Spectral Init
    labels = _spectral_init(adj)
    labels = _greedy_flip(adj, labels.copy())
    labels = _two_swap(adj, labels.copy())
    cut = _cut_size(adj, labels)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Check time
    if time.time() > deadline:
        return best_labels
    
    # 2. Random Restarts with Simulated Annealing or just Greedy+Swap
    # Let's do a few random restarts with greedy flip and 2-swap.
    
    num_restarts = 20 # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        labels = np.random.randint(0, 2, n)
        
        # Greedy Flip
        labels = _greedy_flip(adj, labels.copy())
        
        # 2-Swap
        labels = _two_swap(adj, labels.copy())
        
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
