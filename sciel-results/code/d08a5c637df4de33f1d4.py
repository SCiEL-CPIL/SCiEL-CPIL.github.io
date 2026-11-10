"""
Heuristic matrix for ACO CVRP using a graph-partitioning (clustering) approach.
The idea: partition customers into capacity-feasible clusters, then create a
heuristic that strongly encourages edges within the same cluster and between
a cluster and the depot, while suppressing inter-cluster edges.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using a clustering-based approach.
    
    Strategy:
    1. Greedily partition customers into capacity-feasible clusters using
       distance-based grouping.
    2. Build a heuristic matrix that:
       - Strongly encourages edges within the same cluster
       - Encourages edges from depot to any customer (start of route)
       - Encourages edges from any customer back to depot (end of route)
       - Suppresses inter-cluster edges
    3. Scale by inverse distance within each allowed region.
    """
    deadline = time.time() + 320.0  # 320s safety margin before 330s kill
    n = distance_matrix.shape[0]
    
    # Validate and prepare
    if distance_matrix.shape != (n, n):
        return 1.0 / (distance_matrix + 1e-9)
    
    # Base: inverse distance
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Greedy clustering: assign customers to clusters based on proximity
    # and capacity feasibility
    customers = np.arange(1, n)
    unassigned = set(customers.tolist())
    clusters = []
    
    # Sort customers by distance from depot (closest first)
    depot_dists = distance_matrix[0, 1:]
    sorted_customers = customers[np.argsort(depot_dists)]
    
    for start_node in sorted_customers:
        if time.time() > deadline - 30:
            break
        if start_node not in unassigned:
            continue
        
        # Start a new cluster with start_node
        cluster = [start_node]
        load = demands[start_node]
        unassigned.remove(start_node)
        
        # Greedily add nearby customers that fit capacity
        # Compute distances from start_node to all unassigned
        while load < capacity:
            if time.time() > deadline - 30:
                break
            if not unassigned:
                break
            
            # Find nearest unassigned customer to any node in current cluster
            # For efficiency, use start_node as proxy
            best_node = None
            best_dist = float('inf')
            for j in unassigned:
                if load + demands[j] > capacity:
                    continue
                # Distance to nearest node in cluster (use start_node for speed)
                d = distance_matrix[start_node, j]
                if d < best_dist:
                    best_dist = d
                    best_node = j
            
            if best_node is None or best_dist > 0.5:  # threshold to stop growing
                break
            
            cluster.append(best_node)
            load += demands[best_node]
            unassigned.remove(best_node)
        
        clusters.append(cluster)
    
    # Assign any remaining unassigned customers to their nearest cluster
    for j in list(unassigned):
        if time.time() > deadline - 30:
            break
        # Find nearest cluster by distance to any member
        best_cluster_idx = 0
        best_dist = float('inf')
        for ci, cl in enumerate(clusters):
            min_d = min(distance_matrix[j, k] for k in cl)
            if min_d < best_dist:
                best_dist = min_d
                best_cluster_idx = ci
        clusters[best_cluster_idx].append(j)
        unassigned.remove(j)
    
    # Build cluster assignment array
    cluster_of = np.zeros(n, dtype=int)
    for ci, cl in enumerate(clusters):
        for node in cl:
            cluster_of[node] = ci
    
    # Build the heuristic matrix
    # Within same cluster: high attractiveness (base * boost)
    # Depot to/from any: moderate attractiveness
    # Between different clusters: suppressed
    
    INTRA_BOOST = 3.0    # Boost for same-cluster edges
    DEPOT_BOOST = 1.5    # Boost for depot edges
    INTER_PENALTY = 0.05 # Penalty for inter-cluster edges
    
    heuristic = np.ones((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristic[i, j] = 1e-9
                continue
            
            if i == 0 or j == 0:
                # Depot edge
                heuristic[i, j] = base[i, j] * DEPOT_BOOST
            else:
                if cluster_of[i] == cluster_of[j]:
                    # Same cluster
                    heuristic[i, j] = base[i, j] * INTRA_BOOST
                else:
                    # Different clusters
                    heuristic[i, j] = base[i, j] * INTER_PENALTY
    
    # Ensure all entries are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
