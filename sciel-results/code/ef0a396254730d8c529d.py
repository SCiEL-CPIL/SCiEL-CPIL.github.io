"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Global cache for pre-computed tour orders
# Key: tuple of first row of distance matrix
# Value: list of node indices in the order of the space-filling curve tour
_sfc_cache = {}

def _compute_mds_coords(dist_matrix):
    """
    Recovers 2D coordinates from a Euclidean distance matrix using Classical MDS.
    """
    n = dist_matrix.shape[0]
    # Double centering
    # B = -0.5 * J * D^2 * J
    D_sq = dist_matrix ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ D_sq @ J)
    
    # Eigen decomposition
    # We only need the top 2 eigenvalues and vectors
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2
    # Ensure eigenvalues are positive (numerical noise can make them slightly negative)
    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1]
    l1 = max(eigenvalues[0], 0)
    l2 = max(eigenvalues[1], 0)
    
    coords = np.column_stack((v1 * np.sqrt(l1), v2 * np.sqrt(l2)))
    return coords

def _interleave_bits(x, y, bits=15):
    """
    Interleaves bits of x and y to create a Morton code.
    x, y are integers.
    """
    # Spread bits of x
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    
    # Spread bits of y
    y = (y | (y << 8)) & 0x00FF00FF
    y = (y | (y << 4)) & 0x0F0F0F0F
    y = (y | (y << 2)) & 0x33333333
    y = (y | (y << 1)) & 0x55555555
    
    return x | (y << 1)

def _compute_sfc_order(dist_matrix):
    """
    Computes a tour order based on Morton Space-Filling Curve.
    Falls back to Nearest Neighbor if MDS fails.
    """
    try:
        coords = _compute_mds_coords(dist_matrix)
        
        # Check for NaNs or infinite values
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            raise ValueError("MDS produced invalid coordinates")
        
        # Normalize coordinates to [0, 2^15 - 1]
        min_vals = np.min(coords, axis=0)
        max_vals = np.max(coords, axis=0)
        ranges = max_vals - min_vals
        # Avoid division by zero
        ranges[ranges == 0] = 1.0
        
        normalized = (coords - min_vals) / ranges
        # Scale to 15 bits (32768)
        scaled = normalized * (2**15 - 1)
        int_coords = np.floor(scaled).astype(np.int32)
        
        # Calculate Morton codes
        morton_codes = []
        for i in range(len(int_coords)):
            x, y = int_coords[i]
            code = _interleave_bits(x, y)
            morton_codes.append((code, i))
        
        # Sort by Morton code
        morton_codes.sort(key=lambda x: x[0])
        
        # Extract order
        order = [idx for _, idx in morton_codes]
        return order
        
    except Exception:
        # Fallback: Simple Nearest Neighbor construction
        # This is a safe fallback if MDS fails
        n = dist_matrix.shape[0]
        visited = [False] * n
        order = []
        current = 0
        visited[0] = True
        order.append(0)
        
        for _ in range(n - 1):
            # Find nearest unvisited
            min_dist = np.inf
            next_node = -1
            for i in range(n):
                if not visited[i]:
                    d = dist_matrix[current, i]
                    if d < min_dist:
                        min_dist = d
                        next_node = i
            
            if next_node != -1:
                visited[next_node] = True
                order.append(next_node)
                current = next_node
        
        return order

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Space-Filling Curve (Morton Order) heuristic.
    The tour is pre-computed based on the global structure of the distance matrix
    (approximated via MDS coordinates), providing a non-greedy, global ordering.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
        
    # Fingerprint for caching
    # Using the first row as a key is efficient and unique enough for these instances
    key = tuple(distance_matrix[0].astype(np.float64))
    
    if key not in _sfc_cache:
        _sfc_cache[key] = _compute_sfc_order(distance_matrix)
    
    order = _sfc_cache[key]
    
    # Find the position of current_node in the pre-computed order
    # We want the next node in this order that is still unvisited.
    
    try:
        # np.where is fast for small arrays
        idx_current = np.where(np.array(order) == current_node)[0][0]
    except IndexError:
        # If current_node is not in the order (should not happen), fallback to nearest
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    n = len(order)
    unvisited_set = set(unvisited_nodes)
    
    # Scan forward in the circular order
    for k in range(1, n + 1):
        next_idx = (idx_current + k) % n
        candidate = order[next_idx]
        if candidate in unvisited_set:
            return int(candidate)
            
    # Fallback if something went wrong (should not happen)
    return int(unvisited_nodes[0])
# EVOLVE-BLOCK-END
