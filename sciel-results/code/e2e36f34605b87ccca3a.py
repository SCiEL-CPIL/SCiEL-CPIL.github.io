from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

def check_feasibility(branches, n, adj, adj_set):
    """Check if vertex-disjoint paths exist for all pairs using max-flow."""
    branches_set = set(branches)
    k = len(branches)
    if k < 2:
        return True
    
    # Map vertices to flow network nodes
    # Non-branch vertices: 2 nodes each (in, out)
    # Branch vertices: 2 nodes each (in, out)
    node_id = {}
    next_id = 0
    for v in range(n):
        if v in branches_set:
            node_id[(v, 'in')] = next_id; next_id += 1
            node_id[(v, 'out')] = next_id; next_id += 1
        else:
            node_id[(v, 'in')] = next_id; next_id += 1
            node_id[(v, 'out')] = next_id; next_id += 1
    
    # Super source and sink
    S = next_id; next_id += 1
    T = next_id; next_id += 1
    total_nodes = next_id
    
    # Build flow network
    rows, cols, caps = [], [], []
    
    # Vertex capacity edges
    for v in range(n):
        if v in branches_set:
            # Branch: capacity k-1 (can be used for k-1 paths)
            rows.append(node_id[(v, 'in')])
            cols.append(node_id[(v, 'out')])
            caps.append(k - 1)
        else:
            # Non-branch: capacity 1
            rows.append(node_id[(v, 'in')])
            cols.append(node_id[(v, 'out')])
            caps.append(1)
    
    # Original edges
    for u, v in P['edges']:
        if u in branches_set and v in branches_set:
            # Branch to branch
            rows.append(node_id[(u, 'out')])
            cols.append(node_id[(v, 'in')])
            caps.append(1)
        elif u in branches_set:
            rows.append(node_id[(u, 'out')])
            cols.append(node_id[(v, 'in')])
            caps.append(1)
        elif v in branches_set:
            rows.append(node_id[(v, 'out')])
            cols.append(node_id[(u, 'in')])
            caps.append(1)
        else:
            rows.append(node_id[(u, 'out')])
            cols.append(node_id[(v, 'in')])
            caps.append(1)
    
    # Super source to branch in-nodes
    for b in branches:
        rows.append(S)
        cols.append(node_id[(b, 'in')])
        caps.append(k - 1)
    
    # Branch out-nodes to super sink
    for b in branches:
        rows.append(node_id[(b, 'out')])
        cols.append(T)
        caps.append(k - 1)
    
    if not rows:
        return True
    
    flow_matrix = csr_matrix((caps, (rows, cols)), shape=(total_nodes, total_nodes))
    flow, _, _, _ = maximum_flow(flow_matrix, S, T)
    
    return flow >= k * (k - 1)
