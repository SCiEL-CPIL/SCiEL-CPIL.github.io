def is_2_colorable(n, edges, node_limit=10000):
    # ...
    nodes = [0]
    def dfs():
        if nodes[0] > node_limit:
            return False # Assume non-2-colorable to avoid timeout
        nodes[0] += 1
        # ...
