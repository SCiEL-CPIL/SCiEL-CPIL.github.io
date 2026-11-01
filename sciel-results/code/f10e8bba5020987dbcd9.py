        node_count = 0
        max_nodes = 100000 # Limit search
        
        def backtrack(idx):
            nonlocal node_count
            if time.time() > deadline:
                return False
            node_count += 1
            if node_count > max_nodes:
                return False
            # ... rest of logic
