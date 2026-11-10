<<<<<<< SEARCH
    def local_search(adj_mask, max_iters=400):
        repair(adj_mask)
        adj_mask = greedy_add(adj_mask)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj_mask[i] & (1 << j))]
            if not missing:
                break
            
            random.shuffle(missing)
            improved = False
            for u, v in missing:
                if time.time() > deadline:
                    break
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    improved = True
                    break
            
            if improved:
                continue
            
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_mask[i] & (1 << j)]
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            # Try 1-for-1 swaps
            for i in range(min(40, len(all_edges), len(missing))):
                if time.time() > deadline:
                    break
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                adj_mask[r_u] &= ~(1 << r_v)
                adj_mask[r_v] &= ~(1 << r_u)
                adj_mask[a_u] |= (1 << a_v)
                adj_mask[a_v] |= (1 << a_u)
                
                if is_k33_free(adj_mask):
                    adj_mask = greedy_add(adj_mask)
                    improved = True
                    break
                else:
                    adj_mask[r_u] |= (1 << r_v)
                    adj_mask[r_v] |= (1 << r_u)
                    adj_mask[a_u] &= ~(1 << a_v)
                    adj_mask[a_v] &= ~(1 << a_u)
            
            if improved:
                continue
            
            # Try 2-for-1 swaps: remove 2 edges, add 1
            for i in range(min(25, len(all_edges) - 1)):
                if time.time() > deadline:
                    break
                r1_u, r1_v = all_edges[i]
                r2_u, r2_v = all_edges[i + 1]
                
                # Skip if edges share a vertex
                if r1_u == r2_u or r1_u == r2_v or r1_v == r2_u or r1_v == r2_v:
                    continue
                    
                a_u, a_v = missing[i % len(missing)]
                
                # Remove both edges
                adj_mask[r1_u] &= ~(1 << r1_v)
                adj_mask[r1_v] &= ~(1 << r1_u)
                adj_mask[r2_u] &= ~(1 << r2_v)
                adj_mask[r2_v] &= ~(1 << r2_u)
                # Add new edge
                adj_mask[a_u] |= (1 << a_v)
                adj_mask[a_v] |= (1 << a_u)
                
                if is_k33_free(adj_mask):
                    adj_mask = greedy_add(adj_mask)
                    improved = True
                    break
                else:
                    # Revert
                    adj_mask[r1_u] |= (1 << r1_v)
                    adj_mask[r1_v] |= (1 << r1_u)
                    adj_mask[r2_u] |= (1 << r2_v)
                    adj_mask[r2_v] |= (1 << r2_u)
                    adj_mask[a_u] &= ~(1 << a_v)
                    adj_mask[a_v] &= ~(1 << a_u)
            
            if improved:
                continue
            
            # Try removing a random edge to escape local optimum
            if all_edges:
                r_u, r_v = all_edges[0]
                adj_mask[r_u] &= ~(1 << r_v)
                adj_mask[r_v] &= ~(1 << r_u)
                adj_mask = greedy_add(adj_mask)
        
        return adj_mask
=======
    def local_search(adj_mask, max_iters=400):
        repair(adj_mask)
        adj_mask = greedy_add(adj_mask)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj_mask[i] & (1 << j))]
            if not missing:
                break
            
            random.shuffle(missing)
            improved = False
            for u, v in missing:
                if time.time() > deadline:
                    break
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    improved = True
                    break
            
            if improved:
                continue
            
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_mask[i] & (1 << j)]
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            # Try 1-for-1 swaps - more aggressive with more attempts
            for i in range(min(60, len(all_edges), len(missing))):
                if time.time() > deadline:
                    break
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                adj_mask[r_u] &= ~(1 << r_v)
                adj_mask[r_v] &= ~(1 << r_u)
                adj_mask[a_u] |= (1 << a_v)
                adj_mask[a_v] |= (1 << a_u)
                
                if is_k33_free(adj_mask):
                    adj_mask = greedy_add(adj_mask)
                    improved = True
                    break
                else:
                    adj_mask[r_u] |= (1 << r_v)
                    adj_mask[r_v] |= (1 << r_u)
                    adj_mask[a_u] &= ~(1 << a_v)
                    adj_mask[a_v] &= ~(1 << a_u)
            
            if improved:
                continue
            
            # Try 2-for-1 swaps: remove 2 edges, add 1
            for i in range(min(40, len(all_edges) - 1)):
                if time.time() > deadline:
                    break
                r1_u, r1_v = all_edges[i]
                r2_u, r2_v = all_edges[i + 1]
                
                # Skip if edges share a vertex
                if r1_u == r2_u or r1_u == r2_v or r1_v == r2_u or r1_v == r2_v:
                    continue
                    
                a_u, a_v = missing[i % len(missing)]
                
                # Remove both edges
                adj_mask[r1_u] &= ~(1 << r1_v)
                adj_mask[r1_v] &= ~(1 << r1_u)
                adj_mask[r2_u] &= ~(1 << r2_v)
                adj_mask[r2_v] &= ~(1 << r2_u)
                # Add new edge
                adj_mask[a_u] |= (1 << a_v)
                adj_mask[a_v] |= (1 << a_u)
                
                if is_k33_free(adj_mask):
                    adj_mask = greedy_add(adj_mask)
                    improved = True
                    break
                else:
                    # Revert
                    adj_mask[r1_u] |= (1 << r1_v)
                    adj_mask[r1_v] |= (1 << r1_u)
                    adj_mask[r2_u] |= (1 << r2_v)
                    adj_mask[r2_v] |= (1 << r2_u)
                    adj_mask[a_u] &= ~(1 << a_v)
                    adj_mask[a_v] &= ~(1 << a_u)
            
            if improved:
                continue
            
            # Try 1-for-2 swaps: remove 1 edge, add 2 edges
            for i in range(min(30, len(all_edges))):
                if time.time() > deadline:
                    break
                r_u, r_v = all_edges[i]
                
                # Try to find two missing edges that can be added when r_u-r_v is removed
                adj_mask[r_u] &= ~(1 << r_v)
                adj_mask[r_v] &= ~(1 << r_u)
                
                added_count = 0
                added_edges = []
                for j in range(len(missing)):
                    if time.time() > deadline:
                        break
                    a_u, a_v = missing[j]
                    if can_add(a_u, a_v, adj_mask):
                        adj_mask[a_u] |= (1 << a_v)
                        adj_mask[a_v] |= (1 << a_u)
                        added_edges.append((a_u, a_v))
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 1:
                    adj_mask = greedy_add(adj_mask)
                    improved = True
                    break
                else:
                    # Revert
                    adj_mask[r_u] |= (1 << r_v)
                    adj_mask[r_v] |= (1 << r_u)
                    for a_u, a_v in added_edges:
                        adj_mask[a_u] &= ~(1 << a_v)
                        adj_mask[a_v] &= ~(1 << a_u)
            
            if improved:
                continue
            
            # Try removing a random edge to escape local optimum
            if all_edges:
                r_u, r_v = all_edges[0]
                adj_mask[r_u] &= ~(1 << r_v)
                adj_mask[r_v] &= ~(1 << r_u)
                adj_mask = greedy_add(adj_mask)
        
        return adj_mask
>>>>>>> REPLACE

