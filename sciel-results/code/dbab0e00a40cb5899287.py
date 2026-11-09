import random
import time

def construct():
    start_time = time.time()
    
    def is_valid_addition(G, k, M):
        # Check if adding v with mask M creates a regular subgraph of size >= 6
        # S' is subset of 0..k-1, size j >= 5
        # New subgraph S' U {v} has size j+1 >= 6
        # Condition: for all u in S', deg_{G[S']}(u) + (1 if u in M else 0) == popcount(S' & M)
        
        # Iterate j from 5 to k-1 (since S' is from old graph, max size k-1? No, S' can be all k vertices? 
        # If S' is all k vertices, new subgraph is size k+1.
        # So j goes from 5 to k.
        
        # Optimization: Use bit_count()
        
        for j in range(5, k + 1):
            # Generate masks of size j
            mask = (1 << j) - 1
            limit = 1 << k
            while mask < limit:
                N_v = mask & M
                d = N_v.bit_count()
                
                # Check degrees
                m = mask
                valid = True
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    deg = (G[u] & mask).bit_count()
                    add = 1 if (N_v & lsb) else 0
                    if deg + add != d:
                        valid = False
                        break
                    m -= lsb
                
                if valid:
                    return False
                
                # Next mask
                c = mask & -mask
                r = mask + c
                mask = (((r ^ mask) >> 2) // c) | r
                
        return True

    def risk_score(G, k, M):
        # Count near-misses: subsets S' of size 5 where S' U {v} is almost regular
        # (max deg - min deg <= 1)
        score = 0
        j = 5
        mask = (1 << j) - 1
        limit = 1 << k
        while mask < limit:
            N_v = mask & M
            d_v = N_v.bit_count()
            
            degs = []
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                deg = (G[u] & mask).bit_count()
                add = 1 if (N_v & lsb) else 0
                degs.append(deg + add)
                m -= lsb
            degs.append(d_v)
            
            if max(degs) - min(degs) <= 1:
                score += 1
                
            c = mask & -mask
            r = mask + c
            mask = (((r ^ mask) >> 2) // c) | r
            
        return score

    def verify_graph(G, n):
        for mask in range(1, 1 << n):
            if mask.bit_count() < 6:
                continue
            m = mask
            first_deg = -1
            regular = True
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                deg = (G[u] & mask).bit_count()
                if first_deg == -1:
                    first_deg = deg
                elif deg != first_deg:
                    regular = False
                    break
                m -= lsb
            if regular:
                return False
        return True

    best_n = 5
    best_graph = {'n': 5, 'edges': []}
    
    for attempt in range(50):
        random.seed(attempt)
        G = [0] * 5
        edges = []
        for i in range(5):
            for j in range(i+1, 5):
                if random.random() < 0.5:
                    G[i] |= (1 << j)
                    G[j] |= (1 << i)
                    edges.append([i, j])
        
        current_n = 5
        
        for target_n in range(6, 23):
            k = current_n
            best_mask = None
            best_score = float('inf')
            
            # Try 100 random masks
            for _ in range(100):
                cand_mask = random.getrandbits(k)
                if is_valid_addition(G, k, cand_mask):
                    score = risk_score(G, k, cand_mask)
                    if score < best_score:
                        best_score = score
                        best_mask = cand_mask
            
            if best_mask is None:
                break
            
            G.append(best_mask)
            for i in range(k):
                if (best_mask >> i) & 1:
                    G[i] |= (1 << k)
                    edges.append([i, k])
            
            current_n += 1
            
            if current_n > best_n:
                if verify_graph(G, current_n):
                    best_n = current_n
                    best_graph = {'n': current_n, 'edges': edges.copy()}
                    if current_n >= 20:
                        return best_graph
            
            if time.time() - start_time > 110:
                return best_graph
        
    return best_graph
