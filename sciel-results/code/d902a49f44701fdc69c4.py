import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_n = 37
    best_edges = None
    
    # Helper to generate Paley graph
    def gen_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        adj = [0] * q
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, q

    # Helper to find K5 or Ind5
    def find_violation(adj, n):
        # Find K5
        for i in range(n):
            Ni = adj[i]
            # Iterate j > i in Ni
            j_mask = Ni & ~((1 << (i + 1)) - 1)
            while j_mask:
                j_lsb = j_mask & -j_mask
                j = j_lsb.bit_length() - 1
                j_mask -= j_lsb
                
                Nij = Ni & adj[j]
                k_mask = Nij & ~((1 << (j + 1)) - 1)
                while k_mask:
                    k_lsb = k_mask & -k_mask
                    k = k_lsb.bit_length() - 1
                    k_mask -= k_lsb
                    
                    Nijk = Nij & adj[k]
                    l_mask = Nijk & ~((1 << (k + 1)) - 1)
                    while l_mask:
                        l_lsb = l_mask & -l_mask
                        l = l_lsb.bit_length() - 1
                        l_mask -= l_lsb
                        
                        Nijkl = Nijk & adj[l]
                        m_mask = Nijkl & ~((1 << (l + 1)) - 1)
                        if m_mask:
                            m_lsb = m_mask & -m_mask
                            m = m_lsb.bit_length() - 1
                            return [i, j, k, l, m]
        
        # Find Independent Set 5 (Clique in complement)
        comp_adj = [0] * n
        for i in range(n):
            # comp_adj[i] has bits for j where NOT adj[i] has j, and j != i
            # Full mask for n bits
            full = (1 << n) - 1
            comp_adj[i] = (full ^ adj[i]) & ~(1 << i)
            
        for i in range(n):
            Ni = comp_adj[i]
            j_mask = Ni & ~((1 << (i + 1)) - 1)
            while j_mask:
                j_lsb = j_mask & -j_mask
                j = j_lsb.bit_length() - 1
                j_mask -= j_lsb
                
                Nij = Ni & comp_adj[j]
                k_mask = Nij & ~((1 << (j + 1)) - 1)
                while k_mask:
                    k_lsb = k_mask & -k_mask
                    k = k_lsb.bit_length() - 1
                    k_mask -= k_lsb
                    
                    Nijk = Nij & comp_adj[k]
                    l_mask = Nijk & ~((1 << (k + 1)) - 1)
                    while l_mask:
                        l_lsb = l_mask & -l_mask
                        l = l_lsb.bit_length() - 1
                        l_mask -= l_lsb
                        
                        Nijkl = Nijk & comp_adj[l]
                        m_mask = Nijkl & ~((1 << (l + 1)) - 1)
                        if m_mask:
                            m_lsb = m_mask & -m_mask
                            m = m_lsb.bit_length() - 1
                            return [i, j, k, l, m]
                            
        return None

    def prune_graph(adj, n):
        # Local search to remove vertices
        removed = set()
        current_n = n
        current_adj = adj[:]
        
        while time.time() < deadline:
            violation = find_violation(current_adj, current_n)
            if violation is None:
                break
            
            # Choose vertex to remove
            # Heuristic: Remove vertex with highest degree (for K5) or lowest (for Ind5)
            # We don't know which violation it is, but we can check.
            # Actually, find_violation returns the set.
            # We need to know if it's a clique or ind set.
            # Let's modify find_violation to return type.
            
            # For now, assume it's a clique and remove max degree.
            # If it's an ind set, removing max degree might be bad?
            # In an ind set, vertices have low connectivity to each other.
            # Removing a vertex with high degree in the *whole graph* might break other cliques.
            # Let's just remove the first one or random.
            
            v_to_remove = random.choice(violation)
            
            # Remove v_to_remove
            # Shift indices? Or keep a mask?
            # Keeping a mask of active vertices is easier.
            # But find_violation expects contiguous 0..n-1.
            # So we must rebuild adj list after removal.
            
            new_n = current_n - 1
            new_adj = [0] * new_n
            
            # Map old indices to new
            # If we remove v, indices > v shift down by 1.
            for i in range(current_n):
                if i == v_to_remove:
                    continue
                new_i = i if i < v_to_remove else i - 1
                
                # Copy bits
                old_bits = current_adj[i]
                # We need to shift bits > v_to_remove down by 1.
                # Bits < v_to_remove stay.
                # Bit v_to_remove is cleared.
                
                # Lower part: bits 0..v_to_remove-1
                lower_mask = (1 << v_to_remove) - 1
                lower_bits = old_bits & lower_mask
                
                # Upper part: bits v_to_remove+1..n-1
                upper_mask = ~((1 << (v_to_remove + 1)) - 1)
                upper_bits = (old_bits & upper_mask) >> 1
                
                new_adj[new_i] = lower_bits | upper_bits
                
            current_n = new_n
            current_adj = new_adj
            
            if current_n <= best_n:
                break

        # Reconstruct edges from current_adj
        edges = []
        for i in range(current_n):
            bits = current_adj[i]
            while bits:
                lsb = bits & -bits
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append((i, j))
                bits -= lsb
        
        return current_n, edges

    # Try Paley 53
    try:
        adj_53, n_53 = gen_paley(53)
        n_res, edges_res = prune_graph(adj_53, n_53)
        if n_res > best_n:
            best_n = n_res
            best_edges = edges_res
    except Exception:
        pass

    # If still 37, try Paley 41 (safer, might be already free or close)
    if best_n < 38:
        try:
            adj_41, n_41 = gen_paley(41)
            n_res, edges_res = prune_graph(adj_41, n_41)
            if n_res > best_n:
                best_n = n_res
                best_edges = edges_res
        except Exception:
            pass

    # Fallback to seed
    if best_edges is None:
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        best_n = q
        best_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

    return best_n, best_edges
