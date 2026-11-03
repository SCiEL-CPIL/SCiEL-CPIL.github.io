            # Compute upper bound using greedy coloring
            U = P
            max_color = 0
            while U:
                if time.time() > deadline - 5.0:
                    return
                
                max_color += 1
                if bound + max_color <= best_size:
                    return
                
                # Pick a vertex from U (first one for speed)
                v = (U & -U).bit_length() - 1
                U &= ~(1 << v)
                U &= ~compat_bits[v]
                
            # If we reach here, bound + max_color > best_size
            # Branch on a vertex in P
            # Pick max degree vertex in P
            max_deg = -1
            best_v = -1
            v = 0
            p_copy = P
            while p_copy:
                if p_copy & 1:
                    deg = bin(compat_bits[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
                p_copy >>= 1
                v += 1
            
            if best_v == -1:
                return

            # Include best_v
            new_P = P & compat_bits[best_v]
            X.append(best_v)
            expand(new_P, X, bound + 1)
            X.pop()
            
            # Exclude best_v
            P &= ~(1 << best_v)
            
            # Check bound again (simple popcount)
            if bound + bin(P).count('1') <= best_size:
                return

            # Continue
            expand(P, X, bound)
