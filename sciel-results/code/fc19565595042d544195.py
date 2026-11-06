            key = (u, v) if u < v else (v, u)
            affected = cache.get(key, [])
            # u is min, v is max? No, u, v are random.
            # If u < v, u is min, v is max.
            # If u > v, u is max, v is min.
            is_u_min = u < v
            
            for item in affected:
                idx, pos_min, pos_max = item
                c = subs_6[idx]
                mask = masks_6[idx]
                
                # Determine pos_u and pos_v
                if is_u_min:
                    pos_u = pos_min
                    pos_v = pos_max
                else:
                    pos_u = pos_max
                    pos_v = pos_min
                
                # ... rest of delta calc
