            current_set = []
            cs_arr = np.empty((0, D), dtype=np.int64)
            
            for idx in indices:
                v = sub_candidates[idx]
                
                if cs_arr.shape[0] == 0:
                    current_set.append(v)
                    cs_arr = v[None, :]
                    continue
                    
                diffs = v[None, :] - cs_arr
                dist_sqs = np.sum(diffs**2, axis=1)
                
                if np.min(dist_sqs) >= R_sq:
                    current_set.append(v)
                    cs_arr = np.vstack([cs_arr, v[None, :]])
