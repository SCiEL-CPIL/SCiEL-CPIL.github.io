            old_pt = pts[idx].copy()
            pts[idx] = new_pt
            new_r = _score(pts)
            
            if new_r >= curr_r or (np.random.random() < np.exp(-(curr_r - new_r) / T)):
                curr_r = new_r
                if curr_r > best_r_local:
                    best_r_local = curr_r
                    best_pts_local = pts.copy()
            else:
                pts[idx] = old_pt
