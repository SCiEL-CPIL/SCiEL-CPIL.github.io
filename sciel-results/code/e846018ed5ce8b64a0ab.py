        # compute delta without modifying state
        nd = num_distinct
        if freq[old_c]==1: nd-=1  # old color disappears
        if freq[new_c]==0: nd+=1  # new color appears
        dv = 0
        affected = POS_APS[x]
        new_states = []
        for ap_i in affected:
            ap = APS[ap_i]
            # temporarily consider new color
            cv = [colors[ap[0]],colors[ap[1]],colors[ap[2]],colors[ap[3]]]
            # replace x's color
            # find index of x in ap
            # x is one of ap[0..3]
            # set that to new_c
            for idx in range(4):
                if ap[idx]==x:
                    cv[idx]=new_c
                    break
            nv = 1 if len(set(cv))<3 else 0
            new_states.append((ap_i, nv))
        for ap_i, nv in new_states:
            if nv != is_viol[ap_i]:
                dv += 1 if nv else -1
        e_new = (total_viol+dv)*100 + nd
        delta = e_new - e_old
        if delta <= 0 or random.random() < math.exp(-delta/T):
            # accept: apply
            colors[x]=new_c
            freq[old_c]-=1
            freq[new_c]+=1
            num_distinct=nd
            for ap_i, nv in new_states:
                if nv != is_viol[ap_i]:
                    is_viol[ap_i]=nv
            total_viol+=dv
            if e_new < best_energy:
                best_energy=e_new
                best_colors=list(colors)
        T *= cooling
        it+=1
