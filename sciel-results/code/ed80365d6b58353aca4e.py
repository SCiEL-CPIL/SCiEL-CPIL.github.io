import time, math, random

def _triples(n):
    T=[]
    for x in range(1,n+1):
        x2=x*x
        d=1
        while d*d<=x2:
            if x2%d==0:
                e=x2//d
                if d<e:
                    y=x+d; z=x+e
                    if y<=n and z<=n and y!=z:
                        T.append((x,y,z))
            d+=1
    return T

def construct():
    t0=time.time()
    DEADLINE=82.0
    n=180
    T=_triples(n)
    m=len(T)
    
    # Precompute: for each element, list of (triple_idx, other1, other2)
    elem_triples=[[] for _ in range(n+1)]
    for ti,(a,b,c) in enumerate(T):
        elem_triples[a].append((ti,b,c))
        elem_triples[b].append((ti,a,c))
        elem_triples[c].append((ti,a,b))
    
    def conflict_count(S, x):
        c=0
        for (ti,o1,o2) in elem_triples[x]:
            if S[o1] and S[o2]:
                c+=1
        return c
    
    def count_violated(S):
        """Count violated triples efficiently using per-element tracking."""
        # For each triple, check if all 3 are in S
        nv = 0
        for (a,b,c) in T:
            if S[a] and S[b] and S[c]:
                nv += 1
        return nv
    
    def feasible(S):
        for (a,b,c) in T:
            if S[a] and S[b] and S[c]:
                return False
        return True

    # Greedy construction: descending order
    S=[False]*(n+1)
    for x in range(n,0,-1):
        if conflict_count(S,x)==0:
            S[x]=True
    size=sum(S)
    
    # Simulated annealing phase with swap moves
    def simulated_annealing(S_init, t0, deadline):
        S=S_init[:]
        best_S=S[:]
        best_size=sum(S)
        
        PENALTY=500
        
        # Compute initial objective
        nv = count_violated(S)
        cur_obj = sum(S) - PENALTY*nv
        best_obj = cur_obj
        
        num_restarts=0
        max_restarts=100
        
        while time.time()-t0 < deadline - 4.0:
            T_cur=3.0
            T_end=0.005
            iterations=0
            max_iter=300000
            no_improve=0
            
            while time.time()-t0 < deadline - 4.0 and iterations < max_iter:
                iterations+=1
                
                # Choose move type: 70% flip, 30% swap
                if random.random() < 0.7:
                    # Flip move
                    x=random.randint(1,n)
                    if S[x]:
                        removed_viol=0
                        for (ti,o1,o2) in elem_triples[x]:
                            if S[o1] and S[o2]:
                                removed_viol+=1
                        delta_obj = -1 + PENALTY*removed_viol
                    else:
                        new_viol=0
                        for (ti,o1,o2) in elem_triples[x]:
                            if S[o1] and S[o2]:
                                new_viol+=1
                        delta_obj = 1 - PENALTY*new_viol
                    
                    new_obj=cur_obj+delta_obj
                    
                    if new_obj >= cur_obj or (T_cur>0 and random.random() < math.exp(-(cur_obj-new_obj)/T_cur)):
                        S[x]=not S[x]
                        cur_obj=new_obj
                        if cur_obj>best_obj:
                            best_obj=cur_obj
                            best_S=S[:]
                            no_improve=0
                        else:
                            no_improve+=1
                    else:
                        no_improve+=1
                else:
                    # Swap move: pick two elements, one in S and one not
                    # Try to find a beneficial swap
                    in_elems=[i for i in range(1,n+1) if S[i]]
                    out_elems=[i for i in range(1,n+1) if not S[i]]
                    if in_elems and out_elems:
                        x=random.choice(in_elems)  # will remove
                        y=random.choice(out_elems)  # will add
                        
                        # Compute delta: remove x, add y
                        # Removing x: -1 size, +PENALTY*viol_removed
                        viol_removed_x=0
                        for (ti,o1,o2) in elem_triples[x]:
                            if S[o1] and S[o2] and o1!=y and o2!=y:
                                viol_removed_x+=1
                            elif S[o1] and S[o2]:
                                # y is one of o1,o2 - need to be careful
                                # After swap, y is in S, x is not
                                # Triple (x,o1,o2) was violated if o1,o2 in S
                                # After: x not in S, so not violated regardless
                                viol_removed_x+=1
                        
                        # Adding y: +1 size, -PENALTY*viol_added
                        # But we need to account for interaction with x being removed
                        viol_added_y=0
                        for (ti,o1,o2) in elem_triples[y]:
                            # o1,o2 in S after swap (x removed, y added)
                            # If o1==x or o2==x, they won't be in S after
                            o1_in = S[o1] and o1 != x
                            o2_in = S[o2] and o2 != x
                            if o1_in and o2_in:
                                viol_added_y+=1
                        
                        delta_obj = (-1 + PENALTY*viol_removed_x) + (1 - PENALTY*viol_added_y)
                        delta_obj = PENALTY*(viol_removed_x - viol_added_y)
                        
                        new_obj=cur_obj+delta_obj
                        
                        if new_obj >= cur_obj or (T_cur>0 and random.random() < math.exp(-(cur_obj-new_obj)/T_cur)):
                            S[x]=False
                            S[y]=True
                            cur_obj=new_obj
                            if cur_obj>best_obj:
                                best_obj=cur_obj
                                best_S=S[:]
                                no_improve=0
                            else:
                                no_improve+=1
                        else:
                            no_improve+=1
                
                T_cur*=0.9997
                if T_cur<T_end:
                    T_cur=T_end
                
                if no_improve>8000:
                    break
            
            if best_obj>cur_obj:
                S=best_S[:]
                cur_obj=best_obj
            
            num_restarts+=1
            if num_restarts>=max_restarts:
                break
        
        return best_S, best_size
    
    # Run SA
    if time.time()-t0 < DEADLINE - 8.0:
        S_sa, size_sa = simulated_annealing(S, t0, DEADLINE)
        if feasible(S_sa) and size_sa > size:
            S=S_sa
            size=size_sa
        elif feasible(S_sa):
            S=S_sa
            size=size_sa
    
    # Final local improvement: try adding elements
    if time.time()-t0 < DEADLINE - 3.0:
        improved=True
        while improved and time.time()-t0 < DEADLINE - 3.0:
            improved=False
            for x in range(1,n+1):
                if not S[x]:
                    if conflict_count(S,x)==0:
                        S[x]=True
                        size+=1
                        improved=True
    
    out=[i for i in range(1,n+1) if S[i]]
    if not out:
        out=[180]
    return out
