import numpy as np
import random
import time
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 147.0

    def remaining():
        return DEADLINE - (time.time() - start)

    # ---------- Helpers ----------
    def build_adj(n, bits):
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                b = bits[idx]; idx += 1
                if b == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
        return adj

    def adj_to_bits(n, adj):
        bits = []
        for i in range(n):
            for j in range(i+1, n):
                bits.append(0 if adj[i][j] == 1 else 1)
        return bits

    def out_masks(n, adj):
        om = [0]*n
        for i in range(n):
            m = 0
            row = adj[i]
            for j in range(n):
                if i != j and row[j]:
                    m |= (1 << j)
            om[i] = m
        return om

    def paley_bits(p):
        sq = set((x*x) % p for x in range(1, p))
        bits = []
        for i in range(p):
            for j in range(i+1, p):
                if (j - i) % p in sq:
                    bits.append(0)
                else:
                    bits.append(1)
        return bits

    def is_valid(n, adj, om):
        # Check every 4-subset has a common dominator outside the 4-set
        for quad in combinations(range(n), 4):
            a,b,c,d = quad
            qm = (1<<a)|(1<<b)|(1<<c)|(1<<d)
            ok = False
            for v in range(n):
                if v in (a,b,c,d):
                    continue
                if (om[v] & qm) == qm:
                    ok = True
                    break
            if not ok:
                return False
        return True

    def count_violations_fast(n, adj, om):
        """
        Count 4-subsets with no common dominator.
        A 4-set {a,b,c,d} is valid iff there exists v not in {a,b,c,d} with v->a, v->b, v->c, v->d.
        For each vertex v, let S_v = out-neighborhood of v.
        The 4-sets dominated by v are the 4-subsets of S_v that do NOT contain v (always true since v not in S_v).
        But a 4-set is "covered" if it's a subset of S_v for some v not in the 4-set.
        
        Alternative: total 4-sets = C(n,4). 
        Count 4-sets that are subsets of S_v for some v, but careful with inclusion-exclusion.
        
        Better approach: for each 4-set, check if any vertex outside it dominates all 4.
        This is O(n * C(n,4)) which for n=40 is 40 * 91390 = 3.6M, feasible.
        """
        cnt = 0
        for quad in combinations(range(n), 4):
            a,b,c,d = quad
            qm = (1<<a)|(1<<b)|(1<<c)|(1<<d)
            ok = False
            for v in range(n):
                if v in (a,b,c,d):
                    continue
                if (om[v] & qm) == qm:
                    ok = True
                    break
            if not ok:
                cnt += 1
        return cnt

    # ---------- Simulated Annealing ----------
    def solve_with_sa(n, budget_seconds):
        """
        Simulated annealing on a random tournament.
        Energy = number of 4-subsets with no common dominator.
        """
        random.seed(42)
        # Random tournament
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 0
                else:
                    adj[i][j] = 0
                    adj[j][i] = 1
        om = out_masks(n, adj)
        
        # Precompute list of all 4-subsets for faster energy eval
        all_quads = list(combinations(range(n), 4))
        
        energy = count_violations_fast(n, adj, om)
        best_energy = energy
        best_adj = [row[:] for row in adj]
        
        # Temperature schedule
        T = 2.0
        T_min = 0.01
        cooling = 0.995
        max_steps = int(budget_seconds * 500)  # ~500 steps per second target
        
        step_budget = max_steps
        t0 = time.time()
        
        for step in range(step_budget):
            if time.time() - t0 > budget_seconds - 0.5:
                break
            if energy == 0:
                break
            if T < T_min:
                # Restart with higher temperature
                T = 1.0
                continue
            
            # Pick random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Compute delta energy
            # Flipping edge (u,v): if adj[u][v] was 1, it becomes 0, so u no longer dominates v, v now dominates u.
            # This affects 4-sets that contain v but not u (u was dominating them via v) and 4-sets that contain u but not v (v was dominating them via u).
            # Actually, a 4-set Q is affected if the set of dominators of Q changes.
            # Dominators of Q: vertices w not in Q such that w->x for all x in Q.
            # Flipping (u,v) changes adj[u][v] and adj[v][u].
            # For a 4-set Q containing v but not u: u could be a dominator only if u->v and u->x for all other x in Q.
            #   Before: if adj[u][v]=1, u might dominate; after: adj[u][v]=0, u cannot dominate.
            #   Also, v's out-neighborhood changes: v now points to u. But v is in Q, so v's domination of other sets doesn't matter for Q's validity.
            #   Wait, the dominators of Q are vertices outside Q. So if u is outside Q and Q contains v, then:
            #   Before flip: u dominates Q if adj[u][v]=1 and adj[u][x]=1 for all x in Q\{v}.
            #   After flip: u dominates Q only if adj[u][v]=0 (which it now is) -> so u cannot dominate Q after.
            #   So if u was a dominator of Q before and no other dominator exists, Q becomes invalid.
            #   Similarly, if u was NOT a dominator but becomes one? No, adj[u][v] goes from 1 to 0, so u loses domination of v.
            
            # For a 4-set Q containing u but not v: 
            #   Before: v dominates Q if adj[v][u]=0 (i.e., adj[u][v]=1) and adj[v][x]=1 for all x in Q\{u}.
            #   After: adj[v][u]=1, so v cannot dominate Q.
            
            # So flipping (u,v) from u->v to v->u:
            #   - For Q containing v not u: u loses ability to dominate Q. If u was the ONLY dominator of Q, Q becomes a new violation.
            #   - For Q containing u not v: v loses ability to dominate Q. If v was the ONLY dominator of Q, Q becomes a new violation.
            #   - No Q gains a dominator from this flip? 
            #     Wait, could u gain domination of some Q? No, u now points to v instead of being pointed to. 
            #     For Q containing v not u: u was pointing to v before, now not. So u loses.
            #     For Q containing u not v: v was not pointing to u before, now is. But v is in Q? No, v not in Q. 
            #     Wait, for Q containing u not v: v is a candidate dominator. Before: adj[v][u]=0, so v couldn't dominate u. After: adj[v][u]=1, so v might dominate Q if v->x for all other x in Q.
            #     So v GAINS the ability to dominate Q!
            
            # So delta = (# Q containing v not u where u was sole dominator) + (# Q containing u not v where v was sole dominator) - (# Q containing u not v where v was not a dominator but becomes one) ... 
            # This is complex. Let's just recompute energy fully for simplicity, but optimize.
            
            # For n up to 40, count_violations_fast takes ~50-100ms. With 500 steps/sec that's too slow.
            # Let's do incremental update.
            
            # Store dominator counts for each quad? That's C(n,4) ints, ~91K for n=40. Feasible.
            # Let's maintain dom_count[q] = number of vertices outside q that dominate all of q.
            # A quad is valid iff dom_count[q] > 0.
            
            # Precompute all_quads and their index
            quad_index = {}
            quad_list = []
            for qi, q in enumerate(all_quads):
                quad_index[q] = qi
                quad_list.append(q)
            
            # Initialize dom_count
            dom_count = [0]*len(all_quads)
            for qi, q in enumerate(quad_list):
                a,b,c,d = q
                qm = (1<<a)|(1<<b)|(1<<c)|(1<<d)
                cnt = 0
                for v in range(n):
                    if v in (a,b,c,d):
                        continue
                    if (om[v] & qm) == qm:
                        cnt += 1
                dom_count[qi] = cnt
            energy = sum(1 for c in dom_count if c == 0)
            best_energy = energy
            best_adj = [row[:] for row in adj]
            
            # Now incremental updates
            for step in range(step_budget):
                if time.time() - t0 > budget_seconds - 0.5:
                    break
                if energy == 0:
                    break
                if T < T_min:
                    T = 1.0
                    continue
                
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                # Currently adj[u][v] is either 0 or 1
                # If 1: u->v. After flip: v->u.
                # If 0: v->u. After flip: u->v.
                
                old_dir = adj[u][v]  # 1 means u->v, 0 means v->u
                
                # Affected quads:
                # Case 1: old_dir == 1 (u->v, becoming v->u)
                #   - Quads containing v, not u: u was pointing to v, now not. u may lose domination.
                #   - Quads containing u, not v: v was not pointing to u, now is. v may gain domination.
                # Case 2: old_dir == 0 (v->u, becoming u->v)
                #   - Quads containing u, not v: v was pointing to u, now not. v may lose domination.
                #   - Quads containing v, not u: u was not pointing to v, now is. u may gain domination.
                
                delta_e = 0
                
                if old_dir == 1:
                    # u->v becomes v->u
                    # For quads containing v not u:
                    for w1 in range(n):
                        if w1 == v or w1 == u:
                            continue
                        for w2 in range(w1+1, n):
                            if w2 == v or w2 == u:
                                continue
                            for w3 in range(w2+1, n):
                                if w3 == v or w3 == u:
                                    continue
                                q = tuple(sorted((w1, w2, w3, v)))
                                qi = quad_index.get(q)
                                if qi is None:
                                    continue
                                # Check if u dominates this quad
                                # u must point to w1, w2, w3, v
                                # Before: u->v was true. After: u->v is false.
                                if adj[u][w1] and adj[u][w2] and adj[u][w3]:
                                    # u was a dominator, now isn't
                                    dom_count[qi] -= 1
                                    if dom_count[qi] == 0:
                                        delta_e += 1
                    # For quads containing u not v:
                    for w1 in range(n):
                        if w1 == v or w1 == u:
                            continue
                        for w2 in range(w1+1, n):
                            if w2 == v or w2 == u:
                                continue
                            for w3 in range(w2+1, n):
                                if w3 == v or w3 == u:
                                    continue
                                q = tuple(sorted((w1, w2, w3, u)))
                                qi = quad_index.get(q)
                                if qi is None:
                                    continue
                                # Check if v dominates this quad
                                # v must point to w1, w2, w3, u
                                # Before: v->u was false. After: v->u is true.
                                if adj[v][w1] and adj[v][w2] and adj[v][w3]:
                                    # v was NOT a dominator (since v didn't point to u), now is
                                    dom_count[qi] += 1
                                    if dom_count[qi] == 1:
                                        delta_e -= 1
                else:
                    # v->u becomes u->v
                    # For quads containing u not v:
                    for w1 in range(n):
                        if w1 == v or w1 == u:
                            continue
                        for w2 in range(w1+1, n):
                            if w2 == v or w2 == u:
                                continue
                            for w3 in range(w2+1, n):
                                if w3 == v or w3 == u:
                                    continue
                                q = tuple(sorted((w1, w2, w3, u)))
                                qi = quad_index.get(q)
                                if qi is None:
                                    continue
                                # Check if v dominates this quad
                                # v must point to w1, w2, w3, u
                                # Before: v->u was true. After: v->u is false.
                                if adj[v][w1] and adj[v][w2] and adj[v][w3]:
                                    dom_count[qi] -= 1
                                    if dom_count[qi] == 0:
                                        delta_e += 1
                    # For quads containing v not u:
                    for w1 in range(n):
                        if w1 == v or w1 == u:
                            continue
                        for w2 in range(w1+1, n):
                            if w2 == v or w2 == u:
                                continue
                            for w3 in range(w2+1, n):
                                if w3 == v or w3 == u:
                                    continue
                                q = tuple(sorted((w1, w2, w3, v)))
                                qi = quad_index.get(q)
                                if qi is None:
                                    continue
                                # Check if u dominates this quad
                                # u must point to w1, w2, w3, v
                                # Before: u->v was false. After: u->v is true.
                                if adj[u][w1] and adj[u][w2] and adj[u][w3]:
                                    dom_count[qi] += 1
                                    if dom_count[qi] == 1:
                                        delta_e -= 1
                
                new_energy = energy + delta_e
                if new_energy < energy or random.random() < np.exp(-delta_e / max(T, 1e-10)):
                    # Accept flip
                    adj[u][v] = 1 - adj[u][v]
                    adj[v][u] = 1 - adj[v][u]
                    # Update om
                    if old_dir == 1:
                        # u->v became v->u
                        om[u] &= ~(1 << v)
                        om[v] |= (1 << u)
                    else:
                        # v->u became u->v
                        om[v] &= ~(1 << u)
                        om[u] |= (1 << v)
                    energy = new_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_adj = [row[:] for row in adj]
                        if energy == 0:
                            break
                T *= cooling
            
            # After SA, check if best is valid
            if best_energy == 0:
                bits = adj_to_bits(n, best_adj)
                return bits, True
            return None, False

    # ---------- Main ----------
    best_result = None
    
    # Phase 1: Simulated annealing for small n
    for n_try in range(31, 45):
        if remaining() < 8.0:
            break
        budget = min(6.0, remaining() - 3.0)
        if budget <= 1.0:
            break
        bits, success = solve_with_sa(n_try, budget)
        if success:
            best_result = (n_try, bits)
            break
    
    # Phase 2: If not found, fallback to Paley
    if best_result is None:
        p = 83
        bits = paley_bits(p)
        best_result = (p, bits)
    
    n_final, bits_final = best_result
    return {'n': n_final, 'orientation': bits_final}
