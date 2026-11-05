"""Planted 3-SAT solver: bitset-accelerated WalkSAT with adaptive noise."""
# EVOLVE-BLOCK-START
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    pos_mask = [0] * m
    neg_mask = [0] * m
    var_clauses = [[] for _ in range(n)]

    for c, clause in enumerate(clauses):
        pm = 0
        nm = 0
        for lit in clause:
            v = abs(lit) - 1
            if lit > 0:
                pm |= (1 << v)
            else:
                nm |= (1 << v)
        pos_mask[c] = pm
        neg_mask[c] = nm
        for lit in clause:
            var_clauses[abs(lit) - 1].append(c)

    rng = random.Random(12345)

    def is_satisfied(c, assignment):
        return ((assignment & pos_mask[c]) | ((~assignment) & neg_mask[c])) != 0

    total_budget = 500000
    restarts = 10
    for restart in range(restarts):
        assignment = rng.getrandbits(n)

        unsat_count = [0] * n
        unsat_set = set()
        unsat_list = []
        for c in range(m):
            if not is_satisfied(c, assignment):
                unsat_set.add(c)
                unsat_list.append(c)
                bits = pos_mask[c] | neg_mask[c]
                while bits:
                    bit = bits & (-bits)
                    v = bit.bit_length() - 1
                    unsat_count[v] += 1
                    bits ^= bit

        if not unsat_list:
            return [(assignment >> v) & 1 == 1 for v in range(n)]

        budget = total_budget // restarts
        for _ in range(budget):
            if not unsat_list:
                return [(assignment >> v) & 1 == 1 for v in range(n)]

            c = unsat_list[rng.randrange(len(unsat_list))]
            pm = pos_mask[c]
            nm = neg_mask[c]
            vars_in = pm | nm

            clause_vars = []
            lsb = vars_in
            while lsb:
                bit = lsb & (-lsb)
                v = bit.bit_length() - 1
                clause_vars.append(v)
                lsb ^= bit

            if rng.random() < 0.35:
                v = clause_vars[rng.randrange(len(clause_vars))]
            else:
                best_v = clause_vars[0]
                best_cnt = unsat_count[best_v]
                for vv in clause_vars[1:]:
                    if unsat_count[vv] < best_cnt:
                        best_cnt = unsat_count[vv]
                        best_v = vv
                v = best_v

            new_assignment = assignment ^ (1 << v)

            for cc in var_clauses[v]:
                was_unsat = cc in unsat_set
                now_unsat = not is_satisfied(cc, new_assignment)
                if was_unsat and not now_unsat:
                    unsat_set.remove(cc)
                    # Remove from unsat_list (swap with last for O(1)).
                    idx = unsat_list.index(cc)
                    unsat_list[idx] = unsat_list[-1]
                    unsat_list.pop()
                    bits = pos_mask[cc] | neg_mask[cc]
                    while bits:
                        bit = bits & (-bits)
                        vv = bit.bit_length() - 1
                        unsat_count[vv] -= 1
                        bits ^= bit
                elif not was_unsat and now_unsat:
                    unsat_set.add(cc)
                    unsat_list.append(cc)
                    bits = pos_mask[cc] | neg_mask[cc]
                    while bits:
                        bit = bits & (-bits)
                        vv = bit.bit_length() - 1
                        unsat_count[vv] += 1
                        bits ^= bit

            assignment = new_assignment

            if not unsat_list:
                return [(assignment >> v) & 1 == 1 for v in range(n)]

    return [(assignment >> v) & 1 == 1 for v in range(n)]
# EVOLVE-BLOCK-END
