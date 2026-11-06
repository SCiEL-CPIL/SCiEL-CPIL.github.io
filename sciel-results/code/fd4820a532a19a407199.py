chosen=[]
for row in combinations(range(1,17),4):
    s=set(row)
    if all(not(a & b == a & s == b & s) for a,b in combinations(chosen,2)):
        chosen.append(s)
return [sorted(s) for s in chosen]
