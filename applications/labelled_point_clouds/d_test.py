import dionysus as d
import numpy as np

# create filtration
labels = ["a", "b", "c"]
simplices = [([2], .4), ([1,2], .5), ([0,2], .6), ([0], .1),   ([1], .2), ([0,1], .3),
             ([0,1,2], .7)]
f = d.Filtration()

for vertices, time in simplices:
    f.append(d.Simplex(vertices, time))

f.sort()
print("\nsorted simplices")
print([list(s) for s in f])

m = d.homology_persistence(f)

print("\npersistence pairing")
for i in range(len(m)):
    if m.pair(i) < i: continue
    dim = f[i].dimension()
    if m.pair(i) != m.unpaired:
        print(dim, i, m.pair(i))
    else:
        print(dim, i)

print("\ncycles")
cycles = {i: c for i, c in enumerate(m) if len(c)!= 0}
cycles_dict = {i: ([sc.index for sc in c], f[m.pair(i)].data, f[i].data)
               for i, c in enumerate(m) if len(c)!= 0}
print(cycles_dict)
cycles_dict = {i: ([sc.index for sc in c], m.pair(i), i)
                   if len(c)!= 0 }
clean_cycles_dict = {i:v for i, v in cycles_dict.items() if v != []}
print("\nlabelled cycles")
cycle_labels = {}

for key, value in cycles_dict.items():
    for simp_index in value[0]:
        print(simp_index)
        print([labels[v] for v in f[simp_index]])


        cycle_labels[key] = np.unique([labels[v] for simp_index in value[0]
                                       for v in f[simp_index]])

print(cycle_labels)
