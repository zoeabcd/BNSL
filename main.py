from qiskit import transpile
from qiskit_aer import AerSimulator
from hamiltonian import hamiltonian_para, plot
from qannealing import annealing
from data_score import Generate_Data
import heapq
from operator import itemgetter
from analysis_toolkit import res_draw, res_extractor
import numpy as np

M = 100
T = 10
lamda = 1
n = 4
m = 3
delta_max = 90
delta_cons = 90
delta_trans = 90

show_BF = False
onelocal = True
use_y = True

D = Generate_Data(n)
C, h, J = hamiltonian_para(n, m, D, delta_max, delta_cons, delta_trans, show_BF, onelocal, use_y)
print(h)
print(J)


circ = annealing(n, M, h, J, T, lamda, use_y)

print('the circuit is constructed, simulating...')

simulator = AerSimulator()
compiled_circuit = transpile(circ, simulator)

job = simulator.run(compiled_circuit, shots = 10000)
res = job.result()
counts = res.get_counts(compiled_circuit)

print('simulation done')

best_counts = dict(heapq.nlargest(7, counts.items(),key=itemgetter(1)))
print(best_counts)  

d0 = r0 = y0 = is_cons0 = is_dag0 = is_legal0 = G0 = None
for tmp in best_counts:
    d, r, y, is_cons, is_dag, is_legal, G = res_extractor(tmp, n, m)
    if d0 == None and d != list(np.zeros((int(n*(n-1)), 1))) and is_cons and is_dag and is_legal:
        d0, r0, y0, is_cons0, is_dag0, is_legal0, G0 = d, r, y, is_cons, is_dag, is_legal, G
    print(d,r,y, is_cons, is_dag, is_legal)


res_draw(d0, r0, y0, is_cons0, is_dag0, is_legal0, G0)