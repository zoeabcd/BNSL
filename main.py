from qiskit import transpile
from qiskit_aer import AerSimulator
from hamitonian import hamitonian_para
from qannealing import annealing
from data_score import Generate_Data


M = 100
T = 1
lamda = 1
n = 2
m = 1
D = Generate_Data(n)
delta_max = 10
delta_cons = 10
delta_trans = 10

C, h, J = hamitonian_para(n, m, D, delta_max, \
                           delta_cons, delta_trans)
print(h)
print(J)


circ = annealing(n, M, h, J, T, lamda)
circ.draw('mpl')
simulator = AerSimulator()
compiled_circuit = transpile(circ, simulator)
job = simulator.run(compiled_circuit, shots = 100000)
res = job.result()
counts = res.get_counts(compiled_circuit)
ans = max(counts)
print(ans)