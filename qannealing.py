from qiskit import  QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import RXGate, RZZGate, RZGate
from qiskit_aer import AerSimulator
from hamiltonian import hamiltonian_para
from data_score import Generate_Data, score
from tqdm import tqdm
def At(t, T):
    return(1- t/T)

def Bt(t, T):
    return t/T

def annealing(n, M, h, J, T, lamda, use_y = True):
    d_qr = QuantumRegister(n*(n-1))
    r_qr = QuantumRegister(n*(n-1)/2)
    if use_y:
        y_qr = QuantumRegister(2*n)
    N = int(3*n*(n-1) / 2 )
    if use_y:
        N += 2 * n
    circ = QuantumCircuit(d_qr, r_qr, y_qr) if use_y else QuantumCircuit(d_qr, r_qr)
    circ.h(range(N))
    dt = T / M
    t = 0
    cnt = 0

    for i in range(M):
        for j in range(N):
            for k in range(j+1, N):
                if J[j, k] == 0: continue
                circ.append(RZZGate(2 * J[j, k] * dt * Bt(t, T), label = 'U'+str(cnt)), [j,k])

        for j in range(N):
            if h[j] == 0: continue
            circ.append(RZGate(2 * h[j] * dt * Bt(t,T), label = 'V'+str(cnt)), [j])

        
        for j in range(N):
            circ.append(RXGate(-2 * lamda * dt * At(t, T)), [j])

        t = t + dt
        
    circ.measure_all()
    return circ