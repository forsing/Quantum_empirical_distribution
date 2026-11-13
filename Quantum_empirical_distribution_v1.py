# Quantum_empirical_distribution 



import random
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
WINDOW_SIZE = 5
ANSATZ_LAYERS = 3
BANDWIDTH_GRID = (0.35, 0.55, 0.80, 1.15, 1.60)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < WINDOW_SIZE + 20:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def build_transitions(draws: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    x_list = []
    y_list = []
    for idx in range(window, len(draws)):
        x_list.append(draws[idx - window:idx].astype(np.float64))
        y_list.append(draws[idx])
    return np.array(x_list), np.array(y_list)


def compute_empirical_density(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    diffs = (x_train - x_query[None, :, :]) / (MAX_NUM * bandwidth)
    sq_dist = np.sum(diffs ** 2, axis=(1, 2))
    weights = np.exp(-0.5 * sq_dist)
    weights_sum = float(np.sum(weights))
    if weights_sum <= 1e-15:
        weights = np.ones(len(weights), dtype=np.float64) / len(weights)
    else:
        weights = weights / weights_sum

    grid = np.arange(1, MAX_NUM + 1, dtype=np.float64)[None, :]
    densities = np.zeros((NUM_POSITIONS, MAX_NUM), dtype=np.float64)

    for pos in range(NUM_POSITIONS):
        targets = y_train[:, pos][:, None].astype(np.float64)
        kernel_vals = np.exp(-0.5 * ((grid - targets) / bandwidth) ** 2)
        pos_dist = np.sum(weights[:, None] * kernel_vals, axis=0)
        pos_dist = np.maximum(pos_dist, 1e-12)
        pos_dist /= np.sum(pos_dist)
        if abs(float(np.sum(pos_dist)) - 1.0) > 1e-6:
            raise ValueError("Greska u normalizaciji empirijske raspodele.")
        densities[pos] = pos_dist

    return densities


def build_ansatz_circuit(params: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(NUM_QUBITS)
    idx = 0
    for _ in range(ANSATZ_LAYERS):
        for q in range(NUM_QUBITS):
            qc.ry(float(params[idx]), q)
            idx += 1
        for q in range(NUM_QUBITS - 1):
            qc.cx(q, q + 1)
    for q in range(NUM_QUBITS):
        qc.ry(float(params[idx]), q)
        idx += 1
    return qc


def fit_quantum_register_amplitudes(target_prob: np.ndarray) -> np.ndarray:
    target_state = np.zeros(STATE_DIM, dtype=np.float64)
    target_state[1 : MAX_NUM + 1] = target_prob
    target_state /= np.sum(target_state)
    target_amp = np.sqrt(target_state)

    num_params = (ANSATZ_LAYERS + 1) * NUM_QUBITS
    init_params = np.random.uniform(-0.2, 0.2, size=num_params)

    def loss_fn(theta: np.ndarray) -> float:
        qc = build_ansatz_circuit(theta)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        amps = np.sqrt(np.maximum(probs, 0.0))
        return float(np.sum((amps - target_amp) ** 2))

    res = minimize(loss_fn, init_params, method="L-BFGS-B", options={"maxiter": 120})
    opt_qc = build_ansatz_circuit(res.x)
    final_sv = Statevector.from_instruction(opt_qc)
    full_probs = final_sv.probabilities()

    valid_probs = full_probs[1 : MAX_NUM + 1]
    valid_probs = np.maximum(valid_probs, 1e-12)
    valid_probs /= np.sum(valid_probs)
    if abs(float(np.sum(valid_probs)) - 1.0) > 1e-6:
        raise ValueError("Greska u normalizaciji kvantne raspodele.")
    return valid_probs


def decode_combination_from_amplitudes(q_probs: np.ndarray) -> list[int]:
    available = np.ones(MAX_NUM, dtype=bool)
    selected = []
    for pos in range(NUM_POSITIONS):
        masked_probs = np.where(available, q_probs[pos], -1.0)
        best_idx = int(np.argmax(masked_probs))
        selected.append(best_idx + 1)
        available[best_idx] = False
    return sorted(selected)


def evaluate_chronological_nll(
    x_all: np.ndarray,
    y_all: np.ndarray,
    start_idx: int,
    end_idx: int,
    bandwidth: float,
) -> float:
    total_nll = 0.0
    count = 0
    for t in range(start_idx, end_idx):
        densities = compute_empirical_density(x_all[:t], y_all[:t], x_all[t], bandwidth)
        for pos in range(NUM_POSITIONS):
            target_num = int(y_all[t, pos])
            p = max(float(densities[pos, target_num - 1]), 1e-12)
            total_nll -= np.log(p)
        count += 1
    return total_nll / max(count, 1)


def process_csv(file_path: Path) -> None:
    draws = load_draws(file_path)
    x_all, y_all = build_transitions(draws, WINDOW_SIZE)
    n_samples = len(x_all)

    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    best_bw = float(next(iter(BANDWIDTH_GRID)))
    best_val_nll = float("inf")

    for bw in BANDWIDTH_GRID:
        val_nll = evaluate_chronological_nll(x_all, y_all, train_end, val_end, float(bw))
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_bw = float(bw)

    test_nll = evaluate_chronological_nll(x_all, y_all, val_end, n_samples, best_bw)

    latest_window = draws[-WINDOW_SIZE:].astype(np.float64)
    final_densities = compute_empirical_density(x_all, y_all, latest_window, best_bw)

    final_q_probs = np.zeros((NUM_POSITIONS, MAX_NUM), dtype=np.float64)
    for pos in range(NUM_POSITIONS):
        final_q_probs[pos] = fit_quantum_register_amplitudes(final_densities[pos])

    next_comb = decode_combination_from_amplitudes(final_q_probs)

    if len(set(next_comb)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in next_comb):
        raise RuntimeError(f"Nelegalna NEXT kombinacija: {next_comb}")

    print(f"CSV: {file_path.name}")
    print(f"Validacija NLL (h={best_bw}): {best_val_nll:.4f}")
    print(f"Test NLL: {test_nll:.4f}")
    print(f"NEXT: {next_comb}")


def main() -> None:
    print("Metoda: Quantum Variational Empirical Amplitude Distribution (7x6 qubita)")
    for csv_path in FILES:
        process_csv(csv_path)


if __name__ == "__main__":
    main()



"""
Metoda: Quantum Variational Empirical Amplitude Distribution (7x6 qubita)

CSV: loto7_4718_k90.csv
Validacija NLL (h=0.55): 20.5054
Test NLL: 20.4589
NEXT: [2, x, 15, y, 29, z, 39]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (h=0.55): 20.4562
Test NLL: 20.5789
NEXT: [1, x, 14, y, 24, z, 39]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (h=0.8): 20.4230
Test NLL: 20.2962
NEXT: [2, x, 11, y, 27, z, 38]

"""



"""
Probability Distributions: Find the Best Distribution 
decision-making 
predictive modeling 


Pronalaženje najbolje raspodele verovatnoće za podatke (Probability Distributions for Predictive Modeling)

	1. Modelovanje empirijske raspodele iz podataka: Umesto nametanja gotove normalne ili uniformne raspodele, koristi se ocena gustine verovatnoće (KDE) ili prilagođavanje empirijske funkcije raspodele (ECDF) direktno na istorijskim prelazima.
	2. Kvantno kodiranje raspodele preko amplituda: Naučena empirijska gustina verovatnoće za 7 pozicija mapira se u kvantne amplitude: 6 qubita za sedam registara. Kvadrat modula amplitude daje verovatnoću svakog broja, čime se izbegava klasično rangiranje po frekvenciji. 
	3. Merenje odstupanja raspodela (KL-divergencija / Wasserstein distanca): Pri hronološkoj validaciji, ove metrike precizno mere koliko je predviđena raspodela bliska stvarnoj izvučenoj kombinaciji, bez korišćenja klasifikatora.

Ova oblast direktno pogađa suštinu zadatka jer uči stvarnu distribuciju brojeva iz istorije i prirodno se prevodi u kvantni vektor stanja (amplitude) pomoću Qiskita.

Empirijske raspodele verovatnoće iz istorijskih prelaza i njeno kvantno kodiranje putem amplituda u Qiskit-u.

• Kvantno i klasično: 
    Kvantni deo čini 7 nezavisnih kvantnih registara od po 6 qubita (\(2^6 = 64\) bazna stanja po registru, gde indeks \(0\) ostaje prazan, stanja \(1\dots39\) nose kvantne amplitude za brojeve \(1\dots39\), a stanja \(40\dots63\) imaju amplitudu \(0\)). Stanje se priprema putem Statevector i Initialize (amplitudno kodiranje), dok se evaluacija na lokalnom simulatoru (StatevectorSimulator / Statevector.probabilities()) koristi za dobijanje tačne raspodele verovatnoća \(\vert{}\alpha_i\vert{}^2\). 
    Klasični deo čini učenje kernel gustine (KDE) i regresija prelaza nad prozorom prethodnih kola radi formiranja ciljnih amplituda.
• Hronologija i validacija: Podaci se učitavaju hronološki bez mešanja. Hiperparametar širine kernela (\(h\)) bira se hronološkom validacijom (minimizacija negativnog log-verodostojnosti / KL-divergencije), zatim se meri rezultat na odvojenom test skupu, a završni model se trenira na svim dostupnim istorijskim prelazima pre generisanja NEXT kombinacije.
• Izbor 7 brojeva bez ponavljanja: Umesto zaokruživanja proseka ili kozmetičkih filtera, kombinacija se dobija sekvencijalnim izborom maksimalne verovatnoće iz sedam kvantnih registara (uz maskiranje već izabranog broja i renormalizaciju preostalih amplituda).
"""
