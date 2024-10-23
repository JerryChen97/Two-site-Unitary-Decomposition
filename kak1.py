"""
Differentiable KAK decomposition for two-qubit unitaries.
Modified to ensure gradient computation works with autograd.

Ref: https://arxiv.org/pdf/quant-ph/0507171
"""

from typing import List, Optional, Tuple

import pennylane as qml
from pennylane import math
from pennylane import numpy as np


def magic_basis() -> math.ndarray:
    """Returns the magic basis matrix E that transforms between computational and Bell basis."""
    return math.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / math.sqrt(2)


def pauli_matrices() -> Tuple[math.ndarray, math.ndarray, math.ndarray]:
    """Returns the Pauli matrices X, Y, Z."""
    X = math.array([[0, 1], [1, 0]])
    Y = math.array([[0, -1j], [1j, 0]])
    Z = math.array([[1, 0], [0, -1]])
    return X, Y, Z


def kron_pauli() -> Tuple[math.ndarray, math.ndarray, math.ndarray]:
    """Returns tensor products X⊗X, Y⊗Y, Z⊗Z."""
    X, Y, Z = pauli_matrices()
    XX = math.kron(X, X)
    YY = math.kron(Y, Y)
    ZZ = math.kron(Z, Z)
    return XX, YY, ZZ


def get_canonical_parameters_direct(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes KAK parameters directly without eigendecomposition.
    Modified to work with autograd.
    """
    # Transform to magic basis
    E = magic_basis()
    Edag = E.conj().T
    M = Edag @ U @ E

    # Get Pauli matrices
    XX, YY, ZZ = kron_pauli()

    # Compute parameters using traces
    tx = np.trace(M @ XX) / 4
    ty = np.trace(M @ YY) / 4
    tz = np.trace(M @ ZZ) / 4

    # Extract angles using a more stable method
    # Use real part and avoid explicit float conversion
    alpha_x = np.arccos(np.clip(np.abs(np.real(tx)), -1, 1))
    alpha_y = np.arccos(np.clip(np.abs(np.real(ty)), -1, 1))
    alpha_z = np.arccos(np.clip(np.abs(np.real(tz)), -1, 1))

    # Create array of parameters
    alphas = np.stack([alpha_x, alpha_y, alpha_z])

    # Sort parameters (maintaining gradients)
    sorted_indices = np.argsort(-np.abs(alphas))
    alphas = np.stack([alphas[i] for i in sorted_indices])

    # Scale parameters if needed
    scale = np.where(alphas[0] > np.pi / 4, np.pi / (4 * alphas[0]), 1.0)
    alphas = alphas * scale

    return alphas[0], alphas[1], alphas[2]


def project_to_weyl_chamber(alphas: math.ndarray) -> math.ndarray:
    """Projects parameters into the Weyl chamber."""
    # Sort in descending order
    alphas = math.sort(math.abs(alphas))[::-1]

    # Ensure first parameter doesn't exceed π/4
    if alphas[0] > np.pi / 4.0:
        alphas = (np.pi / 4.0) * alphas / alphas[0]

    return alphas


def zyz_decomposition(U: math.ndarray) -> Tuple[float, float, float]:
    """
    Extracts ZYZ Euler angles from a 2x2 unitary matrix.
    More stable than using arctrigonometric functions directly.
    """
    # Ensure U is unitary
    U = U / math.sqrt(math.abs(math.linalg.det(U)))

    # Extract angles
    beta = 2 * math.arctan2(math.abs(U[1, 0]), math.abs(U[0, 0]))
    gamma = math.angle(U[1, 0]) - math.angle(U[0, 0])
    alpha = math.angle(U[1, 1]) - math.angle(U[0, 1])

    return alpha, beta, gamma


def kak_decomposition(U: np.ndarray) -> Tuple[List[qml.operation.Operation], np.ndarray]:
    """
    Performs KAK decomposition of a two-qubit unitary, returning quantum circuit.
    Modified to work with autograd types.
    """
    # Get canonical parameters
    alpha_x, alpha_y, alpha_z = get_canonical_parameters_direct(U)

    # Create circuit
    ops = []

    # Add operations (no explicit float conversion)
    if not np.allclose(alpha_x, 0):
        ops.append(qml.IsingXX(2 * alpha_x, wires=[0, 1]))
    if not np.allclose(alpha_y, 0):
        ops.append(qml.IsingYY(2 * alpha_y, wires=[0, 1]))
    if not np.allclose(alpha_z, 0):
        ops.append(qml.IsingZZ(2 * alpha_z, wires=[0, 1]))

    return ops, np.stack([alpha_x, alpha_y, alpha_z])


def get_full_matrix(op):
    """Gets the full 4x4 matrix representation of an operation."""
    mat = op.matrix()

    # If it's a single-qubit operation, we need to properly expand it
    if mat.shape == (2, 2):
        # Get identity for the other qubit
        I = math.eye(2)

        # Tensor product depends on which wire the operation acts on
        if op.wires[0] == 0:
            return math.kron(mat, I)
        else:
            return math.kron(I, mat)

    return mat


def test_decomposition_accuracy(weights):
    """Tests if the decomposition recreates the original unitary."""
    U_original = circX(weights[0], weights[1], weights[2])

    # Get decomposition
    ops, alphas = kak_decomposition(U_original)

    # Reconstruct unitary from decomposition
    U_reconstructed = np.eye(4)
    for op in reversed(ops):
        U_reconstructed = get_full_matrix(op) @ U_reconstructed

    # Compare
    fidelity = np.abs(np.trace(U_original.conj().T @ U_reconstructed)) / 4
    return np.real(fidelity)


def circX(a, b, g):
    """Creates a parameterized two-qubit unitary matrix."""
    l0 = 2 * a + g
    l1 = -2 * b - g
    l2 = -2 * a + g
    l3 = 2 * b - g

    e0 = 0.25 * np.exp(-1.0j * l0)
    e1 = 0.25 * np.exp(-1.0j * l1)
    e2 = 0.25 * np.exp(-1.0j * l2)
    e3 = 0.25 * np.exp(-1.0j * l3)

    e11 = e0 + e1 + e2 + e3
    e12 = e0 - 1.0j * e1 - e2 + 1.0j * e3
    e13 = e0 - e1 + e2 - e3
    e14 = e0 + 1.0j * e1 - e2 - 1.0j * e3

    return np.array(
        [[e11, e12, e13, e14], [e14, e11, e12, e13], [e13, e14, e11, e12], [e12, e13, e14, e11]]
    )


# Define quantum device
dev = qml.device("lightning.qubit", wires=2)


@qml.qnode(dev, interface="autograd")
def circuit(weights):
    """Enhanced circuit with more flexible structure."""
    # Initial state preparation
    qml.RY(np.pi / 4, wires=0)

    # First layer of single-qubit rotations
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=0)
    qml.RZ(weights[2], wires=0)

    # Two-qubit entangling operations
    qml.IsingXX(weights[3], wires=[0, 1])
    qml.IsingYY(weights[4], wires=[0, 1])
    qml.IsingZZ(weights[5], wires=[0, 1])

    # Final layer of single-qubit rotations
    qml.RX(weights[6], wires=0)
    qml.RY(weights[7], wires=0)
    qml.RZ(weights[8], wires=0)

    # Return multiple measurements
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0))]


def cost_function(weights):
    """Enhanced cost function with weighted penalties."""
    measurements = circuit(weights)
    targets = [0.0, 1.0, 0.0]  # Target: |+⟩ state

    # Add higher weight to the X measurement to prioritize getting it right
    penalties = [1.0, 2.0, 1.0]
    return sum(p * (m - t) ** 2 for m, t, p in zip(measurements, targets, penalties))


def check_gradient(weights):
    """Check if gradients are being computed correctly."""
    grad = qml.grad(cost_function)(weights)
    print("Gradient:", grad)
    return grad


if __name__ == "__main__":
    # Initialize weights with smaller values
    np.random.seed(0)
    weights = 0.1 * np.random.randn(9, requires_grad=True)

    # Test initial conditions
    print("Initial cost:", cost_function(weights))
    print("Initial measurements:", circuit(weights))
    print("Initial gradient:", check_gradient(weights))

    # Optimization with adaptive learning rate
    opt = qml.AdamOptimizer(0.01)  # Switch to Adam optimizer

    print("\nStarting optimization:")
    costs = []
    best_cost = float("inf")
    best_weights = None
    patience = 50  # Number of iterations to wait for improvement
    no_improvement = 0

    for it in range(500):  # More iterations
        if (it + 1) % 20 == 0:
            cost = cost_function(weights)
            costs.append(cost)
            measurements = circuit(weights)
            grad_norm = np.linalg.norm(check_gradient(weights))

            print(f"Iter: {it + 1:5d} | Cost: {cost:0.7f} | Grad norm: {grad_norm:0.7f}")
            print(f"Measurements: {measurements}")

            # Check for improvement
            if cost < best_cost:
                best_cost = cost
                best_weights = weights.copy()
                no_improvement = 0
            else:
                no_improvement += 1

            # Early stopping
            if no_improvement >= patience:
                print(f"\nStopping early due to no improvement for {patience} iterations")
                weights = best_weights
                break

        try:
            weights = opt.step(cost_function, weights)
        except Exception as e:
            print(f"Optimization failed at iteration {it + 1}: {str(e)}")
            break

    # Final results
    print("\nFinal tests:")
    print("Final measurements:", circuit(weights))
    print("Final cost:", cost_function(weights))
    print("Final weights:", weights)

    if len(costs) > 1:
        print("\nCost progress:")
        for i, cost in enumerate(costs):
            print(f"Step {(i+1)*20}: {cost:0.7f}")
