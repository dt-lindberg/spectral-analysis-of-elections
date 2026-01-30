from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "diversity_map_8_96"


def load_soc_matrix(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as soc_file:
        first_line = soc_file.readline().strip()
        if not first_line:
            raise ValueError(f"Empty .soc file: {path}")

        if first_line.startswith("#"):
            num_candidates = int(soc_file.readline().strip())
        else:
            num_candidates = int(first_line)

        for _ in range(num_candidates):
            soc_file.readline()

        header_parts = [
            part.strip() for part in soc_file.readline().split(",") if part.strip()
        ]
        if len(header_parts) < 3:
            raise ValueError(
                f"Unexpected header line in {path}: expected 3 fields, got {header_parts}"
            )

        num_voters = int(header_parts[0])
        num_options = int(header_parts[2])
        votes = np.zeros((num_voters, num_candidates), dtype=int)

        row = 0
        for _ in range(num_options):
            line_parts = [
                part.strip() for part in soc_file.readline().split(",") if part.strip()
            ]
            quantity = int(line_parts[0])
            ranking = np.array(line_parts[1 : 1 + num_candidates], dtype=int)
            for _ in range(quantity):
                votes[row] = ranking
                row += 1

        if row != num_voters:
            raise ValueError(f"Expected {num_voters} voters in {path}, loaded {row}.")

    return votes


def summarize_votes(votes: np.ndarray) -> None:
    num_voters, num_candidates = votes.shape
    num_unique_ballots = np.unique(votes, axis=0).shape[0]

    positions = np.empty_like(votes)
    for voter_index, ranking in enumerate(votes):
        positions[voter_index, ranking] = np.arange(num_candidates)

    print("Election statistics")
    print(f"- voters: {num_voters}")
    print(f"- candidates: {num_candidates}")
    print(f"- example preference ranking: {votes[39, :]}")
    print(f"- Number of unique ballots: {num_unique_ballots}")
    print(f"- matrix shape: {votes.shape}")


def generate_uniform_votes(num_candidates: int) -> np.ndarray:
    rankings = list(itertools.permutations(range(num_candidates)))
    return np.array(rankings, dtype=int)


def compute_preference(v1: np.ndarray, c0: int, c1: int) -> int:
    """Computes which of the two candidates c0,c1 that voter v1 prefers.

    Note that:
        * Candidates c0 and c1 are guaranteed to exist (uniquely) in the v1 array,
        * The candidates indices cannot be tied, but one is strictly lower.

    Approximate running time: O(n) where n is the length of v1
    """
    # np.flatnonzero returns the index of the entry that is non-zero (True)
    # v1 == c0 filters the v1 array to only contain True for entry c0
    # Together, they yield the index of c0
    c0_idx = np.flatnonzero(v1 == c0)[0]
    c1_idx = np.flatnonzero(v1 == c1)[0]
    return c0 if c0_idx < c1_idx else c1


def compute_swap_distance(v1: np.ndarray, v2: np.ndarray) -> int:
    """Given two preference rankings (voters), compute their swap distance"""

    # Sanity check
    assert v1.shape == v2.shape, (
        f"Expected preference rankings to have same shape, but got v1={v1.shape} and v2={v2.shape}"
    )

    swap_distance = 0

    # Iterate all m-choose-2 candidate pairs
    # As each candidate is an integer, we can iterate
    # (0,1), (0,2), ..., (0,m), (1, 2), (1,3), ..., (1, m), ..., (m-1, m)
    for c0 in range(0, v1.shape[-1]):
        for c1 in range(c0 + 1, v1.shape[-1]):
            # Check if the voters preference for c0,c1 are the same
            # Increment the swap distance if not
            preference_v1 = compute_preference(v1, c0, c1)
            preference_v2 = compute_preference(v2, c0, c1)
            if preference_v1 != preference_v2:
                swap_distance += 1

    return swap_distance


def compute_adjacency_matrix(votes: np.ndarray) -> np.ndarray:
    """Given an (N,M) numpy array of voters, returns the adjacency matrix A:

    $A = exp(
        (-d(v_i - v_j)^2) / (2*sigma^2)     for each i!=j, and A_ii=0.
        )$

        where $d(v_i, v_j)$ is the swap distance between voter i and j.

    Note about the normalization:
        If swap_distance=0, exp(-(0**2)/2) = exp(0) = 1
        If swap_distance=1, exp(-(1**2)/2) = exp(-1) = 1/e
        ...
        If swap_distance=C, exp(-(C**2)/2) -> 0 as C -> infty
    """
    sigma = 1.0
    num_voters = votes.shape[0]
    adjacency_matrix = np.zeros((num_voters, num_voters), dtype=float)

    for i in range(num_voters):
        for j in range(i + 1, num_voters):
            swap_distance = compute_swap_distance(votes[i], votes[j])
            similarity = np.exp(-(swap_distance**2) / (2 * sigma**2))
            adjacency_matrix[i, j], adjacency_matrix[j, i] = similarity, similarity

    return adjacency_matrix


def plot_adjacency_histogram(adjacency: np.ndarray, output_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bin_values, _, _ = ax1.hist(adjacency.flatten(), bins="auto")
    ax2.hist(adjacency.flatten(), bins="auto")

    target_index = min(9, len(bin_values) - 1)
    target_height = bin_values[target_index]
    if target_height == 0:
        target_height = 1

    ax2.set_ylim(0, target_height)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_eigenpair_histogram(
    eigenvector: np.ndarray,
    eigenvalue: float,
    index: int,
    election_id: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(eigenvector**2, bins="auto", color="#4C78A8", alpha=0.8)
    ax.set_xlabel("Eigenvector value (squared)")
    ax.set_ylabel("Count")

    header = f"Eigenvector {index} distribution"
    ax.set_title(header)
    details = f"Eigenvalue: {eigenvalue:.6f}\nElection: {election_id}\nIndex: {index}"
    ax.text(
        0.98,
        0.98,
        details,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#CCCCCC"},
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def compute_k_eigenpairs(
    adjacency: np.ndarray, k: int
) -> tuple[list[float], list[np.ndarray]]:
    """Computes the (symmetric) normalized graph Laplacian:

    $D^{-1/2}@L@D^{-1/2} = I - D^{-1/2}@W@D^{-1/2}$

    where:
        * D is the degree matrix,
        * L is the Laplacian,
        * I is the identity matrix,
        * W is the weighted adjacency matrix,
        * @ is the matrix product (between 2d arrays)
            (https://numpy.org/doc/stable/reference/routines.linalg.html#the-operator)
    """
    # Construct the degree matrix
    degree = adjacency.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt_degree = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)

    d_inv_sqrt = np.diag(inv_sqrt_degree)
    identity = np.eye(adjacency.shape[0])
    laplacian = identity - d_inv_sqrt @ adjacency @ d_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    order = np.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[order]
    eigenvectors_sorted = eigenvectors[:, order]

    k = min(k, eigenvalues_sorted.shape[0])
    values = eigenvalues_sorted[:k].tolist()
    vectors = [eigenvectors_sorted[:, idx] for idx in range(k)]

    return values, vectors


def compute_indices(eigenvalues: list[float]) -> tuple[float, float]:
    """Compute diversity and polarization indices based on eigenvalues.

    Spectral indices:
        * Polarization:     $min(1.0, lambda_2 - lambda_1)$
        * Diversity:        $1/N * sum_i((lambda_i - 1)^2)$

    Assumes eigenvalues are sorted in ascending order.
    """
    evals = np.array(eigenvalues)

    # Diversity: Mean Squared Deviation from 1.0
    diversity = float(np.mean((evals - 1.0) ** 2))

    # Polarization: Gap between 2nd and 3rd eigenvalue (indices 1 and 2)
    polarization = evals[2] - evals[1]
    polarization = float(min(1.0, polarization))

    return diversity, polarization


def plot_scree(eigenvalues_dict: dict[str, list[float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, values in eigenvalues_dict.items():
        ax.plot(values, label=label, marker="o", markersize=3, alpha=0.7)

    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue Magnitude")
    ax.set_title("Scree Plot of Graph Laplacian Eigenvalues")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_scree_grid(
    eigenvalues_by_m: dict[int, list[float]], output_path: Path
) -> None:
    num_plots = len(eigenvalues_by_m)
    cols = 2
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))

    axes_list = np.ravel(axes)
    for ax, (m, values) in zip(axes_list, sorted(eigenvalues_by_m.items())):
        ax.plot(values, marker="o", markersize=2, alpha=0.7)
        ax.set_title(f"Uniform election scree (m={m})")
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue Magnitude")
        ax.grid(True, alpha=0.3)

    for ax in axes_list[len(eigenvalues_by_m) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# if __name__ == "__main__":
#     min_m = 3
#     max_m = 6
#
#     eigenvalues_by_m = {}
#
#     for num_candidates in range(min_m, max_m + 1):
#         print(f"Processing uniform election with m={num_candidates}")
#         votes = generate_uniform_votes(num_candidates)
#         adjacency_matrix = compute_adjacency_matrix(votes)
#
#         k_eigenvalues = votes.shape[0]
#         eigenvalues, _ = compute_k_eigenpairs(adjacency_matrix, k_eigenvalues)
#         eigenvalues_by_m[num_candidates] = eigenvalues
#
#         # diversity_score, polarization_score = compute_indices(eigenvalues)
#         # print(f"\tSpectral Diversity:    {diversity_score:.6f}")
#         # print(f"\tSpectral Polarization: {polarization_score:.6f}")
#
#     output_plot_path = (
#         REPO_ROOT / "spectral-clustering" / "uniform_scree_grid_m3_m6.png"
#     )
#     plot_scree_grid(eigenvalues_by_m, output_plot_path)
#     print(f"Saved scree grid to: {output_plot_path}")

if __name__ == "__main__":
    # ELECTIONS_TO_COMPARE = ["ID", "AN", "Impartial Culture_2"]
    # ELECTIONS_TO_COMPARE = ["ID", "AN", "Impartial Culture_2", "10-Cube_5", "Circle_13"]
    ELECTIONS_TO_COMPARE = ["AN"]
    K_EIGENVALUES = 100
    eigenvalues_results = {}
    for election_id in ELECTIONS_TO_COMPARE:
        election_path = (
            REPO_ROOT
            / "dap"
            / "dap-code"
            / "experiments"
            / EXPERIMENT_ID
            / "elections"
            / f"{election_id}.soc"
        )
        if not election_path.exists():
            print(f"Warning: Election file not found: {election_path}")
            continue
        print(f"Processing election: {election_id}")
        votes = load_soc_matrix(election_path)
        # # select a subset of votes
        # k = 8
        # # print(votes[0 : k // 2, :], votes[-(k // 2 + 1) : -1, :])
        # votes = np.concatenate([votes[0 : k // 2, :], votes[-(k // 2 + 1) : -1, :]])
        # # print(votes)
        # assert 1 != 1, "Raised error"
        adjacency_matrix = compute_adjacency_matrix(votes)

        eigenvalues, eigenvectors = compute_k_eigenpairs(
            adjacency_matrix, K_EIGENVALUES
        )
        eigenvalues_results[election_id] = eigenvalues
        # print("Eigenvalues:")
        # print(eigenvalues, "\n")
        # print("Eigenvectors:")
        # print(eigenvectors)
        # Compute and print indices
        # diversity_score, polarization_score = compute_indices(eigenvalues)
        # print(f"\tSpectral Diversity:    {diversity_score:.6f}")
        # print(f"\tSpectral Polarization: {polarization_score:.6f}")
    # Get Scree plot
    output_plot_path = REPO_ROOT / "spectral-clustering" / "scree_plot_3.png"
    plot_scree(eigenvalues_results, output_plot_path)
    print(f"Saved scree plot to: {output_plot_path}")
