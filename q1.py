import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.optimize import linear_sum_assignment

# ============================================================
# Parameters
# ============================================================

N = 20
P = 0.20

DT = 0.06

# Formation-control gains
KP = 4.0
KC = 0.15

# Number of movement frames for each letter
STEPS_PER_LETTER = 100

# Pause after each completed letter
HOLD_FRAMES = 15

# Your name
NAME = "STALIN"

rng = np.random.default_rng(10)


# ============================================================
# Generate a connected Erdos-Renyi graph
# ============================================================

while True:

    G = nx.erdos_renyi_graph(
        N,
        P,
        seed=int(rng.integers(1, 10**9))
    )

    if nx.is_connected(G):
        break

A = nx.to_numpy_array(G)


# ============================================================
# Define the letters explicitly
# ============================================================

LETTERS = {

    # --------------------------------------------------------
    # S
    # --------------------------------------------------------
    "S": [
        [
            (1.6, 1.8),
            (-1.5, 1.8),
            (-1.5, 0.5),
            (0.0, 0.0),
            (1.5, -0.5),
            (1.5, -1.8),
            (-1.6, -1.8)
        ]
    ],

    # --------------------------------------------------------
    # T
    # --------------------------------------------------------
    "T": [
        [
            (-1.7, 1.8),
            (1.7, 1.8)
        ],
        [
            (0.0, 1.8),
            (0.0, -1.8)
        ]
    ],

    # --------------------------------------------------------
    # A
    # --------------------------------------------------------
    "A": [
        [
            (-1.6, -1.8),
            (0.0, 1.8),
            (1.6, -1.8)
        ],
        [
            (-0.9, -0.3),
            (0.9, -0.3)
        ]
    ],

    # --------------------------------------------------------
    # L
    # --------------------------------------------------------
    "L": [
        [
            (-1.3, 1.8),
            (-1.3, -1.8),
            (1.5, -1.8)
        ]
    ],

    # --------------------------------------------------------
    # I
    # --------------------------------------------------------
    "I": [
        [
            (-1.4, 1.8),
            (1.4, 1.8)
        ],
        [
            (0.0, 1.8),
            (0.0, -1.8)
        ],
        [
            (-1.4, -1.8),
            (1.4, -1.8)
        ]
    ],

    # --------------------------------------------------------
    # N
    # --------------------------------------------------------
    "N": [
        [
            (-1.5, -1.8),
            (-1.5, 1.8),
            (1.5, -1.8),
            (1.5, 1.8)
        ]
    ]
}


# ============================================================
# Uniformly sample points along a polyline
# ============================================================

def sample_polyline(points, n_points):

    points = np.asarray(
        points,
        dtype=float
    )

    if len(points) == 1:

        return np.repeat(
            points,
            n_points,
            axis=0
        )

    segment_lengths = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1
    )

    cumulative = np.concatenate(
        ([0.0], np.cumsum(segment_lengths))
    )

    total_length = cumulative[-1]

    if total_length == 0:

        return np.repeat(
            points[:1],
            n_points,
            axis=0
        )

    distances = np.linspace(
        0.0,
        total_length,
        n_points
    )

    sampled = []

    for distance in distances:

        segment = (
            np.searchsorted(
                cumulative,
                distance,
                side="right"
            )
            - 1
        )

        segment = min(
            segment,
            len(points) - 2
        )

        start = points[segment]
        end = points[segment + 1]

        start_distance = cumulative[
            segment
        ]

        segment_length = segment_lengths[
            segment
        ]

        if segment_length == 0:

            alpha = 0.0

        else:

            alpha = (
                distance
                - start_distance
            ) / segment_length

        point = (
            start
            +
            alpha * (end - start)
        )

        sampled.append(point)

    return np.asarray(
        sampled
    )


# ============================================================
# Create exactly N target points for a letter
# ============================================================

def create_letter(letter, n_points=20):

    strokes = LETTERS[letter]

    # Find length of every stroke
    stroke_lengths = []

    for stroke in strokes:

        stroke = np.asarray(
            stroke,
            dtype=float
        )

        if len(stroke) > 1:

            length = np.sum(
                np.linalg.norm(
                    np.diff(
                        stroke,
                        axis=0
                    ),
                    axis=1
                )
            )

        else:

            length = 1.0

        stroke_lengths.append(
            length
        )

    stroke_lengths = np.asarray(
        stroke_lengths
    )

    # Allocate points according to stroke length
    raw_counts = (
        n_points
        *
        stroke_lengths
        /
        stroke_lengths.sum()
    )

    counts = np.floor(
        raw_counts
    ).astype(int)

    # Every stroke gets at least 2 points
    counts[counts < 2] = 2

    # Adjust total to exactly n_points
    while counts.sum() < n_points:

        index = np.argmax(
            raw_counts - counts
        )

        counts[index] += 1

    while counts.sum() > n_points:

        candidates = np.where(
            counts > 2
        )[0]

        index = candidates[
            np.argmax(
                counts[candidates]
            )
        ]

        counts[index] -= 1

    # Sample points
    result = []

    for stroke, count in zip(
        strokes,
        counts
    ):

        sampled = sample_polyline(
            stroke,
            count
        )

        result.append(
            sampled
        )

    result = np.vstack(
        result
    )

    # Center the letter
    result -= result.mean(
        axis=0
    )

    # Scale all letters similarly
    width = np.ptp(
        result[:, 0]
    )

    height = np.ptp(
        result[:, 1]
    )

    scale = max(
        width,
        height
    )

    if scale > 0:

        result *= (
            3.8 / scale
        )

    return result


# ============================================================
# Generate target formations for STALIN
# ============================================================

target_formations = []

for letter in NAME:

    target = create_letter(
        letter,
        N
    )

    target_formations.append(
        target
    )


# ============================================================
# Initial random positions
# ============================================================

positions = rng.uniform(
    -6.0,
    6.0,
    size=(N, 2)
)


# ============================================================
# Store complete trajectory
# ============================================================

trajectory = []
target_history = []
letter_indices = []


# ============================================================
# Formation-control simulation
# ============================================================

for letter_index, letter in enumerate(NAME):

    target = target_formations[
        letter_index
    ]

    # --------------------------------------------------------
    # Assign the nearest possible target point to each agent
    # --------------------------------------------------------

    cost_matrix = np.sum(
        (
            positions[:, None, :]
            -
            target[None, :, :]
        ) ** 2,
        axis=2
    )

    row_ind, col_ind = (
        linear_sum_assignment(
            cost_matrix
        )
    )

    assigned_target = np.zeros_like(
        target
    )

    for r, c in zip(
        row_ind,
        col_ind
    ):

        assigned_target[r] = target[c]

    # --------------------------------------------------------
    # Move agents toward this letter
    # --------------------------------------------------------

    for step in range(
        STEPS_PER_LETTER
    ):

        # Direct formation error
        formation_error = (
            positions
            -
            assigned_target
        )

        # Relative formation error
        relative_error = np.zeros_like(
            positions
        )

        for i in range(N):

            neighbors = np.nonzero(
                A[i]
            )[0]

            for j in neighbors:

                relative_error[i] += (
                    (
                        positions[i]
                        -
                        positions[j]
                    )
                    -
                    (
                        assigned_target[i]
                        -
                        assigned_target[j]
                    )
                )

        # Formation controller
        control = (
            -KP * formation_error
            -
            KC * relative_error
        )

        # Update positions
        positions = (
            positions
            +
            DT * control
        )

        # Save frame
        trajectory.append(
            positions.copy()
        )

        target_history.append(
            assigned_target.copy()
        )

        letter_indices.append(
            letter_index
        )

    # --------------------------------------------------------
    # Hold the completed letter
    # --------------------------------------------------------

    for _ in range(
        HOLD_FRAMES
    ):

        trajectory.append(
            positions.copy()
        )

        target_history.append(
            assigned_target.copy()
        )

        letter_indices.append(
            letter_index
        )


# ============================================================
# Convert to arrays
# ============================================================

trajectory = np.asarray(
    trajectory
)

target_history = np.asarray(
    target_history
)

letter_indices = np.asarray(
    letter_indices
)


# ============================================================
# Animation setup
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.set_xlim(
    -6,
    6
)

ax.set_ylim(
    -5,
    5
)

ax.set_aspect(
    "equal"
)

ax.set_xlabel(
    "x"
)

ax.set_ylabel(
    "y"
)

ax.set_title(
    "Multi-Agent Formation Control: STALIN"
)


# Actual agents
scatter = ax.scatter(
    trajectory[0, :, 0],
    trajectory[0, :, 1],
    s=65
)


# Target positions
target_scatter = ax.scatter(
    target_history[0, :, 0],
    target_history[0, :, 1],
    s=25,
    alpha=0.25
)


# Letter label
letter_text = ax.text(
    0.03,
    0.94,
    "",
    transform=ax.transAxes,
    fontsize=14
)


# ============================================================
# Animation update function
# ============================================================

def update(frame):

    # Update actual agent locations
    scatter.set_offsets(
        trajectory[frame]
    )

    # Update target positions
    target_scatter.set_offsets(
        target_history[frame]
    )

    # Current letter
    current_index = (
        letter_indices[frame]
    )

    current_letter = NAME[
        current_index
    ]

    letter_text.set_text(
        "Target letter: "
        +
        current_letter
    )

    return (
        scatter,
        target_scatter,
        letter_text
    )


# ============================================================
# Create animation
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    frames=len(trajectory),
    interval=40,
    blit=True
)


# ============================================================
# Save animation
# ============================================================

writer = FFMpegWriter(
    fps=25
)

animation.save(
    "formation_animation.mp4",
    writer=writer
)

plt.show()