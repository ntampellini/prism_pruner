"""Algebra utilities."""

from typing import Sequence, cast

import numpy as np

from prism_pruner.typing import Array1D_float, Array2D_float, Array3D_float


def norm_of(vec: Array1D_float) -> float:
    """Return the norm of the vector."""
    return cast("float", np.linalg.norm(vec, axis=None))


def normalize(vec: Array1D_float) -> Array1D_float:
    """Normalize a vector."""
    return vec / norm_of(vec)


def vec_angle(v1: Array1D_float, v2: Array1D_float) -> float:
    """Return the planar angle defined by two 3D vectors."""
    return float(
        np.degrees(
            np.arccos(
                np.clip(np.dot(normalize(v1), normalize(v2)), -1.0, 1.0),
            )
        )
    )


def dihedral(p: Array2D_float) -> float:
    """
    Find dihedral angle in degrees from 4 3D vecs.

    Praxeolitic formula: 1 sqrt, 1 cross product.
    """
    p0, p1, p2, p3 = p

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= norm_of(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return float(np.degrees(np.arctan2(y, x)))


def rot_mat_from_pointer(pointer: Array1D_float, angle: float) -> Array2D_float:
    """
    Get the rotation matrix from the rotation pivot using a quaternion.

    :param pointer: 3D vector representing the rotation pivot
    :param angle: rotation angle in degrees
    :return rotation_matrix: matrix that applied to a point, rotates it along the pointer
    """
    assert pointer.shape[0] == 3

    angle_2 = np.radians(angle) / 2
    sin = np.sin(angle_2)
    pointer = normalize(pointer)
    return quaternion_to_rotation_matrix(
        [
            sin * pointer[0],
            sin * pointer[1],
            sin * pointer[2],
            np.cos(angle_2),
        ]
    )


def quaternion_to_rotation_matrix(quat: Array1D_float | Sequence[float]) -> Array2D_float:
    """
    Convert a quaternion into a full three-dimensional rotation matrix.

    This rotation matrix converts a point in the local reference frame to a
    point in the global reference frame.

    :param quat: 4-element array representing the quaternion (q0, q1, q2, q3)
    :return: 3x3 element array representing the full 3D rotation matrix
    """
    # Extract the values from Q (adjusting for scalar last in input)
    q1, q2, q3, q0 = quat

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])


def kronecker_delta(i: int, j: int) -> int:
    """Kronecker delta."""
    return int(i == j)


def get_inertia_moments(coords: Array3D_float, masses: Array1D_float) -> Array1D_float:
    """
    Find the moments of inertia of the three principal axes.

    :return: diagonal of the diagonalized inertia tensor, that is
    a shape (3,) array with the moments of inertia along the main axes.
    (I_x, I_y and largest I_z last)
    """
    coords -= center_of_mass(coords, masses)
    norms_of_coords = np.linalg.norm(coords, axis=1, keepdims=True)
    inertia_moment_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    for i in range(3):
        for j in range(3):
            k = kronecker_delta(i, j)
            inertia_moment_matrix[i][j] = np.sum(
                [
                    masses[n] * ((norms_of_coords[n] ** 2) * k - coords[n][i] * coords[n][j])
                    for n, _ in enumerate(coords)
                ]
            )

    inertia_moment_matrix = diagonalize(inertia_moment_matrix)

    return np.diag(inertia_moment_matrix)


def diagonalize(a: Array2D_float) -> Array2D_float:
    """Build the diagonalized matrix."""
    eigenvalues_of_a, eigenvectors_of_a = np.linalg.eig(a)
    b = eigenvectors_of_a[:, np.abs(eigenvalues_of_a).argsort()]
    return np.dot(np.linalg.inv(b), np.dot(a, b))  # type: ignore[no-any-return]


def center_of_mass(coords: Array3D_float, masses: Array1D_float) -> Array1D_float:
    """Find the center of mass for the atomic system."""
    total_mass = sum([masses[i] for i in range(len(coords))])
    w = np.array([0.0, 0.0, 0.0])
    for i in range(len(coords)):
        w += coords[i] * masses[i]
    return w / total_mass  # type: ignore[no-any-return]


def get_alignment_matrix(p: Array1D_float, q: Array1D_float) -> Array2D_float:
    """
    Build the rotation matrix that aligns vectors q to p (Kabsch algorithm).

    Assumes centered vector sets (i.e. their mean is the origin).
    """
    # calculate the covariance matrix
    cov_mat = np.ascontiguousarray(p.T) @ q

    # Compute the SVD
    v, _, w = np.linalg.svd(cov_mat)

    if (np.linalg.det(v) * np.linalg.det(w)) < 0.0:
        v[:, -1] = -v[:, -1]

    return np.dot(v, w)  # type: ignore[no-any-return]
