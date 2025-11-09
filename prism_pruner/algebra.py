"""Algebra utilities."""

from typing import Sequence

import numpy as np

from prism_pruner.typing import Array1D_float, Array2D_float, Array3D_float


def normalize(vec: Array1D_float) -> Array1D_float:
    """Normalize a vector."""
    return vec / np.linalg.norm(vec)


def vec_angle(v1: Array1D_float, v2: Array1D_float) -> float:
    """Return the planar angle defined by two 3D vectors."""
    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(
                        v1 / np.linalg.norm(v1),
                        v2 / np.linalg.norm(v2),
                    ),
                    -1.0,
                    1.0,
                ),
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
    b1 /= np.linalg.norm(b1)

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
    pointer = pointer / np.linalg.norm(pointer)
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


def get_inertia_moments(coords: Array3D_float, masses: Array1D_float) -> Array1D_float:
    """Compute the principal moments of inertia of a molecule.

    Returns a length-3 array [I_x, I_y, I_z], sorted ascending.
    """
    # Shift to center of mass
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
    coords = coords - com

    # Compute inertia tensor
    norms_sq = np.einsum("ni,ni->n", coords, coords)
    total = np.sum(masses * norms_sq)
    I_matrix = total * np.eye(3) - np.einsum("n,ni,nj->ij", masses, coords, coords)

    # Principal moments via symmetric eigendecomposition
    moments, _ = np.linalg.eigh(I_matrix)

    return np.sort(moments)


def diagonalize(a: Array2D_float) -> Array2D_float:
    """Build the diagonalized matrix."""
    eigenvalues_of_a, eigenvectors_of_a = np.linalg.eig(a)
    b = eigenvectors_of_a[:, np.abs(eigenvalues_of_a).argsort()]
    return np.dot(np.linalg.inv(b), np.dot(a, b))  # type: ignore[no-any-return]


def get_alignment_matrix(p: Array1D_float, q: Array1D_float) -> Array2D_float:
    """
    Build the rotation matrix that aligns vectors q to p (Kabsch algorithm).

    Assumes centered vector sets (i.e. their mean is the origin).
    """
    # calculate the covariance matrix
    cov_mat = p.T @ q

    # Compute the SVD
    v, _, w = np.linalg.svd(cov_mat)

    # Ensure proper rotation (det = 1, not -1)
    if np.linalg.det(v) * np.linalg.det(w) < 0.0:
        v[:, -1] *= -1

    return v @ w  # type: ignore[no-any-return]
