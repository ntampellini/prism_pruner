"""Algebra utilities."""

from typing import Sequence, cast

import numba as nb
import numpy as np

from prism_pruner.typing import Array1D_float, Array1D_int, Array2D_float, Array3D_float, F


def njit_typed(func: F) -> F:
    """Mypy wrapper preserving type information."""
    return cast("F", nb.njit(func))


def njit_fastmath_typed(func: F) -> F:
    """Mypy wrapper preserving type information."""
    return cast("F", nb.njit(func, fastmath=True))


@njit_typed
def norm(vec: Array1D_int) -> Array1D_int:
    """
    Normalize a vector (3D only).

    Note: a tad faster than Numpy version.
    """
    return vec / np.sqrt((vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]))  # type: ignore[no-any-return]


@njit_typed
def norm_of(vec: Array1D_int) -> float:
    """
    Norm of a vector (3D only).

    Note: a tad faster than Numpy version
    """
    return float(np.sqrt((vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])))


@njit_typed
def vec_angle(v1: Array1D_int, v2: Array1D_int) -> float:
    """Return the planar angle defined by two 3D vectors."""
    return float(
        np.degrees(
            np.arccos(
                clip(np.dot(norm(v1), norm(v2)), -1.0, 1.0),
            )
        )
    )


@njit_typed
def clip[T: (np.floating)](n: T, lower: T, higher: T) -> T:
    """Jittable version of np.clip for single values."""
    if n > higher:
        return higher
    elif n < lower:
        return lower

    return n


@njit_typed
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


@njit_typed
def rot_mat_from_pointer(pointer: Array1D_int, angle: float) -> Array2D_float:
    """
    Get the rotation matrix from the rotation pivot using a quaternion.

    :param pointer: 3D vector representing the rotation pivot
    :param angle: rotation angle in degrees
    :return rotation_matrix: matrix that applied to a point, rotates it along the pointer
    """
    assert pointer.shape[0] == 3

    angle_2 = np.radians(angle) / 2
    sin = np.sin(angle_2)
    pointer = norm(pointer)
    return quaternion_to_rotation_matrix(
        [
            sin * pointer[0],
            sin * pointer[1],
            sin * pointer[2],
            np.cos(angle_2),
        ]
    )


@njit_typed
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


@njit_fastmath_typed
def all_dists(a: Array2D_float, b: Array2D_float) -> Array2D_float:
    """Return a 2D matrix of all the pairwise distances between the two vector sets."""
    assert a.shape[1] == b.shape[1]
    C = np.empty((a.shape[0], b.shape[0]), a.dtype)
    I_BLK = 32
    J_BLK = 32

    # workaround to get the right datatype for acc
    init_val_arr = np.zeros(1, a.dtype)
    init_val = init_val_arr[0]

    # Blocking and partial unrolling
    # Beneficial if the second dimension is large -> computationally bound problem
    for ii in nb.prange(a.shape[0] // I_BLK):
        for jj in range(b.shape[0] // J_BLK):
            for i in range(I_BLK // 4):
                for j in range(J_BLK // 2):
                    acc_0 = init_val
                    acc_1 = init_val
                    acc_2 = init_val
                    acc_3 = init_val
                    acc_4 = init_val
                    acc_5 = init_val
                    acc_6 = init_val
                    acc_7 = init_val
                    for k in range(a.shape[1]):
                        acc_0 += (a[ii * I_BLK + i * 4 + 0, k] - b[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_1 += (a[ii * I_BLK + i * 4 + 0, k] - b[jj * J_BLK + j * 2 + 1, k]) ** 2
                        acc_2 += (a[ii * I_BLK + i * 4 + 1, k] - b[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_3 += (a[ii * I_BLK + i * 4 + 1, k] - b[jj * J_BLK + j * 2 + 1, k]) ** 2
                        acc_4 += (a[ii * I_BLK + i * 4 + 2, k] - b[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_5 += (a[ii * I_BLK + i * 4 + 2, k] - b[jj * J_BLK + j * 2 + 1, k]) ** 2
                        acc_6 += (a[ii * I_BLK + i * 4 + 3, k] - b[jj * J_BLK + j * 2 + 0, k]) ** 2
                        acc_7 += (a[ii * I_BLK + i * 4 + 3, k] - b[jj * J_BLK + j * 2 + 1, k]) ** 2
                    C[ii * I_BLK + i * 4 + 0, jj * J_BLK + j * 2 + 0] = np.sqrt(acc_0)
                    C[ii * I_BLK + i * 4 + 0, jj * J_BLK + j * 2 + 1] = np.sqrt(acc_1)
                    C[ii * I_BLK + i * 4 + 1, jj * J_BLK + j * 2 + 0] = np.sqrt(acc_2)
                    C[ii * I_BLK + i * 4 + 1, jj * J_BLK + j * 2 + 1] = np.sqrt(acc_3)
                    C[ii * I_BLK + i * 4 + 2, jj * J_BLK + j * 2 + 0] = np.sqrt(acc_4)
                    C[ii * I_BLK + i * 4 + 2, jj * J_BLK + j * 2 + 1] = np.sqrt(acc_5)
                    C[ii * I_BLK + i * 4 + 3, jj * J_BLK + j * 2 + 0] = np.sqrt(acc_6)
                    C[ii * I_BLK + i * 4 + 3, jj * J_BLK + j * 2 + 1] = np.sqrt(acc_7)

        # Remainder j
        for i in range(I_BLK):
            for j in range((b.shape[0] // J_BLK) * J_BLK, b.shape[0]):
                acc_0 = init_val
                for k in range(a.shape[1]):
                    acc_0 += (a[ii * I_BLK + i, k] - b[j, k]) ** 2
                C[ii * I_BLK + i, j] = np.sqrt(acc_0)

    # Remainder i
    for i in range((a.shape[0] // I_BLK) * I_BLK, a.shape[0]):
        for j in range(b.shape[0]):
            acc_0 = init_val
            for k in range(a.shape[1]):
                acc_0 += (a[i, k] - b[j, k]) ** 2
            C[i, j] = np.sqrt(acc_0)

    return C


@njit_typed
def kronecker_delta(i: int, j: int) -> int:
    """Kronecker delta."""
    return int(i == j)


@njit_typed
def get_inertia_moments(coords: Array3D_float, masses: Array1D_float) -> Array1D_float:
    """
    Find the moments of inertia of the three principal axes.

    :return: diagonal of the diagonalized inertia tensor, that is
    a shape (3,) array with the moments of inertia along the main axes.
    (I_x, I_y and largest I_z last)
    """
    coords -= center_of_mass(coords, masses)
    inertia_moment_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    for i in range(3):
        for j in range(3):
            k = kronecker_delta(i, j)
            inertia_moment_matrix[i][j] = sum(
                [
                    masses[n] * ((norm_of(coords[n]) ** 2) * k - coords[n][i] * coords[n][j])
                    for n, _ in enumerate(coords)
                ]
            )

    inertia_moment_matrix = diagonalize(inertia_moment_matrix)

    return np.diag(inertia_moment_matrix)


@njit_typed
def diagonalize(a: Array2D_float) -> Array2D_float:
    """Build the diagonalized matrix."""
    eigenvalues_of_a, eigenvectors_of_a = np.linalg.eig(a)
    b = eigenvectors_of_a[:, np.abs(eigenvalues_of_a).argsort()]
    return np.dot(np.linalg.inv(b), np.dot(a, b))  # type: ignore[no-any-return]


@njit_typed
def center_of_mass(coords: Array3D_float, masses: Array1D_float) -> Array1D_float:
    """Find the center of mass for the atomic system."""
    total_mass = sum([masses[i] for i in range(len(coords))])
    w = np.array([0.0, 0.0, 0.0])
    for i in range(len(coords)):
        w += coords[i] * masses[i]
    return w / total_mass  # type: ignore[no-any-return]


@njit_typed
def get_moi_deviation_vec(
    coords1: Array2D_float, coords2: Array2D_float, masses: Array1D_float
) -> Array1D_float:
    """Determine the relative difference of the three principal axes moments of inertia."""
    im_1 = get_inertia_moments(coords1, masses)
    im_2 = get_inertia_moments(coords2, masses)

    return np.abs(im_1 - im_2) / im_1


@njit_typed
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
