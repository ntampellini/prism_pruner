"""PRISM - PRuning Interface for Similar Molecules."""

from typing import cast

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
    """Return the normalized vector.

    A tad faster than Numpy version.
    Only for 3D vectors.
    """
    norm: Array1D_int = vec / np.sqrt((vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]))

    return norm


@njit_typed
def norm_of(vec: Array1D_int) -> float:
    """Return the norm of the vector.

    A tad faster than Numpy version, but
    only compatible with 3D vectors.
    """
    norm_ang: float = np.sqrt((vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]))

    return norm_ang


@njit_typed
def vec_angle(v1: Array1D_int, v2: Array1D_int) -> float:
    """Return the planar angle defined by two 3D vectors."""
    v1_u = norm(v1)
    v2_u = norm(v2)
    angle_deg: float = (
        np.arccos(clip(np.dot(v1_u, v2_u), np.float32(-1.0), np.float32(+1.0))) * 180 / np.pi
    )
    return angle_deg


@njit_typed
def clip(n: np.float32, lower: np.float32, higher: np.float32) -> np.float32:
    """Jittable version of np.clip for single values."""
    if n > higher:
        return higher
    elif n < lower:
        return lower
    else:
        return n


@njit_typed
def dihedral(p: Array2D_float) -> float:
    """Return dihedral angle in degrees from 4 3D vecs.

    Praxeolitic formula: 1 sqrt, 1 cross product.
    """
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

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

    dihedral_deg: float = np.degrees(np.arctan2(y, x))

    return dihedral_deg


@njit_typed
def rot_mat_from_pointer(pointer: Array1D_int, angle: float) -> Array2D_float:
    """Get the rotation matrix from the rotation pivot.

    Return the rotation matrix that rotates a system around the given pointer
    of angle degrees. The algorithm is based on scipy quaternions.
    :params pointer: a 3D vector
    :params angle: an int/float, in degrees
    :return rotation_matrix: matrix that applied to a point, rotates it along the pointer
    """
    assert pointer.shape[0] == 3

    pointer = norm(pointer)
    angle *= np.pi / 180
    quat = np.array(
        [
            np.sin(angle / 2) * pointer[0],
            np.sin(angle / 2) * pointer[1],
            np.sin(angle / 2) * pointer[2],
            np.cos(angle / 2),
        ]
    )
    # normalized quaternion, scalar last (i j k w)

    return quaternion_to_rotation_matrix(quat)


@njit_typed
def quaternion_to_rotation_matrix(quat: Array1D_int) -> Array2D_float:
    """Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
    This rotation matrix converts a point in the local reference
    frame to a point in the global reference frame.
    """
    # Extract the values from Q (adjusting for scalar last in input)
    q0 = quat[3]
    q1 = quat[0]
    q2 = quat[1]
    q3 = quat[2]

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
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return np.ascontiguousarray(rot_matrix)


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
    if i == j:
        return 1
    return 0


@njit_typed
def get_inertia_moments(coords: Array3D_float, masses: Array1D_float) -> Array1D_int:
    """Return the moments of inertia of the three principal axes.

    Return the diagonal of the diagonalized inertia tensor, that is
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
    """Return the diagonalized matrix."""
    eigenvalues_of_a, eigenvectors_of_a = np.linalg.eig(a)
    b = eigenvectors_of_a[:, np.abs(eigenvalues_of_a).argsort()]
    diagonal_matrix: Array2D_float = np.dot(np.linalg.inv(b), np.dot(a, b))
    return diagonal_matrix


@njit_typed
def center_of_mass(coords: Array3D_float, masses: Array1D_float) -> Array1D_int:
    """Return the center of mass for the atomic system."""
    total_mass = sum([masses[i] for i in range(len(coords))])
    w = np.array([0.0, 0.0, 0.0])
    for i in range(len(coords)):
        w += coords[i] * masses[i]
    com: Array1D_int = w / total_mass
    return com


@njit_typed
def get_moi_deviation_vec(
    coords1: Array2D_float, coords2: Array2D_float, masses: Array1D_float
) -> Array1D_float:
    """Return the realative difference of the three principal axes moments of inertia."""
    im_1 = get_inertia_moments(coords1, masses)
    im_2 = get_inertia_moments(coords2, masses)

    vec: Array1D_float = np.abs(im_1 - im_2) / im_1

    return vec


@njit_typed
def get_alignment_matrix(p: Array1D_float, q: Array1D_float) -> Array2D_float:
    """Return the rotation matrix that aligns vectors q to p (Kabsch algorithm).

    Assumes centered vector sets (i.e. their mean is the origin).
    """
    # calculate the covariance matrix
    cov_mat = np.ascontiguousarray(p.T) @ q

    # Compute the SVD
    v, _, w = np.linalg.svd(cov_mat)
    d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0

    if d:
        v[:, -1] = -v[:, -1]

    # Create Rotation matrix u
    rot_mat: Array2D_float = np.dot(v, w)

    return rot_mat
