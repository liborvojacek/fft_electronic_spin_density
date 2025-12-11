"""
qe_d_orbital_wavefunctions.py

Utilities to build real-space d-orbital wavefunctions from a 5x5
Hubbard occupation matrix in Quantum ESPRESSO's real d-orbital basis.

Basis order (QE convention for 5 d orbitals):
    0: dz2
    1: dzx  (dxz)
    2: dzy  (dyz)
    3: dx2-y2
    4: dxy

Main things you might want:

- diagonalize_occupation(occ):
    → natural occupation numbers and orbitals in this basis.

- make_wavefunction_from_coeffs(coeffs, radial_func=None):
    → scalar function psi(x, y, z).

- get_hole_wavefunction(occ, radial_func=None):
    → psi_hole(x, y, z) for a "one-hole" situation (e.g. 4 electrons in 5 d).

- get_occupied_wavefunctions(occ, nelect=4, radial_func=None):
    → list of psi_i(x, y, z) for the nelect most-occupied orbitals.

Optional radial part:
    - hydrogenic_radial_3d(r, Z=1.0, n=3)

All angular dependence uses normalized real cubic harmonics.
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
# Optional: hydrogenic-like radial part for a 3d orbital
# ----------------------------------------------------------------------

def hydrogenic_radial_3d(r, Z: float = 12.0, n: int = 3):
    """
    Hydrogenic 3d radial wavefunction R_3d(r) for nuclear charge Z and
    principal quantum number n.

    Parameters
    ----------
    r : array-like
        Radial coordinate(s).
    Z : float
        Effective nuclear charge.
    n : int
        Principal quantum number (3 for 3d).

    Returns
    -------
    R : ndarray
        Radial part R_3d(r) (normalized in atomic-unit conventions).
    """
    r = np.array(r, dtype=float)
    rho = 2.0 * Z * r / n
    # R_3d = (1 / (9 * sqrt(30))) * rho^2 * Z^(3/2) * exp(-rho/2)
    return (1.0 / (9.0 * np.sqrt(30.0))) * (rho**2) * (Z**1.5) * np.exp(-rho / 2.0)


# ----------------------------------------------------------------------
# Real d-orbital cubic harmonics in QE's order:
# [dz2, dzx, dzy, dx2-y2, dxy]
# ----------------------------------------------------------------------

def cubic_harmonics_d_qe(x, y, z):
    """
    Real d-orbital cubic harmonics in Quantum ESPRESSO's convention:

        index  orbital
        -----  -------------------
        0      dz2
        1      dzx  (dxz)
        2      dzy  (dyz)
        3      dx2-y2
        4      dxy

    Input
    -----
    x, y, z : array-like or scalars
        Cartesian coordinates (broadcastable).

    Output
    ------
    Y : ndarray, shape (5, ...)
        Stack of [dz2, dzx, dzy, dx2-y2, dxy] evaluated at (x,y,z).
        These are normalized real spherical harmonics (purely angular).
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    r2 = x * x + y * y + z * z
    eps = 1e-18
    safe_r2 = np.where(r2 > eps, r2, eps)

    inv_r2 = 1.0 / safe_r2
    fac = 1.0 / np.sqrt(4.0 * np.pi)

    # Y_{2,0}  (dz2)  ∝ (3 z^2 - r^2) / r^2
    Y_dz2 = np.sqrt(5.0 / 4.0) * (3.0 * z * z - r2) * inv_r2 * fac

    # Y_{2,±1} real combinations → dxz, dyz ∝ xz / r^2, yz / r^2
    Y_dxz = np.sqrt(60.0 / 4.0) * x * z * inv_r2 * fac  # = sqrt(15) * xz / r^2 * fac
    Y_dyz = np.sqrt(60.0 / 4.0) * y * z * inv_r2 * fac  # = sqrt(15) * yz / r^2 * fac

    # Y_{2,±2} real combinations → dxy, dx2-y2
    Y_dxy = np.sqrt(15.0 / 4.0) * (2.0 * x * y) * inv_r2 * fac
    Y_dx2y2 = np.sqrt(15.0 / 4.0) * (x * x - y * y) * inv_r2 * fac

    # QE order: dz2, dzx, dzy, dx2-y2, dxy
    return np.stack([Y_dz2, Y_dxz, Y_dyz, Y_dx2y2, Y_dxy], axis=0)


# ----------------------------------------------------------------------
# Diagonalizing the occupation matrix: natural orbitals & occupations
# ----------------------------------------------------------------------

def diagonalize_occupation(occ: np.ndarray):
    """
    Diagonalize a 5x5 Hubbard occupation matrix in the QE d-orbital basis.

    Parameters
    ----------
    occ : array-like, shape (5, 5)
        Hermitian occupation matrix in the basis:
        [dz2, dzx, dzy, dx2-y2, dxy].

    Returns
    -------
    eigvals_sorted : ndarray, shape (5,)
        Natural occupation numbers n_i, sorted ascending (smallest first).
    eigvecs_sorted : ndarray, shape (5, 5)
        Corresponding natural orbitals (columns).
        Column j is the orbital with occupation eigvals_sorted[j],
        expressed as coefficients in the QE d basis.
    """
    occ = np.asarray(occ, dtype=complex)
    if occ.shape != (5, 5):
        raise ValueError(f"Expected a 5x5 matrix, got {occ.shape}")

    eigvals, eigvecs = np.linalg.eigh(occ)

    # Sort by occupation (ascending)
    order = np.argsort(eigvals.real)
    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]

    # Fix global phase of each orbital for reproducibility:
    for j in range(5):
        v = eigvecs_sorted[:, j]
        k = np.argmax(np.abs(v))
        phase = np.exp(-1j * np.angle(v[k]))
        eigvecs_sorted[:, j] = v * phase

    return eigvals_sorted, eigvecs_sorted


# ----------------------------------------------------------------------
# From orbital coefficients to a real-space wavefunction psi(x, y, z)
# ----------------------------------------------------------------------

def make_wavefunction_from_coeffs(coeffs: np.ndarray, center: tuple = (0, 0, 0), radial_func=hydrogenic_radial_3d):
    """
    Build a wavefunction psi(x, y, z) from a coefficient vector in QE's
    5-d-orbital basis.

    Parameters
    ----------
    coeffs : array-like, shape (5,)
        Coefficients in the QE d basis:
        [dz2, dzx, dzy, dx2-y2, dxy].
    radial_func : callable or None, optional
        Function R(r) for the radial dependence, r = sqrt(x^2 + y^2 + z^2).
        If None, the wavefunction is purely angular.

        Example:
            radial_func = lambda r: hydrogenic_radial_3d(r, Z=Z_eff)

    Returns
    -------
    psi : callable
        psi(x, y, z) → real array, same broadcast shape as x,y,z.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.shape != (5,):
        raise ValueError(f"Expected shape (5,), got {coeffs.shape}")

    def psi(x, y, z):
        x_arr = np.array(x, dtype=float) - center[0]
        y_arr = np.array(y, dtype=float) - center[1]
        z_arr = np.array(z, dtype=float) - center[2]

        r = np.sqrt(x_arr * x_arr + y_arr * y_arr + z_arr * z_arr)

        # Angular part in QE order
        ang = cubic_harmonics_d_qe(x_arr, y_arr, z_arr)  # shape (5, ...)

        if radial_func is None:
            R = 1.0
        else:
            R = radial_func(r)

        # Contract coefficients with angular basis:
        # coeffs: (5,), ang: (5, ...) → wf_ang: (...)
        wf_ang = np.tensordot(coeffs, ang, axes=(0, 0))
        return R * wf_ang

    return psi


# ----------------------------------------------------------------------
# Convenience wrappers for occupation matrices: hole & occupied orbitals
# ----------------------------------------------------------------------

def get_hole_coeffs(occ: np.ndarray):
    """
    Given a 5x5 Hubbard occupation matrix (e.g. trace ≈ 4 → one hole),
    return the coefficients for the hole orbital, i.e. the eigenvector
    with the smallest occupation number.

    Parameters
    ----------
    occ : array-like, shape (5,5)
        Hubbard occupation matrix in QE d basis.

    Returns
    -------
    c_hole : ndarray, shape (5,)
        Coefficients of the hole orbital in the QE d basis.
    eigvals : ndarray, shape (5,)
        Natural occupation numbers (ascending).
    """
    eigvals, eigvecs = diagonalize_occupation(occ)
    # Smallest occupation → hole
    c_hole = eigvecs[:, 0]
    return c_hole, eigvals


def get_occupied_coeffs(occ: np.ndarray, nelect: int = 4):
    """
    Given a 5x5 Hubbard occupation matrix, return the coefficients for
    the nelect most-occupied natural orbitals.

    Parameters
    ----------
    occ : array-like, shape (5,5)
        Hubbard occupation matrix in QE d basis.
    nelect : int
        Number of most-occupied orbitals to return.

    Returns
    -------
    c_occ : ndarray, shape (5, nelect)
        Columns are the nelect most-occupied orbitals (largest occupations).
    eigvals : ndarray, shape (5,)
        Natural occupation numbers (ascending).
    """
    eigvals, eigvecs = diagonalize_occupation(occ)
    if not (1 <= nelect <= 5):
        raise ValueError("nelect must be between 1 and 5")

    # eigvals sorted ascending, so last `nelect` are most occupied
    c_occ = eigvecs[:, -nelect:]
    return c_occ, eigvals


def get_hole_wavefunction(occ: np.ndarray, radial_func=None):
    """
    Construct the real-space wavefunction psi_hole(x, y, z) corresponding
    to the hole orbital, i.e. eigenvector of occ with the smallest
    occupation number.

    For a 3d^9-like configuration (4 electrons in 5 d orbitals), this is
    the "missing" orbital.

    Parameters
    ----------
    occ : array-like, shape (5,5)
        Hubbard occupation matrix in QE d basis.
    radial_func : callable or None
        Radial function R(r). If None, only angular part is used.

    Returns
    -------
    psi_hole : callable
        psi_hole(x, y, z) → complex array.
    eigvals : ndarray, shape (5,)
        Natural occupation numbers (ascending).
    """
    c_hole, eigvals = get_hole_coeffs(occ)
    psi_hole = make_wavefunction_from_coeffs(c_hole, radial_func=radial_func)
    return psi_hole, eigvals


def get_occupied_wavefunctions(occ: np.ndarray, nelect: int = 4, radial_func=None):
    """
    Construct the real-space wavefunctions psi_i(x, y, z) for the nelect
    most-occupied natural orbitals.

    Parameters
    ----------
    occ : array-like, shape (5,5)
        Hubbard occupation matrix in QE d basis.
    nelect : int
        Number of electron orbitals to return.
    radial_func : callable or None
        Radial function R(r). If None, only angular part is used.

    Returns
    -------
    psis : list of callables
        List [psi_1, ..., psi_nelect]; each psi_i(x, y, z) → complex array.
        Ordered from less-occupied to more-occupied among the top nelect.
    eigvals : ndarray, shape (5,)
        Natural occupation numbers (ascending).
    """
    c_occ, eigvals = get_occupied_coeffs(occ, nelect=nelect)

    # c_occ shape: (5, nelect) with columns = orbitals
    psis = []
    for j in range(nelect):
        coeffs_j = c_occ[:, j]
        psis.append(make_wavefunction_from_coeffs(coeffs_j, radial_func=radial_func))

    return psis, eigvals


# ----------------------------------------------------------------------
# Optional: a simple test/demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Example: pure dz2 occupation matrix (1 electron, 4 empty)
    occ_dz2 = np.zeros((5, 5), dtype=complex)
    occ_dz2[0, 0] = 1.0

    # Hole orbital in this artificial case → some combination of others
    psi_hole, eigvals = get_hole_wavefunction(occ_dz2)
    print("Eigenvalues (occupations, ascending):", eigvals.real)

    # Evaluate hole wavefunction at a couple of points (purely angular)
    print("psi_hole(1, 0, 0) =", psi_hole(1.0, 0.0, 0.0))
    print("psi_hole(0, 0, 1) =", psi_hole(0.0, 0.0, 1.0))

    # Now: example with 4 electrons (trace ≈ 4 → 1 hole)
    occ_4e = np.diag([0.95, 0.98, 0.99, 0.97, 0.11])  # toy model
    print("\nTrace(occ_4e) =", np.trace(occ_4e).real)

    psi_hole_4e, eigvals_4e = get_hole_wavefunction(occ_4e)
    print("Eigenvalues (4e case, ascending):", eigvals_4e.real)

    # Include a hydrogenic radial part for a more realistic orbital shape
    psi_hole_4e_full, _ = get_hole_wavefunction(
        occ_4e,
        radial_func=lambda r: hydrogenic_radial_3d(r, Z=5.0, n=3)
    )

    val = psi_hole_4e_full(0.3, 0.4, 0.5)
    print("psi_hole_4e_full(0.3, 0.4, 0.5) =", val)
