import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

 


#Jeffery shape parameter, c is the aspect ratio of the sheroid
def lambda_ar(c: float) -> float:
    c2 = c * c
    lamb = (c2 - 1.0) / (c2 + 1.0)
    return lamb


#Shperical angles to unit vector. 
def sph_to_vec(theta: float, phi: float) -> np.ndarray:
    return np.array([
        np.sin(theta) * np.cos(phi), #d1 = rcos(phi), r = sin(theta)
        np.sin(theta) * np.sin(phi), #d2 = rsin(phi), r = sin(theta)
        np.cos(theta)  #d3, d3´=-theta´sin(theta)
    ], dtype=float)

# Clausius-Mossotti
def alpha(V:float, n_p: float, n_m: float) -> float:
    er = (n_p/n_m)**2
    Alpha = 3*V*(8.8541878128e-12)*((er-1)/(er+2))
    return Alpha



#Velocity gradient 3x3 matrix G= nabla u, G[i, j] =du_x/dx_j. (x,y,z).
def grad_u_simple_shear(gamma: float, plane: str) -> np.ndarray:
    G = np.zeros((3,3), dtype=float)
    if plane == "xy":
        G[0,1] = gamma
    elif plane == "xz":
        G[0,2] = gamma
    else:
        raise ValueError("plane must be 'xy' or 'xz'")
    return G




def decompose_grad_u(G):
    E = 0.5 * (G + G.T)
    W = 0.5 * (G - G.T)
    return E, W

def normalize_rows(P):
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    return P / norms

#Jeffery equation in vector form: 
#dp = (Omega x d) + lam (E p - (p^T E p) p)

def jeffery_rhs_vector(t: float, p: np.ndarray, E: np.ndarray, W: np.ndarray, lam: float) -> np.ndarray:

    p = np.asarray(p, dtype=float)
    Ep = E @ p #Ejk d
    Wp = W @ p #omega * d
    dEp = p @ Ep 
    dp = Wp + lam * (Ep - dEp * p)
    return dp

#Unit vector -> (theta, phi). Uses atan2 for robust quadrant.
def vec_to_sph(p: np.ndarray) -> tuple[float, float]:
    p = np.asarray(p, dtype=float)
    p = p / np.linalg.norm(p)
    theta = np.arccos(np.clip(p[2], -1.0, 1.0))
    phi = np.arctan2(p[1], p[0])
    return theta, phi

#continuous angles in radians
def unwrap_angle_rad(angle_rad:np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(angle_rad, dtype=float))

#continuous angles in degrees
def unwrap_angle_deg(angle_deg:np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(angle_deg, dtype=float))))
   
#Rayleigh range, zr 
def rayleigh_range(w0:float, lam: float) -> float: 
    return (np.pi * (w0**2))/lam 

#U0
def trap_depth(alpha: float,P:float,w0:float) -> float:
    return (2*alpha*P)/(8.8541878128e-12*299792458.0*np.pi*(w0**2))

def harmonic_stiffness_from_gaussian(
    U0: float,
    w0: float,
    zR: float
) -> tuple[float, float, float]:
    if U0 < 0.0:
        raise ValueError("U0 must be non-negative.")
    if w0 <= 0.0:
        raise ValueError("w0 must be positive.")
    if zR <= 0.0:
        raise ValueError("zR must be positive.")

    kx = 4.0 * U0 / w0**2
    ky = kx
    kz = 2.0 * U0 / zR**2

    return kx, ky, kz

def rotational_drag_sphere(
    eta: float,
    effective_radius: float
) -> float:
    """
    Aproximación esférica del coeficiente de
    arrastre rotacional.

        zeta_r = 8*pi*eta*a^3

    Es una aproximación inicial.
    """

    if eta <= 0.0:
        raise ValueError(
            "eta must be positive."
        )

    if effective_radius <= 0.0:
        raise ValueError(
            "effective_radius must be positive."
        )

    return (
        8.0
        * np.pi
        * eta
        * effective_radius**3
    )

def alignment_rate_from_torque(
    kappa: float,
    rotational_drag: float
) -> float:
    """
    Convierte una intensidad de torque kappa [J]
    en una tasa de alineamiento xi [1/s].

        xi = 2 * kappa / zeta_r
    """

    if rotational_drag <= 0.0:
        raise ValueError(
            "rotational_drag must be positive."
        )

    return (
        2.0
        * kappa
        / rotational_drag
    )
def uniaxial_alignment_drift(
    p: np.ndarray,
    axis: np.ndarray,
    alignment_rate: float
) -> np.ndarray:
    """
    Contribución orientacional que alinea p con ±axis.

    p_dot = xi * (p·axis) * [axis - (p·axis)p]

    Parameters
    ----------
    p : np.ndarray
        Vector director.
    axis : np.ndarray
        Eje preferido de alineamiento.
    alignment_rate : float
        Tasa de alineamiento xi [1/s].
    """
    p = normalize_vector(np.asarray(p, dtype=float))
    axis = normalize_vector(np.asarray(axis, dtype=float))

    c = float(np.dot(p, axis))

    return alignment_rate * c * (
        axis - c * p
    )

























def integrate_director_vector(
    d0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray, 
    E: np.ndarray,
    W: np.ndarray,
    lam: float,
    rtol: float = 1e-9,
    atol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray,np.ndarray]:

    sol = solve_ivp(
        fun=lambda t, p: jeffery_rhs_vector(t, p, E, W, lam),
        t_span=t_span,
        y0=np.asarray(d0, dtype=float),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )

    P_raw = sol.y.T  # (N,3)
    norm_raw = np.linalg.norm(P_raw, axis=1)

    P = normalize_rows(P_raw)
    return sol.t, P, norm_raw

 
#returns [dtheta/dt, dphi/dt]
def jeffery_rhs_theta_phi(t: float, y: np.ndarray, E: np.ndarray, W: np.ndarray, B: float) -> np.ndarray:
   
    theta, phi = y

    # unitary vector
    d = sph_to_vec(theta, phi)
    d_dot = jeffery_rhs_vector(t, d, E, W, B)

    # d3' = -theta'sin(theta) -> thta'=-d3'/sin(theta)
    sin_th = np.sin(theta)
    eps = 1e-12
    if abs(sin_th) < eps:
        theta_dot = 0.0
        phi_dot = 0.0
        return np.array([theta_dot, phi_dot], dtype=float)

    theta_dot = -d_dot[2] / sin_th

    # d1 * d2' - d1' * d2 = phi' sin^2(theta)
    sin2 = sin_th * sin_th
    phi_dot = (d[0] * d_dot[1] - d_dot[0] * d[1]) / sin2

    return np.array([theta_dot, phi_dot], dtype=float)


def integrate_theta_phi(
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    E: np.ndarray,
    W: np.ndarray,
    B: float,
    rtol: float = 1e-9,
    atol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    sol = solve_ivp(
        fun=lambda t, y: jeffery_rhs_theta_phi(t, y, E, W, B),
        t_span=t_span,
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )
    Y = sol.y.T  # (N,2)
    return sol.t, Y

#Extensional flow
def grad_u_extensional(epsilon_dot: float, plane: str) -> np.ndarray:
    G = np.zeros((3,3), dtype=float)
    if plane == "xy":
        G[0,0]= epsilon_dot
        G[1,1]=-epsilon_dot
    elif plane == "xz":
        G[0,0]= epsilon_dot
        G[2,2]=-epsilon_dot
    elif plane == "yz":
        G[1,1] = epsilon_dot
        G[2,2] = -epsilon_dot
    else:
        raise ValueError("plane must be 'xy', 'xz' or 'yz'")
    return G

#Rigid rotation, all contribution is in the antisimetric part
def grad_u_rotation(omega: float, plane: str) -> np.ndarray:
    G = np.zeros((3,3),dtype=float)
    if plane == "xy": # u = (-omega*y, omega*x,0), rotates around z
        G[0,1]=-omega
        G[1,0]=omega
    elif plane == "xz": # u = (omega*z,0, -omega*x), rotates around y
        G[0,2] = omega
        G[2,0] = -omega
    elif plane == "yz": # u = (0,-omega*z, omega*y), rotates around x
        G[1,2] = -omega
        G[2,1] = omega
    else:
        raise ValueError("plane must be 'xy', 'xz' or 'yz'")
    return G

#mixed flow shear + stretching
def grad_u_mixed_shear_stretch(gamma: float, s: float) -> np.ndarray: 
    G=np.zeros((3,3), dtype=float)
    G[0,0] = s
    G[1,1]= -s
    G[0,2] = gamma
    return G

#Normalize a 3D vector, n =||v|| 
def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-15:
        raise ValueError("Too close to 0")
    return v/n



#converts directors P of (N,3) to angles
def directors_to_angles(P:np.ndarray):
    P=np.asarray(P, dtype=float)
    d1 = P[:,0]
    d2 = P[:,1]
    d3 = P[:,2]

    theta_rad = np.arccos(np.clip(d3,-1.0,1.0))
    phi_rad_wrapped = np.arctan2(d2,d1)
    phi_rad_unwrapped = np.unwrap(phi_rad_wrapped)

    return theta_rad, phi_rad_wrapped, phi_rad_unwrapped


def optical_angular_velocity_effective(
    p: np.ndarray,
    e_pol: np.ndarray = None,
    lambda_opt: float = 0.0
) -> np.ndarray:
    """

    Model:
        U(p) = -(1/2) * Delta_alpha * E0^2 * (p · e_pol)^2

    Uses:

        lambda_opt   [1/s]

    Effective angular velocity:

        Omega_opt = lambda_opt * (p · e_pol) * (p x e_pol)

    Where:
        - p      : unit vector
        - e_pol  : unidirectional linear polarization
        - lambda_opt > 0 promotes alignment with ± e_pol

    Interpretation:
    - If p is already aligned with e_pol, then p x e_pol = 0
      and there is no additional optical rotation.
    - If p is tilted, an effective rotation occurs that tends
      to align p with the polarization direction.

    Parameters:
    - p: direction vector.
    - e_pol: Polarization direction. If None: x = (1,0,0).
    - lambda_opt: effective optical alignment intensity [1/s].

    Return:
    - Omega_opt: Effective angular velocity.
    """
    p = normalize_vector(np.asarray(p, dtype=float))

    if e_pol is None:
        e_pol = np.array([1.0, 0.0, 0.0], dtype=float)
    e_pol = normalize_vector(np.asarray(e_pol, dtype=float))

    c = float(np.dot(p, e_pol))
    Omega_opt = lambda_opt * c * np.cross(p, e_pol)
    return Omega_opt

def optical_alignment_drift(
    p: np.ndarray,
    e_pol: np.ndarray = None,
    lambda_opt: float = 0.0
) -> np.ndarray:
    """
    Devuelve la contribución óptica directa a p_dot.

    uses:
        p_dot_opt = Omega_opt x p

    with:
        Omega_opt = lambda_opt * (p · e_pol) * (p x e_pol)

    gets:

        p_dot_opt = lambda_opt * (p·e_pol) * [ e_pol - (p·e_pol) p ]

    """
    p = normalize_vector(np.asarray(p, dtype=float))

    Omega_opt = optical_angular_velocity_effective(
        p=p,
        e_pol=e_pol,
        lambda_opt=lambda_opt
    )

    p_dot_opt = np.cross(Omega_opt, p)
    return p_dot_opt


def jeffery_rhs_vector_optical(
    t: float,
    d: np.ndarray,
    E: np.ndarray,
    W: np.ndarray,
    lam: float,
    e_pol: np.ndarray = None,
    lambda_opt: float = 0.0
) -> np.ndarray:
    """
    right-hand side of the angular momentum equation:

        d_dot = d_dot_Jeffery + d_dot_opt

    Where:
        d_dot_Jeffery = W d + lam (E d - (d^T E d) d)
        d_dot_opt     = optical_alignment_drift(...)

    """
    d = normalize_vector(np.asarray(d, dtype=float))

    d_dot_jeffery = jeffery_rhs_vector(t, d, E, W, lam)
    d_dot_opt = optical_alignment_drift(
        p=d,
        e_pol=e_pol,
        lambda_opt=lambda_opt
    )

    return d_dot_jeffery + d_dot_opt

def u_poiseuille_rectangular(x: float, y:float, U: float, A: float) -> float:
    #Velocity profile 
    return U * (1.0 - (x/A)**2) * (1.0 - y**2)

def grad_u_poiseuille_rectangular(x: float, y: float, U: float, A: float) -> np.ndarray:

    G = np.zeros((3, 3), dtype=float)

    #Partial derivatives of u with respect x and y 
    du_z_dx = U * (-2.0 * x / (A**2)) * (1.0 - y**2)
    du_z_dy = U * (1.0 - (x/A)**2) * (-2.0 * y)

    G[2, 0] = du_z_dx
    G[2, 1] = du_z_dy

    return G



#beam waist
def waist_beam(w0:float, z: float, zR: float) -> float:
    return w0*((1+((z/zR)**2))**(1/2))

def gaussian_intensity_factor(
    R: np.ndarray,
    w0: float,
    zR: float
) -> float:
    """
    Perfil espacial normalizado de intensidad de un haz gaussiano.

    I(R)/I0 = (w0/w(z))^2 * exp[-2(x^2+y^2)/w(z)^2]

    """
    R = np.asarray(R, dtype=float)

    if R.shape != (3,):
        raise ValueError("R must be a three-dimensional vector.")
    if w0 <= 0.0:
        raise ValueError("w0 must be positive.")
    if zR <= 0.0:
        raise ValueError("zR must be positive.")

    x, y, z = R
    wz = waist_beam(w0, z, zR)
    r2 = x**2 + y**2

    return (w0 / wz)**2 * np.exp(-2.0 * r2 / wz**2)





def gaussian_com_potential(
    R: np.ndarray,
    U0: float,
    w0: float,
    zR: float
) -> float:
    """
    Potencial gaussiano del centro de masa.

    U(R) = -U0 * I(R)/I0

    U0 > 0 representa la profundidad del potencial.
    """
    if U0 < 0.0:
        raise ValueError("U0 must be non-negative.")

    intensity_factor = gaussian_intensity_factor(
        R=R,
        w0=w0,
        zR=zR
    )

    return -U0 * intensity_factor

def gaussian_com_force(
    R: np.ndarray,
    U0: float,
    w0: float,
    zR: float
) -> np.ndarray:
    """
    Fuerza óptica asociada al potencial gaussiano del COM.

    F_opt = -grad_R U

    Parameters
    ----------
    R : np.ndarray
        Posición [x, y, z].
    U0 : float
        Profundidad positiva del potencial.
    w0 : float
        Radio del haz en el foco.
    zR : float
        Rango de Rayleigh.

    Returns
    -------
    np.ndarray
        Fuerza óptica [Fx, Fy, Fz].
    """
    R = np.asarray(R, dtype=float)

    if R.shape != (3,):
        raise ValueError("R must be a three-dimensional vector.")
    if U0 < 0.0:
        raise ValueError("U0 must be non-negative.")
    if w0 <= 0.0:
        raise ValueError("w0 must be positive.")
    if zR <= 0.0:
        raise ValueError("zR must be positive.")

    x, y, z = R

    wz = waist_beam(w0, z, zR)
    w2 = wz**2
    r2 = x**2 + y**2

    exponential = np.exp(-2.0 * r2 / w2)
    common = U0 * w0**2 * exponential

    # Fuerzas transversales
    Fx = -4.0 * common * x / w2**2
    Fy = -4.0 * common * y / w2**2

    # Derivada de w(z)^2 respecto de z
    dw2_dz = 2.0 * w0**2 * z / zR**2

    U = -common / w2

    Fz = U * dw2_dz * (
        1.0 / w2
        - 2.0 * r2 / w2**2
    )

    return np.array([Fx, Fy, Fz], dtype=float)

def harmonic_com_potential(
    R: np.ndarray,
    kx: float,
    ky: float,
    kz: float
) -> float:
    """
    Potencial armónico tridimensional para el centro de masa.
    """
    R = np.asarray(R, dtype=float)

    if R.shape != (3,):
        raise ValueError("R must be a three-dimensional vector.")

    x, y, z = R

    return 0.5 * (
        kx * x**2
        + ky * y**2
        + kz * z**2
    )

def harmonic_com_force(
    R: np.ndarray,
    kx: float,
    ky: float,
    kz: float
) -> np.ndarray:
    """
    Fuerza restauradora del potencial armónico.

    F = (-kx*x, -ky*y, -kz*z)
    """
    R = np.asarray(R, dtype=float)

    if R.shape != (3,):
        raise ValueError("R must be a three-dimensional vector.")

    x, y, z = R

    return np.array([
        -kx * x,
        -ky * y,
        -kz * z
    ], dtype=float)



def overdamped_translational_drift(
    force: np.ndarray,
    mobility
) -> np.ndarray:
    """
    Convierte una fuerza en velocidad sobreamortiguada.

    R_dot = mobility * force

    mobility puede ser:
    - un escalar isotrópico;
    - una matriz de movilidad 3x3.
    """
    force = np.asarray(force, dtype=float)

    if force.shape != (3,):
        raise ValueError("force must be a three-dimensional vector.")

    mobility_array = np.asarray(mobility, dtype=float)

    if mobility_array.ndim == 0:
        return float(mobility_array) * force

    if mobility_array.shape == (3, 3):
        return mobility_array @ force

    raise ValueError(
        "mobility must be a scalar or a 3x3 matrix."
    )



# Squared electric-field amplitude of the Gaussian beam
def gaussian_field_squared(
    R: np.ndarray,
    P: float,
    w0: float,
    zR: float
) -> float:

    epsilon_0 = 8.8541878128e-12
    c_light = 299792458.0

    E0_squared = (
        4.0 * P
        / (
            epsilon_0
            * c_light
            * np.pi
            * w0**2
        )
    )

    intensity_factor = gaussian_intensity_factor(
        R=R,
        w0=w0,
        zR=zR
    )

    return E0_squared * intensity_factor

# Depolarization factors for a prolate spheroid
def prolate_depolarization_factors(
    aspect_ratio: float
) -> tuple[float, float]:

    if aspect_ratio < 1.0:
        raise ValueError(
            "aspect_ratio must be >= 1 for a prolate spheroid."
        )

    # Sphere
    if abs(aspect_ratio - 1.0) < 1.0e-8:
        return 1.0 / 3.0, 1.0 / 3.0

    eccentricity = np.sqrt(
        1.0 - 1.0 / aspect_ratio**2
    )

    N_parallel = (
        (1.0 - eccentricity**2)
        / (2.0 * eccentricity**3)
        * (
            np.log(
                (1.0 + eccentricity)
                / (1.0 - eccentricity)
            )
            - 2.0 * eccentricity
        )
    )

    N_perpendicular = (
        1.0 - N_parallel
    ) / 2.0

    return N_parallel, N_perpendicular

# Rayleigh polarizabilities of a prolate spheroid
def prolate_polarizabilities(
    V: float,
    n_p: float,
    n_m: float,
    aspect_ratio: float
) -> tuple[float, float]:

    epsilon_0 = 8.8541878128e-12

    epsilon_r = (n_p / n_m)**2

    N_parallel, N_perpendicular = (
        prolate_depolarization_factors(
            aspect_ratio
        )
    )

    contrast = epsilon_r - 1.0

    alpha_parallel = (
        epsilon_0
        * V
        * contrast
        / (
            1.0
            + N_parallel * contrast
        )
    )

    alpha_perpendicular = (
        epsilon_0
        * V
        * contrast
        / (
            1.0
            + N_perpendicular * contrast
        )
    )

    return (
        alpha_parallel,
        alpha_perpendicular
    )

# Points inside a prolate spheroid whose long axis is initially z
def prolate_spheroid_points(
    Length: float,
    diameter: float,
    n_grid: int = 13
) -> np.ndarray:

    a = Length / 2.0
    b = diameter / 2.0

    q = np.linspace(
        -1.0,
        1.0,
        n_grid
    )

    points = []

    for qx in q:
        for qy in q:
            for qz in q:

                if (
                    qx**2
                    + qy**2
                    + qz**2
                    <= 1.0
                ):
                    points.append([
                        b * qx,
                        b * qy,
                        a * qz
                    ])

    return np.asarray(
        points,
        dtype=float
    )

def rotate_points_from_z(
    points: np.ndarray,
    axis: np.ndarray
) -> np.ndarray:
    """
    Rota un conjunto de puntos cuyo eje principal original
    está orientado en z, de manera que dicho eje quede
    orientado según 'axis'.

    Parameters
    ----------
    points : np.ndarray
        Arreglo de forma (N, 3).

    axis : np.ndarray
        Nueva dirección del eje principal.

    Returns
    -------
    np.ndarray
        Puntos rotados, con forma (N, 3).
    """

    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "points must have shape (N, 3)."
        )

    axis = normalize_vector(
        np.asarray(axis, dtype=float)
    )

    z_axis = np.array(
        [0.0, 0.0, 1.0],
        dtype=float
    )

    c = float(
        np.dot(z_axis, axis)
    )

    # Already aligned with +z
    if c > 1.0 - 1.0e-12:
        return points.copy()

    # Aligned with -z
    if c < -1.0 + 1.0e-12:
        rotation = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ])

        return points @ rotation.T

    v = np.cross(
        z_axis,
        axis
    )

    s = np.linalg.norm(v)

    K = np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ])

    rotation = (
        np.eye(3)
        + K
        + K @ K
        * ((1.0 - c) / s**2)
    )

    return points @ rotation.T

def gradient_potential(
    R: np.ndarray,
    axis: np.ndarray,
    alpha_v: float,
    volume: float,
    body_points: np.ndarray,
    field_squared_fn
) -> float:
    """
    Potencial orientacional debido al gradiente de intensidad.

    Implementa:

        U_grad = -(alpha_v / 2) * integral_V |E|^2 dV

    Los puntos de body_points representan uniformemente
    el volumen de la partícula en su sistema propio.

    Parameters
    ----------
    R : np.ndarray
        Posición del centro de masa.

    axis : np.ndarray
        Orientación del eje largo de la partícula.

    alpha_v : float
        Polarizabilidad por unidad de volumen.

    volume : float
        Volumen total de la partícula.

    body_points : np.ndarray
        Puntos que muestrean el volumen de la partícula.

    field_squared_fn : callable
        Función:

            field_squared_fn(position) -> |E|^2

        Esto permite usar Gaussian, Dark Focus u otro campo.

    Returns
    -------
    float
        Energía U_grad [J].
    """

    R = np.asarray(R, dtype=float)

    if R.shape != (3,):
        raise ValueError(
            "R must be a three-dimensional vector."
        )

    if volume <= 0.0:
        raise ValueError(
            "volume must be positive."
        )

    rotated_points = rotate_points_from_z(
        points=body_points,
        axis=axis
    )

    laboratory_points = (
        R[None, :]
        + rotated_points
    )

    field_squared = np.array([
        field_squared_fn(point)
        for point in laboratory_points
    ])

    integral_E2 = (
        volume
        * np.mean(field_squared)
    )

    return (
        -0.5
        * alpha_v
        * integral_E2
    )

def gradient_torque_strength(
    R: np.ndarray,
    alpha_v: float,
    volume: float,
    body_points: np.ndarray,
    field_squared_fn,
    preferred_axis: np.ndarray = None,
    perpendicular_axis: np.ndarray = None
) -> float:
    """
    Calcula la intensidad kappa_grad del potencial
    orientacional de gradiente.

    Se compara la energía de la partícula orientada
    con el eje preferido y perpendicular a éste.

    Returns
    -------
    float
        kappa_grad [J].
    """

    if preferred_axis is None:
        preferred_axis = np.array(
            [0.0, 0.0, 1.0],
            dtype=float
        )

    if perpendicular_axis is None:
        perpendicular_axis = np.array(
            [1.0, 0.0, 0.0],
            dtype=float
        )

    U_parallel = gradient_potential(
        R=R,
        axis=preferred_axis,
        alpha_v=alpha_v,
        volume=volume,
        body_points=body_points,
        field_squared_fn=field_squared_fn
    )

    U_perpendicular = gradient_potential(
        R=R,
        axis=perpendicular_axis,
        alpha_v=alpha_v,
        volume=volume,
        body_points=body_points,
        field_squared_fn=field_squared_fn
    )

    return float(
        abs(
            U_perpendicular
            - U_parallel
        )
    )

def polarization_torque_strength(
    alpha_parallel: float,
    alpha_perpendicular: float,
    E_squared: float
) -> float:
    """
    Intensidad del potencial orientacional debido
    a la polarización.

    kappa_pol =
        0.5 * (alpha_parallel - alpha_perpendicular)
        * |E|^2
    """

    delta_alpha = (
        alpha_parallel
        - alpha_perpendicular
    )

    return (
        0.5
        * delta_alpha
        * E_squared
    )






def local_alignment_rate(
    R: np.ndarray,
    maximum_rate: float,
    intensity_factor_fn=None,
    position_dependent: bool = True
) -> float:
    """
    Calcula una tasa local de alineamiento orientacional.

    Si position_dependent es False:

        xi(R) = xi_0

    Si position_dependent es True:

        xi(R) = xi_0 * f_I(R)

    donde f_I(R) es un factor de intensidad normalizado,
    por ejemplo:

        f_I(R) = I(R) / I0

    Parameters
    ----------
    R : np.ndarray
        Posición del centro de masa [x, y, z].

    maximum_rate : float
        Tasa de alineamiento de referencia xi_0 [1/s].

    intensity_factor_fn : callable or None
        Función con la forma:

            intensity_factor_fn(R) -> float

        Debe devolver el factor de intensidad normalizado.

    position_dependent : bool
        Si es True, la tasa depende de la posición.
        Si es False, devuelve directamente maximum_rate.

    Returns
    -------
    float
        Tasa local de alineamiento xi(R) [1/s].
    """
    R = np.asarray(R, dtype=float)

    if R.shape != (3,):
        raise ValueError(
            "R must be a three-dimensional vector."
        )

    if not np.isfinite(maximum_rate):
        raise ValueError(
            "maximum_rate must be finite."
        )

    if not position_dependent:
        return float(maximum_rate)

    if intensity_factor_fn is None:
        raise ValueError(
            "intensity_factor_fn is required when "
            "position_dependent is True."
        )

    intensity_factor = float(
        intensity_factor_fn(R)
    )

    if not np.isfinite(intensity_factor):
        raise ValueError(
            "intensity_factor_fn must return a finite value."
        )

    if intensity_factor < 0.0:
        raise ValueError(
            "The normalized intensity factor cannot be negative."
        )

    return float(maximum_rate) * intensity_factor

def coupled_com_orientation_rhs(
    t: float,
    state: np.ndarray,
    beta: float,
    translational_mobility,
    swimming_speed: float = 0.0,
    flow_velocity_fn=None,
    flow_gradient_fn=None,
    optical_force_fn=None,
    orientation_drift_fn=None
) -> np.ndarray:
    """
    Ecuación general acoplada para la posición del centro
    de masa y la orientación de una partícula activa.

    State
    -----
    state = [x, y, z, px, py, pz]

    Translational dynamics
    ----------------------
    R_dot = u(t, R)
          + v_s p
          + mobility * F_opt(t, R, p)

    Orientational dynamics
    ----------------------
    p_dot = p_dot_Jeffery
          + p_dot_extra(t, R, p)

    Optional functions
    ------------------
    flow_velocity_fn(t, R)
        Devuelve la velocidad local del flujo.

    flow_gradient_fn(t, R)
        Devuelve el gradiente local de velocidad, grad(u).

    optical_force_fn(t, R, p)
        Devuelve la fuerza óptica sobre el centro de masa.

    orientation_drift_fn(t, R, p)
        Devuelve contribuciones orientacionales adicionales,
        por ejemplo torque de gradiente o polarización.

    Parameters
    ----------
    t : float
        Tiempo [s].

    state : np.ndarray
        Estado [x, y, z, px, py, pz].

    beta : float
        Parámetro de Bretherton de la partícula.

    translational_mobility : float or np.ndarray
        Movilidad traslacional escalar o tensor 3x3.

    swimming_speed : float
        Rapidez de autopropulsión v_s [m/s].

    Returns
    -------
    np.ndarray
        Derivada temporal
        [x_dot, y_dot, z_dot, px_dot, py_dot, pz_dot].
    """
    state = np.asarray(state, dtype=float)

    if state.shape != (6,):
        raise ValueError(
            "state must contain "
            "[x, y, z, px, py, pz]."
        )

    if not np.isfinite(beta):
        raise ValueError(
            "beta must be finite."
        )

    if not np.isfinite(swimming_speed):
        raise ValueError(
            "swimming_speed must be finite."
        )

    # ========================================================
    # STATE VARIABLES
    # ========================================================

    R = state[0:3]

    p = normalize_vector(
        state[3:6]
    )

    # ========================================================
    # LOCAL FLOW VELOCITY
    # ========================================================

    if flow_velocity_fn is None:
        u_flow = np.zeros(3, dtype=float)

    else:
        u_flow = np.asarray(
            flow_velocity_fn(t, R),
            dtype=float
        )

        if u_flow.shape != (3,):
            raise ValueError(
                "flow_velocity_fn must return "
                "a three-dimensional vector."
            )

    # ========================================================
    # LOCAL VELOCITY GRADIENT
    # ========================================================

    if flow_gradient_fn is None:
        G = np.zeros((3, 3), dtype=float)

    else:
        G = np.asarray(
            flow_gradient_fn(t, R),
            dtype=float
        )

        if G.shape != (3, 3):
            raise ValueError(
                "flow_gradient_fn must return "
                "a 3x3 matrix."
            )

    E, W = decompose_grad_u(G)

    # ========================================================
    # OPTICAL FORCE
    # ========================================================

    if optical_force_fn is None:
        F_opt = np.zeros(3, dtype=float)

    else:
        F_opt = np.asarray(
            optical_force_fn(t, R, p),
            dtype=float
        )

        if F_opt.shape != (3,):
            raise ValueError(
                "optical_force_fn must return "
                "a three-dimensional vector."
            )

    v_opt = overdamped_translational_drift(
        force=F_opt,
        mobility=translational_mobility
    )

    # ========================================================
    # ACTIVE SWIMMING
    # ========================================================

    v_swim = swimming_speed * p

    # ========================================================
    # TRANSLATIONAL EQUATION
    # ========================================================

    R_dot = (
        u_flow
        + v_swim
        + v_opt
    )

    # ========================================================
    # JEFFERY ORIENTATION EQUATION
    # ========================================================

    p_dot = jeffery_rhs_vector(
        t=t,
        d=p,
        E=E,
        W=W,
        lam=beta
    )

    # ========================================================
    # ADDITIONAL ORIENTATIONAL DRIFT
    # ========================================================

    if orientation_drift_fn is not None:
        p_dot_extra = np.asarray(
            orientation_drift_fn(t, R, p),
            dtype=float
        )

        if p_dot_extra.shape != (3,):
            raise ValueError(
                "orientation_drift_fn must return "
                "a three-dimensional vector."
            )

        p_dot = p_dot + p_dot_extra

    return np.concatenate(
        (R_dot, p_dot)
    )


