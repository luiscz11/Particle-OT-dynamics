import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pyvista as pv

import src_Jeffery as jf


# ============================================================
# 1) MODEL SWITCHES
# ============================================================

USE_COM_TRAP = True

# "gaussian" or "harmonic"
TRAP_MODEL = "gaussian"

USE_GRADIENT_ALIGNMENT = True
USE_POLARIZATION_ALIGNMENT = True

# Makes the optical alignment rates depend on COM position
USE_POSITION_DEPENDENT_TORQUE = True

# Optional intrinsic swimming of the microalga
USE_SWIMMING = True


# ============================================================
# 2) SIMPLE SHEAR FLOW
# ============================================================

gamma = 0.90                 # Shear rate [1/s]
plane = "xy"                 # "xy" or "xz"

G = jf.grad_u_simple_shear(
    gamma=gamma,
    plane=plane
)

E, W = jf.decompose_grad_u(G)


# ============================================================
# 3) PARTICLE PARAMETERS
# ============================================================
Large = 500.0e-9
diameter = 70.0e-9
aspect_ratio = Large / diameter    # Monoraphidium griffithii
beta = jf.lambda_ar(aspect_ratio)

# Approximate scalar translational mobility.
# This is only a first isotropic approximation.
eta = 0.953e-3               # Water viscosity [Pa s]
effective_radius = diameter/2    # Effective radius [m]

translational_mobility = 1.0 / (
    6.0 * np.pi * eta * effective_radius
)

#active swimming speed 
swimming_speed = 10.0e-6     # [m/s]

if not USE_SWIMMING:
    swimming_speed = 0.0


# ============================================================
# 4) INITIAL CENTER-OF-MASS POSITION
# ============================================================

R0 = np.array([
     0,    # x0 [m]
    0,    # y0 [m]
     0     # z0 [m]
], dtype=float)


# ============================================================
# 5) INITIAL ORIENTATION
# ============================================================

theta0_deg = 25.0
phi0_deg = 25.0

theta0 = np.deg2rad(theta0_deg)
phi0 = np.deg2rad(phi0_deg)

p0 = jf.sph_to_vec(theta0, phi0)


# ============================================================
# 6) GAUSSIAN BEAM PARAMETERS
# ============================================================

#Representative values
wavelength = 976.0e-9        # [m]
w0 = 10.0e-6                 # Beam radius at focus [m]
laser_power = 140.0e-3
Volume = (4/3)*np.pi*(Large/2)*((diameter/2)**2)
n_particle = 1.38
n_medium = 1.33
k_B = 1.380649e-23      # Boltzmann constant [J/K]
temperature = 298.15    # [K]
kBT = k_B * temperature


zR = jf.rayleigh_range(
    w0=w0,
    lam=wavelength
)

alpha_scalar = jf.alpha(
    V=Volume,
    n_p=n_particle,
    n_m=n_medium
)

U0 = jf.trap_depth(
    alpha=alpha_scalar,
    P=laser_power,
    w0=w0
)         # Gaussian potential depth [J]

# Harmonic approximation derived from the Gaussian potential
kx, ky, kz = jf.harmonic_stiffness_from_gaussian(
    U0=U0,
    w0=w0,
    zR=zR
)


# ============================================================
# 7) ORIENTATIONAL OPTICAL PARAMETERS
# ============================================================

# Optical beam axis: gradient torque tends to align with z
beam_axis = np.array([0.0, 0.0, 1.0], dtype=float)

# Linear polarization direction
e_pol = np.array([0.0, 1.0, 0.0], dtype=float)

# Axis perpendicular to the beam axis, used to calculate
# Delta U_grad from the two limiting orientations.
gradient_perpendicular_axis = np.array([1.0,0.0,0.0],dtype=float)

alpha_v = alpha_scalar / Volume

# ------------------------------------------------------------
# Anisotropic polarizabilities

alpha_parallel, alpha_perpendicular = (
    jf.prolate_polarizabilities(
        V=Volume,
        n_p=n_particle,
        n_m=n_medium,
        aspect_ratio=aspect_ratio
    )
)

# ------------------------------------------------------------
# Discretized particle volume for Eq. (4)


body_points = jf.prolate_spheroid_points(
    Length=Large,
    diameter=diameter,
    n_grid=13
)

# ------------------------------------------------------------
# Rotational viscous drag
# Initial spherical approximation
# ------------------------------------------------------------

rotational_drag = jf.rotational_drag_sphere(
    eta=eta,
    effective_radius=effective_radius
)

# ------------------------------------------------------------
# Local squared electric field
# ------------------------------------------------------------

def field_squared(R: np.ndarray) -> float:

    return jf.gaussian_field_squared(
        R=R,
        P=laser_power,
        w0=w0,
        zR=zR
    )

# ============================================================
# 8) TIME
# ============================================================

t0 = 0.0
tf = 2000.0

n_points = 4000
t_eval = np.linspace(t0, tf, n_points)


# ============================================================
# 9) OPTICAL FORCE
# ============================================================

def optical_force(R: np.ndarray) -> np.ndarray:
    """
    Returns the optical force acting on the center of mass.
    """
    if not USE_COM_TRAP:
        return np.zeros(3, dtype=float)

    if TRAP_MODEL == "gaussian":
        return jf.gaussian_com_force(
            R=R,
            U0=U0,
            w0=w0,
            zR=zR
        )

    if TRAP_MODEL == "harmonic":
        return jf.harmonic_com_force(
            R=R,
            kx=kx,
            ky=ky,
            kz=kz
        )

    raise ValueError(
        "TRAP_MODEL must be 'gaussian' or 'harmonic'."
    )


def optical_potential(R: np.ndarray) -> float:
    """
    Returns the optical potential acting on the center of mass.
    """
    if not USE_COM_TRAP:
        return 0.0

    if TRAP_MODEL == "gaussian":
        return jf.gaussian_com_potential(
            R=R,
            U0=U0,
            w0=w0,
            zR=zR
        )

    if TRAP_MODEL == "harmonic":
        return jf.harmonic_com_potential(
            R=R,
            kx=kx,
            ky=ky,
            kz=kz
        )

    raise ValueError(
        "TRAP_MODEL must be 'gaussian' or 'harmonic'."
    )


# ============================================================
# 10) POSITION-DEPENDENT ALIGNMENT RATES
# ============================================================
# Retained for future phenomenological tests.
# The current model calculates the optical torque directly
# from the equations

def local_alignment_rate(
    R: np.ndarray,
    maximum_rate: float
) -> float:
    """
    Phenomenological first approximation:

        xi(R) = xi_0 * I(R)/I0

    Thus, the optical torque is strongest near the focus and
    decreases away from the high-intensity region.
    """
    if not USE_POSITION_DEPENDENT_TORQUE:
        return maximum_rate

    intensity_factor = jf.gaussian_intensity_factor(
        R=R,
        w0=w0,
        zR=zR
    )

    return maximum_rate * intensity_factor


# ============================================================
# 11) COUPLED COM-ORIENTATION EQUATION
# ============================================================

def coupled_rhs(
    t: float,
    state: np.ndarray
) -> np.ndarray:
    """
    Coupled state:

        state = [x, y, z, px, py, pz]

    Translational dynamics:

        R_dot = u(R) + v_s p + mobility F_opt(R)

    Orientational dynamics:

        p_dot = p_dot_Jeffery
              + p_dot_gradient
              + p_dot_polarization
    """
    state = np.asarray(state, dtype=float)

    if state.shape != (6,):
        raise ValueError(
            "state must contain [x, y, z, px, py, pz]."
        )

    R = state[0:3]
    p = jf.normalize_vector(state[3:6])

    # --------------------------------------------------------
    # Translational motion
    # --------------------------------------------------------

    # Local background-flow velocity.
    # For simple shear in xy:
    # u(R) = (gamma*y, 0, 0)
    u_flow = G @ R

    F_opt = optical_force(R)

    v_opt = jf.overdamped_translational_drift(
        force=F_opt,
        mobility=translational_mobility
    )

    v_swim = swimming_speed * p

    R_dot = u_flow + v_swim + v_opt

    # --------------------------------------------------------
    # Jeffery orientation dynamics
    # --------------------------------------------------------

    p_dot = jf.jeffery_rhs_vector(
        t=t,
        d=p,
        E=E,
        W=W,
        lam=beta
    )

    # --------------------------------------------------------
    # Gradient-induced optical alignment
    # --------------------------------------------------------

    if USE_GRADIENT_ALIGNMENT:

        if USE_POSITION_DEPENDENT_TORQUE:
            R_torque = R
        else:
            R_torque = np.zeros(3, dtype=float)

        kappa_grad = jf.gradient_torque_strength(
            R=R_torque,
            alpha_v=alpha_v,
            volume=Volume,
            body_points=body_points,
            field_squared_fn=field_squared,
            preferred_axis=beam_axis,
            perpendicular_axis=gradient_perpendicular_axis
        )

        xi_grad = jf.alignment_rate_from_torque(
            kappa=kappa_grad,
            rotational_drag=rotational_drag
        )

        p_dot_grad = jf.uniaxial_alignment_drift(
            p=p,
            axis=beam_axis,
            alignment_rate=xi_grad
        )

        p_dot = p_dot + p_dot_grad

    # --------------------------------------------------------
    # Polarization-induced optical alignment
    # --------------------------------------------------------

    if USE_POLARIZATION_ALIGNMENT:

        if USE_POSITION_DEPENDENT_TORQUE:
            R_torque = R
        else:
            R_torque = np.zeros(3, dtype=float)

        E_squared = field_squared(
            R_torque
        )

        kappa_pol = jf.polarization_torque_strength(
            alpha_parallel=alpha_parallel,
            alpha_perpendicular=alpha_perpendicular,
            E_squared=E_squared
        )

        xi_pol = jf.alignment_rate_from_torque(
            kappa=kappa_pol,
            rotational_drag=rotational_drag
        )

        p_dot_pol = jf.uniaxial_alignment_drift(
            p=p,
            axis=e_pol,
            alignment_rate=xi_pol
        )

        p_dot = p_dot + p_dot_pol
    return np.concatenate((R_dot, p_dot))


# ============================================================
# 12) INITIAL STATE AND INTEGRATION
# ============================================================

state0 = np.concatenate((R0, p0))

sol = solve_ivp(
    fun=coupled_rhs,
    t_span=(t0, tf),
    y0=state0,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-12
)

if not sol.success:
    raise RuntimeError(
        f"The integration failed: {sol.message}"
    )


# ============================================================
# 13) EXTRACT RESULTS
# ============================================================

state_history = sol.y.T

R = state_history[:, 0:3]
P_raw = state_history[:, 3:6]

norm_raw = np.linalg.norm(P_raw, axis=1)
norm_error = np.abs(norm_raw - 1.0)

P = jf.normalize_rows(P_raw)

px = P[:, 0]
py = P[:, 1]
pz = P[:, 2]

theta_rad, phi_wrapped, phi_unwrapped = (
    jf.directors_to_angles(P)
)

theta_deg = np.rad2deg(theta_rad)
phi_deg = np.rad2deg(phi_unwrapped)

# Center-of-mass coordinates in micrometers
R_um = 1.0e6 * R

x_um = R_um[:, 0]
y_um = R_um[:, 1]
z_um = R_um[:, 2]

distance_from_focus_um = np.linalg.norm(
    R_um,
    axis=1
)

potential_history = np.array([
    optical_potential(Ri)
    for Ri in R
])

force_history = np.array([
    optical_force(Ri)
    for Ri in R
])

force_magnitude_pN = (
    np.linalg.norm(force_history, axis=1) * 1.0e12
)

intensity_history = np.array([
    jf.gaussian_intensity_factor(
        R=Ri,
        w0=w0,
        zR=zR
    )
    for Ri in R
])


# ============================================================
# 14) INITIAL DRIFT DIAGNOSTICS
# ============================================================

F0 = optical_force(R0)

v_flow_0 = G @ R0

v_opt_0 = jf.overdamped_translational_drift(
    force=F0,
    mobility=translational_mobility
)

v_swim_0 = swimming_speed * p0

p_dot_h_0 = jf.jeffery_rhs_vector(
    t=0.0,
    d=p0,
    E=E,
    W=W,
    lam=beta
)

if USE_POSITION_DEPENDENT_TORQUE:
    R_torque_0 = R0
else:
    R_torque_0 = np.zeros(3, dtype=float)


kappa_grad_initial = jf.gradient_torque_strength(
    R=R_torque_0,
    alpha_v=alpha_v,
    volume=Volume,
    body_points=body_points,
    field_squared_fn=field_squared,
    preferred_axis=beam_axis,
    perpendicular_axis=gradient_perpendicular_axis
)

xi_grad_initial = jf.alignment_rate_from_torque(
    kappa=kappa_grad_initial,
    rotational_drag=rotational_drag
)


E_squared_initial = field_squared(
    R_torque_0
)

kappa_pol_initial = jf.polarization_torque_strength(
    alpha_parallel=alpha_parallel,
    alpha_perpendicular=alpha_perpendicular,
    E_squared=E_squared_initial
)

xi_pol_initial = jf.alignment_rate_from_torque(
    kappa=kappa_pol_initial,
    rotational_drag=rotational_drag
)

p_dot_grad_0 = jf.uniaxial_alignment_drift(
    p=p0,
    axis=beam_axis,
    alignment_rate=xi_grad_initial
)

p_dot_pol_0 = jf.uniaxial_alignment_drift(
    p=p0,
    axis=e_pol,
    alignment_rate=xi_pol_initial
)

print()
print("====================================================")
print("INITIAL PARAMETERS")
print("====================================================")
print("aspect ratio =", aspect_ratio)
print("beta =", beta)
print("w0 =", w0, "m")
print("zR =", zR, "m")
print("U0/kBT =", U0 / kBT)
print("kx =", kx, "N/m")
print("ky =", ky, "N/m")
print("kz =", kz, "N/m")
print("translational mobility =", translational_mobility)
print()

print("====================================================")
print("INITIAL TRANSLATIONAL DRIFTS")
print("====================================================")
print("R0 =", R0, "m")
print("F_opt(R0) =", F0, "N")
print("u_flow(R0) =", v_flow_0, "m/s")
print("v_opt(R0) =", v_opt_0, "m/s")
print("v_swim(R0) =", v_swim_0, "m/s")
print()

print("====================================================")
print("INITIAL ORIENTATIONAL DRIFTS")
print("====================================================")
print("||p_dot_Jeffery|| =", np.linalg.norm(p_dot_h_0))
print("||p_dot_gradient|| =", np.linalg.norm(p_dot_grad_0))
print("||p_dot_polarization|| =", np.linalg.norm(p_dot_pol_0))
print("xi_grad(R0) =", xi_grad_initial, "1/s")
print("xi_pol(R0) =", xi_pol_initial, "1/s")
print()


# ============================================================
# 15) INITIAL AND FINAL STATES
# ============================================================

R_initial = R[0]
R_final = R[-1]

p_initial = jf.normalize_vector(P[0])
p_final = jf.normalize_vector(P[-1])

theta_initial = np.rad2deg(
    np.arccos(np.clip(p_initial[2], -1.0, 1.0))
)

phi_initial = np.rad2deg(
    np.arctan2(p_initial[1], p_initial[0])
)

theta_final = np.rad2deg(
    np.arccos(np.clip(p_final[2], -1.0, 1.0))
)

phi_final = np.rad2deg(
    np.arctan2(p_final[1], p_final[0])
)

print("====================================================")
print("INITIAL AND FINAL STATES")
print("====================================================")
print("R0 =", R_initial, "m")
print("Rf =", R_final, "m")
print()
print("p0 =", p_initial)
print("||p0|| =", np.linalg.norm(p_initial))
print("theta0 =", theta_initial, "deg")
print("phi0 =", phi_initial, "deg")
print()
print("pf =", p_final)
print("||pf|| =", np.linalg.norm(p_final))
print("thetaf =", theta_final, "deg")
print("phif =", phi_final, "deg")
print()
print("maximum norm error =", np.max(norm_error))

print("===================================================")
print("alpha_scalar =", alpha_scalar)
print("alpha_parallel =", alpha_parallel)
print("alpha_perpendicular =", alpha_perpendicular)
print("rotational_drag =", rotational_drag)
print()

print("orientational drifts")
print("kappa_grad(R0) =", kappa_grad_initial, "J")
print("kappa_pol(R0) =", kappa_pol_initial, "J")


# ============================================================
# 16) MATPLOTLIB GRAPHICS
# ============================================================

if plane == "xy":
    flow_label = rf"$\mathbf{{u}}=({gamma:g}y,0,0)$"
elif plane == "xz":
    flow_label = rf"$\mathbf{{u}}=({gamma:g}z,0,0)$"
else:
    flow_label = r"$\mathbf{u}$"

gradient_label = (
    "ON" if USE_GRADIENT_ALIGNMENT else "OFF"
)

polarization_label = (
    "ON" if USE_POLARIZATION_ALIGNMENT else "OFF"
)

trap_label = (
    f"{TRAP_MODEL}, ON"
    if USE_COM_TRAP
    else "OFF"
)

title = (
    rf"$\mathbf{{R}}(t),\mathbf{{p}}(t)$"
    "\n"
    + flow_label
    + "\n"
    + rf"$r={aspect_ratio:.3f},\ "
      rf"\beta={beta:.3f},\ "
      rf"\xi_{{grad}}(R_0)={xi_grad_initial:.2e},\ "
      rf"\xi_{{pol}}(R_0)={xi_pol_initial:.2e}$"
    "\n"
    + rf"$\mathrm{{COM\ trap}}={trap_label},\ "
      rf"\mathrm{{gradient}}={gradient_label},\ "
      rf"\mathrm{{polarization}}={polarization_label}$"
)

fig, axs = plt.subplots(
    3,
    2,
    figsize=(12, 10)
)

# ------------------------------------------------------------
# COM components
# ------------------------------------------------------------

axs[0, 0].plot(
    sol.t,
    x_um,
    label=r"$x$",
    linewidth=1.7
)

axs[0, 0].plot(
    sol.t,
    y_um,
    label=r"$y$",
    linewidth=1.7
)

axs[0, 0].plot(
    sol.t,
    z_um,
    label=r"$z$",
    linewidth=1.7
)

axs[0, 0].set_title(
    "Center-of-mass position"
)

axs[0, 0].set_xlabel(r"$t\ [\mathrm{s}]$")
axs[0, 0].set_ylabel(r"$R_i\ [\mu\mathrm{m}]$")
axs[0, 0].grid(True, alpha=0.8)
axs[0, 0].legend()


# ------------------------------------------------------------
# Orientation components
# ------------------------------------------------------------

axs[0, 1].plot(
    sol.t,
    px,
    label=r"$p_x$",
    linewidth=1.7
)

axs[0, 1].plot(
    sol.t,
    py,
    label=r"$p_y$",
    linewidth=1.7
)

axs[0, 1].plot(
    sol.t,
    pz,
    label=r"$p_z$",
    linewidth=1.7
)

axs[0, 1].set_title(
    r"Components of $\mathbf{p}(t)$"
)

axs[0, 1].set_xlabel(r"$t\ [\mathrm{s}]$")
axs[0, 1].set_ylabel(r"$p_i$")
axs[0, 1].grid(True, alpha=0.8)
axs[0, 1].legend()


# ------------------------------------------------------------
# theta(t)
# ------------------------------------------------------------

axs[1, 0].plot(
    sol.t,
    theta_deg,
    label=r"$\theta(t)$",
    linewidth=1.8
)

axs[1, 0].set_title(r"$\theta(t)$")
axs[1, 0].set_xlabel(r"$t\ [\mathrm{s}]$")
axs[1, 0].set_ylabel(r"$\theta\ [^\circ]$")
axs[1, 0].grid(True, alpha=0.8)
axs[1, 0].legend()


# ------------------------------------------------------------
# phi(t)
# ------------------------------------------------------------

axs[1, 1].plot(
    sol.t,
    phi_deg,
    label=r"$\phi(t)$",
    linewidth=1.8
)

axs[1, 1].set_title(r"$\phi(t)$")
axs[1, 1].set_xlabel(r"$t\ [\mathrm{s}]$")
axs[1, 1].set_ylabel(r"$\phi\ [^\circ]$")
axs[1, 1].grid(True, alpha=0.8)
axs[1, 1].legend()


# ------------------------------------------------------------
# Optical potential
# ------------------------------------------------------------

axs[2, 0].plot(
    sol.t,
    potential_history / kBT,
    label=r"$U_{\mathrm{COM}}/k_BT$",
    linewidth=1.8
)

axs[2, 0].set_title(
    "Optical COM potential"
)

axs[2, 0].set_xlabel(r"$t\ [\mathrm{s}]$")
axs[2, 0].set_ylabel(r"$U/k_BT$")
axs[2, 0].grid(True, alpha=0.8)
axs[2, 0].legend()


# ------------------------------------------------------------
# Orientation norm error
# ------------------------------------------------------------

axs[2, 1].plot(
    sol.t,
    np.maximum(norm_error, 1.0e-16),
    label=r"$|\|\mathbf{p}\|-1|$",
    linewidth=1.6
)

axs[2, 1].set_yscale("log")

axs[2, 1].set_title(
    "Orientation norm error"
)

axs[2, 1].set_xlabel(r"$t\ [\mathrm{s}]$")
axs[2, 1].set_ylabel(
    r"$|\|\mathbf{p}\|-1|$"
)

axs[2, 1].grid(True, alpha=0.8)
axs[2, 1].legend()

fig.suptitle(
    title,
    fontsize=14,
    y=0.995
)

plt.tight_layout(
    rect=[0.0, 0.0, 1.0, 0.91]
)

plt.show()


# ============================================================
# 17) ADDITIONAL COM DIAGNOSTICS
# ============================================================

fig_com, axs_com = plt.subplots(
    1,
    2,
    figsize=(11, 4.3)
)

axs_com[0].plot(
    sol.t,
    distance_from_focus_um,
    linewidth=1.8
)

axs_com[0].set_title(
    "Distance from trap center"
)

axs_com[0].set_xlabel(r"$t\ [\mathrm{s}]$")
axs_com[0].set_ylabel(r"$\|\mathbf{R}\|\ [\mu\mathrm{m}]$")
axs_com[0].grid(True, alpha=0.8)

axs_com[1].plot(
    sol.t,
    force_magnitude_pN,
    label=r"$\|\mathbf{F}_{opt}\|$",
    linewidth=1.8
)

axs_com[1].set_title(
    "Optical-force magnitude"
)

axs_com[1].set_xlabel(r"$t\ [\mathrm{s}]$")
axs_com[1].set_ylabel(r"$\|\mathbf{F}_{opt}\|\ [\mathrm{pN}]$")
axs_com[1].grid(True, alpha=0.8)
axs_com[1].legend()

plt.tight_layout()
plt.show()


# ============================================================
# 18) 3D COM TRAJECTORY
# ============================================================

plotter_com = pv.Plotter(
    window_size=(1100, 800)
)

plotter_com.set_background("white")

com_path = pv.lines_from_points(R_um)

plotter_com.add_mesh(
    com_path,
    color="red",
    line_width=5,
    label="R(t)"
)

R0_um = R_um[0]
Rf_um = R_um[-1]

start_com = pv.PolyData(
    R0_um.reshape(1, 3)
)

end_com = pv.PolyData(
    Rf_um.reshape(1, 3)
)

trap_center = pv.PolyData(
    np.zeros((1, 3))
)

plotter_com.add_mesh(
    start_com,
    color="green",
    point_size=12,
    render_points_as_spheres=True
)

plotter_com.add_mesh(
    end_com,
    color="blue",
    point_size=12,
    render_points_as_spheres=True
)

plotter_com.add_mesh(
    trap_center,
    color="black",
    point_size=14,
    render_points_as_spheres=True
)

plotter_com.add_axes(
    line_width=3,
    xlabel="x",
    ylabel="y",
    zlabel="z"
)

plotter_com.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
    color="black",
    xtitle="x [um]",
    ytitle="y [um]",
    ztitle="z [um]",
    font_size=16
)

plotter_com.add_text(
    "Center-of-mass trajectory\n"
    f"trap = {trap_label}\n"
    f"shear plane = {plane}",
    position="upper_left",
    font_size=16,
    color="black"
)

plotter_com.add_legend(
    labels=[
        ["R(t)", "red"],
        ["t_0", "green"],
        ["t_f", "blue"],
        ["trap center", "black"]
    ],
    bcolor="white",
    border=False,
    face="circle",
    size=(0.18, 0.16)
)

plotter_com.camera_position = "iso"
plotter_com.camera.zoom(1.15)

plotter_com.show()


# ============================================================
# 19) ORIENTATION TRAJECTORY ON S^2
# ============================================================

plotter_orientation = pv.Plotter(
    window_size=(1100, 800)
)

plotter_orientation.set_background("white")

sphere = pv.Sphere(
    radius=1.0,
    theta_resolution=80,
    phi_resolution=80
)

plotter_orientation.add_mesh(
    sphere,
    color="lightgray",
    opacity=0.22,
    smooth_shading=True,
    specular=0.15,
    show_edges=False
)

orientation_path = pv.lines_from_points(P)

plotter_orientation.add_mesh(
    orientation_path,
    color="red",
    line_width=4,
    label="p(t)"
)

start_orientation = pv.PolyData(
    p_initial.reshape(1, 3)
)

end_orientation = pv.PolyData(
    p_final.reshape(1, 3)
)

plotter_orientation.add_mesh(
    start_orientation,
    color="green",
    point_size=8,
    render_points_as_spheres=True
)

plotter_orientation.add_mesh(
    end_orientation,
    color="blue",
    point_size=8,
    render_points_as_spheres=True
)

arrow_t0 = pv.Arrow(
    start=(0.0, 0.0, 0.0),
    direction=p_initial,
    tip_length=0.22,
    tip_radius=0.05,
    shaft_radius=0.018,
    scale=0.98
)

arrow_tf = pv.Arrow(
    start=(0.0, 0.0, 0.0),
    direction=p_final,
    tip_length=0.22,
    tip_radius=0.05,
    shaft_radius=0.018,
    scale=0.98
)

plotter_orientation.add_mesh(
    arrow_t0,
    color="green"
)

plotter_orientation.add_mesh(
    arrow_tf,
    color="blue"
)

# Beam axis
beam_arrow = pv.Arrow(
    start=(0.0, 0.0, 0.0),
    direction=beam_axis,
    tip_length=0.18,
    tip_radius=0.04,
    shaft_radius=0.012,
    scale=0.85
)

plotter_orientation.add_mesh(
    beam_arrow,
    color="orange"
)

# Polarization axis
polarization_arrow = pv.Arrow(
    start=(0.0, 0.0, 0.0),
    direction=e_pol,
    tip_length=0.18,
    tip_radius=0.04,
    shaft_radius=0.012,
    scale=0.85
)

plotter_orientation.add_mesh(
    polarization_arrow,
    color="purple"
)

plotter_orientation.add_axes(
    line_width=3,
    xlabel="x",
    ylabel="y",
    zlabel="z"
)

plotter_orientation.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
    color="black",
    xtitle="px",
    ytitle="py",
    ztitle="pz",
    font_size=18
)

plotter_orientation.add_text(
    "p(t) on S^2\n"
    f"gradient alignment = {gradient_label}\n"
    f"polarization alignment = {polarization_label}",
    position="upper_left",
    font_size=16,
    color="black"
)

plotter_orientation.add_legend(
    labels=[
        ["p(t)", "red"],
        ["t_0", "green"],
        ["t_f", "blue"],
        ["beam axis", "orange"],
        ["polarization", "purple"]
    ],
    bcolor="white",
    border=False,
    face="circle",
    size=(0.20, 0.20)
)

plotter_orientation.camera_position = "iso"
plotter_orientation.camera.zoom(1.15)

plotter_orientation.show()