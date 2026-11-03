<h1 align="center">Particle OT dynamics</h1>

<p align="center">
    <strong>Luis Ivan Cerezo Zepeda <sup>1</sup></strong>,
    <strong>Raúl Ignacio Hernández Aranda<sup>1</sup></strong><br><br>

    <small>
        <sup>1</sup> School of Engineering and Sciences, Tecnológico de Monterrey, Mexico
    </small>
</p>

This repository contains numerical models developed to study the translational and orientational dynamics of elongated microalgae subjected to fluid flow and trapped with an Optical Tweezer. 

Depending the case, the model combines hydrodynamic rotation, active swimming, optical trapping and optical induced aligment to investigate the competition between fluid-particle and light-particle interactions. The diagram below represents how must be interpreted the simulations.

<p align="center">
  <img src="steps/Imagenes/Imagen1.svg"
       alt="Overview of the microalgae optical dynamics model"
       width="700">
</p>

The microalgae is represented by its center of mass position

$$
\mathbf{R}(t)=(x,y,z)
$$
and its orientation vector 
$$\mathbf{p}=(p_x, p_y, p_z),\qquad ||\mathbf{p}|| = 1 $$

Its orientational dynamics are modeled from Jeffery's equation and progressively extended to include optical aligment torques, while its translational dynamics account for background flow, active swimming and optical trapping forces. 

## Relevant parameters 

### Microalgae parameters

$$ \mathbf{L} \mathbf{ }(\text{Length}), \qquad \mathbf{D} \mathbf{ }(\text{Diameter})$$

This parameters controls the geometry and Jeffery through the aspect ratio.

$$ \mathbf{v}_s \mathbf{ }(\text{velocity of the active particle})$$

$$ \mathbf{\eta} \mathbf{ }(\text{viscosity})$$

### Flux parameter

$$ \dot\gamma \mathbf{ }(\text{Shear rate})$$

More shear rate makes the Jeffery dynamics happen faster.

### Laser parameters 
$$ \mathbf{P} \mathbf{ }(\text{Laser power})$$

$$ \mathbf{w}_0 \mathbf{ }(\text{Beam waist})$$

$\mathbf{P}$ controls how the intensity increases or decreseas, hence the intensity of the optical torque and alligment. 

$\mathbf{w}_0$ controls the intensity and the gradients experimented, specially the perpendicular stiffness $k_\perp$ ($k_x$, $k_y$)

### Optic parameters
$$ \mathbf{n}_p \mathbf{ }(\text{particle refractive index})$$

$$ \mathbf{n}_m \mathbf{ }(\text{Medium refractive index})$$

### Initial parameters 
$$ \mathbf{R} \mathbf{ }(\text{Initial position})$$

$$ \mathbf{\theta}_0 \mathbf{ }(\text{Azimutal angle})$$

$$ \mathbf{\phi}_0 \mathbf{ }(\text{angle with respect to z})$$

R is important in gaussian simulations, getting away from the origin will change the intensity experimented by the particle.

$\theta_0$ and $\phi_0$ determines which orbit the jeffery orbit will follow; which is the base of the simulations. 


