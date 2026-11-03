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