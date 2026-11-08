# Bird Sim

A Manim animation of a flock of birds (or a school of fish) that moves by three local rules. It is based on:

> Couzin, Krause, James, Ruxton & Franks (2002). _Collective Memory and Spatial Sorting in Animal Groups._ J. Theor. Biol. 218, 1–11. doi:[10.1006/jtbi.2002.3065](https://doi.org/10.1006/jtbi.2002.3065)

## Model

Each animal reacts only to the neighbours it can see:

1. **Repulsion** (r < r<sub>r</sub>): turn away. This rule overrides the other two.
2. **Orientation** (r<sub>r</sub> ≤ r < r<sub>o</sub>): align with the neighbours.
3. **Attraction** (r<sub>o</sub> ≤ r < r<sub>a</sub>): turn toward the neighbours.

Animals cannot see behind them (a blind angle). They turn at a limited rate, move at a constant speed and have random noise in their heading. The yellow arrow shows the group's mean heading **d**<sub>group</sub>, drawn from the group centre **c**<sub>group</sub>.

## Parameters

|                                  | Paper (Fig. 3) | `birds`     | `fish`      |
| -------------------------------- | -------------- | ----------- | ----------- |
| Repulsion r<sub>r</sub>          | 1              | 1 BL        | 1 BL        |
| Orientation width Δr<sub>o</sub> | 0–15           | 7           | 1           |
| Attraction width Δr<sub>a</sub>  | 0–15           | 12          | 23          |
| Field of view α                  | 270°           | 300°        | 240°        |
| Noise σ (per 0.1 s)              | 0.05 rad       | ≈ 0.095 rad | ≈ 0.063 rad |

BL = body length.

The `birds` preset gives a moving parallel flock. The `fish` preset gives a rotating ring ("torus"). The simulation is in 2D, inside a box. The paper's model is 3D, in open space.

## Run

Set `species`, `n_birds` and `seed` in `BirdSimulation`. Example videos are in `Animations/`.
