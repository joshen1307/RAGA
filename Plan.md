
## Machine Learning for Radio Astronomy.

*Simulating realistic representations of the radio sky at varying resolutions becomes increasingly challenging as radio interferometers increase in both sensitivity and spatial dynamic range. In this project the student will design and implement a deep generative model for different classes of galaxies in radio surveys, with a view to producing SKA-scale survey simulations. The project will work initially with data from the NVSS and FIRST radio surveys and will aim to produce simulated versions of these surveys using a generative model. Depending on progress, this work can be extended to include the development of a generative adversarial network for classification of sources within next generation radio surveys.*
---

### Work Plan

1. Adapt network from arXiv:1903.11921 to include classical attention gates;
2. Adapt network from arXiv:1903.11921 to include sononet attention gates;
3. Compare performance of (1) & (2);
4. Examine spatial attention differences between FIRST and NVSS datasets;
5. Implement *distraction* gates to map spatial regions that detract from accurate classification;
6. Examine spatial distribution of distraction to determine optimal region size for NVSS/FIRST samples.

#### Potential publications

* MNRAS (or similar) astronomy paper on FR classification using astronomy
* NIPS (or similar) paper on dynamic image resizing based on distraction
* \[Potential extension\] *NIPS (or similar) paper on Bayesian attention gates using weighted compatibility functions*

#### Useful Links

* [FRDEEP PyTorch dataset](https://hongmingtang060313.github.io/FR-DEEP/)
* [FRDeep PyTorch Tutorial](https://as595.github.io/frdeepcnn/)
* [NIPS Conference Papers](https://papers.nips.cc)
