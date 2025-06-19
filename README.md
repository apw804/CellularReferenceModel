# Cellular Reference Model (CRM)

A lightweight, fast tool for simulating wireless cellular networks at a system level, designed for researchers in industry and academia. Aligns with 5G principles while providing efficient computation through vectorized operations and 3GPP compliance.

## Project Overview

The Cellular Reference Model (CRM) is a streamlined system-level cellular network simulation framework specifically designed for research applications in both industry and academia. This project is inspired by the [AIMM Simulator](https://github.com/keithbriggs/AIMM-simulator), authored by Keith Briggs, and represents a collaboration between academic and industrial research perspectives.

While CRM aligns with 5G principles and incorporates relevant 3GPP standards, it is intentionally not a full 5G system recreation. Instead, it provides researchers with a lightweight, computationally efficient tool for rapid prototyping, algorithm development, and network analysis.

CRM focuses on the essential elements of cellular network modeling - cell and user equipment (UE) layouts, path loss calculations, and energy efficiency analysis - while maintaining high computational performance through vectorized operations. This makes it ideal for researchers who need fast iteration cycles and the ability to test concepts without the complexity and computational overhead of full system simulators.

## Key Features

### Research-Oriented Design
- **Lightweight Framework**: Streamlined implementation focused on core research needs.
- **Fast Execution**: Optimized for rapid prototyping and iterative algorithm development.
- **Academic & Industrial Use**: Designed specifically for research environments and requirements.
- **5G Principles Alignment**: Incorporates relevant 5G concepts without full system complexity.

### Efficient Network Modeling
- **Hexagonal Cell Grid**: Standard 3GPP-compliant hexagonal cell layout for realistic network topologies.
- **Flexible UE Distribution**: Configurable user equipment placement using Poisson point process or uniform distribution.
- **Vectorized Operations**: Optimized computational performance through vectorized calculations for rapid analysis.
- **Scalable Simulations**: Efficient handling of various network sizes suitable for research scenarios.

### Research-Focused 3GPP Compliance
- **Essential Path Loss Models**: Support for key models including free-space, RMa (Rural Macrocell), UMa (Urban Macrocell).
- **LOS/NLOS Scenarios**: Line-of-sight and non-line-of-sight propagation modeling for comprehensive analysis.
- **Standards Alignment**: Implementation follows relevant 3GPP simulation protocols for research validity.
- **5G-Relevant Parameters**: Incorporates 5G-aligned parameters without full system complexity.

### Research Visualization & Analysis
- **Quick Visualization**: Fast rendering of UEs, cells, and their attachment relationships for immediate feedback.
- **Research Metrics**: System performance indicators tailored for academic and industrial research needs.
- **Rapid Prototyping Support**: Visual tools designed for iterative research and development.

### Energy Efficiency Research Tools
- **Energy Score Calculation**: Assessment of energy efficiency for each base station-to-UE link.
- **Research-Grade Metrics**: Analysis based on radiated transmit power suitable for academic publications.
- **Throughput-to-Power Analysis**: Essential efficiency metrics for energy research applications.

### Algorithm Research Platform
- **Optimization Testing Ground**: Framework for testing and comparing optimization algorithms.
- **Research Algorithm Implementation**: Built-in random search, hill climbing, and simulated annealing.
 for benchmarking.
- **Custom Algorithm Integration**: Easy integration of novel optimization approaches for research.
- **Benchmarking Capabilities**: Standardized platform for algorithm performance comparison.

## Research Applications

### Academic Research
- Algorithm development and testing.
- Energy efficiency studies.
- Network optimization research.
- Comparative analysis of optimization techniques.
- Rapid concept validation.

### Industrial Research
- Prototype algorithm development.
- Proof-of-concept demonstrations.
- Benchmarking studies.
- Research collaboration with academia.
- Fast iteration for R&D projects.

## Performance Characteristics

- **Research-Speed Computation**: Vectorized operations optimized for research iteration cycles.
- **Lightweight Architecture**: Minimal overhead for maximum research productivity.
- **Fast Simulation Cycles**: Quick turnaround for hypothesis testing and validation.
- **Memory Efficient**: Optimized for running multiple experiments and parameter sweeps.
- **Research-Scale Ready**: Designed for typical research scenario sizes and computational resources.

## Contributors

This project is a collaboration between:
- **Kishan Sthankiya** - Queen Mary University of London
- **Keith Briggs** - BT

## Acknowledgments

Special thanks to Keith Briggs for his invaluable contributions and expertise in cellular network simulation. This project builds upon concepts and methodologies developed in the [AIMM Simulator](https://github.com/keithbriggs/AIMM-simulator), which provided inspiration and foundational insights for the CRM framework.

The collaboration between academic and industrial perspectives has been essential in creating a tool that serves both research communities effectively.
