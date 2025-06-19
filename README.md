# Cellular Reference Model (CRM)

A lightweight, fast tool for simulating wireless cellular networks at a system level, designed for researchers in industry and academia. Aligns with 5G principles while providing efficient computation through vectorized operations and 3GPP compliance.

## Project Overview

The Cellular Reference Model (CRM) is a streamlined system-level cellular network simulation framework specifically designed for research applications in both industry and academia. This project is inspired by the [AIMM Simulator](https://github.com/keithbriggs/AIMM-simulator), authored by [Keith Briggs](https://keithbriggs.info/), and represents a collaboration between academic and industrial research perspectives.

While CRM aligns with 5G principles and incorporates relevant 3GPP standards, it is intentionally not a full 5G system recreation. Instead, it provides researchers with a lightweight, computationally efficient tool for rapid prototyping, algorithm development, and network analysis.

CRM focuses on the essential elements of cellular network modeling - cell and user equipment (UE) layouts, path loss calculations, and energy efficiency analysis - while maintaining high computational performance through vectorized operations. This makes it ideal for researchers who need fast iteration cycles and the ability to test concepts without the complexity and computational overhead of full system simulators.

## Key Features

### Core Framework
CRM provides a **lightweight, research-focused simulation environment** that incorporates 5G principles without the computational overhead of full system simulators. The framework is designed for both academic and industrial research environments, enabling fast iteration cycles through **vectorized operations** and optimized performance. This makes it ideal for algorithm development, energy efficiency studies, and rapid prototyping where researchers need quick results without sacrificing technical accuracy.

### Network Modeling & 3GPP Compliance
The simulator features **hexagonal cell grids** with configurable UE distribution using Poisson or uniform placement strategies. CRM implements **3GPP-compliant path loss models** including free-space, RMa (Rural Macrocell), and UMa (Urban Macrocell) scenarios with complete LOS/NLOS propagation modeling. This standards alignment ensures research validity while maintaining the computational efficiency needed for multiple experiments.

### Research Applications & Performance
CRM serves as a comprehensive platform for **optimization algorithm testing and comparison**, including built-in implementations of random search, hill climbing, and simulated annealing with support for custom algorithm integration. CRM delivers fast execution suitable for typical research computational resources while enabling scalable simulations across various network sizes.

## Contributors

This project is a collaboration between:
- **[Kishan Sthankiya](https://orcid.org/0009-0002-1935-0476)** - Queen Mary University of London
- **[Keith Briggs](https://keithbriggs.info/)** - BT

## Academic Integrity Declaration

The concepts, algorithms, and source code implementation in this project are the original intellectual contributions of the authors listed above. However, in the interest of full transparency and academic integrity, we declare that:

- **Documentation**: This README file has been rewritten and formatted with assistance from an AI language model (GitHub Copilot/Claude) to improve clarity, structure, and presentation.
- **Code Documentation**: Docstrings throughout the source code have been enhanced and standardized with AI assistance to improve readability and completeness.
- **Content Verification**: All AI-generated documentation has been thoroughly reviewed and verified for technical accuracy by the authors.
- **Original Work**: The underlying research contributions, methodologies, algorithmic implementations, and technical work remain the work of the human authors.

This approach ensures clear, professional documentation while maintaining the authenticity and integrity of the research contributions.

## Acknowledgments

Special thanks to Keith Briggs for his invaluable contributions and expertise in cellular network simulation. This project builds upon concepts and methodologies developed in the [AIMM Simulator](https://github.com/keithbriggs/AIMM-simulator), which provided inspiration and foundational insights for the CRM framework. The collaboration between academic and industrial perspectives has been essential in creating a tool that serves both research communities effectively.
