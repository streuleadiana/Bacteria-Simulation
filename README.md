# Bacteria Colony Evolution - MPI Simulation

This project represents a simulation of the evolution of a bacterial colony on a rectangular culture surface, implemented in parallel using **MPI (Message Passing Interface)**.

## Problem Description

Simulate the evolution of a colony of bacteria that is grown on a culture surface of rectangular shape. The culture surface is a **2D grid** and bacteria can grow in grid points.

Initially, bacteria are seeded in certain points of the grid. At each point of the grid there can be at most one bacterium. The bacteria colony evolves in time: bacteria can multiply or can die.

### Rules of Evolution
Multiplication and deaths of bacteria happen **synchronously**, and time passes discretely in successive generations. Every generation is a result of the preceding one based on the following rules:

1.  **Multiplication (Spawn):** If there is a group of **3 bacteria** around an empty grid point, a new bacterium will spawn in that point.
2.  **Death (Isolation):** A bacterium dies if it is isolated (it has **less than 2** bacteria around).
3.  **Death (Suffocation):** A bacterium dies if it is suffocated (it has **more than 3** bacteria around).
4.  **Survival:** A bacterium survives if it has exactly 2 or 3 neighbors.

*Note:* Neighbors are considered the 8 cells surrounding a grid point (Moore neighborhood).

## Standard Requirements

The goal of this project is to determine the final configuration of the bacterial colony after a number of `G` generations, given an initial configuration loaded from a file.

**Key Technical Requirements:**
* **Parallel Implementation:** Implement a parallel version using **MPI** and **1D data decomposition** (row-wise split).
* **Portability:** The program must work if deployed on a single localhost or distributed across several hosts.
* **Consistency:** Serial and parallel versions must produce the exact same results (deterministic computation).
* **Communication:** Implement efficient patterns of communication (e.g., handling ghost cells/halos).
* **Input/Output:**
    * Initial configuration read from an input file.
    * Final configuration saved to an output file.
