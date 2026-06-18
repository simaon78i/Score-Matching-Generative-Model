# Score Matching Generative Model

This repository provides a modular PyTorch implementation of a Score-Matching Generative Model. It is designed based on the principles of denoising diffusion models (following the framework established by Karras et al. 2022). The codebase is structured to facilitate flexible training, configurable noise scheduling, and multiple sampling strategies.

## Project Structure

* **main.py**: The central entry point for training the model and evaluating its generative performance.
* **score_matching.py**: Contains the core `ScoreMatchingModel` class, implementing Karras et al. pre-conditioning, loss calculation, and the denoising process.
* **samplers.py**: Implements various ODE/SDE solvers (`EulerSampler`, `MultistepDPMSampler`) used for the image generation process.

## Key Features

* **Advanced Pre-conditioning**: Implements stable pre-conditioning strategies for noise-conditional models.
* **Flexible Training**: Supports `l1` and `l2` loss functions with customizable weighting strategies (`ones`, `snr`, `karras`, `min_snr`).
* **Versatile Sampling**: Allows switching between deterministic ODE solvers and stochastic SDE sampling methods.
* **Modularity**: The system is designed with a clear separation between the neural network architecture, noise schedules, and sampling logic, allowing for easy experimentation.

## Prerequisites

Ensure that you have the following libraries installed in your environment:

* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Einops
* Scikit-learn

## Usage

### Training the Model
To start the training process, run the `main.py` script. You can customize the training process by passing arguments directly to the script, such as the number of iterations, batch size, and the compute device (e.g., cuda, cpu, mps).

### Sampling and Evaluation
Once the training is complete, `main.py` automatically performs:
1. **Denoising Visualization**: Generates a grid showing the progression from noisy input to the final clean image.
2. **Distribution Analysis**: Computes statistical metrics (such as pixel variance and Total Variation) to compare the statistics of the generated images against the ground truth data.

## Customization

* **Adding New Samplers**: You can implement custom samplers by inheriting from the `Sampler` base class defined in `samplers.py` and implementing the required step method.
* **Noise Scheduling**: The model supports interchangeable noise schedules. You can define custom scheduling logic by ensuring compatibility with the expected interface and updating the `ScoreMatchingModelConfig`.

## References
The architecture and implementation methodology are based on the work: *Elucidating the Design Space of Diffusion-Based Generative Models* (Karras et al., 2022).
