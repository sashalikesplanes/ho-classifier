# HO-Classifier: Classification of Human Inputs
This project is a research experiment conducted at Control & Simulation, TU Delft during the academic years 2021-2022. It explores the possibility of a machine learning algorithm classifying the type of single integrator control task being completed by a human operator. The project was conducted on 10 participants, with the data gathered, cleaned, processed, and labeled to create a dataset for further research. A neural network classifier was implemented and trained to analyze performance and potential applications in control software.

## Tech Stack
The project was implemented using Python, MATLAB, PyTorch, Numpy, and Pandas. Google Cloud Platform and Gradient Paperspace were leveraged for computational resources.

## Project Structure
The project is organized as follows:

- `.`: Main files for running the neural network training and inference.
- `Artifacts`: This directory contains Weights and Biases files which keep track of the model training runs.
- `Confusion_Matrices`: This directory contains the raw data used to produce the confusion matrices for the classifier.
- `Data`: Here you'll find the experimental data in MATLAB format that was gathered during the research.
- `Data_Visualisation`: This directory hosts Jupyter notebooks used to produce graphs for research analysis.
- `Learners`: This directory is an archive of some of the models which were trained and saved during the course of the project.
- `Misc`: A collection of various helper scripts used throughout the project.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Acknowledgements
This research project was conducted during the academic years 2021-2022 as part of a program at Control & Simulation, TU Delft.

