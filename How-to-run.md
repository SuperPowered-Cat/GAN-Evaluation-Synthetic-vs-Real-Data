# HOW TO RUN THE CODE
## Clone the repository
```
git clone https://github.com/SuperPowered-Cat/gan-svm-evaluation.git
cd gan-svm-evaluation
```
## Install Dependancies
First ensure you have Python  3.8+ installed. Then, install the necessary packages:
```
pip install -r requirements.txt
```
## Data Preprocessing
Run the data_preprocessing.py script to preprocess and scale the data:
```
python scripts/data_preprocessing.py
```
This will generate the scaled data arrays and store them in the /data directory.
It is recommended to use various datasets for seeing different results.
## Train the GAN and Generate Synthetic Data
Run the gan_model.py script to define, compile, and train the GAN. This script will also generate synthetic data:
```
python scripts/gan_model.py
```
The synthetic data will be saved in the /data directory as synthetic_data.npy and synthetic_y.npy.
## Evaluate SVM on Real vs Synthetic Data
Run the evaluation.py script to train and evaluate the SVM on real and synthetic data:
```
python scripts/evaluation.py
```
The results will be saved in the /results directory.
## Evaluate SVM on Mixed Real and Synthetic Data
Run the evaluation_mixed.py script to evaluate the performance of SVM when trained on a mix of real and synthetic data:
```
python scripts/evaluation_mixed.py
```
## Output
- The evaluation scripts will produce accuracy metrics for different ratios of training data, as well as comparisons between real and synthetic data.
- The final results will be plotted and saved as images in the /results directory, showcasing the performance differences.
