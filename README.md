# Predicting Cardiovascular Disease Using Machine Learning
This project leverages machine learning techniques to predict the likelihood of an individual developing Major Ischemic Heart Disease (MICHD) based on various lifestyle factors. Cardiovascular diseases, such as heart attacks, pose a significant global health threat. Early detection and prevention can save lives, and machine learning offers powerful tools to analyze complex health data and identify at-risk individuals. Our project involves comprehensive data preprocessing, implementation of several classification algorithms, and rigorous evaluation to determine the most effective model for predicting heart disease risk.

## Objective
The objective of this project is to implement machine learning models to analyze a health dataset and predict the risk of heart disease. The project aims to:
1. Perform data preprocessing and cleaning.
2. Implement and evaluate various binary classification models.
3. Compare the performance of different models.
4. Document the findings and methodology in a concise report.

## Dataset
The dataset used in this project comes from the Behavioral Risk Factor Surveillance System (BRFSS), a health-related telephone survey system collecting data on the health behaviors and conditions of U.S. residents.

## File List

1. **README.md**
   - Provides an overview of the project and instructions on how to test the implementations.

2. **implementations.py**
   - Contains six functions, fully commented as specified in the project statement. This module is entirely independent of the rest of the project.

3. **run.py**
   - Generates the submitted .csv file for the AIcrowd competition. To test it, execute the command: `python run.py`. The script will take some time to complete, with a progress report shown in the console. Note that the script loads data from the local root path "data/dataset_to_release_2."

4. **best_model.ipynb**
   - The core of our project, this Jupyter notebook includes code and details about each algorithm tried, leading to our final algorithm used in `run.py`. Some cells may take a while to run. The notebook is fully executed, and all cells are evaluated.

5. **comparison.ipynb**
   - In this notebook, we compare our model with the one from `sklearn`. We make claims and draw conclusions, separating this comparison into a distinct file to emphasize that `best_model.ipynb` has no dependencies on other libraries. All cells are fully executed and evaluated.

6. **report.pdf**
   - Contains detailed findings, explaining our assumptions, claims, conclusions, findings, and any mistakes made during the project. Unlike `best_model.ipynb`, this document does not contain any code but provides an in-depth explanation of our thought process and intuition behind implementing the model.
  
## Testing Instructions

To test our project, follow these steps:

1. Execute `python run.py` to generate the AIcrowd submission .csv file. Be patient, as it might take some time.

2. Refer to the `best_model.ipynb` and `comparison.ipynb` notebooks for detailed insights and comparisons.

3. Review the `report.pdf` for a comprehensive explanation of our findings.

If you have any questions or encounter issues, please don't hesitate to reach out to the authors for assistance. We hope you find our project informative and insightful.


