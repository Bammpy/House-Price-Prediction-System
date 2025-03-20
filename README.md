A machine learning tool to predict house prices using Random Forest and SHAP explanations.

Features
GUI built with Tkinter
Predicts prices based on bedrooms, location, utilities, etc.
Visualizes feature importance with SHAP
Setup
Clone: `git clone https://github.com/yourusername/housepred.git\`
Create env: `python3 -m venv vm`
Activate: `source vm/bin/activate`
Install: `pip install -r requirements.txt`
Run: `jupyter notebook`
Files
`House Price Prediction System.ipynb`: Main notebook with GUI.
`train_model.py`: Script to train the Random Forest model.
`house_price_model.pkl`: Pre-trained model file. " > README.md
Replace `yourusername` with your GitHub username.

Commit and Push: 
git add README.md
git commit -m "Added README file"
git push origin master

uppress Tkinter Warning: The Glyph 128269 warning might appear when others run your code. Add this to the top of your .ipynb:
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
then re-commit:
git add "House Price Prediction System.ipynb"
git commit -m "Suppressed Tkinter glyph warning"
git push
The model currently achieves perfect R² scores (1.00) due to a small synthetic dataset. For production, a larger dataset is recommended to avoid overfitting." >> README.md
git add README.md
git commit -m "Added note about R² scores"
git push




