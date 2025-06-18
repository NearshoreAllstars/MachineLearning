# Multi-GPU Exact Gaussian Process Regression

This Python script performs **exact Gaussian Process (GP) regression** using **GPyTorch** with support for **multiple GPUs**. It normalizes the input data, splits it for training and testing, and uses an L-BFGS optimizer to train the GP model. It also supports large-scale predictions efficiently.

## 📍 Author

**Zhenqiang Wang**  
College of Earth, Ocean, and Atmospheric Sciences  
Oregon State University, Corvallis, OR 97331  
📧 [zhenqiang.wang@oregonstate.edu](mailto:zhenqiang.wang@oregonstate.edu)

---

## 🧠 Features

- Exact GP regression using GPyTorch
- Multi-GPU kernel parallelism
- Custom training with L-BFGS optimizer
- Support for high-volume prediction datasets
- Data normalization and train/test split
- Computes test RMSE and prediction time

---

## 📂 Input Files

Ensure the following CSV files are available in the script's working directory:

- `Forcing_750sim_SF.csv`: Feature matrix for training
- `TWL_750sim_SF.csv`: Target vector for training
- `Inputs_GPR_predict_25yr_sim_1.csv`: Feature matrix for prediction

---

## 🛠 Dependencies

- Python 3.8+
- `torch`
- `gpytorch`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `scipy`

Install via:

```bash
pip install torch gpytorch numpy pandas scikit-learn matplotlib scipy
Also ensure the FullBatchLBFGS optimizer is available (likely a custom module under ../LBFGS.py).

🚀 How to Run
Make sure your CUDA environment is set up and GPUs are accessible.

bash
Copy
Edit
python3 multi_gpu_gp_regression.py
The script will:

Normalize the dataset

Train a GP model using all available GPUs

Compute RMSE on a test set

Perform prediction on a large dataset

Report prediction time

📈 Output
Console logs showing:

Training loss, kernel lengthscale, and noise

Test RMSE

Prediction timing

Model is trained using 80% of the input data; 20% is used for testing.

⚙️ Notes
The number of GPUs is detected automatically with torch.cuda.device_count().

The script is adapted from the GPyTorch example on Multi-GPU Exact GP Regression.

📄 License
This software is free to use and distribute. There is no warranty for fitness or merchantability.

📌 Project Structure
bash
Copy
Edit
.
├── multi_gpu_gp_regression.py     # Main script
├── Forcing_750sim_SF.csv          # Input features for training
├── TWL_750sim_SF.csv              # Output values for training
├── Inputs_GPR_predict_25yr_sim_1.csv  # Input features for prediction
├── LBFGS.py                       # Custom L-BFGS optimizer
└── README.md
🙋 Support
For questions or collaboration inquiries, contact zhenqiang.wang@oregonstate.edu.

