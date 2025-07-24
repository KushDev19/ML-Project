# 🎯 Student Performance Predictor

A full-stack machine learning application that predicts student math scores based on demographic factors and test performance. Built with production-grade architecture.

## 🚀 Overview

This project implements a complete ML pipeline for predicting student performance using multiple algorithms with automated model selection. The system includes a web interface, comprehensive data processing, and production deployment configurations.

**Live Demo**: **[https://ml-project-nu2y.onrender.com/](https://ml-project-nu2y.onrender.com/)**

## ✨ Features

- **Smart Predictions**: Predicts math scores using reading/writing performance and demographics
- **Multiple ML Models**: Tests 8 different algorithms and selects the best performer
- **Web Interface**: Clean, responsive UI with real-time predictions
- **Production Ready**: Complete with error handling, logging, and cloud deployment
- **Automated Pipeline**: End-to-end data processing and model training

## 📊 Dataset

- **Source**: Students Performance Dataset
- **Size**: 1,000 student records
- **Features**: 8 input features (gender, ethnicity, education, etc.)
- **Target**: Math score (0-100)

## 🤖 Machine Learning Pipeline

### Data Processing
- **Numerical Features**: StandardScaler + SimpleImputer
- **Categorical Features**: OneHotEncoder + SimpleImputer
- **Pipeline**: Automated preprocessing with proper train/test separation

### Models Tested
- Linear Regression
- K-Neighbors Regressor
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost
- AdaBoost

### Model Selection
Automated selection based on R² score with cross-validation to prevent overfitting.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+

### Installation

Clone the repository
git clone https://github.com/KushDev19/ML-Project.git
cd ML-Project

Install dependencies
pip install -r requirements.txt

Run training pipeline
python src/components/data_ingestion.py

Start web application
python application.py


### Usage
1. Open your browser to `http://localhost:5000` (or visit the live demo above)
2. Fill in the student information form
3. Click "Predict Math Score" to get results

## 📈 Model Performance

The system automatically selects the best-performing model based on:
- **R² Score**: Primary evaluation metric
- **Cross-validation**: 5-fold CV for robust performance estimation
- **Mean Squared Error**: Secondary metric for regression quality

## 🎨 Web Interface

- **Modern Design**: Gradient-based UI with responsive layout
- **Real-time Validation**: Input validation with user feedback
- **Error Handling**: Graceful error messages and recovery
- **Mobile Friendly**: Works across different screen sizes

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Flask | Web framework and API |
| **ML Libraries** | Scikit-learn, XGBoost, CatBoost | Model training and inference |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **Frontend** | HTML5, CSS3, JavaScript | User interface |
| **Deployment** | Render.com | Cloud hosting and CI/CD |
| **Model Storage** | Pickle | Model serialization |

## 📋 Input Features

| Feature | Type | Description |
|---------|------|-------------|
| **Gender** | Categorical | Student gender (Male/Female) |
| **Race/Ethnicity** | Categorical | Ethnic background (Groups A-E) |
| **Parental Education** | Categorical | Highest parental education level |
| **Lunch** | Categorical | Lunch type (Standard/Free-Reduced) |
| **Test Preparation** | Categorical | Completion of test prep course |
| **Reading Score** | Numerical | Reading test score (0-100) |
| **Writing Score** | Numerical | Writing test score (0-100) |

## 🌐 Deployment

This application is deployed on **Render.com** with automatic builds from the main branch. The deployment includes:
- Automatic dependency installation
- Python 3.11 runtime for compatibility
- Zero-downtime deployments
- HTTPS encryption
- Global CDN distribution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Kush Dev**
- GitHub: [@KushDev19](https://github.com/KushDev19)
- LinkedIn: [Connect with me](https://linkedin.com/in/kushdev19)

## 🙏 Acknowledgments

- Kaggle for the Students Performance Dataset
- Scikit-learn community for excellent ML tools
- Flask community for the lightweight web framework
- Render.com for seamless deployment experience

## 📚 Additional Resources

- [Jupyter Notebooks](./notebook/) - Detailed EDA and model experiments
- [API Documentation](./docs/api.md) - Endpoint specifications
- [Deployment Guide](./docs/deployment.md) - Step-by-step deployment instructions

---

⭐ **Star this repository if you found it helpful!**

