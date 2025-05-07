# 🎥 **Face-to-Engage: AI-Driven Emotion Detection in Virtual Classrooms Using Video, CNN, and SVM**

## 📌 **Project Overview:**

**Face-to-Engage** builds an AI-powered solution to detect **engagement, boredom, frustration, and confusion** from facial expressions in virtual settings like online classrooms and meetings. Using the **DAiSEE dataset**, we applied and compared **handcrafted feature extraction (Dlib 68 facial landmarks)** and **CNN-based deep features**, feeding both into **SVM classifiers** to evaluate emotion recognition performance.

This project has practical applications for **e-learning platforms, virtual training, and online collaboration tools** aiming to measure real-time user attention and emotional states.

---

## 🏗️ **Tech Stack & Their Roles:**

✅ **AWS S3** – Stored 15GB DAiSEE video dataset and processed image frames  
✅ **AWS SageMaker** – Modeled, trained, and evaluated ML pipelines via Jupyter notebooks  
✅ **Python** – Core programming language for data wrangling, modeling, evaluation  
✅ **TensorFlow (Keras API)** – Built custom CNN architecture for deep feature extraction  
✅ **OpenCV** – Processed video frames: reading, writing, image manipulation  
✅ **Dlib** – Extracted **68 facial landmarks** as handcrafted features  
✅ **Scikit-learn (SVM-SVC)** – Classifier for predicting emotional states  
✅ **Matplotlib** – Visualized training loss, validation loss, confusion matrices  

---

## 💡 **What Makes This Project Unique:**

✨ Explored **dual feature extraction pipelines** → handcrafted features via Dlib vs CNN features via Keras  
✨ Tackled **multi-label, multi-intensity classification problem** (emotion + intensity levels 0–3)  
✨ Deployed and ran models fully on **AWS cloud (S3 + SageMaker)**  
✨ Compared per-emotion accuracy (engagement, boredom, confusion, frustration) with actionable visualizations  
✨ Evaluated **interpretable models (SVM) instead of black-box deep neural networks** for better explainability

---

## 📊 **Key Results & Insights:**

✔️ **CNN+SVM outperformed handcrafted features → 50% accuracy vs 45%**  
✔️ Frustration detection achieved **highest individual class accuracy (80%)**  
✔️ Confusion was **most difficult emotion to classify** across both models  
✔️ Validation loss higher than training → dataset underfitting, room for improvement  
✔️ Combining CNN + handcrafted features (hybrid models in literature) shows benchmark accuracy of 53%

---

## ✅ **What Went Well:**

🌟 Successfully implemented **two complete ML pipelines: Dlib landmarks + SVM, CNN features + SVM**  
🌟 Leveraged **AWS SageMaker and S3 for scalable cloud ML workflow**  
🌟 Achieved consistent performance despite small, noisy dataset  
🌟 Generated per-emotion **confusion matrices and insights into which emotions are harder to predict**

---

## ⚠️ **Challenges:**

😅 Processing **large video dataset (15GB)** → required frame extraction and batching  
😅 **Parsing nested intensity labels** for multi-class SVM setup  
😅 SVM doesn’t natively handle multi-label → implemented **one-vs-one classifiers per emotion**  
😅 Limited annotated data → test set excluded due to missing labels  
😅 Balancing training/validation splits to handle **class imbalance**

---

## 🧩 **Methodology Overview:**

1. Converted video → image frames
2. **Facial Landmark Pipeline:**
   - Extracted **68 landmarks via Dlib**
   - Computed Euclidean distances
   - Fed vectors into **SVM-SVC classifier**
3. **CNN Pipeline:**
   - Built CNN (Conv2D layers, LeakyReLU, Softmax)
   - Extracted deep features
   - Fed features into **SVM-SVC classifier**
4. Evaluated both models on accuracy, precision, recall, confusion matrices

---

## 📝 **Evaluation Metrics:**

| Model                     | Accuracy | Precision | Recall |
|--------------------------|----------|-----------|--------|
| 68 Facial Landmarks + SVM | 0.45     | 0.45      | 0.45   |
| CNN Features + SVM        | 0.50     | 0.50      | 0.50   |

✅ Frustration class achieved best accuracy; confusion class remained hardest to detect.

