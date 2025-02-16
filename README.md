# Computer-Vision

## 🎯 研究所電腦視覺課程作業專案
此專案展示研究所 **電腦視覺 (Computer Vision)** 課程的作業內容與結果。每個作業 (HW) 皆包含 **Motivation（動機）、Idea（想法）、Technical Challenges（技術挑戰）、Results（結果）** 等，並附上相關參考資料。

---

## 📂 專案內容
- [HW1](#hw1)：車輛與行人偵測與追蹤
- [HW2](#hw2)：車牌辨識
- [HW3](#hw3)：森林火災影像分類
- [HW4](#hw4)：Stable Diffusion 訓練 Pokemon 風格模型

---

## 📌 HW1：車輛與行人偵測與追蹤

📌 **參考來源**：[YouTube 教學影片](https://www.youtube.com/watch?v=O3b8lVF93jU)

### 🔹 Motivation（動機）
台灣的交通壅塞與秩序問題普遍存在，透過提升 **車輛與行人追蹤技術**，可以幫助交通管理、減少擁擠、提升城市安全，並促進智慧城市發展。

### 🔹 Idea（概念）
本作業主要參考 OpenCV 提供的 **背景去除演算法**，特別是 **MOG2** 來偵測並追蹤車輛與行人，希望透過參數調整與方法改良來提升偵測準確度。

### 🔹 Technical Challenges（技術挑戰與解決方案）
- **挑戰**：MOG2 受 **光影變化** 影響較大，導致誤判。
- **解決方案**：
  1. 測試其他演算法，如 **KNN 背景去除**，以提升準確度。
  2. 調整參數，以減少光影影響，提高追蹤精度。

### 🔹 Results（結果與發現）
- 儘管進行了參數調整與演算法替換，但光影影響仍然存在。
- 物件偵測與追蹤的準確度表現不錯，顯示選擇合適的演算法與參數微調對系統效能有幫助。

### 🔹 Contribution & Impact（貢獻與影響）
- **貢獻**：比較不同方法提升車輛與行人追蹤的準確性。
- **影響**：可應用於 **智慧交通管理、城市規劃**，提高城市安全性。

---

## 📌 HW2：車牌辨識

📌 **參考來源**：[YouTube 教學影片](https://www.youtube.com/watch?v=NApYP_5wlKY)

### 🔹 Motivation（動機）
車牌辨識技術已廣泛應用於 **停車管理、交通違規監測** 等領域，但傳統的人工收費方式費時費力，若能全面自動化，可大幅節省資源。

### 🔹 Idea（概念）
- 使用 **YOLO** 進行車輛偵測，但 YOLO 無法辨識文字，故需結合 **OCR（光學字符辨識）** 來讀取車牌文字。
- 透過 **OpenCV + EasyOCR** 來實現車牌辨識，而不依賴預訓練模型。

### 🔹 Technical Challenges（技術挑戰與解決方案）
- **挑戰**：如何處理傾斜或模糊的車牌？
- **解決方案**：
  1. **透視變換** (Perspective Transform)：校正傾斜的車牌。
  2. **Canny 邊緣偵測** 閥值調整，提高辨識準確率。

### 🔹 Results（結果與發現）
- 針對 **正面車牌**，系統辨識效果良好。
- 偏斜或模糊的車牌辨識效果較差，未來可嘗試 **專用的車牌辨識模型** 來提升準確度。

### 🔹 Contribution & Impact（貢獻與影響）
- **貢獻**：探索無需訓練模型的車牌辨識方式。
- **影響**：自動化車牌辨識可減少人力成本，提高停車與交通管理效率。

---

## 📌 HW3：森林火災影像分類

📌 **數據集來源**：[Kaggle - Forest Fire Images](https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images/data)

### 🔹 Motivation（動機）
森林火災在台灣較少發生，但在澳洲等國家卻是嚴重的災害。因此，**早期偵測** 對於減少損害至關重要。

### 🔹 Idea（概念）
- 透過 **卷積神經網絡 (CNN)** 訓練影像分類模型。
- 使用 **數據增強技術**（Data Augmentation）提升模型的泛化能力。

### 🔹 Technical Challenges（技術挑戰與解決方案）
- 無重大技術困難，但數據集中部分影像損壞，需先行清理。

### 🔹 Results（結果與發現）
- 訓練準確度達 **97%**，驗證準確度約 **93%**。
- 但在 **紅色調影像**（如楓葉林）容易誤判，可能是因為訓練數據主要為綠色森林。

### 🔹 Contribution & Impact（貢獻與影響）
- **貢獻**：透過數據增強技術提高森林火災辨識準確率。
- **影響**：若擴充數據集並持續改進，該技術可應用於 **災害預警系統**。

---

## 📌 HW4：Stable Diffusion 訓練 Pokemon 風格模型

📌 **模型來源**：[Hugging Face - SD Pokemon](https://huggingface.co/lambdalabs/sd-pokemon-diffusers)  
📌 **數據集來源**：[Hugging Face - Pokemon Dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)

### 🔹 Motivation（動機）
本作業選擇訓練 **Stable Diffusion** 模型，以 **Pokemon 風格** 的圖片進行訓練，目標是透過 **文字 (prompt) 輸入** 來生成不同風格的寶可夢圖片。

### 🔹 Results（結果示例）
![](https://github.com/pincheng0523/Computer-Vision/blob/main/HW4/totoro.png)  
**Totoro**

| ![](https://github.com/pincheng0523/Computer-Vision/blob/main/HW4/dog.png) | ![](https://github.com/pincheng0523/Computer-Vision/blob/main/HW4/cat.png) |
|:--:|:--:|
| **Dog** | **Cat** |


---

## 📌 結論
本專案涵蓋了 **車輛追蹤、車牌辨識、影像分類、Stable Diffusion 生成影像** 等多個電腦視覺應用，並透過不同技術探索解決方案。未來可進一步提升演算法效能，並整合更先進的 AI 模型來解決各種挑戰。 🚀

