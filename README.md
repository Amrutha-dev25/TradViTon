# Marathi / Maharashtra-Style Saree Classifier

An academic computer-vision project (TensorFlow/Keras) that takes a saree image and answers
**MARATHI / MAHARASHTRA STYLE** or **NON-MARATHI STYLE**, with a confidence score.

> No accuracy figure is claimed anywhere in this project. Run `python src/evaluate.py` to
> see the real numbers for *your* data.

---

## 1. Project objective
Build a small, readable CNN that shows the core ideas: convolution, pooling, flattening, dense
layers, a **128-dimensional feature vector**, and a final classification layer.

## 2. What "Marathi saree classification" means here
It is a **visual** classification task: does the picture *look like* the images in your
`maharashtra` folder (drape, pattern, colour, border) compared with the images in your
`non_maharashtra` folder? It is **not** a cultural or ethnographic identifier. Regional saree
styles overlap visually (e.g. Nauvari vs. some Karnataka/Goa/Telugu drapes), and a plain
six-yard saree can look similar across regions.

## 3. Dataset folder structure
```
dataset/
├── maharashtra/        <- your ~500 Marathi images go here (class 1)
└── non_maharashtra/    <- add later: other saree styles (class 0)
```
Sub-folders are searched recursively. Supported: `.jpg .jpeg .png .webp`.

## 4. Automatic folder-based labelling
The folder name *is* the label: `maharashtra` = 1, `non_maharashtra` = 0. No CSV, no manual labels.

## 5. Why `resize`, not `reshape`
Source images have different resolutions (e.g. 900×600×3, 1920×1080×3). `numpy.reshape` only
re-arranges the *same* number of values and fails (or scrambles pixels) when the counts differ.
`cv2.resize` interpolates pixels to a new size, so any image becomes 128×128.

## 6. The 128×128×3 preprocessing
`read file → BGR→RGB → cv2.resize to 128×128 → divide by 255 → shape (128,128,3)`.
Training, evaluation, prediction and feature extraction all call the same function
(`src/preprocessing.py`). Resizing ignores the aspect ratio, so tall photos are squeezed; that
is applied identically everywhere.

## 7. CNN architecture
```
Input 128×128×3
 → Conv2D(32) → MaxPooling2D → Conv2D(64) → MaxPooling2D → Conv2D(128) → MaxPooling2D
 → Flatten → Dense(256, relu) → Dense(128, relu)  == "feature_vector"
 → Dropout(0.5) → Dense(1, sigmoid)
```
Print it with `python src/model.py`.

## 8. MaxPooling
Keeps the strongest response in each 2×2 window. It halves height/width (128→64→32→16), cuts
computation and makes the features tolerant to small shifts.

## 9. Flatten
Turns the final 16×16×128 feature maps into one 32,768-long vector so Dense layers can use it.

## 10. Dense layers
Fully connected layers that combine the visual features found by the convolutions.

## 11. The 128-dimensional feature vector
The `Dense(128)` layer is named `feature_vector`. A separate model
(`models/feature_extractor.keras`) maps image → that layer, giving a compact 128-number
description of the image (useful for similarity search or the positive-only mode).

## 12. Training
```
python src/train.py
```
Automatically detects the mode:
* **Two classes** → supervised binary CNN (Adam, binary cross-entropy, class weights,
  EarlyStopping, ModelCheckpoint, ReduceLROnPlateau).
* **One class** → positive-only fallback (section 16).

Split: 70 % train / 15 % validation / 15 % test, stratified per class. The split is saved to
`models/split_manifest.json` so evaluation uses the identical test images.

Augmentation (training only): optional horizontal flip, ±~11° rotation, ±10 % zoom, ±8 %
translation, ±15 % brightness. No vertical flips or hue shifts, which would distort the drape.
A horizontal flip changes the side of the pallu; set `USE_HORIZONTAL_FLIP = False` in
`src/config.py` if that matters to you.

**Data leakage:** byte-identical duplicates are removed, and near-identical images (perceptual
hash) are kept together in the same split. Nearly identical photos (same shoot, same saree) in
both train and test artificially inflate accuracy; also remove them by hand when you can.
Splitting by *photo shoot / person / source* is even safer than splitting by image.

## 13. Evaluation
```
python src/evaluate.py
```
Reports accuracy, precision, recall, F1, ROC AUC, confusion matrix, TP/TN/FP/FN and the
classification report; saves `outputs/confusion_matrix.png`, `outputs/evaluation_report.txt`.
Training curves: `outputs/training_history.png`. With a small test set the numbers are noisy.

## 14. External image testing
Put any image in `test_images/` (never in `dataset/`):
```
python src/predict.py --image test_images/saree1.jpg
```
If confidence < `CONFIDENCE_THRESHOLD` (0.70, in `config.py`) the answer is **UNCERTAIN**.

## 15. Feature extraction
```
python src/feature_extractor.py --image test_images/saree1.jpg --save
```
Prints `Feature vector shape: (128,)`, a short preview, and saves `outputs/saree1_feature.npy`.

## 16. Positive-only limitation
With only `maharashtra/`, a sigmoid classifier would see a single label and learn "always
Marathi". So this project does **not** train one. The fallback:
1. Train a **convolutional autoencoder** (same encoder, same 128-d `feature_vector`) on the Marathi
   images to learn a representation without any negatives.
2. Compute the **centroid** of the training feature vectors.
3. Compute the **cosine distance** of validation (held-out) Marathi images to the centroid; the
   threshold is the 95th percentile of those distances.
4. A new image is `LIKELY MARATHI` if its distance is within the threshold, otherwise
   `LIKELY NON-MARATHI / OUTSIDE TRAINING DISTRIBUTION` (`UNCERTAIN` near the boundary).

Why it is weaker: the autoencoder is rewarded for *reconstructing* images, not for separating
Marathi from other styles. It captures colours, textures, background and layout in general, so
"different from the training photos" is not the same as "not a Marathi saree" (e.g. a Marathi
saree in a different photo setting may be rejected, and a similar Kannada saree accepted). The
false-positive rate cannot be measured without negatives (you may pass
`--extra-negatives <folder>` to `evaluate.py` to estimate it).

## 17. How to add non-Marathi images
Copy other-style sarees (Kanjeevaram, Banarasi, Bengali, Gujarati Nivi, etc.) into
`dataset/non_maharashtra/`, run `python src/check_dataset.py`, then `python src/train.py` again.
Aim for a similar count and similar photo conditions as your Marathi set.

## 18. How to improve the dataset
* Match conditions between classes (background, lighting, pose, resolution, camera), or the CNN
  will learn those instead of the drape.
* Use many different people, shops and sources; avoid many photos of the same saree.
* Include full-drape views, not only close-ups of fabric.
* Remove duplicates, watermarked and tiny images.
* Include hard negatives (visually similar regional styles).
* Have a knowledgeable person verify the labels.

## 19. How to interpret confidence
Binary mode: confidence = `max(p, 1-p)` of the sigmoid output. It is *not* a calibrated
real-world probability, and CNNs can be confidently wrong on images unlike the training data
(a non-saree image can still get 90 %). Positive-only mode shows a distance, not a probability.

## 20. Limitations
Results can be affected by pose, occlusion, blouse design, pallu arrangement, image quality,
lighting, background and similar regional styles. Small datasets overfit. A 128×128 input loses
fine weave and border detail. The model can learn shortcuts (background, colour, photographer)
instead of the drape. Use `check_dataset.py --preview 12` and inspect your images.

---

## Commands (Windows / macOS / Linux)
```
python -m venv .venv
.venv\Scripts\activate            (Windows)   |   source .venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt

python src/check_dataset.py                       # inspect data (add --preview 12 / --augmentation)
python src/train.py                               # train (auto mode)
python src/evaluate.py                            # test-set metrics
python src/predict.py --image test_images/example.jpg
python src/feature_extractor.py --image test_images/example.jpg --save
python src/model.py                               # print model.summary()
```
Run everything from the `marathi_saree_classifier` folder. Use Python 3.10–3.12 (TensorFlow ≥ 2.16).
