Scene Classification with Bag of Visual Words and Spatial Pyramids

------------------------------------------------------------

OVERVIEW

This project implements a scene classification pipeline using local image
descriptors and histogram-based representations. Images are described using
SIFT features, quantized into a Bag of Visual Words (BoW) representation via
k-means clustering. Classification performance is evaluated using multiple
models, including a nearest-neighbor baseline, a linear SVM, and a
chi-squared kernel SVM combined with a spatial pyramid representation.

The goal of the project is to compare a simple BoW baseline against a
spatially enhanced representation and a kernel better suited for histogram
data.

------------------------------------------------------------

PROJECT STRUCTURE

data/
not included due to copyright concerns

src/
Source code for feature extraction, vocabulary construction, encoding, and
classification.

outputs/
Generated descriptors, visual vocabularies, encoded histograms, and result
figures.

outputs/descriptors/
Saved SIFT keypoints and descriptors for each image.

outputs/vocab/
Learned visual vocabularies.

outputs/histograms/
Encoded BoW and spatial pyramid histograms and labels.

outputs/results/
Accuracy plots and confusion matrices for the final models.

------------------------------------------------------------

DEPENDENCIES

Required Python packages:

- numpy
- opencv-python
- scikit-learn
- matplotlib
- tqdm

Install all dependencies with:

pip install -r requirements.txt

------------------------------------------------------------

HOW TO RUN

Run the full pipeline from the project root directory in the following order:

1. Extract SIFT descriptors
   python src/extract_sift.py

2. Build the visual vocabulary (K = 400)
   python src/build_vocab.py 400

3. Encode images using Bag of Visual Words with spatial pyramid
   python src/encode_bow.py 400

4. Run baseline classifiers (Nearest Neighbor and Linear SVM, plain BoW)
   python src/classify_plain.py 400 1

5. Run chi-squared kernel SVM with spatial pyramid
   python src/classify_chi2.py 400 1

------------------------------------------------------------

FEATURE REPRESENTATIONS

1. Plain Bag of Visual Words
   - Single global histogram per image
   - L1-normalized

2. Spatial Pyramid Bag of Visual Words
   - 1x1 global histogram
   - 2x2 spatial grid
   - Concatenated and normalized histogram

------------------------------------------------------------

MODELS EVALUATED

1. Nearest Neighbor (baseline)
   - Feature representation: Plain BoW
   - Distance metric: Euclidean

2. Linear Support Vector Machine
   - Feature representation: Plain BoW
   - Kernel: Linear

3. Chi-squared Kernel SVM
   - Feature representation: Spatial Pyramid BoW
   - Kernel: Chi-squared (precomputed)

------------------------------------------------------------

RESULTS

All results reported below use a vocabulary size of K = 400:

- Nearest Neighbor (BoW): 0.307
- Linear SVM (BoW): 0.426
- Chi-squared SVM + Spatial Pyramid: 0.610

The chi-squared kernel combined with a spatial pyramid representation
provides a significant improvement over the plain BoW baselines by
preserving coarse spatial layout and using a kernel well suited to histogram
features.

------------------------------------------------------------

OUTPUT FILES

Final figures and results are stored in outputs/results/, including:

- Confusion matrices for all evaluated models
- Accuracy plots for selected parameter settings

Only final results relevant to model comparison are included in the
submission.

------------------------------------------------------------

NOTES

- The visual vocabulary is learned using descriptors from the training set
  only.
- Parameter sweeps over vocabulary size and SVM regularization were performed
  during development but are not included in the final submission.
- Intermediate files generated during experimentation were removed to keep
  the repository clean and focused on final results.

------------------------------------------------------------

END OF README
