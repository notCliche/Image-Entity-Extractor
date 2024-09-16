# Image-Entity-Extractor
# ML Approach Documentation

## 1. Problem Overview

The task at hand is to extract entity values from product images. These entities include physical attributes such as width, depth, height, weight, voltage, wattage, and volume. This capability is crucial for e-commerce platforms, content moderation, and various other applications where detailed product information is essential.

## 2. ML Approach

### 2.1 Model Architecture

For this task, we've chosen to use a Vision Transformer (ViT) model. Specifically, we're using the `google/vit-base-patch16-224` pretrained model from the Hugging Face Transformers library.

Vision Transformers have shown remarkable performance in image classification tasks and can be adapted for our specific use case of entity extraction. The ViT model processes images as a sequence of fixed-size patches, applying self-attention mechanisms to capture global dependencies in the image.

### 2.2 Model Adaptation

While the original ViT model is designed for image classification, we've adapted it for our multi-label classification task:

1. We've replaced the final classification layer with a new linear layer that outputs probabilities for each possible unit across all entity types.
2. The number of output units is determined by summing the number of possible units for each entity type as defined in our `ENTITY_UNIT_MAP`.

### 2.3 Data Preprocessing

Images are preprocessed using the following steps:

1. Resizing to 224x224 pixels (the input size expected by the ViT model).
2. Converting to RGB format to ensure consistency.
3. Normalizing pixel values using mean and standard deviation values typical for ImageNet-trained models.

### 2.4 Inference

During inference:
1. The image is preprocessed and passed through the model.
2. The model outputs logits for each possible unit.
3. We select the unit with the highest probability.

## 3. Evaluation

The model's performance is evaluated using the F1 score, which is the harmonic mean of precision and recall. This metric is particularly useful for imbalanced datasets and multi-label classification problems.
We've implemented a custom F1 score calculation that takes into account the specific requirements of our task, including handling empty predictions and mismatched units.

## 4. Challenges and Future Improvements

1. **Number Prediction**: The current implementation uses random number generation as a placeholder. Developing an accurate method for predicting the numeric value is a crucial next step.
2. **Multi-task Learning**: Instead of treating this as a single multi-label classification problem, we could explore multi-task learning approaches where we have separate outputs for entity type and unit prediction.
3. **Data Augmentation**: Implementing data augmentation techniques such as rotations, flips, and color jittering could help improve the model's robustness.
4. **Ensemble Methods**: Combining predictions from multiple models (e.g., ViT, ResNet, EfficientNet) could potentially improve overall accuracy.
5. **Optical Character Recognition (OCR)**: Incorporating OCR techniques could help in directly reading numeric values and units from product images.
6. **Few-shot Learning**: Given the potential for new product categories and units, exploring few-shot learning techniques could help the model adapt more quickly to new entities.

## 5. Contributors
- Om Prakash Behera
- Sai Nikhita Palisetty
- Sayali Khamitkar
- Harsh Maurya
