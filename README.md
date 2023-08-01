# trimble_test

## Method
1. cross validation to choose the right pretrained model based on accuracy and F1 score (model_choice.py): I used all the training dataset for this purpose and tested many pretrained models from pytorch image models(timm), for each model the dataset is splitted into folds (one for test and the others for training). The best results obtained with efficientnet_b5. so for the rest of experiments I used this model
2. training (train.py): the dataset is splitted into training and validation (0.7, 0.3) and I used also weighted loss because the dataset is unbalanced, road images are almost twice field images that's why I used pos_weight=2 in the loss function. I also used early stopping to prevent model overfiting.
3. test(test.py): test using the best weights(that give best loss on validation data) results are saved into image
4. inference (inference.py): to test a specific image
## Results
### Results on test images
![test_results](https://github.com/melloulid/trimble_test/assets/141152003/5ed7c57d-b047-4216-8eeb-11da3d5bc34b)
### Results on other images
