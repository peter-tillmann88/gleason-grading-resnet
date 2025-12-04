# gleason-grading-resnet
Deep CNN (ResNet-50) for classifying Gleason grades (G3, G4, G5, and NC) on the SICAPv2 prostate histopathology dataset

Dataset from: https://data.mendeley.com/datasets/9xxm58dvs3/1

How to run: 

1. Download Dataset from above link and keep note of where it is located on computer.

2. Open prepare_sicapv2.py
    - Update the input directory (line 14, data_dir) to where you downloaded that dataset
    - Update the output directory (line 19, out_dir) to a folder where processed ImageFolder data will go.

3. Run prepare_sicapv2.py

4. Open train_resnet.py or train_alexnet.py
   - Update the input (line 12, DATA_DIR) to where the preprocessed data is located from step 3.
  
5. Run train_resnet.py or train_alexnet.py.

6. Run evaluate_resnet.py or evaluate_alexnet.py depending on which train file you ran in step 5. 
