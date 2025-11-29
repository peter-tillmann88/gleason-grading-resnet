import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
)
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


#config 
#change this dpeenidng on where yours is 
DATA_DIR = r"C:\Users\Peter\Desktop\SICAPv2_imagefolder"

#where the best model checkpoint was saved 
OUTPUT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model_alexnet.pt")

#basic hpyerparameters 
BATCH_SIZE = 16 #how many images per batch    
NUM_WORKERS = 4  #number of dataloader worker processes 


def build_model(num_classes) :
    
    #load alexnet w imagenet weights 
    base = models.alexnet(weights =  models.AlexNet_Weights.IMAGENET1K_V1)

    # numnber of features coming out of the oringinal final fc layer 
    in_features = base.classifier[6].in_features

    #replace the oringinal classification layer with identity
    #basically replace the final 1000-class head layer with a nothing layer 
    base.classifier[6] = nn.Identity()

    #our own classifier which maps alexnet feature to num_classes 
    classifier = nn.Linear(in_features, num_classes)


    #final model
    model = nn.Sequential(base, classifier)
    return model 

#create dataloader for the test split, using simialr steps as training/validation 
def get_test_loader(): 
    test_transform =  transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
    ])
    #image folder 
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), test_transform)

    test_loader =  DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = False, num_workers= NUM_WORKERS)

    return test_dataset, test_loader


#evaluation 
def evaluate_metrics(model, loader, class_names, device): 
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    #no gradient tracking for evaluation 
    with torch.no_grad(): 
        #iterate through the dataloader 
        for inputs, labels in loader: 
            inputs = inputs.to(device) #use gpu if available 
            labels = labels.to(device) 

            #forward pass through model get unnormalized scores 
            outputs = model(inputs)
            
            #covert unnormalized scores into probabilties per class
            probs = torch.softmax(outputs, dim=1)

            #for each image find index of largest probability (0 = noncanersou, 1 = gleason_3, 2 = gleason_4, 3 = gleason_5)
            _, preds = torch.max(probs, 1)

            #store true labels
            all_labels.append(labels.cpu().numpy())

            #store predicted labels 
            all_preds.append(preds.cpu().numpy())

            #store predicted probabilties 
            all_probs.append(probs.cpu().numpy())

    #concatenatte into array form needed for Sklearn 
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs) 

        #basic accuracy 
    acc = (y_true == y_pred).mean()
    print(f"test accuracy: {acc:.4f}")

        #classification report 
    print("classification report")
    print(classification_report(y_true, y_pred, target_names=class_names))

        #confusion matrix
    confm = confusion_matrix(y_true, y_pred)
    print("confusion matrix")
    print(confm)

        #area under curve
    auc_marco = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    print(f"\nMacro ROC-AUC: {auc_marco:.4f}")

        #quadratic weighted cappa 
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    print(f"quadaratic weighted kappa: {kappa:.4f}")

    return y_true, y_pred, y_prob, confm
    
#get the extracted features and clusteirng
def extract_features_pca(model, loader, class_names, device): 

    #enage evaluation mode 
    model.eval() 

    #these hold all feature vectors, and labels for test set 
    features = []
    labels_list = []

    # the first part of alexnet
    backbone = model[0]

    #do not track gradients for testing 
    with torch.no_grad(): 
        #iterate over dataloader (inputs = images, labels = integers)
        for inputs, labels in loader: 

            #move imges onto same device as model 
            inputs = inputs.to(device) 
            
            #pass through backbone to get features, this gets feature vector per image inst of classification  
            feats = backbone(inputs) 

            #flatten out the baackbone 
            feats = feats.view(feats.size(0), -1) 

            #move back to cpu from gpu (memory problems) 
            features.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    #turn again into long array X = (# of test images, feature dimension) Y= (int labels) 
    X = np.concatenate(features, axis =0 )
    y = np.concatenate(labels_list, axis= 0)


    #project using pca to 2d space 
    pca = PCA(n_components=2, random_state=0)
    X_2d = pca.fit_transform(X)


    plt.figure(figsize=(6,6))

    #loop over each class and plot its ponits 
    for cls_idx in np.unique(y):
        #find which points below to this class if true= sample is this class, false otherwise 
        idx = (y == cls_idx)

        #draw these points as one group on the plot 
        #dot size s = 8 
        #visitbiltiy = 0.7 
        plt.scatter(X_2d[idx, 0], X_2d[idx,1], s=8, label=class_names[cls_idx], alpha=0.7,) 
    
    plt.title("PCA of AlexNet features on SICAPv2 (test set)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "feature_pca_alexnet.png")
    plt.savefig(out_path, dpi = 200)
    plt.close()

    print("saved pca plot")


#main
def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load test data
    test_dataset, test_loader = get_test_loader()
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    #rebuild model and load the trained weights from train_alexnet 
    model = build_model(num_classes).to(device)

    #use model trained in train_alexnet.py
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    #eval metrics
    y_true, y_pred, y_prob, confm = evaluate_metrics(model, test_loader, class_names, device,)

    #extract features for clsutering visuale
    extract_features_pca(model, test_loader, class_names, device,)


if __name__ == "__main__":
    main()



 