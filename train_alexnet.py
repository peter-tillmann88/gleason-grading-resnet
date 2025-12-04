import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

#Config

#change this dpeenidng on where yours is 
DATA_DIR = r"C:\Users\Peter\Desktop\SICAPv2_imagefolder"

#where to save checkpoints 
OUTPUT_DIR = "checkpoints"

#basic hpyerparameters 
BATCH_SIZE = 16 #how many images per batch
EPOCHS = 20 #how many full passes over training data
LR = 3e-4 #learning rate for optimizer  
WEIGHT_DECAY = 1e-4 #regulaitation strength     
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

#training script 
def main(): 
    os.makedirs(OUTPUT_DIR, exist_ok = True) 

    #just check if device running has a GPU to make it go faster 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #standard transformers w additional rezisedcrop to ensure model sees different things each epoch, and flips to prevent overfitting
    #basically just defining how preprocess and augmentation for images should work 
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
    }

    #dataset loaders 

    #use imagefolder, expecting the proper folder structure laid out by prepare_sicapv2
    dset = { split: datasets.ImageFolder(os.path.join(DATA_DIR, split), data_transforms[split])
               
               for split in ["train", "val"]
               }
    #wrap dataset in loaders to get batches 
    dataset_loaders = {split : DataLoader(dset[split], batch_size=BATCH_SIZE, shuffle = (split == "train"), num_workers= NUM_WORKERS)
                      
                      for split in ["train", "val"]
                      }
    
    #store the dataseet sizes and class names 
    dset_sizes = {split: len(dset[split]) for split in ["train", "val"]}
    class_names = dset["train"].classes
    num_classes = len(class_names)

    #model/loss/optimizer



    model = build_model(num_classes)
    model = model.to(device)

    #define cross entropy loss for multiclass classification 
    criterion = nn.CrossEntropyLoss()

    #adamW optimizer with learning rate, and weight decay 
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay= WEIGHT_DECAY)

    #track the best validation accuracyt and epoch for checkpointing 
    best_val_acc = 0.0
    best_epoch = -1
    best_path = os.path.join(OUTPUT_DIR, "best_model_alexnet.pt")

    #training loop
    for epoch in range(1, EPOCHS + 1): 
        print(f"\nepoch {epoch}/{EPOCHS}")

        #each epochs training and val stage 
        for phase in ["train", "val"]:
            if phase == "train": 
                model.train() #set model to train
            else: 
                model.eval() #set model to eval mode 
            
            running_loss = 0.0
            running_corrects = 0

            #loop over all batches in this phase 
            for inputs, labels in dataset_loaders[phase]: 
                
                #this just moves the inputs/labels to CPU/GPU depending what you have 
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #zero out graidents from previous step 
                optimizer.zero_grad()

                #forweard pass and backward passs if in training 
                if phase == "train": 
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else : 
                    #no gradients needed for validation 
                    with torch.no_grad(): 
                        outputs = model(inputs)
                        loss = criterion(outputs, labels) 

                #puts model outputs to predicted class indices 
                _, preds = torch.max(outputs, 1)

                #accumulate loss and # of correct predicitons 
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += (preds == labels).sum().item() 

            #compute avg loss and accuracy for this phase 
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Accuracy:{epoch_acc:.4f}")



            #track best val model
            #if validation and accuracy have improved save the new model checkpoint 
            if phase == "val" and epoch_acc > best_val_acc: 
                best_val_acc = epoch_acc 
                best_epoch = epoch 
                torch.save( { "epoch": epoch, "model_state_dict": model.state_dict(), "classes": class_names,}, best_path)

                print(f"best model saved (val_acc= {epoch_acc:.4f})")
    
    print(f"Training finished. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print("Best model checkpoint saved to: ", best_path)


if __name__ == "__main__":
    main()

                




