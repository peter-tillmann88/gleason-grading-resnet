import os
import shutil
import pandas as pd


#in order to use pytorch imagefolder trainer we need file structure like this 
#basically all im doing here is take the SICAPv2 partition tables (Val1 + Test) and physically organize the image files into the folder structure the CNN expects 


#edit these paths for ur personal machine

#where you downloaded sicapv2
#edit this line
data_dir =  r"C:\Users\Peter\Desktop\Data\SICAPv2"

#where the train/val/test split is 
#you need a folder with Sicapv2_imagefolder contain train/test/val each of those files containing non_cancerous/gleason_3/gleason_4/gleason_5
#edit this line 
out_dir  = r"C:\Users\Peter\Desktop\SICAPv2_imagefolder"  

#path to raw patch images 
images_dir = os.path.join(data_dir, "images" )

#path to those partitions 
partition_dir = os.path.join(data_dir,  "partition" ) 


#map index (0,1,2,3) to class folder name 
label_names = { 0: "non_cancerous",   1: "gleason_3", 2: "gleason_4",  3:  "gleason_5", }

#use val/val1 as train+val split 
train_table_path  = os.path.join(partition_dir,  "Validation", "Val1",  "Train.xlsx")
val_table_path   =  os.path.join(partition_dir,   "Validation", "Val1", "Test.xlsx")

# final test set from test/test,xslx
test_table_path   = os.path.join(partition_dir,  "Test", "Test.xlsx")

#given row with NC/G3/G4/G5 figure out which class it belongs to 
def row_to_class(row): 
    #convert one row from the excel into a single class label. each row has four columns NC,G3,G4,G5 one of them tells us gleason grade.
    
    flags = [row["NC"],  row["G3"],  row["G4"], row["G5"]]

    #if multilabbeleld or unlabelled skip it 
    if sum(flags)  != 1:

        return None 
    
    #get index ex. 0 -> NC, 1 -> G3. etc
    idx =  flags.index(1)

    #return the folder name we want to use for this cclass 
    return  label_names[idx]


#take a df and copy each image into the right class folder 
def copy_split_rows(df, split_name):


    #basically for every patch listed in the excel file run the code, wehre _ is the row index (we do not need it) r is the actual row 
    for _, r in  df.iterrows():

        #take the file name associate with the image_name column in that row 
        img_name =  r["image_name"]

        #cls just becomes the string name of class folder noncanerous/G3/G4/G5 
        cls =  row_to_class(r)

        #forget labels that are unlabbeled 
        if cls is None:

            print(" skipping (bad label): ", img_name)
            continue

        #just specified where that image should be copied to what folder in particular 
        src =  os.path.join(images_dir,  img_name) #where file is 
        dst_folder =  os.path.join(out_dir,  split_name, cls) #which split/class foldder 
        #create split/class folders 
        os.makedirs(dst_folder, exist_ok=True)

        dst =  os.path.join(dst_folder, img_name)#full dest path


        
        #copy utility basically just takes file from src and writes it to dst 
        shutil.copy2(src,  dst)

def main():

    #create the root output folder 
    os.makedirs(out_dir, exist_ok=True)
    print( "loading train/test tables")

    #read all excel files that define which images belong to which split
    train_df  = pd.read_excel(train_table_path)
    test_df  =  pd.read_excel(test_table_path)
    val_df   = pd.read_excel(val_table_path)  

    #only keep basic grade labels, drop all other columns 
    keep_cols = ["image_name", "NC", "G3", "G4", "G5"]
    train_df =  train_df[keep_cols]
    test_df  = test_df[keep_cols]
    val_df   = val_df[keep_cols]


    print(f"train rows: {len(train_df)}   val rows: {len(val_df)}  test rows:  {len(test_df)}")

    #build imagefolder structuere for each split 
    print("copying train images")
    copy_split_rows(train_df, "train")

    print("copying val images")
    copy_split_rows(val_df, "val")

    print("copying test images")
    copy_split_rows(test_df, "test")


    print("  imagefolder dataset at: ", out_dir)

if __name__ == "__main__":
    main()
