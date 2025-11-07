from pathlib import Path 
import random, shutil
import uuid

def split_dataset(src, dst, train_p=0.8, val_p=0.1, test_p=0.1):
    src = Path(src)
    dst = Path(dst)
    IMG_EXS = {".jpg", ".jpeg", ".png"}


    for class_dir in src.iterdir():
        print('Class Courrant =', class_dir.name)
        if(class_dir.is_dir):
            images = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXS]
            random.shuffle(images) #mixage d'image du repertoire

            nbrImg = len(images)
            print(f"nombre d'image: {nbrImg}")
            nbr_train = int(nbrImg * train_p)
            print(f"nombre d'image pour entrainement: {nbr_train}")
            n_val = int(nbrImg * val_p)

            # data_split = {
            #      'train': images[:nbr_train],
            #      'val' : images[:nbr_train+n_val],
            #      'test': images[:nbr_train+n_val]
            # }
            data_split = {
            'train': images[:nbr_train],
            'val': images[nbr_train:nbr_train + n_val],
            'test': images[nbr_train + n_val:]
             }
            print(f"Total images: {len(images)}")
            print(f"Train: {len(data_split['train'])}")
            print(f"Val: {len(data_split['val'])}")
            print(f"Test: {len(data_split['test'])}")
            print(f"Somme: {sum(len(v) for v in data_split.values())}")

            for split_item, imgs in data_split.items():
                for img in imgs:
                    dest = dst / split_item / class_dir.name / img.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if dest.exists():
                         unique_id = uuid.uuid4()
                         dest = dst / split_item / class_dir.name / str(unique_id)+img.name
                         dest.parent.mkdir(parents=True, exist_ok=True)

                        print(f"Fichier déjà existant : {dest.name} à été renomé en ")
                        shutil.copy2(img, dest)
                    except Exception as e:
                        print(f"Erreur sur {img}: {e}")
                
        
    print("Separation (Train, Validation et test) et Copy Terminée !")
            

split_dataset("data", "dataset")