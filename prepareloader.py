import torch
import json
from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
    load_decathlon_datalist,
    ThreadDataLoader,
    decollate_batch,
    list_data_collate,
)
from transform.transform import train_transforms, val_transforms, test_transforms


# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, dataframe, transform):
#         self.dataframe = dataframe
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, index):
#         row = self.dataframe.iloc[index]
#         return (
#             self.transform((Image.open(row["file_path"]))),
#             row["class"])



def prepare_loaders(fold, df, debug=False):

    train_df = df.query("fold!=@fold&fold!=4").reset_index(drop=True)
    train_df = train_df.loc[:,["image_path","mask_multiclass_path"]]
    train_img = train_df['image_path'].values.tolist()
    train_msk = train_df['mask_multiclass_path'].values.tolist()


    valid_df = df.query("fold==@fold").reset_index(drop=True)
    valid_df = valid_df.loc[:,["image_path","mask_multiclass_path"]]
    val_img = valid_df['image_path'].values.tolist()
    val_msk = valid_df['mask_multiclass_path'].values.tolist()

    test_df = df.query("fold==4").reset_index(drop=True)
    test_df = test_df.loc[:,["image_path","mask_multiclass_path"]]
    test_img = test_df['image_path'].values.tolist()
    test_msk = test_df['mask_multiclass_path'].values.tolist()

    # print(val_json)


    train_dicts = [{'image': image_name, 'label': label_name}
                  for image_name, label_name in zip(train_img, train_msk)]

    val_dicts = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(val_img, val_msk)]

    test_dicts = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(test_img, test_msk)]
    
    # print(test_dicts)
    print("train data:", len(train_dicts), "val data", len(val_dicts), "test data:", len(test_dicts))
    myDict = {
        "description": "umwgi",
        "labels": {
            "0":"Background",
            "1":"large bowel",
            "2":"small bowel",
            "3":"stomach"
        },
        "licence":"yt",
        "moality": {
            "0":"CT"
        },
        "name": "umwgi",
        "numTest": 20,
        "numTraining": 80,
        "reference": "uw-university",
        "release": "1.0 06/08/2022",
        "tensorImageSize": "3D",
        "training": train_dicts,
        "validation": val_dicts,
        "test": test_dicts
    }
    json_object = json.dumps(myDict, indent = 4) 
    with open(f"dataset.json", 'w') as f:
        json.dump(myDict, f)

    
    datasets = "./dataset.json"
    train_json = load_decathlon_datalist(datasets, True, "training")
    val_json = load_decathlon_datalist(datasets, True, "validation")
    test_json = load_decathlon_datalist(datasets, True, "test")

    # print(valid_df)
    # if debug:
    #     train_json = train_json.head(32*5).query("empty==0")
    #     val_json = val_json.head(32*5).query("empty==0")

    # train_dataset = Dataset(train_json, transform= train_transforms)
    # valid_dataset = Dataset(val_json, transform= val_transforms)

    train_dataset = CacheDataset(data=train_json, transform=train_transforms, cache_num=24, cache_rate=1.0, num_workers=8,)
    valid_dataset = CacheDataset(data=val_json, transform=val_transforms, cache_num=24, cache_rate=1.0, num_workers=8,)
    test_dataset = CacheDataset(data=test_json, transform=test_transforms, cache_num=24, cache_rate=1.0, num_workers=8,)
    train_loader = ThreadDataLoader(train_dataset, num_workers=8, batch_size=1, shuffle=True)
    val_loader = ThreadDataLoader(valid_dataset, num_workers=4, batch_size=1)  
    test_loader = ThreadDataLoader(test_dataset, num_workers=4, batch_size=1) 
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, drop_last=False, persistent_workers=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, drop_last=False, persistent_workers=True)

    return train_loader, val_loader, test_loader, test_dataset